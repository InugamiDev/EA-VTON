# intent: two-tower embedding model for garment recommendation with FAISS ANN retrieval
# status: done
# next: integrate with size_mlp.py for size-aware re-ranking, build training pipeline
# blockers: none
# confidence: high

"""Two-tower recommendation model for garment retrieval.

Architecture follows DSSM / YouTube DNN style: separate user and item towers
produce 64-d L2-normalised embeddings, scored via temperature-scaled dot
product and trained with InfoNCE (in-batch negatives).

At serving time, item embeddings are pre-indexed in a FAISS IVF index for
sub-millisecond approximate nearest-neighbour retrieval.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

USER_BODY_DIM: int = 128
USER_BEHAVIOR_DIM: int = 64
USER_INPUT_DIM: int = USER_BODY_DIM + USER_BEHAVIOR_DIM  # 192

ITEM_RESNET_DIM: int = 256
ITEM_CATEGORY_DIM: int = 13
ITEM_SIZE_DIM: int = 7
ITEM_COLOR_DIM: int = 32
ITEM_INPUT_DIM: int = (
    ITEM_RESNET_DIM + ITEM_CATEGORY_DIM + ITEM_SIZE_DIM + ITEM_COLOR_DIM
)  # 308

EMBED_DIM: int = 64
TAU: float = 0.07


# ------------------------------------------------------------------
# Towers
# ------------------------------------------------------------------

class UserTower(nn.Module):
    """Encodes user body shape + browsing behaviour into a 64-d embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(USER_INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, EMBED_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised user embedding (B, 64)."""
        return F.normalize(self.net(x), p=2, dim=-1)


class ItemTower(nn.Module):
    """Encodes garment visual + categorical features into a 64-d embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ITEM_INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, EMBED_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised item embedding (B, 64)."""
        return F.normalize(self.net(x), p=2, dim=-1)


# ------------------------------------------------------------------
# Two-tower model
# ------------------------------------------------------------------

class TwoTowerModel(nn.Module):
    """Two-tower retrieval model with InfoNCE training and FAISS serving."""

    _instance: Optional["TwoTowerModel"] = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        super().__init__()
        self.user_tower = UserTower()
        self.item_tower = ItemTower()
        self.tau = TAU

        # Populated at serving time via load_index()
        self._faiss_index = None
        self._item_ids: list[str] = []
        self._item_metadata: list[dict] = []

    # ------------------------------------------------------------------
    # Singleton lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "TwoTowerModel":
        """Return the singleton, loading weights + FAISS index on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    model = cls()
                    model_dir = os.environ.get("MODEL_DIR")
                    if model_dir:
                        weights_path = Path(model_dir) / "two_tower.pt"
                        if weights_path.exists():
                            state = torch.load(
                                weights_path,
                                map_location="cpu",
                                weights_only=True,
                            )
                            model.load_state_dict(state)
                            logger.info("TwoTowerModel weights loaded from %s", weights_path)
                        else:
                            logger.warning(
                                "TwoTowerModel weights not found at %s — random init",
                                weights_path,
                            )

                        index_path = Path(model_dir) / "item_index.faiss"
                        if index_path.exists():
                            model._load_faiss_index(index_path)
                    else:
                        logger.warning("MODEL_DIR not set — TwoTowerModel random init")

                    model.eval()
                    cls._instance = model
        return cls._instance

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._instance is not None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, u_user: torch.Tensor, u_item: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled dot-product similarity.

        Parameters
        ----------
        u_user : (B, 64) L2-normalised user embeddings
        u_item : (B, 64) or (B, N, 64) L2-normalised item embeddings

        Returns
        -------
        Similarity logits scaled by 1/tau.
        """
        if u_item.dim() == 3:
            # (B, N) — user queries multiple items
            return torch.einsum("bd,bnd->bn", u_user, u_item) / self.tau
        return (u_user * u_item).sum(dim=-1) / self.tau

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def info_nce_loss(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE contrastive loss with in-batch negatives.

        Each (user_i, item_i) pair is a positive; all other items in the
        batch serve as negatives for user_i.

        Parameters
        ----------
        user_features : (B, 192) raw user inputs
        item_features : (B, 308) raw item inputs

        Returns
        -------
        Scalar loss.
        """
        u_user = self.user_tower(user_features)  # (B, 64)
        u_item = self.item_tower(item_features)  # (B, 64)

        # Full similarity matrix: (B, B)
        logits = torch.mm(u_user, u_item.t()) / self.tau

        # Positive pairs lie on the diagonal
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------
    # FAISS index management
    # ------------------------------------------------------------------

    def _load_faiss_index(self, index_path: Path) -> None:
        """Load a pre-built FAISS index and accompanying metadata."""
        try:
            import faiss

            self._faiss_index = faiss.read_index(str(index_path))
            logger.info(
                "FAISS index loaded: %d vectors, dim=%d",
                self._faiss_index.ntotal,
                self._faiss_index.d,
            )

            meta_path = index_path.with_suffix(".meta.npz")
            if meta_path.exists():
                meta = np.load(str(meta_path), allow_pickle=True)
                self._item_ids = meta["item_ids"].tolist()
                self._item_metadata = meta["item_metadata"].tolist()
        except ImportError:
            logger.warning("faiss not installed — ANN retrieval unavailable")
        except Exception:
            logger.exception("Failed to load FAISS index from %s", index_path)

    @torch.inference_mode()
    def build_index(
        self,
        item_features: torch.Tensor,
        item_ids: list[str],
        item_metadata: list[dict] | None = None,
    ) -> None:
        """Build a FAISS index from a batch of item features.

        Parameters
        ----------
        item_features : (N, 308) raw item feature matrix
        item_ids : list of N item identifiers
        item_metadata : optional list of N metadata dicts
        """
        import faiss

        embeddings = self.item_tower(item_features).cpu().numpy()  # (N, 64)
        n, d = embeddings.shape

        # IVF with flat quantiser; nlist heuristic: sqrt(N)
        nlist = max(1, int(n ** 0.5))
        quantiser = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantiser, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)

        self._faiss_index = index
        self._item_ids = list(item_ids)
        self._item_metadata = list(item_metadata or [{} for _ in item_ids])
        logger.info("FAISS index built with %d items (nlist=%d)", n, nlist)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def recommend(
        self,
        body_embedding: list[float],
        behavior_embedding: list[float] | None = None,
        top_k: int = 10,
        category: str | None = None,
    ) -> list[dict]:
        """Retrieve the top-K most relevant items for a user.

        Parameters
        ----------
        body_embedding : 128-d body-shape vector
        behavior_embedding : 64-d browsing/purchase history vector (None = cold start)
        top_k : number of results to return
        category : optional category filter applied post-retrieval

        Returns
        -------
        List of dicts: [{item_id, score, metadata}, ...]
        """
        if self._faiss_index is None:
            logger.warning("No FAISS index available — returning empty results")
            return []

        if len(body_embedding) != USER_BODY_DIM:
            raise ValueError(
                f"body_embedding must be {USER_BODY_DIM}-d, got {len(body_embedding)}"
            )

        # Cold-start fallback: zero behaviour signal
        if behavior_embedding is None:
            behavior_embedding = [0.0] * USER_BEHAVIOR_DIM
        elif len(behavior_embedding) != USER_BEHAVIOR_DIM:
            raise ValueError(
                f"behavior_embedding must be {USER_BEHAVIOR_DIM}-d, got {len(behavior_embedding)}"
            )

        user_input = torch.tensor(
            body_embedding + behavior_embedding, dtype=torch.float32,
        ).unsqueeze(0)  # (1, 192)

        u_user = self.user_tower(user_input).cpu().numpy()  # (1, 64)

        # Over-fetch when filtering by category
        search_k = top_k * 3 if category else top_k
        scores, indices = self._faiss_index.search(u_user, search_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._item_metadata[idx] if idx < len(self._item_metadata) else {}

            if category and meta.get("category", "").lower() != category.lower():
                continue

            results.append({
                "item_id": self._item_ids[idx] if idx < len(self._item_ids) else str(idx),
                "score": round(float(score), 4),
                "metadata": meta,
            })
            if len(results) >= top_k:
                break

        return results
