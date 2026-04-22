# intent: PyTorch MLP for learned size prediction, replacing rule-based estimation
# status: done
# next: train on CAESAR/RTR + Vietnamese anthropometric data, evaluate vs size_engine.py baseline
# blockers: none
# confidence: high

"""SizeMLP — learned size prediction from body measurements + pose embedding.

Replaces the linear chest-estimation heuristics in size_engine.py with a
3-layer MLP that ingests 134 features (6 derived anthropometric scalars +
128-d body embedding from MediaPipe PoseLandmarker) and outputs a probability
distribution over 7 standard size classes (XXS through XXL).

Training uses ordinal cross-entropy to penalise predictions that are far
from the ground-truth size on the ordinal scale.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SIZE_LABELS: list[str] = ["XXS", "XS", "S", "M", "L", "XL", "XXL"]
NUM_CLASSES: int = len(SIZE_LABELS)
INPUT_DIM: int = 134  # 6 anthropometric scalars + 128-d body embedding
LAMBDA_ORD: float = 0.1


class SizeMLP(nn.Module):
    """Three-layer MLP mapping body features → size-class probabilities."""

    _instance: Optional["SizeMLP"] = None
    _lock: Lock = Lock()

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list[int] | None = None,
        output_dim: int = NUM_CLASSES,
        dropout: list[float] | float = 0.3,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        if isinstance(dropout, (int, float)):
            dropout = [dropout] * len(hidden_dims)

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout[i] if i < len(dropout) else dropout[-1]),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, input_dim) → (B, output_dim) logits."""
        return self.net(x)

    # ------------------------------------------------------------------
    # Singleton lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "SizeMLP":
        """Return the singleton, loading weights on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    model = cls()
                    model_dir = os.environ.get("MODEL_DIR")
                    if model_dir:
                        # Check both MODEL_DIR/size_mlp.pt and MODEL_DIR/best.pt
                        weights_path = Path(model_dir) / "size_mlp.pt"
                        if not weights_path.exists():
                            weights_path = Path(model_dir) / "best.pt"
                        if weights_path.exists():
                            state = torch.load(
                                weights_path,
                                map_location="cpu",
                                weights_only=True,
                            )
                            model.load_state_dict(state)
                            logger.info("SizeMLP weights loaded from %s", weights_path)
                        else:
                            logger.warning(
                                "SizeMLP weights not found at %s — running with random init",
                                weights_path,
                            )
                    else:
                        logger.warning("MODEL_DIR not set — SizeMLP running with random init")
                    model.eval()
                    cls._instance = model
        return cls._instance

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._instance is not None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_features(
        height_cm: float,
        weight_kg: float,
        age: float = 25.0,
        waist_cm: float | None = None,
        hip_cm: float | None = None,
        shoulder_cm: float | None = None,
    ) -> list[float]:
        """Compute the 6 scalar features from raw body measurements."""
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m ** 2) if height_m > 0 else 0.0

        # Fallback ratios when circumferences are unavailable
        waist_hip_ratio = (waist_cm / hip_cm) if (waist_cm and hip_cm) else 0.80
        shoulder_hip_ratio = (shoulder_cm / hip_cm) if (shoulder_cm and hip_cm) else 1.05

        return [height_cm, weight_kg, bmi, age, waist_hip_ratio, shoulder_hip_ratio]

    @torch.inference_mode()
    def predict(
        self,
        height_cm: float,
        weight_kg: float,
        body_embedding: list[float],
        size_chart: list[dict] | None = None,
        fit_preference: str = "regular",
        *,
        age: float = 25.0,
        waist_cm: float | None = None,
        hip_cm: float | None = None,
        shoulder_cm: float | None = None,
    ) -> dict:
        """Run inference and return a SizeResponse-compatible dict.

        Parameters
        ----------
        height_cm, weight_kg : raw body measurements
        body_embedding : 128-d vector from MediaPipe PoseLandmarker
        size_chart : optional garment size chart (list of {size, chestMin, chestMax})
        fit_preference : "slim" | "regular" | "relaxed"
        """
        scalars = self._derive_features(
            height_cm, weight_kg, age, waist_cm, hip_cm, shoulder_cm,
        )

        if len(body_embedding) != 128:
            raise ValueError(
                f"body_embedding must be 128-d, got {len(body_embedding)}"
            )

        features = torch.tensor(
            scalars + list(body_embedding), dtype=torch.float32,
        ).unsqueeze(0)  # (1, 134)

        logits = self.net(features)  # (1, 7)
        probs = F.softmax(logits, dim=-1).squeeze(0)  # (7,)

        # Apply fit-preference shift before argmax
        fit_offsets = {"slim": -1, "regular": 0, "relaxed": 1}
        offset = fit_offsets.get(fit_preference, 0)

        predicted_idx = int(probs.argmax().item()) + offset
        predicted_idx = max(0, min(predicted_idx, NUM_CLASSES - 1))
        predicted_size = SIZE_LABELS[predicted_idx]

        # Build probability map
        prob_map = {
            label: round(probs[i].item(), 4) for i, label in enumerate(SIZE_LABELS)
        }

        # Match against size chart when available
        recommended_size = predicted_size
        confidence = "high" if probs[predicted_idx].item() > 0.5 else (
            "medium" if probs[predicted_idx].item() > 0.25 else "low"
        )
        alternatives: list[dict] = []

        if size_chart:
            chart_sizes = [entry["size"].upper() for entry in size_chart]
            if predicted_size in chart_sizes:
                recommended_size = predicted_size
            else:
                # Fall back to the highest-probability size present in the chart
                for idx in probs.argsort(descending=True).tolist():
                    if SIZE_LABELS[idx] in chart_sizes:
                        recommended_size = SIZE_LABELS[idx]
                        break

            # Suggest adjacent sizes as alternatives
            chart_idx = (
                chart_sizes.index(recommended_size)
                if recommended_size in chart_sizes
                else None
            )
            if chart_idx is not None:
                if chart_idx > 0:
                    smaller = size_chart[chart_idx - 1]
                    alternatives.append({
                        "size": smaller["size"],
                        "note": f"Size down ({smaller.get('chestMin', '?')}-{smaller.get('chestMax', '?')} cm)",
                    })
                if chart_idx < len(size_chart) - 1:
                    larger = size_chart[chart_idx + 1]
                    alternatives.append({
                        "size": larger["size"],
                        "note": f"Size up ({larger.get('chestMin', '?')}-{larger.get('chestMax', '?')} cm)",
                    })

        return {
            "recommended_size": recommended_size,
            "confidence": confidence,
            "probabilities": prob_map,
            "fit_preference": fit_preference,
            "alternatives": alternatives,
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def ordinal_cross_entropy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        lambda_ord: float = LAMBDA_ORD,
    ) -> torch.Tensor:
        """Ordinal cross-entropy loss.

        L_ordinal = -sum(y_k * log(y_hat_k))
                    + lambda_ord * sum((k - k*)^2 * y_hat_k)

        Parameters
        ----------
        logits : (B, C) raw network outputs
        targets : (B,) integer class labels in [0, C)
        lambda_ord : weight for the ordinal penalty term
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        # Standard cross-entropy component
        ce_loss = F.nll_loss(log_probs, targets)

        # Ordinal penalty: weighted distance from true class
        batch_size, num_classes = logits.shape
        class_indices = torch.arange(num_classes, device=logits.device).float()
        target_indices = targets.float().unsqueeze(1)  # (B, 1)
        ordinal_weights = (class_indices.unsqueeze(0) - target_indices) ** 2  # (B, C)
        ordinal_penalty = (ordinal_weights * probs).sum(dim=-1).mean()

        return ce_loss + lambda_ord * ordinal_penalty
