"""Lightweight CNN for body shape embedding extraction.

Encodes a person's body shape into a compact 128-d vector (e_b) by
fusing pose heatmaps (18 channels) and human parsing (20-ch one-hot)
into a single 38-channel spatial tensor, then running it through a
small convolutional encoder.

Architecture:
  Input: concat(pose_heatmap[18ch], parsing_one_hot[20ch]) = 38 channels
    → Conv(38,  64, 3, stride=2) + BatchNorm + ReLU    (32, 24)
    → Conv(64, 128, 3, stride=2) + BatchNorm + ReLU    (16, 12)
    → Conv(128, 256, 3, stride=2) + BatchNorm + ReLU   (8,  6)
    → Conv(256, 256, 3, stride=2) + BatchNorm + ReLU   (4,  3)
    → AdaptiveAvgPool2d(1)                              (1,  1)
    → Linear(256, 128) → L2-normalize → e_b ∈ R^128

The embedding captures body proportions, posture, and clothing layout,
enabling the try-on generator to condition garment deformation on the
target person's body shape.

Training: triplet loss with online hard mining on body-shape similarity
groups derived from anthropometric measurements.
"""

# intent: body embedding CNN fusing pose + parsing into a 128-d shape vector
# status: done
# next: train on paired body data and export checkpoint
# confidence: high

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import ClassVar

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent.parent / "models"))
# Check both direct path and body_embedding subdirectory
_CKPT_DIRECT = MODEL_DIR / "body_embedding_cnn.pt"
_CKPT_SUBDIR = MODEL_DIR / "body_embedding" / "best.pt"
CHECKPOINT_PATH = _CKPT_SUBDIR if _CKPT_SUBDIR.exists() else _CKPT_DIRECT

EMBEDDING_DIM = 128
NUM_POSE_CHANNELS = 18
NUM_PARSING_CLASSES = 20
TOTAL_INPUT_CHANNELS = NUM_POSE_CHANNELS + NUM_PARSING_CLASSES  # 38

# Spatial size matching HRNet heatmap output
SPATIAL_H = 64
SPATIAL_W = 48


# ── PyTorch nn.Module definition ──

def _build_network():
    """Construct the BodyEmbeddingCNN as a torch.nn.Module.

    Separated into a factory function so the import of torch is deferred
    until the model is actually needed.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class BodyEmbeddingCNN(nn.Module):
        """Lightweight CNN that maps (38, 64, 48) body features to R^128."""

        def __init__(self) -> None:
            super().__init__()

            self.encoder = nn.Sequential(
                # Block 1: (38, 64, 48) → (64, 32, 24)
                nn.Conv2d(TOTAL_INPUT_CHANNELS, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # Block 2: (64, 32, 24) → (128, 16, 12)
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # Block 3: (128, 16, 12) → (256, 8, 6)
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                # Block 4: (256, 8, 6) → (256, 4, 3)
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(256, EMBEDDING_DIM)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: (B, 38, 64, 48) float tensor.

            Returns:
                (B, 128) L2-normalized embedding.
            """
            x = self.encoder(x)       # (B, 256, 4, 3)
            x = self.pool(x)          # (B, 256, 1, 1)
            x = x.view(x.size(0), -1) # (B, 256)
            x = self.fc(x)            # (B, 128)
            x = F.normalize(x, p=2, dim=1)  # L2-normalize
            return x

    return BodyEmbeddingCNN


class BodyEmbeddingNet:
    """Body embedding extraction with pose + parsing fusion.

    Singleton — use ``BodyEmbeddingNet.get_instance()`` to obtain the
    shared model.  Falls back gracefully to zero vectors when PyTorch
    is unavailable or weights are missing.
    """

    _instance: ClassVar[BodyEmbeddingNet | None] = None
    _lock: ClassVar[Lock] = Lock()
    _loaded: ClassVar[bool] = False

    def __init__(self) -> None:
        self._model = None
        self._device = "cpu"

        try:
            import torch

            # Select device
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"

            # Build network
            NetworkClass = _build_network()
            self._model = NetworkClass()

            # Load pretrained weights if available
            if CHECKPOINT_PATH.exists():
                logger.info("Loading body embedding weights from %s", CHECKPOINT_PATH)
                state_dict = torch.load(
                    str(CHECKPOINT_PATH),
                    map_location=self._device,
                    weights_only=True,
                )
                self._model.load_state_dict(state_dict)
                BodyEmbeddingNet._loaded = True
                logger.info("Body embedding model loaded on %s", self._device)
            else:
                logger.warning(
                    "Body embedding checkpoint not found at %s — "
                    "model initialized with random weights",
                    CHECKPOINT_PATH,
                )

            self._model.to(self._device)
            self._model.eval()

        except ImportError:
            logger.warning(
                "PyTorch not available — body embedding will return zero vectors"
            )
            self._model = None

    @classmethod
    def get_instance(cls) -> BodyEmbeddingNet:
        """Return the singleton BodyEmbeddingNet instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def is_loaded(cls) -> bool:
        """Check whether the model weights are loaded (vs random init)."""
        return cls._loaded

    # ── Public API ──

    def predict(
        self,
        heatmaps: np.ndarray | None,
        parsing: np.ndarray | None,
    ) -> np.ndarray:
        """Extract 128-d body embedding from pose heatmaps and parsing map.

        Args:
            heatmaps: (18, 64, 48) float32 pose heatmaps, or None.
            parsing:  (H, W) uint8 semantic label map (0–19), or None.

        Returns:
            ndarray of shape (128,) — L2-normalized body embedding.
            Returns zeros(128) if both inputs are None or the model
            is unavailable.
        """
        if heatmaps is None and parsing is None:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        if self._model is None:
            logger.warning("No model available — returning zero embedding")
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        input_tensor = self._prepare_input(heatmaps, parsing)
        embedding = self._infer(input_tensor)
        return embedding

    # ── Internal methods ──

    def _prepare_input(
        self,
        heatmaps: np.ndarray | None,
        parsing: np.ndarray | None,
    ) -> np.ndarray:
        """Prepare the 38-channel input tensor.

        - Heatmaps: use directly if (18, 64, 48), otherwise zero-fill.
        - Parsing:  convert (H, W) label map to (20, H, W) one-hot,
          then resize to (20, 64, 48) to match heatmap spatial dims.

        Returns:
            ndarray of shape (1, 38, 64, 48) — batch of 1.
        """
        # Pose heatmaps: (18, 64, 48)
        if heatmaps is not None:
            pose_channels = heatmaps.astype(np.float32)
            if pose_channels.shape != (NUM_POSE_CHANNELS, SPATIAL_H, SPATIAL_W):
                logger.warning(
                    "Unexpected heatmap shape %s, expected (%d, %d, %d) — zero-filling",
                    pose_channels.shape, NUM_POSE_CHANNELS, SPATIAL_H, SPATIAL_W,
                )
                pose_channels = np.zeros(
                    (NUM_POSE_CHANNELS, SPATIAL_H, SPATIAL_W), dtype=np.float32
                )
        else:
            pose_channels = np.zeros(
                (NUM_POSE_CHANNELS, SPATIAL_H, SPATIAL_W), dtype=np.float32
            )

        # Parsing one-hot: (20, 64, 48)
        if parsing is not None:
            parsing_onehot = self._parsing_to_onehot(parsing)
        else:
            parsing_onehot = np.zeros(
                (NUM_PARSING_CLASSES, SPATIAL_H, SPATIAL_W), dtype=np.float32
            )

        # Concatenate along channel axis: (38, 64, 48)
        combined = np.concatenate([pose_channels, parsing_onehot], axis=0)

        # Add batch dimension: (1, 38, 64, 48)
        return np.expand_dims(combined, axis=0)

    def _parsing_to_onehot(self, parsing: np.ndarray) -> np.ndarray:
        """Convert (H, W) label map to (20, 64, 48) one-hot float tensor.

        Resizes to the target spatial dimensions using nearest-neighbor
        interpolation to preserve discrete labels before one-hot encoding.
        """
        from PIL import Image as PILImage

        # Resize to heatmap spatial dims if needed
        h, w = parsing.shape[:2]
        if h != SPATIAL_H or w != SPATIAL_W:
            parsing_pil = PILImage.fromarray(parsing.astype(np.uint8), mode="L")
            parsing_pil = parsing_pil.resize(
                (SPATIAL_W, SPATIAL_H), PILImage.NEAREST
            )
            parsing = np.array(parsing_pil, dtype=np.uint8)

        # One-hot encode: (H, W) → (20, H, W)
        onehot = np.zeros(
            (NUM_PARSING_CLASSES, SPATIAL_H, SPATIAL_W), dtype=np.float32
        )
        for cls_id in range(NUM_PARSING_CLASSES):
            onehot[cls_id] = (parsing == cls_id).astype(np.float32)

        return onehot

    def _infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run forward pass through the CNN.

        Args:
            input_tensor: (1, 38, 64, 48) float32.

        Returns:
            ndarray of shape (128,) — L2-normalized embedding.
        """
        import torch

        with torch.no_grad():
            x = torch.from_numpy(input_tensor).to(self._device)
            embedding = self._model(x)  # (1, 128)

        return embedding.cpu().numpy().squeeze(0)  # (128,)
