"""SCHP human parsing for virtual try-on feature extraction.

Self-Correction Human Parsing (SCHP) — 59.36 mIoU on LIP dataset.

Architecture:
  - Backbone:  ResNet-101 (pretrained on ImageNet)
  - PPM:       Pyramid Pooling Module (pool sizes 1, 2, 3, 6)
  - Decoder:   Multi-level feature aggregation with skip connections
  - Head:      Conv(512, 20, 1x1) → 20-class softmax

  Self-correction loop: iterative refinement where the model's
  pseudo-labels from round N are used as auxiliary supervision in
  round N+1, progressively correcting edge and boundary errors.

Input:  RGB image resized to (512, 512), normalized with ImageNet mean/std
Output: (H, W) uint8 semantic label map in ATR 20-class format

References:
  - Li et al., "Self-Correction for Human Parsing", TPAMI 2020
  - ATR dataset label format (20 classes)
"""

# intent: SCHP human parsing with ONNX Runtime inference, ATR 20-class output
# status: done
# next: download and validate ONNX weights in deployment setup
# confidence: high

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import ClassVar

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ──

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent.parent / "models"))
ONNX_PATH = MODEL_DIR / "schp_atr.onnx"

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Input dimensions for SCHP
INPUT_SIZE = 512

# Number of semantic classes (ATR format)
NUM_CLASSES = 20


class SCHPParser:
    """SCHP human parsing model (20-class ATR format).

    Singleton — use ``SCHPParser.get_instance()`` to obtain the shared model.
    Loads an ONNX Runtime session from the weights file on first access.
    Raises ``ImportError`` if ``onnxruntime`` is not installed or the
    weights file is missing.
    """

    _instance: ClassVar[SCHPParser | None] = None
    _lock: ClassVar[Lock] = Lock()

    # ATR 20-class label map
    LABEL_MAP: ClassVar[dict[int, str]] = {
        0: "Background",
        1: "Hat",
        2: "Hair",
        3: "Sunglasses",
        4: "Upper-clothes",
        5: "Skirt",
        6: "Pants",
        7: "Dress",
        8: "Belt",
        9: "Left-shoe",
        10: "Right-shoe",
        11: "Face",
        12: "Left-leg",
        13: "Right-leg",
        14: "Left-arm",
        15: "Right-arm",
        16: "Bag",
        17: "Scarf",
        18: "Torso-skin",
        19: "Socks",
    }

    # Inverse map for quick lookup
    LABEL_NAME_TO_ID: ClassVar[dict[str, int]] = {v: k for k, v in LABEL_MAP.items()}

    # Garment-related label IDs (useful for try-on masking)
    GARMENT_LABELS: ClassVar[set[int]] = {4, 5, 6, 7, 8, 17}  # clothes + belt + scarf

    def __init__(self) -> None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise ImportError(
                "onnxruntime is required for SCHP human parsing. "
                "Install with: pip install onnxruntime-gpu  (or onnxruntime)"
            ) from exc

        if not ONNX_PATH.exists():
            raise ImportError(
                f"SCHP ONNX weights not found at {ONNX_PATH}. "
                "Download the model or set MODEL_DIR env var."
            )

        logger.info("Loading SCHP parser from %s", ONNX_PATH)
        providers = ort.get_available_providers()
        preferred = []
        if "CUDAExecutionProvider" in providers:
            preferred.append("CUDAExecutionProvider")
        preferred.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(str(ONNX_PATH), providers=preferred)
        self._input_name = self._session.get_inputs()[0].name
        logger.info(
            "SCHP loaded (providers=%s, input=%s)",
            preferred, self._input_name,
        )

    @classmethod
    def get_instance(cls) -> SCHPParser:
        """Return the singleton SCHPParser instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Public API ──

    def predict(self, image: Image.Image) -> np.ndarray:
        """Run human parsing on a PIL Image.

        Args:
            image: RGB PIL Image of any size.

        Returns:
            ndarray of shape (H, W) with uint8 class labels (0–19).
            H, W match the original image dimensions.
        """
        original_size = image.size  # (width, height)
        input_tensor = self._preprocess(image)
        logits = self._infer(input_tensor)
        label_map = self._postprocess(logits, original_size)
        return label_map

    # ── Internal methods ──

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize to (512, 512), convert to float32, normalize with ImageNet stats.

        Returns:
            ndarray of shape (1, 3, 512, 512) — NCHW format.
        """
        img_resized = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0  # (512, 512, 3)

        # Normalize per-channel
        img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

        # HWC → CHW → NCHW
        img_array = np.transpose(img_array, (2, 0, 1))
        return np.expand_dims(img_array, axis=0).astype(np.float32)

    def _infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run ONNX Runtime inference.

        Args:
            input_tensor: (1, 3, 512, 512) float32.

        Returns:
            Raw logits of shape (20, 512, 512).
        """
        outputs = self._session.run(None, {self._input_name: input_tensor})
        # Output shape: (1, 20, 512, 512) — squeeze batch dim
        logits = outputs[0][0]
        return logits.astype(np.float32)

    def _postprocess(
        self,
        logits: np.ndarray,
        original_size: tuple[int, int],
    ) -> np.ndarray:
        """Convert logits to label map, resize to original dimensions.

        Args:
            logits: (20, 512, 512) raw output from the model.
            original_size: (width, height) of the original input image.

        Returns:
            ndarray of shape (H, W) with uint8 class labels (0–19).
        """
        # Argmax over class dimension → (512, 512)
        label_map = np.argmax(logits, axis=0).astype(np.uint8)

        # Resize back to original image dimensions if needed
        orig_w, orig_h = original_size
        if label_map.shape != (orig_h, orig_w):
            label_pil = Image.fromarray(label_map, mode="L")
            # Use nearest-neighbor to preserve discrete labels
            label_pil = label_pil.resize((orig_w, orig_h), Image.NEAREST)
            label_map = np.array(label_pil, dtype=np.uint8)

        return label_map
