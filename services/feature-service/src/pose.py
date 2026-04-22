"""HRNet-W32 pose estimation for virtual try-on feature extraction.

Architecture — HRNet-W32 (High-Resolution Net, 28.5M params, 74.9 AP on COCO):
  4-stage parallel multi-resolution branches with repeated multi-scale fusion.

  Stage 1: 4x BasicBlock(64)                                    → (64, 64, 48)
  Stage 2: Branch1 (64, 64, 48) + Branch2 (128, 32, 24)         → fuse
  Stage 3: + Branch3 (256, 16, 12)                               → fuse
  Stage 4: + Branch4 (512, 8, 6)                                 → fuse
  Head:    Conv(64, 18, 1x1)                                     → (18, 64, 48) heatmaps

  Multi-scale fusion: each branch exchanges information by upsampling
  (low→high) or strided convolution (high→low) at every stage transition.

Input:  RGB image resized to (256, 192), normalized with ImageNet mean/std
Output: 18 COCO keypoints (x, y, confidence) + (18, 64, 48) heatmaps

References:
  - Sun et al., "Deep High-Resolution Representation Learning for
    Visual Recognition", TPAMI 2019
  - Pretrained weights: HRNet-W32 trained on COCO keypoints
"""

# intent: HRNet-W32 pose estimation with ONNX Runtime inference
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
ONNX_PATH = MODEL_DIR / "hrnet_w32_coco.onnx"

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Input dimensions expected by HRNet-W32
INPUT_HEIGHT = 256
INPUT_WIDTH = 192

# Output heatmap dimensions (1/4 of input)
HEATMAP_HEIGHT = 64
HEATMAP_WIDTH = 48

# 18 COCO keypoints
NUM_KEYPOINTS = 18

# Confidence threshold for keypoint detection
CONFIDENCE_THRESHOLD = 0.3

# COCO keypoint names (for reference / debugging)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "neck",
]


class HRNetPose:
    """HRNet-W32 pose estimation model.

    Singleton — use ``HRNetPose.get_instance()`` to obtain the shared model.
    Loads an ONNX Runtime session from the weights file on first access.
    Raises ``ImportError`` if ``onnxruntime`` is not installed or the
    weights file is missing.
    """

    _instance: ClassVar[HRNetPose | None] = None
    _lock: ClassVar[Lock] = Lock()

    def __init__(self) -> None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise ImportError(
                "onnxruntime is required for HRNet pose estimation. "
                "Install with: pip install onnxruntime-gpu  (or onnxruntime)"
            ) from exc

        if not ONNX_PATH.exists():
            raise ImportError(
                f"HRNet-W32 ONNX weights not found at {ONNX_PATH}. "
                "Download the model or set MODEL_DIR env var."
            )

        logger.info("Loading HRNet-W32 from %s", ONNX_PATH)
        providers = ort.get_available_providers()
        # Prefer CUDA > CPU
        preferred = []
        if "CUDAExecutionProvider" in providers:
            preferred.append("CUDAExecutionProvider")
        preferred.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(str(ONNX_PATH), providers=preferred)
        self._input_name = self._session.get_inputs()[0].name
        logger.info(
            "HRNet-W32 loaded (providers=%s, input=%s)",
            preferred, self._input_name,
        )

    @classmethod
    def get_instance(cls) -> HRNetPose:
        """Return the singleton HRNetPose instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Public API ──

    def predict(self, image: Image.Image) -> dict[str, np.ndarray]:
        """Run pose estimation on a PIL Image.

        Args:
            image: RGB PIL Image of any size.

        Returns:
            dict with:
              - "keypoints": ndarray of shape (18, 3) — [x_norm, y_norm, confidence]
              - "heatmaps":  ndarray of shape (18, 64, 48) — raw heatmap activations
        """
        input_tensor = self._preprocess(image)
        heatmaps = self._infer(input_tensor)
        keypoints = self._postprocess(heatmaps, image.size)
        return {"keypoints": keypoints, "heatmaps": heatmaps}

    # ── Internal methods ──

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize to (256, 192), convert to float32, normalize with ImageNet stats.

        Returns:
            ndarray of shape (1, 3, 256, 192) — NCHW format.
        """
        img_resized = image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0  # (H, W, 3)

        # Normalize per-channel
        img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

        # HWC → CHW → NCHW
        img_array = np.transpose(img_array, (2, 0, 1))  # (3, 256, 192)
        return np.expand_dims(img_array, axis=0).astype(np.float32)

    def _infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run ONNX Runtime inference.

        Args:
            input_tensor: (1, 3, 256, 192) float32.

        Returns:
            Heatmaps of shape (18, 64, 48).
        """
        outputs = self._session.run(None, {self._input_name: input_tensor})
        # Output shape: (1, 18, 64, 48) — squeeze batch dim
        heatmaps = outputs[0][0]
        return heatmaps.astype(np.float32)

    def _postprocess(
        self,
        heatmaps: np.ndarray,
        original_size: tuple[int, int],
    ) -> np.ndarray:
        """Extract keypoint coordinates from heatmaps via argmax.

        For each of the 18 channels, find the peak location and its
        confidence value.  Coordinates are normalized to [0, 1] relative
        to the original image dimensions.

        Args:
            heatmaps: (18, 64, 48) activation maps.
            original_size: (width, height) of the original input image.

        Returns:
            ndarray of shape (18, 3) — [x_normalized, y_normalized, confidence].
        """
        num_joints = heatmaps.shape[0]
        keypoints = np.zeros((num_joints, 3), dtype=np.float32)

        for j in range(num_joints):
            hm = heatmaps[j]  # (64, 48)

            # Flatten, find argmax, convert to 2D index
            flat_idx = np.argmax(hm)
            y_hm, x_hm = np.unravel_index(flat_idx, hm.shape)
            confidence = float(hm[y_hm, x_hm])

            if confidence < CONFIDENCE_THRESHOLD:
                keypoints[j] = [0.0, 0.0, 0.0]
                continue

            # Sub-pixel refinement: shift towards second-highest neighbor
            # (standard HRNet post-processing)
            if 0 < x_hm < HEATMAP_WIDTH - 1:
                x_hm += 0.25 * np.sign(hm[y_hm, x_hm + 1] - hm[y_hm, x_hm - 1])
            if 0 < y_hm < HEATMAP_HEIGHT - 1:
                y_hm += 0.25 * np.sign(hm[y_hm + 1, x_hm] - hm[y_hm - 1, x_hm])

            # Normalize to [0, 1] relative to heatmap grid
            x_norm = float(x_hm) / HEATMAP_WIDTH
            y_norm = float(y_hm) / HEATMAP_HEIGHT

            keypoints[j] = [x_norm, y_norm, confidence]

        return keypoints
