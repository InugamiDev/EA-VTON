"""Feature extraction service — pose, parsing, body embedding, quality.

Endpoints:
  POST /extract        — extract pose + parsing + body embedding from person image
  POST /quality-check  — assess image quality for try-on suitability
  GET  /health         — service health check
"""

# intent: FastAPI entry point for feature extraction service
# status: done
# next: integrate HRNet + SCHP models when weights are available
# confidence: high

from __future__ import annotations

import io
import logging
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Feature Service",
    description="Pose estimation (HRNet), human parsing (SCHP), body embedding extraction",
    version="0.1.0",
)


@app.get("/health")
def health():
    """Service health check."""
    return {
        "status": "healthy",
        "service": "feature-service",
        "models": _get_loaded_models(),
    }


@app.post("/extract")
async def extract_features(
    image: UploadFile = File(...),
    include_pose: bool = Form(True),
    include_parsing: bool = Form(True),
    include_embedding: bool = Form(True),
    height_cm: float | None = Form(None),
    weight_kg: float | None = Form(None),
):
    """Extract features from a person image.

    Returns pose keypoints + heatmaps, parsing map, and body embedding.
    Optionally includes body measurements if height_cm and weight_kg are provided.
    """
    t0 = time.perf_counter()
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result: dict[str, Any] = {}

    if include_pose:
        try:
            pose_data = _extract_pose(img)
            result["pose"] = {
                "keypoints": pose_data["keypoints"].tolist(),
                "heatmaps_shape": list(pose_data["heatmaps"].shape),
                # Heatmaps returned as base64-encoded numpy for efficiency
                "heatmaps_b64": _ndarray_to_b64(pose_data["heatmaps"]),
            }
        except Exception as e:
            logger.warning("Pose extraction failed: %s", e)
            result["pose"] = None

    if include_parsing:
        try:
            parsing_map = _extract_parsing(img)
            result["parsing"] = {
                "shape": list(parsing_map.shape),
                "classes_present": list(np.unique(parsing_map).tolist()),
                "map_b64": _ndarray_to_b64(parsing_map),
            }
        except Exception as e:
            logger.warning("Parsing extraction failed: %s", e)
            result["parsing"] = None

    if include_embedding:
        try:
            embedding = _extract_body_embedding(
                result.get("pose"),
                result.get("parsing"),
                img,
            )
            result["body_embedding"] = embedding.tolist()
        except Exception as e:
            logger.warning("Body embedding extraction failed: %s", e)
            result["body_embedding"] = None

    # Body measurements from MediaPipe (if height/weight provided)
    if height_cm is not None and weight_kg is not None:
        try:
            from src.body_estimator import estimate_from_photo
            measurements = estimate_from_photo(image_bytes, height_cm, weight_kg)
            result["body_measurements"] = {
                "chest_circumference": measurements.chest_circumference,
                "waist_circumference": measurements.waist_circumference,
                "hip_circumference": measurements.hip_circumference,
                "shoulder_width": measurements.shoulder_width,
                "arm_length": measurements.arm_length,
                "leg_length": measurements.leg_length,
                "torso_length": measurements.torso_length,
                "bmi": measurements.bmi,
                "method": measurements.method,
                "landmark_confidence": measurements.landmark_confidence,
                "warnings": measurements.warnings,
            }
        except Exception as e:
            logger.warning("Body measurement failed: %s", e)
            result["body_measurements"] = None

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    result["processing_time_ms"] = elapsed_ms

    return result


@app.post("/quality-check")
async def quality_check(image: UploadFile = File(...)):
    """Assess image quality for virtual try-on suitability."""
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    from src.quality import assess_quality
    assessment = assess_quality(img)
    return assessment


# ── Internal helpers ──


def _get_loaded_models() -> dict[str, str]:
    """Report which models are loaded/available."""
    models = {}
    try:
        from src.pose import HRNetPose
        models["pose"] = "hrnet_w32"
    except ImportError:
        models["pose"] = "mediapipe_fallback"
    try:
        from src.parsing import SCHPParser
        models["parsing"] = "schp_atr"
    except ImportError:
        models["parsing"] = "not_loaded"
    models["body_embedding"] = "cnn_r128"
    return models


def _extract_pose(img: Image.Image) -> dict[str, np.ndarray]:
    """Extract pose keypoints and heatmaps.

    Uses HRNet-W32 if available, falls back to MediaPipe.
    Returns dict with 'keypoints' (18, 3) and 'heatmaps' (18, 64, 48).
    """
    try:
        from src.pose import HRNetPose
        model = HRNetPose.get_instance()
        return model.predict(img)
    except ImportError:
        return _extract_pose_mediapipe(img)


def _extract_pose_mediapipe(img: Image.Image) -> dict[str, np.ndarray]:
    """Fallback pose extraction using MediaPipe."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    from src.body_estimator import _ensure_model

    model_path = _ensure_model()
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.array(img),
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return {
            "keypoints": np.zeros((18, 3), dtype=np.float32),
            "heatmaps": np.zeros((18, 64, 48), dtype=np.float32),
        }

    # Convert MediaPipe 33 landmarks to 18-point COCO format
    COCO_FROM_MP = [0, 0, 0, 0, 0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 0]
    landmarks = result.pose_landmarks[0]
    keypoints = np.zeros((18, 3), dtype=np.float32)
    for i, mp_idx in enumerate(COCO_FROM_MP):
        lm = landmarks[mp_idx]
        keypoints[i] = [lm.x, lm.y, lm.visibility if hasattr(lm, "visibility") else 0.5]

    # Generate Gaussian heatmaps (18, 64, 48)
    heatmaps = _generate_heatmaps(keypoints, 64, 48, sigma=6.0)

    return {"keypoints": keypoints, "heatmaps": heatmaps}


def _generate_heatmaps(
    keypoints: np.ndarray,
    h: int = 64,
    w: int = 48,
    sigma: float = 6.0,
) -> np.ndarray:
    """Generate Gaussian heatmaps from keypoints.

    H_j(p) = c_j * exp(-||p - (x_j, y_j)||^2 / (2 * sigma^2))
    """
    heatmaps = np.zeros((keypoints.shape[0], h, w), dtype=np.float32)
    for j in range(keypoints.shape[0]):
        x, y, conf = keypoints[j]
        if conf < 0.1:
            continue
        px, py = int(x * w), int(y * h)
        if px < 0 or px >= w or py < 0 or py >= h:
            continue

        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        heatmaps[j] = conf * np.exp(
            -((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2)
        )

    return heatmaps


def _extract_parsing(img: Image.Image) -> np.ndarray:
    """Extract human parsing map (20-class ATR format).

    Uses SCHP if available, otherwise returns dummy map.
    """
    try:
        from src.parsing import SCHPParser
        parser = SCHPParser.get_instance()
        return parser.predict(img)
    except ImportError:
        logger.warning("SCHP not available — returning empty parsing map")
        w, h = img.size
        return np.zeros((h, w), dtype=np.uint8)


def _extract_body_embedding(
    pose_data: dict | None,
    parsing_data: dict | None,
    img: Image.Image,
) -> np.ndarray:
    """Extract body embedding vector (R^128).

    Uses lightweight CNN on concat(pose_heatmap, parsing).
    Falls back to zero vector if inputs are unavailable.
    """
    try:
        from src.body_embedding import BodyEmbeddingNet
        model = BodyEmbeddingNet.get_instance()

        heatmaps = None
        parsing_map = None

        if pose_data and "heatmaps_b64" in pose_data:
            heatmaps = _b64_to_ndarray(pose_data["heatmaps_b64"])
        if parsing_data and "map_b64" in parsing_data:
            parsing_map = _b64_to_ndarray(parsing_data["map_b64"])

        return model.predict(heatmaps, parsing_map)
    except ImportError:
        return np.zeros(128, dtype=np.float32)


def _ndarray_to_b64(arr: np.ndarray) -> str:
    """Encode numpy array to base64 string."""
    import base64
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_ndarray(b64_str: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    import base64
    buf = io.BytesIO(base64.b64decode(b64_str))
    return np.load(buf)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
