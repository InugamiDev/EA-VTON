"""Pose tracking via MediaPipe PoseLandmarker (tasks API).

// intent: track 33 body landmarks per frame, extract garment-relevant subset
// status: done
// confidence: high

Tracks 8 garment-relevant landmarks: shoulders (11,12), elbows (13,14),
wrists (15,16), hips (23,24). Returns (33, 3) landmark arrays.
~15 FPS on M2 CPU.

Uses MediaPipe tasks API (>=0.10.21) — the legacy mp.solutions API was removed.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Garment-relevant landmark indices (MediaPipe Pose)
# Shoulders: 11, 12 | Elbows: 13, 14 | Wrists: 15, 16 | Hips: 23, 24
GARMENT_LANDMARK_IDS = [11, 12, 13, 14, 15, 16, 23, 24]

# Model path
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
_MODEL_DIR = Path("/tmp/mediapipe_models")
_MODEL_PATH = _MODEL_DIR / "pose_landmarker_lite.task"

# Singleton pose model
_landmarker = None


@dataclass
class PoseResult:
    """Pose detection result for a single frame."""
    landmarks: np.ndarray | None  # (33, 3) x,y,visibility or None if no pose
    garment_landmarks: np.ndarray | None  # (8, 3) subset
    detected: bool


def _ensure_model():
    """Download pose landmarker model if not present."""
    if _MODEL_PATH.exists():
        return
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MediaPipe PoseLandmarker model...")
    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    logger.info("PoseLandmarker model ready at %s", _MODEL_PATH)


def _get_landmarker():
    """Get or create singleton PoseLandmarker instance."""
    global _landmarker
    if _landmarker is not None:
        return _landmarker

    import mediapipe as mp

    _ensure_model()

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    logger.info("MediaPipe PoseLandmarker initialized")
    return _landmarker


def detect_pose(frame: Image.Image) -> PoseResult:
    """Detect pose landmarks in a single frame.

    Args:
        frame: PIL Image (RGB).

    Returns:
        PoseResult with (33, 3) landmarks array or None if no person detected.
    """
    import mediapipe as mp

    landmarker = _get_landmarker()
    frame_arr = np.array(frame)

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_arr)
    results = landmarker.detect(mp_image)

    if not results.pose_landmarks or len(results.pose_landmarks) == 0:
        return PoseResult(landmarks=None, garment_landmarks=None, detected=False)

    # Extract (33, 3) array: [x, y, visibility]
    pose_lm = results.pose_landmarks[0]  # First (and only) person
    landmarks = np.array([
        [lm.x, lm.y, lm.visibility]
        for lm in pose_lm
    ], dtype=np.float32)

    garment_landmarks = landmarks[GARMENT_LANDMARK_IDS]

    return PoseResult(
        landmarks=landmarks,
        garment_landmarks=garment_landmarks,
        detected=True,
    )


def compute_pose_delta(
    prev_garment_lm: np.ndarray,
    curr_garment_lm: np.ndarray,
) -> float:
    """Compute pose change between two frames using garment-relevant landmarks.

    Uses normalized L2 distance of the 8 garment landmarks (x, y only).
    Ignores visibility for delta computation.

    Args:
        prev_garment_lm: (8, 3) previous frame landmarks.
        curr_garment_lm: (8, 3) current frame landmarks.

    Returns:
        Scalar delta (0 = identical, higher = more movement).
    """
    prev_xy = prev_garment_lm[:, :2]
    curr_xy = curr_garment_lm[:, :2]
    return float(np.mean(np.linalg.norm(curr_xy - prev_xy, axis=1)))


def compute_torso_rotation(garment_landmarks: np.ndarray) -> float:
    """Compute torso rotation angle from shoulder vector.

    Uses the angle of the left_shoulder → right_shoulder vector relative to horizontal.
    Returns angle in radians.
    """
    # Landmarks indices within garment_landmarks array:
    # 0=left_shoulder(11), 1=right_shoulder(12)
    left_shoulder = garment_landmarks[0, :2]
    right_shoulder = garment_landmarks[1, :2]

    delta = right_shoulder - left_shoulder
    angle = float(np.arctan2(delta[1], delta[0]))
    return angle


def release():
    """Release the MediaPipe PoseLandmarker instance."""
    global _landmarker
    if _landmarker is not None:
        _landmarker.close()
        _landmarker = None
        logger.info("MediaPipe PoseLandmarker released")
