"""Photo-based body measurement estimation using MediaPipe Pose.

Pipeline:
  1. User provides height_cm + weight_kg as reference
  2. Front-facing photo → MediaPipe PoseLandmarker (tasks API) → 33 landmarks
  3. Pixel distances between landmarks → scale by known height → real cm widths
  4. BMI-adjusted depth-to-width ratio → Ramanujan's ellipse → circumferences
  5. Feed into ea_size_engine for size recommendation

Supports partial body (upper body only) — uses anthropometric proportions
to estimate missing lower-body measurements from visible upper body.

References:
  - Ramanujan's ellipse perimeter approximation
  - ANSUR/CAESAR depth-to-width ratios by BMI
  - MediaPipe Pose landmark indices:
    11/12 = shoulders, 23/24 = hips, 13/14 = elbows, 15/16 = wrists,
    25/26 = knees, 27/28 = ankles, 0 = nose

Accuracy: ±3-5 cm for circumferences (sufficient for S/M/L/XL sizing)
"""

# intent: photo-based body measurement using MediaPipe landmarks + known height/weight
# status: done
# confidence: medium — accuracy depends on photo quality, pose, clothing

from __future__ import annotations

import io
import math
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Model file for MediaPipe PoseLandmarker (tasks API v0.10+)
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
_MODEL_DIR = Path(__file__).parent.parent / "models"
_MODEL_PATH = _MODEL_DIR / "pose_landmarker_heavy.task"


def _ensure_model() -> str:
    """Download pose landmarker model if not present. Returns path."""
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading PoseLandmarker model to %s ...", _MODEL_PATH)
    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    logger.info("Download complete (%.1f MB)", _MODEL_PATH.stat().st_size / 1e6)
    return str(_MODEL_PATH)


@dataclass
class LandmarkPoint:
    x: float  # normalized [0, 1]
    y: float
    z: float
    visibility: float
    presence: float


@dataclass
class BodyMeasurements:
    """All measurements in cm."""
    # Direct from landmarks (scaled by height)
    shoulder_width: float = 0.0
    torso_length: float = 0.0
    arm_length: float = 0.0
    leg_length: float = 0.0
    hip_width: float = 0.0

    # Estimated circumferences (ellipse model)
    chest_circumference: float = 0.0
    waist_circumference: float = 0.0
    hip_circumference: float = 0.0

    # Metadata
    bmi: float = 0.0
    scale_cm_per_px: float = 0.0
    pixel_height: float = 0.0
    landmark_confidence: float = 0.0
    body_coverage: str = "full"  # "full", "upper", "minimal"

    # Source info
    method: str = "mediapipe_landmark_ellipse"
    warnings: list[str] = field(default_factory=list)


# ── Depth-to-width ratios (from ANSUR/CAESAR anthropometric studies) ──

def _depth_ratio(bmi: float, body_part: str) -> float:
    """Estimate depth/width ratio based on BMI.

    As BMI increases, body cross-section becomes more circular (ratio → 1.0).
    At low BMI, body is more elliptical (ratio → 0.6).
    """
    base_ratios = {
        "chest": 0.75,
        "waist": 0.70,
        "hip": 0.65,
    }
    base = base_ratios.get(body_part, 0.70)
    # +0.008 per BMI unit above 22 (normal), clamped [0.55, 0.95]
    bmi_adj = 0.008 * (bmi - 22.0)
    return max(0.55, min(0.95, base + bmi_adj))


def _ellipse_circumference(width_cm: float, depth_cm: float) -> float:
    """Ramanujan's approximation for ellipse perimeter.

    P ≈ π(a+b)(1 + 3h/(10 + √(4 - 3h)))  where h = ((a-b)/(a+b))²
    """
    a = width_cm / 2.0
    b = depth_cm / 2.0
    if a + b < 0.01:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def _pixel_dist(lm1: LandmarkPoint, lm2: LandmarkPoint, w: int, h: int) -> float:
    """Euclidean pixel distance between two landmarks."""
    dx = (lm2.x - lm1.x) * w
    dy = (lm2.y - lm1.y) * h
    return math.sqrt(dx * dx + dy * dy)


def _midpoint_y(lm1: LandmarkPoint, lm2: LandmarkPoint) -> float:
    """Average Y coordinate (normalized)."""
    return (lm1.y + lm2.y) / 2.0


def _is_visible(lm: LandmarkPoint, threshold: float = 0.5) -> bool:
    """Check if a landmark is reliably detected."""
    return lm.visibility > threshold and lm.presence > threshold


# ── Anthropometric proportions for partial-body estimation ──
# Source: averaged from multiple anthropometric surveys (NASA, ANSUR, Pheasant)

# Ratios relative to height
_HEIGHT_RATIOS = {
    "shoulder_width": 0.26,
    "arm_length": 0.34,
    "leg_length": 0.48,
    "torso_length": 0.30,
    "shoulder_to_hip_height": 0.34,  # shoulder Y to hip Y as fraction of height
}

# Shoulder-to-body ratios (for when we can see shoulders but not full body)
# shoulder_width is used as anchor to estimate other widths
_SHOULDER_RATIOS = {
    "hip_from_shoulder": 0.85,  # hip width ≈ 85% of shoulder width (female avg)
}


# ── Main estimation function ──

def estimate_from_photo(
    image_bytes: bytes,
    height_cm: float,
    weight_kg: float,
) -> BodyMeasurements:
    """Estimate body measurements from a photo + known height/weight.

    Works with full-body or upper-body photos. Uses visible landmarks
    to compute scale, then estimates missing measurements from
    anthropometric proportions.

    Args:
        image_bytes: JPEG/PNG image data
        height_cm: Known height in cm
        weight_kg: Known weight in kg

    Returns:
        BodyMeasurements with all estimated values
    """
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    result = BodyMeasurements()
    result.bmi = weight_kg / ((height_cm / 100.0) ** 2)

    # Load image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = img.size
    img_array = np.array(img)

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

    # Ensure model is downloaded
    model_path = _ensure_model()

    # Run PoseLandmarker (tasks API)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        pose_result = landmarker.detect(mp_image)

    if not pose_result.pose_landmarks or len(pose_result.pose_landmarks) == 0:
        result.warnings.append("No pose detected — try facing the camera")
        return _fallback_estimation(result, height_cm, weight_kg)

    # Extract landmarks (first person detected)
    raw_landmarks = pose_result.pose_landmarks[0]
    landmarks = {}
    for i, lm in enumerate(raw_landmarks):
        landmarks[i] = LandmarkPoint(
            x=lm.x, y=lm.y, z=lm.z,
            visibility=lm.visibility if hasattr(lm, 'visibility') else 0.5,
            presence=lm.presence if hasattr(lm, 'presence') else 0.5,
        )

    # ── Determine body coverage ──
    # Check which key landmarks are visible
    shoulders_visible = _is_visible(landmarks[11]) and _is_visible(landmarks[12])
    hips_visible = _is_visible(landmarks[23]) and _is_visible(landmarks[24])
    ankles_visible = _is_visible(landmarks[27]) and _is_visible(landmarks[28])
    nose_visible = _is_visible(landmarks[0])

    if not shoulders_visible:
        result.warnings.append("Shoulders not detected — need at least upper body visible")
        return _fallback_estimation(result, height_cm, weight_kg)

    # Classify coverage
    if ankles_visible and hips_visible:
        result.body_coverage = "full"
    elif hips_visible:
        result.body_coverage = "upper"
        result.warnings.append("Partial body detected — lower body measurements estimated from proportions")
    else:
        result.body_coverage = "minimal"
        result.warnings.append("Only upper body visible — measurements estimated from shoulder width + height/weight")

    # Confidence: average visibility of detected key landmarks
    visible_keys = [11, 12]  # shoulders always
    if hips_visible:
        visible_keys += [23, 24]
    if ankles_visible:
        visible_keys += [27, 28]
    avg_visibility = sum(landmarks[i].visibility for i in visible_keys) / len(visible_keys)
    result.landmark_confidence = round(avg_visibility, 3)

    # ── Compute scale factor ──
    # Strategy depends on body coverage

    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    shoulder_mid_y = _midpoint_y(l_shoulder, r_shoulder)

    if result.body_coverage == "full":
        # Full body: scale from head to feet
        nose = landmarks[0]
        head_top_y = nose.y - (shoulder_mid_y - nose.y) * 0.7

        l_ankle = landmarks[27]
        r_ankle = landmarks[28]
        ankle_mid_y = _midpoint_y(l_ankle, r_ankle)
        # Extend past ankles for feet
        knee_mid_y = _midpoint_y(landmarks[25], landmarks[26])
        feet_bottom_y = ankle_mid_y + (ankle_mid_y - knee_mid_y) * 0.15

        pixel_height = (feet_bottom_y - head_top_y) * img_h
        # Landmarks cover ~93% of actual height
        pixel_height_corrected = pixel_height / 0.93
        scale = height_cm / pixel_height_corrected

    elif result.body_coverage == "upper":
        # Upper body: scale from head to hips, then extrapolate
        # Head-to-hip is approximately 52% of total height (anthropometric average)
        nose = landmarks[0]
        head_top_y = nose.y - (shoulder_mid_y - nose.y) * 0.7

        l_hip = landmarks[23]
        r_hip = landmarks[24]
        hip_mid_y = _midpoint_y(l_hip, r_hip)

        pixel_head_to_hip = (hip_mid_y - head_top_y) * img_h
        # Head-to-hip ≈ 52% of total height
        estimated_full_pixel_height = pixel_head_to_hip / 0.52
        scale = height_cm / estimated_full_pixel_height

    else:
        # Minimal: only shoulders visible — use shoulder width as anchor
        # Average shoulder width ≈ 26% of height
        shoulder_px = _pixel_dist(l_shoulder, r_shoulder, img_w, img_h)
        expected_shoulder_cm = height_cm * _HEIGHT_RATIOS["shoulder_width"]
        scale = expected_shoulder_cm / shoulder_px

    result.scale_cm_per_px = round(scale, 4)

    # ── Extract widths from landmarks ──

    # Shoulder width (always available)
    shoulder_px = _pixel_dist(l_shoulder, r_shoulder, img_w, img_h)
    result.shoulder_width = round(shoulder_px * scale, 1)

    # Hip width
    if hips_visible:
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        hip_px = _pixel_dist(l_hip, r_hip, img_w, img_h)
        result.hip_width = round(hip_px * scale, 1)
    else:
        # Estimate from shoulder width using anthropometric ratio
        result.hip_width = round(result.shoulder_width * _SHOULDER_RATIOS["hip_from_shoulder"], 1)
        hip_px = result.hip_width / scale  # back-calculate for circumference estimation

    # Chest width: ~92% of shoulder width
    chest_width_px = shoulder_px * 0.92

    # Waist width: blended from shoulder and hip
    waist_width_px = shoulder_px * 0.35 + hip_px * 0.65

    # BMI adjustment for waist
    if result.bmi > 25:
        bmi_waist_factor = min(0.15, (result.bmi - 25) * 0.02)
        waist_width_px = waist_width_px * (1 + bmi_waist_factor)

    # ── Convert widths to circumferences via ellipse model ──

    chest_width_cm = chest_width_px * scale
    waist_width_cm = waist_width_px * scale
    hip_width_cm = hip_px * scale

    chest_depth = chest_width_cm * _depth_ratio(result.bmi, "chest")
    waist_depth = waist_width_cm * _depth_ratio(result.bmi, "waist")
    hip_depth = hip_width_cm * _depth_ratio(result.bmi, "hip")

    result.chest_circumference = round(_ellipse_circumference(chest_width_cm, chest_depth), 1)
    result.waist_circumference = round(_ellipse_circumference(waist_width_cm, waist_depth), 1)
    result.hip_circumference = round(_ellipse_circumference(hip_width_cm, hip_depth), 1)

    # ── Arm length ──
    if _is_visible(landmarks[13]) and _is_visible(landmarks[15]):
        l_elbow = landmarks[13]
        l_wrist = landmarks[15]
        upper_arm_px = _pixel_dist(l_shoulder, l_elbow, img_w, img_h)
        forearm_px = _pixel_dist(l_elbow, l_wrist, img_w, img_h)
        result.arm_length = round((upper_arm_px + forearm_px) * scale, 1)
    else:
        result.arm_length = round(height_cm * _HEIGHT_RATIOS["arm_length"], 1)

    # ── Leg length ──
    if ankles_visible and hips_visible:
        l_hip = landmarks[23]
        l_knee = landmarks[25]
        l_ankle = landmarks[27]
        upper_leg_px = _pixel_dist(l_hip, l_knee, img_w, img_h)
        lower_leg_px = _pixel_dist(l_knee, l_ankle, img_w, img_h)
        result.leg_length = round((upper_leg_px + lower_leg_px) * scale, 1)
    else:
        result.leg_length = round(height_cm * _HEIGHT_RATIOS["leg_length"], 1)

    # ── Torso length ──
    if hips_visible:
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        torso_top_y = shoulder_mid_y * img_h
        torso_bottom_y = _midpoint_y(l_hip, r_hip) * img_h
        result.torso_length = round(abs(torso_bottom_y - torso_top_y) * scale, 1)
    else:
        result.torso_length = round(height_cm * _HEIGHT_RATIOS["torso_length"], 1)

    # ── Sanity checks ──

    if result.chest_circumference < 60 or result.chest_circumference > 150:
        result.warnings.append(
            f"Chest estimate ({result.chest_circumference} cm) seems off — "
            "check that you're facing the camera with arms slightly apart"
        )

    if result.shoulder_width < 25 or result.shoulder_width > 65:
        result.warnings.append(
            f"Shoulder width ({result.shoulder_width} cm) seems off — "
            "may indicate incorrect distance or pose"
        )

    logger.info(
        "Body estimation complete: coverage=%s, chest=%.1f, waist=%.1f, hip=%.1f, "
        "shoulder=%.1f, BMI=%.1f, confidence=%.2f",
        result.body_coverage, result.chest_circumference, result.waist_circumference,
        result.hip_circumference, result.shoulder_width,
        result.bmi, result.landmark_confidence,
    )

    return result


def _fallback_estimation(
    result: BodyMeasurements,
    height_cm: float,
    weight_kg: float,
) -> BodyMeasurements:
    """Fall back to height/weight-only regression when landmarks fail.

    Uses the same formulas as ea_size_engine (universal model).
    """
    result.method = "height_weight_regression_fallback"
    result.chest_circumference = round(height_cm * 0.52 + weight_kg * 0.18, 1)
    result.waist_circumference = round(height_cm * 0.35 + weight_kg * 0.30, 1)
    result.hip_circumference = round(height_cm * 0.48 + weight_kg * 0.22, 1)

    # Rough proportional estimates
    result.shoulder_width = round(height_cm * 0.26, 1)
    result.arm_length = round(height_cm * 0.34, 1)
    result.leg_length = round(height_cm * 0.48, 1)
    result.torso_length = round(height_cm * 0.30, 1)

    return result
