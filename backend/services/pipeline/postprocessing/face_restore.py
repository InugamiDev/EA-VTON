"""Face restoration for try-on results.

Fixes VTON face blurring using SSIM-guided face swap:
- Detect face region via mediapipe (lighter than dlib, cross-platform)
- Compare face SSIM between original and result
- When SSIM < threshold, paste original face with feathered alpha blending
- No heavy model (CodeFormer) needed — the original face is already high quality
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# SSIM threshold below which we swap the face
_FACE_SSIM_THRESHOLD = 0.70

# Default feather radius for alpha blending
_DEFAULT_FEATHER_PX = 15


_FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"


def _get_face_model_path() -> str:
    """Download and cache the mediapipe face detection model."""
    import os
    import urllib.request

    model_dir = os.path.expanduser("~/.cache/mediapipe/models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "blaze_face_short_range.tflite")
    if not os.path.exists(path):
        logger.info("Downloading mediapipe face detection model...")
        urllib.request.urlretrieve(_FACE_MODEL_URL, path)
    return path


def detect_face_region(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Detect face bounding box using mediapipe FaceDetector (tasks API).

    Returns:
        (x1, y1, x2, y2) bounding box or None if no face detected.
    """
    try:
        import mediapipe as mp
    except ImportError:
        logger.warning("mediapipe not available, falling back to heuristic face region")
        return _heuristic_face_region(image)

    img_arr = np.array(image)
    h, w = img_arr.shape[:2]

    try:
        model_path = _get_face_model_path()
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5,
        )
        detector = mp.tasks.vision.FaceDetector.create_from_options(options)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_arr)
        result = detector.detect(mp_image)
        detector.close()

        if not result.detections:
            logger.info("No face detected by mediapipe")
            return None

        detection = result.detections[0]
        bbox = detection.bounding_box
        x1 = max(0, bbox.origin_x)
        y1 = max(0, bbox.origin_y)
        x2 = min(w, bbox.origin_x + bbox.width)
        y2 = min(h, bbox.origin_y + bbox.height)
    except Exception as e:
        logger.warning("mediapipe face detection failed: %s, using heuristic", e)
        return _heuristic_face_region(image)

    # Add padding (20% on each side) to capture hair/forehead
    pad_x = int((x2 - x1) * 0.20)
    pad_y = int((y2 - y1) * 0.20)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return (x1, y1, x2, y2)


def _heuristic_face_region(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Heuristic face region: top-center 30% of image.

    Used when mediapipe is unavailable.
    """
    w, h = image.size
    # Assume face is in top 30%, center 50% of image
    x1 = int(w * 0.25)
    y1 = 0
    x2 = int(w * 0.75)
    y2 = int(h * 0.30)
    return (x1, y1, x2, y2)


def _compute_face_ssim(
    result: Image.Image,
    original: Image.Image,
    bbox: tuple[int, int, int, int],
) -> float:
    """Compute SSIM of the face region between result and original."""
    from skimage.metrics import structural_similarity as ssim

    x1, y1, x2, y2 = bbox

    orig_face = original.crop(bbox).resize((128, 128), Image.LANCZOS)
    result_face = result.crop(bbox).resize((128, 128), Image.LANCZOS)

    orig_arr = np.array(orig_face)
    result_arr = np.array(result_face)

    if orig_arr.shape != result_arr.shape:
        return 1.0  # Can't compare, skip restoration

    try:
        score = ssim(orig_arr, result_arr, channel_axis=2, data_range=255)
        return float(score)
    except Exception as e:
        logger.warning("Face SSIM computation failed: %s", e)
        return 1.0  # Skip restoration on error


def restore_face(
    result: Image.Image,
    original: Image.Image,
    face_bbox: tuple[int, int, int, int] | None = None,
    feather_px: int = _DEFAULT_FEATHER_PX,
) -> tuple[Image.Image, dict]:
    """Restore the original face onto the try-on result if degraded.

    Uses SSIM to decide whether restoration is needed. If SSIM < threshold,
    pastes the original face with feathered alpha blending for seamless edges.

    Returns:
        (restored_image, details_dict)
    """
    details = {"face_detected": False, "restored": False}

    # Detect face if bbox not provided
    if face_bbox is None:
        face_bbox = detect_face_region(original)

    if face_bbox is None:
        details["reason"] = "no_face_detected"
        return result, details

    details["face_detected"] = True
    details["face_bbox"] = list(face_bbox)

    # Ensure both images are same size for face swap
    if result.size != original.size:
        original_resized = original.resize(result.size, Image.LANCZOS)
    else:
        original_resized = original

    # Scale bbox if images were different sizes
    orig_w, orig_h = original.size
    res_w, res_h = result.size
    if (orig_w, orig_h) != (res_w, res_h):
        sx, sy = res_w / orig_w, res_h / orig_h
        face_bbox = (
            int(face_bbox[0] * sx), int(face_bbox[1] * sy),
            int(face_bbox[2] * sx), int(face_bbox[3] * sy),
        )

    # Compute face SSIM
    face_ssim = _compute_face_ssim(result, original_resized, face_bbox)
    details["face_ssim"] = round(face_ssim, 4)

    if face_ssim >= _FACE_SSIM_THRESHOLD:
        details["reason"] = "face_quality_acceptable"
        return result, details

    # Face is degraded — swap it from the original
    logger.info("Face SSIM %.3f < %.2f, restoring face", face_ssim, _FACE_SSIM_THRESHOLD)

    x1, y1, x2, y2 = face_bbox
    face_w, face_h = x2 - x1, y2 - y1

    if face_w <= 0 or face_h <= 0:
        details["reason"] = "zero_size_face_bbox"
        return result, details

    # Create feathered alpha mask for smooth blending
    mask = Image.new("L", (face_w, face_h), 255)

    # Apply Gaussian blur to create feathered edges
    if feather_px > 0:
        # Create a slightly smaller white rectangle, then blur
        inner_mask = Image.new("L", (face_w, face_h), 0)
        inner_x1 = feather_px
        inner_y1 = feather_px
        inner_x2 = face_w - feather_px
        inner_y2 = face_h - feather_px

        if inner_x2 > inner_x1 and inner_y2 > inner_y1:
            inner_arr = np.array(inner_mask)
            inner_arr[inner_y1:inner_y2, inner_x1:inner_x2] = 255
            mask = Image.fromarray(inner_arr)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))
        # else: face too small for feathering, use hard mask

    # Extract face from original and paste onto result
    original_face = original_resized.crop(face_bbox)

    result_copy = result.copy()
    result_copy.paste(original_face, (x1, y1), mask)

    details["restored"] = True
    details["feather_px"] = feather_px

    return result_copy, details
