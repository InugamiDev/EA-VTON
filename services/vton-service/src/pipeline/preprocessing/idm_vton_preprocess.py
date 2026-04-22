"""IDM-VTON-specific preprocessing.

IDM-VTON requires DensePose UV maps + agnostic masks + human parsing that
the standard preprocessor doesn't produce.

Simplified approach (fixes heavy preprocessing flaw):
- Use detectron2 DensePose (R_50_FPN, lighter than R_101) for body UV map
- Skip OpenPose — derive agnostic mask from DensePose body part labels
- Skip separate parsing model — map DensePose 24 body parts to semantic regions
- Mask refinement: morphological close + Gaussian blur edges + dilation

Fallback: mediapipe pose estimation if detectron2 unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

PREPROCESS_DIR = Path("/tmp/fitview_preprocess")
PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)

# DensePose body part indices for torso region
_TORSO_PARTS = {1, 2}  # Front torso, back torso
_UPPER_BODY_PARTS = {1, 2, 3, 4}  # Torso + upper arms
_LOWER_BODY_PARTS = {7, 8, 9, 10}  # Upper legs, lower legs
_ARM_PARTS = {3, 4, 5, 6}  # Upper/lower arms

# Garment type → which DensePose parts to mask
_GARMENT_PART_MAP = {
    "upper_body": _TORSO_PARTS | _ARM_PARTS,
    "lower_body": _LOWER_BODY_PARTS,
    "full_body": _TORSO_PARTS | _ARM_PARTS | _LOWER_BODY_PARTS,
}


def _has_detectron2() -> bool:
    """Check if detectron2 is available."""
    try:
        import detectron2  # noqa: F401
        return True
    except ImportError:
        return False


def _has_mediapipe() -> bool:
    """Check if mediapipe is available."""
    try:
        import mediapipe  # noqa: F401
        return True
    except ImportError:
        return False


def generate_densepose(person_image: Image.Image) -> np.ndarray | None:
    """Generate DensePose IUV map from person image.

    Returns:
        IUV array of shape (H, W, 3) where:
        - Channel 0: body part index (0-24)
        - Channel 1: U coordinate
        - Channel 2: V coordinate
        Returns None if detectron2 is unavailable.
    """
    if not _has_detectron2():
        logger.warning("detectron2 not available, skipping DensePose")
        return None

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from densepose import add_densepose_config
    from densepose.vis.extractor import DensePoseResultExtractor

    img_arr = np.array(person_image)[:, :, ::-1]  # RGB → BGR for detectron2

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(
        "detectron2://densepose_rcnn_R_50_FPN_s1x.yaml"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = (
        "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    )

    predictor = DefaultPredictor(cfg)
    outputs = predictor(img_arr)

    instances = outputs["instances"]
    if len(instances) == 0:
        logger.warning("No person detected by DensePose")
        return None

    # Extract DensePose results for the first detected person
    extractor = DensePoseResultExtractor()
    densepose_data = extractor(instances)[0]

    # Build IUV map at image resolution
    h, w = img_arr.shape[:2]
    iuv = np.zeros((h, w, 3), dtype=np.float32)

    # densepose_data contains labels and UV coordinates
    if hasattr(densepose_data, "labels") and hasattr(densepose_data, "uv"):
        box = instances.pred_boxes.tensor[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        labels = densepose_data.labels.cpu().numpy()
        uv = densepose_data.uv.cpu().numpy()

        # Resize labels/UV to box size
        bh, bw = y2 - y1, x2 - x1
        if bh > 0 and bw > 0:
            labels_resized = cv2.resize(
                labels.astype(np.float32), (bw, bh), interpolation=cv2.INTER_NEAREST
            )
            u_resized = cv2.resize(uv[0], (bw, bh), interpolation=cv2.INTER_LINEAR)
            v_resized = cv2.resize(uv[1], (bw, bh), interpolation=cv2.INTER_LINEAR)

            # Place into full image
            y1c, y2c = max(0, y1), min(h, y2)
            x1c, x2c = max(0, x1), min(w, x2)
            sy, sx = y1c - y1, x1c - x1
            eh, ew = y2c - y1c, x2c - x1c

            iuv[y1c:y2c, x1c:x2c, 0] = labels_resized[sy : sy + eh, sx : sx + ew]
            iuv[y1c:y2c, x1c:x2c, 1] = u_resized[sy : sy + eh, sx : sx + ew]
            iuv[y1c:y2c, x1c:x2c, 2] = v_resized[sy : sy + eh, sx : sx + ew]

    return iuv


_POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


def _get_pose_model_path() -> str:
    """Download and cache the mediapipe pose landmarker model."""
    import os
    import urllib.request

    model_dir = os.path.expanduser("~/.cache/mediapipe/models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "pose_landmarker_lite.task")
    if not os.path.exists(path):
        logger.info("Downloading mediapipe pose landmarker model...")
        urllib.request.urlretrieve(_POSE_MODEL_URL, path)
    return path


def _generate_mediapipe_body_mask(
    person_image: Image.Image, garment_type: str = "upper_body"
) -> Image.Image:
    """Fallback: generate body region mask using mediapipe pose estimation (tasks API).

    Less accurate than DensePose but works on macOS without detectron2.
    """
    import mediapipe as mp

    img_arr = np.array(person_image)
    h, w = img_arr.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    try:
        model_path = _get_pose_model_path()
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
        )
        landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_arr)
        result = landmarker.detect(mp_image)
        landmarker.close()

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            logger.warning("MediaPipe detected no pose, returning center mask")
            y_start, y_end = int(h * 0.15), int(h * 0.70)
            x_start, x_end = int(w * 0.20), int(w * 0.80)
            mask[y_start:y_end, x_start:x_end] = 255
            return Image.fromarray(mask)

        landmarks = result.pose_landmarks[0]
    except Exception as e:
        logger.warning("MediaPipe pose detection failed: %s, returning center mask", e)
        y_start, y_end = int(h * 0.15), int(h * 0.70)
        x_start, x_end = int(w * 0.20), int(w * 0.80)
        mask[y_start:y_end, x_start:x_end] = 255
        return Image.fromarray(mask)

    # Extract key body landmarks as pixel coordinates
    def lm_px(idx):
        lm = landmarks[idx]
        return int(lm.x * w), int(lm.y * h)

    # Key points
    left_shoulder = lm_px(11)
    right_shoulder = lm_px(12)
    left_hip = lm_px(23)
    right_hip = lm_px(24)
    left_elbow = lm_px(13)
    right_elbow = lm_px(14)
    left_wrist = lm_px(15)
    right_wrist = lm_px(16)

    if garment_type in ("upper_body", "full_body"):
        # Create polygon covering torso + arms
        torso_poly = np.array([
            right_shoulder, left_shoulder,
            left_elbow, left_wrist,
            left_hip, right_hip,
            right_wrist, right_elbow,
        ], dtype=np.int32)

        # Also fill a simpler torso rectangle with padding
        all_x = [p[0] for p in [left_shoulder, right_shoulder, left_hip, right_hip,
                                  left_elbow, right_elbow, left_wrist, right_wrist]]
        all_y = [p[1] for p in [left_shoulder, right_shoulder, left_hip, right_hip,
                                  left_elbow, right_elbow, left_wrist, right_wrist]]

        pad_x = int(w * 0.05)
        pad_y = int(h * 0.03)

        x_min = max(0, min(all_x) - pad_x)
        x_max = min(w, max(all_x) + pad_x)
        y_min = max(0, min(all_y) - pad_y)
        y_max = min(h, max(all_y) + pad_y)

        mask[y_min:y_max, x_min:x_max] = 255

        # Also fill the torso polygon for more precision
        cv2.fillPoly(mask, [torso_poly], 255)

    if garment_type in ("lower_body", "full_body"):
        left_knee = lm_px(25)
        right_knee = lm_px(26)
        left_ankle = lm_px(27)
        right_ankle = lm_px(28)

        lower_poly = np.array([
            left_hip, right_hip,
            right_knee, right_ankle,
            left_ankle, left_knee,
        ], dtype=np.int32)
        cv2.fillPoly(mask, [lower_poly], 255)

    return Image.fromarray(mask)


def generate_agnostic_mask(
    densepose_iuv: np.ndarray | None,
    person_image: Image.Image,
    garment_type: str = "upper_body",
) -> Image.Image:
    """Generate agnostic mask from DensePose body part labels.

    The agnostic mask marks which pixels to replace with the garment.
    Falls back to mediapipe if DensePose unavailable.
    """
    if densepose_iuv is None:
        return _generate_mediapipe_body_mask(person_image, garment_type)

    parts_to_mask = _GARMENT_PART_MAP.get(garment_type, _GARMENT_PART_MAP["upper_body"])
    part_labels = densepose_iuv[:, :, 0].astype(int)

    # Create binary mask from body part labels
    mask = np.zeros(part_labels.shape, dtype=np.uint8)
    for part_id in parts_to_mask:
        mask[part_labels == part_id] = 255

    # Mask refinement (fixes mask sensitivity flaw)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    mask = cv2.dilate(mask, kernel, iterations=1)  # 5px expansion
    mask = cv2.GaussianBlur(mask, (11, 11), 0)  # Smooth edges
    mask = (mask > 127).astype(np.uint8) * 255  # Re-binarize

    return Image.fromarray(mask)


def generate_agnostic_image(
    person_image: Image.Image, agnostic_mask: Image.Image
) -> Image.Image:
    """Create agnostic person image (garment region erased).

    Fills masked region with gray to match IDM-VTON's expected input.
    """
    person_arr = np.array(person_image)
    mask_arr = np.array(agnostic_mask.resize(person_image.size, Image.NEAREST))

    # Normalize mask to 0-1
    mask_float = mask_arr.astype(np.float32) / 255.0

    # Fill masked region with neutral gray (128, 128, 128)
    gray = np.full_like(person_arr, 128)
    result = (person_arr * (1 - mask_float[:, :, None]) + gray * mask_float[:, :, None])

    return Image.fromarray(result.astype(np.uint8))


def prepare_idm_vton_inputs(
    person_path: str,
    garment_path: str | None,
    config,
) -> dict:
    """Orchestrate all IDM-VTON-specific preprocessing.

    Generates DensePose, agnostic mask, and agnostic image needed by the model.

    Returns:
        dict with paths to generated files and details
    """
    person_img = Image.open(person_path).convert("RGB")
    garment_type = getattr(config, "garment_type", "upper_body")

    run_id = str(uuid4())
    details = {"method": "densepose" if _has_detectron2() else "mediapipe"}

    # Step 1: Generate DensePose IUV map (or None if unavailable)
    densepose_iuv = None
    if _has_detectron2():
        try:
            densepose_iuv = generate_densepose(person_img)
            details["densepose"] = "success" if densepose_iuv is not None else "no_person_detected"
        except Exception as e:
            logger.warning("DensePose generation failed: %s", e)
            details["densepose"] = f"failed: {e}"

    if densepose_iuv is None and not _has_mediapipe():
        logger.warning("Neither detectron2 nor mediapipe available, skipping IDM-VTON preprocess")
        return {"skipped": True, "details": details}

    # Step 2: Generate agnostic mask
    agnostic_mask = generate_agnostic_mask(densepose_iuv, person_img, garment_type)
    mask_path = str(PREPROCESS_DIR / f"agnostic_mask_{run_id}.png")
    agnostic_mask.save(mask_path, "PNG")
    details["agnostic_mask_path"] = mask_path

    # Step 3: Generate agnostic image (person with garment erased)
    agnostic_img = generate_agnostic_image(person_img, agnostic_mask)
    agnostic_path = str(PREPROCESS_DIR / f"agnostic_{run_id}.png")
    agnostic_img.save(agnostic_path, "PNG")
    details["agnostic_image_path"] = agnostic_path

    # Step 4: Save DensePose visualization if available
    if densepose_iuv is not None:
        densepose_vis = (densepose_iuv[:, :, 0] * 10).clip(0, 255).astype(np.uint8)
        densepose_path = str(PREPROCESS_DIR / f"densepose_{run_id}.png")
        Image.fromarray(densepose_vis).save(densepose_path, "PNG")
        details["densepose_path"] = densepose_path

    return {
        "skipped": False,
        "agnostic_mask_path": mask_path,
        "agnostic_image_path": agnostic_path,
        "densepose_iuv": densepose_iuv,
        "details": details,
    }
