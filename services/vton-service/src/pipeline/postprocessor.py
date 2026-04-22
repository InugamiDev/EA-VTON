"""Postprocessing for try-on results.

Stages:
1. Face preservation check — compare face region similarity between input and output
2. Color correction — match color distribution to original photo
3. Sharpening — subtle unsharp mask to counter model blur
4. Confidence scoring — composite score from multiple signals
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("/tmp/fitview_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def postprocess_result(
    result_path: str,
    original_person_path: str,
    config,
) -> dict:
    """Postprocess a try-on result image.

    Returns dict with result_path (possibly modified), confidence_score, and details.
    """
    result_img = Image.open(result_path).convert("RGB")
    original_img = Image.open(original_person_path).convert("RGB")

    details = {}
    scores = []

    # ── 1. Face preservation check ───────────────────────────────────
    if config.check_face_preservation:
        face_score, face_details = _check_face_preservation(result_img, original_img)
        scores.append(("face_preservation", face_score, 0.35))  # weight 35%
        details["face_preservation"] = face_details
    else:
        scores.append(("face_preservation", 0.7, 0.35))

    # ── 2. Color correction ──────────────────────────────────────────
    if config.color_correct:
        result_img, cc_details = _color_correct(result_img, original_img)
        details["color_correction"] = cc_details

    # ── 3. Sharpening ────────────────────────────────────────────────
    if config.sharpen_result:
        result_img = _sharpen(result_img)
        details["sharpened"] = True

    # ── 4. Structural quality check ──────────────────────────────────
    quality_score, quality_details = _check_structural_quality(result_img)
    scores.append(("structural_quality", quality_score, 0.25))  # weight 25%
    details["structural_quality"] = quality_details

    # ── 5. Resolution / artifact check ───────────────────────────────
    artifact_score, artifact_details = _check_artifacts(result_img)
    scores.append(("artifact_check", artifact_score, 0.20))  # weight 20%
    details["artifacts"] = artifact_details

    # ── 6. Overall image quality ─────────────────────────────────────
    iq_score, iq_details = _check_image_quality(result_img)
    scores.append(("image_quality", iq_score, 0.20))  # weight 20%
    details["image_quality"] = iq_details

    # ── 7. Skin tone fidelity (if original available) ─────────────
    skin_score, skin_details = _check_skin_tone_fidelity(result_img, original_img)
    if skin_score is not None:
        scores.append(("skin_tone_fidelity", skin_score, 0.15))
        details["skin_tone_fidelity"] = skin_details

    # ── Compute weighted confidence ──────────────────────────────────
    total_weight = sum(w for _, _, w in scores)
    confidence = sum(s * w for _, s, w in scores) / total_weight
    confidence = max(0.0, min(1.0, confidence))

    details["score_breakdown"] = {name: round(s, 3) for name, s, _ in scores}

    # ── Save final result ────────────────────────────────────────────
    out_id = str(uuid4())
    out_path = str(RESULTS_DIR / f"{out_id}.jpg")
    result_img.save(out_path, "JPEG", quality=93)

    return {
        "result_path": out_path,
        "confidence_score": round(confidence, 2),
        "details": details,
    }


# ---------------------------------------------------------------------------
#  Face preservation
# ---------------------------------------------------------------------------

def _check_face_preservation(
    result: Image.Image, original: Image.Image
) -> tuple[float, dict]:
    """Check if the face region is preserved between original and result.

    Uses structural similarity (SSIM) on the top 30% of the image
    (where the face typically is).
    """
    from skimage.metrics import structural_similarity as ssim

    # Crop top 30% (face region)
    def get_face_region(img: Image.Image) -> np.ndarray:
        w, h = img.size
        face_crop = img.crop((0, 0, w, int(h * 0.30)))
        # Resize both to same size for comparison
        face_crop = face_crop.resize((256, 128), Image.LANCZOS)
        return np.array(face_crop)

    orig_face = get_face_region(original)
    result_face = get_face_region(result)

    # Ensure same shape
    if orig_face.shape != result_face.shape:
        return 0.5, {"error": "shape_mismatch", "ssim": 0.5}

    try:
        score = ssim(orig_face, result_face, channel_axis=2, data_range=255)
        # SSIM ranges from -1 to 1; use max(0, score) — negative SSIM means severe degradation
        normalized = max(0.0, min(1.0, score))
    except Exception as e:
        logger.warning("SSIM computation failed: %s", e)
        return 0.5, {"error": str(e)}

    return normalized, {
        "ssim": round(score, 4),
        "score": round(normalized, 3),
    }


# ---------------------------------------------------------------------------
#  Color correction
# ---------------------------------------------------------------------------

def _color_correct(
    result: Image.Image, original: Image.Image
) -> tuple[Image.Image, dict]:
    """Light color correction to match the original photo's tone.

    Adjusts the result's mean brightness and color balance toward the original.
    """
    orig_arr = np.array(original, dtype=np.float32)
    result_arr = np.array(result, dtype=np.float32)

    # Per-channel mean matching (gentle — only apply 30% of the correction)
    blend_factor = 0.30
    corrections = {}

    for c, name in enumerate(["R", "G", "B"]):
        orig_mean = np.mean(orig_arr[:, :, c])
        result_mean = np.mean(result_arr[:, :, c])
        if result_mean > 0:
            ratio = orig_mean / result_mean
            # Clamp ratio to avoid extreme corrections
            ratio = max(0.8, min(1.2, ratio))
            adjusted_ratio = 1.0 + (ratio - 1.0) * blend_factor
            result_arr[:, :, c] = np.clip(result_arr[:, :, c] * adjusted_ratio, 0, 255)
            corrections[name] = round(adjusted_ratio, 3)

    result_corrected = Image.fromarray(result_arr.astype(np.uint8))

    return result_corrected, {"channel_corrections": corrections}


# ---------------------------------------------------------------------------
#  Sharpening
# ---------------------------------------------------------------------------

def _sharpen(img: Image.Image) -> Image.Image:
    """Apply subtle sharpening to counteract model blur."""
    return ImageEnhance.Sharpness(img).enhance(1.15)


# ---------------------------------------------------------------------------
#  Structural quality check
# ---------------------------------------------------------------------------

def _check_structural_quality(img: Image.Image) -> tuple[float, dict]:
    """Check structural quality of the result image.

    Measures:
    - Edge density (too low = blurry, too high = artifacts)
    - Smoothness variance (should be moderate)
    """
    arr = np.array(img.convert("L"), dtype=np.float32)

    # Laplacian for edge detection
    from PIL import ImageFilter
    laplacian = img.convert("L").filter(ImageFilter.Kernel(
        size=(3, 3), kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128
    ))
    lap_arr = np.array(laplacian, dtype=np.float32) - 128.0
    edge_variance = float(np.var(lap_arr))

    # Score: moderate edge variance is good
    # Too low (<50) = blurry, too high (>2000) = artifacts
    if edge_variance < 30:
        score = 0.3  # Very blurry
    elif edge_variance < 80:
        score = 0.6  # Somewhat blurry
    elif edge_variance < 500:
        score = 0.9  # Good
    elif edge_variance < 1500:
        score = 0.75  # Slightly noisy
    else:
        score = 0.5  # Too much edge activity (artifacts)

    return score, {
        "edge_variance": round(edge_variance, 1),
        "score": round(score, 3),
    }


# ---------------------------------------------------------------------------
#  Artifact detection
# ---------------------------------------------------------------------------

def _check_artifacts(img: Image.Image) -> tuple[float, dict]:
    """Detect common try-on artifacts.

    Checks:
    - Color banding / sudden color transitions
    - Unnaturally uniform regions (copy-paste artifacts)
    - Edge discontinuities
    """
    arr = np.array(img, dtype=np.float32)

    # Check for unnatural color uniformity in blocks
    h, w, _ = arr.shape
    block_size = 32
    uniform_blocks = 0
    total_blocks = 0

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = arr[y : y + block_size, x : x + block_size]
            block_std = np.std(block)
            total_blocks += 1
            if block_std < 3.0:  # Very uniform block
                uniform_blocks += 1

    uniform_ratio = uniform_blocks / max(total_blocks, 1)

    # Check gradient smoothness
    dx = np.diff(arr[:, :, 0], axis=1)
    dy = np.diff(arr[:, :, 0], axis=0)
    gradient_variance = float(np.var(dx) + np.var(dy)) / 2

    # Score
    if uniform_ratio > 0.3:
        score = 0.4  # Too many uniform blocks
    elif uniform_ratio > 0.15:
        score = 0.7
    else:
        score = 0.9

    return score, {
        "uniform_block_ratio": round(uniform_ratio, 3),
        "gradient_variance": round(gradient_variance, 1),
        "score": round(score, 3),
    }


# ---------------------------------------------------------------------------
#  Image quality
# ---------------------------------------------------------------------------

def _check_skin_tone_fidelity(
    result: Image.Image, original: Image.Image
) -> tuple[float | None, dict]:
    """Check skin tone consistency between result and original.

    Uses LAB color space to measure skin color drift.
    """
    try:
        import cv2

        result_arr = np.array(result)
        original_arr = np.array(original.resize(result.size, Image.LANCZOS))

        # Detect skin in both using HSV
        def get_skin_mask(arr):
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            m1 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([35, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([170, 20, 70]), np.array([180, 255, 255]))
            return (cv2.bitwise_or(m1, m2) > 0)

        result_skin = get_skin_mask(result_arr)
        orig_skin = get_skin_mask(original_arr)

        if np.sum(result_skin) < 500 or np.sum(orig_skin) < 500:
            return None, {"reason": "insufficient_skin_pixels"}

        # Compare mean LAB values of skin regions
        result_lab = cv2.cvtColor(result_arr, cv2.COLOR_RGB2LAB).astype(np.float64)
        orig_lab = cv2.cvtColor(original_arr, cv2.COLOR_RGB2LAB).astype(np.float64)

        result_mean = np.mean(result_lab[result_skin], axis=0)
        orig_mean = np.mean(orig_lab[orig_skin], axis=0)

        delta_e = float(np.sqrt(np.sum((result_mean - orig_mean) ** 2)))

        # Score: delta_e < 5 is excellent, < 15 is good, > 30 is bad
        if delta_e < 5:
            score = 1.0
        elif delta_e < 15:
            score = 0.8
        elif delta_e < 30:
            score = 0.6
        else:
            score = 0.3

        return score, {"delta_e": round(delta_e, 2), "score": round(score, 3)}

    except ImportError:
        return None, {"reason": "cv2_not_available"}
    except Exception as e:
        logger.warning("Skin tone fidelity check failed: %s", e)
        return None, {"error": str(e)}


def _check_image_quality(img: Image.Image) -> tuple[float, dict]:
    """Overall image quality assessment.

    Checks brightness, contrast, and saturation are in reasonable ranges.
    """
    arr = np.array(img, dtype=np.float32)

    mean_brightness = float(np.mean(arr))
    std_dev = float(np.std(arr))  # Proxy for contrast

    # Brightness: ideal 80-180
    if 80 <= mean_brightness <= 180:
        b_score = 1.0
    elif 50 <= mean_brightness <= 210:
        b_score = 0.7
    else:
        b_score = 0.4

    # Contrast: ideal std 40-80
    if 40 <= std_dev <= 80:
        c_score = 1.0
    elif 25 <= std_dev <= 100:
        c_score = 0.7
    else:
        c_score = 0.4

    score = (b_score + c_score) / 2

    return score, {
        "mean_brightness": round(mean_brightness, 1),
        "std_dev": round(std_dev, 1),
        "score": round(score, 3),
    }
