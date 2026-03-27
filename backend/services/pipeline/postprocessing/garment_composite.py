"""Garment-masked compositing for try-on results.

// intent: mask the NEW garment in the VTON result (not the old garment diff),
//         enhance only the garment, paste original person everywhere else
// status: done
// confidence: high

Pipeline:
1. Run FashnHumanParser on VTON result → semantic segmentation
2. Extract garment labels (top/dress/pants/skirt) → garment mask
3. Enhance garment only (lighting transfer + sharpening) inside mask
4. Composite: original pixels outside mask + enhanced garment inside + feathered boundary
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Map pipeline garment_type → parser category
_TYPE_TO_CATEGORY = {
    "upper_body": "tops",
    "lower_body": "bottoms",
    "full_body": "one-pieces",
    "dress": "one-pieces",
    # passthrough for already-correct values
    "tops": "tops",
    "bottoms": "bottoms",
    "one-pieces": "one-pieces",
}

# Garment label IDs from FashnHumanParser
# 3=top, 4=dress, 5=skirt, 6=pants, 7=belt, 10=scarf
_GARMENT_LABEL_IDS = {
    "tops": {3},                    # top
    "bottoms": {5, 6},              # skirt, pants
    "one-pieces": {3, 4, 5, 6},    # top, dress, skirt, pants
}

# Singleton parser instance (heavy model, load once)
_parser = None
_parser_device = None


def _get_parser(device: str = "cpu"):
    """Get or create the singleton FashnHumanParser."""
    global _parser, _parser_device
    if _parser is None or _parser_device != device:
        from fashn_human_parser import FashnHumanParser
        logger.info("Loading FashnHumanParser on %s...", device)
        _parser = FashnHumanParser(device=device)
        _parser_device = device
        logger.info("FashnHumanParser ready")
    return _parser


def detect_garment_mask(
    result: Image.Image,
    category: str = "tops",
    device: str = "cpu",
    feather_radius: float = 5.0,
) -> np.ndarray:
    """Detect the garment region in the VTON result using semantic segmentation.

    Uses FashnHumanParser to parse the result image into 18 semantic classes,
    then extracts the garment labels matching the category.

    Args:
        result: VTON output image.
        category: Garment category ("tops", "bottoms", "one-pieces").
        device: Device for the parser model.
        feather_radius: Gaussian blur sigma for feathered edges.

    Returns:
        Soft float32 mask [0, 1] where 1 = garment, 0 = not garment.
    """
    parser = _get_parser(device)

    # Run semantic segmentation
    seg_map = parser.predict(result)  # (H, W) with label IDs 0-17

    # Extract garment labels for this category
    label_ids = _GARMENT_LABEL_IDS.get(category, {3})
    garment_binary = np.isin(seg_map, list(label_ids)).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    garment_binary = cv2.morphologyEx(garment_binary, cv2.MORPH_CLOSE, kernel)
    garment_binary = cv2.morphologyEx(garment_binary, cv2.MORPH_OPEN, kernel)

    # Feather edges for seamless compositing
    soft_mask = garment_binary.astype(np.float32) / 255.0
    if feather_radius > 0:
        soft_mask = cv2.GaussianBlur(soft_mask, (0, 0), sigmaX=feather_radius)

    return soft_mask


def detect_garment_mask_by_diff(
    result: Image.Image,
    original: Image.Image,
    threshold: int = 25,
    min_area: int = 5000,
) -> np.ndarray:
    """Fallback: detect garment by diffing result vs original.

    Used when the semantic parser is unavailable.
    """
    if result.size != original.size:
        result = result.resize(original.size, Image.LANCZOS)

    res_arr = np.array(result, dtype=np.float32)
    orig_arr = np.array(original, dtype=np.float32)
    diff = np.mean(np.abs(res_arr - orig_arr), axis=2)
    binary = (diff > threshold).astype(np.uint8) * 255

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(binary)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean_mask, [c], -1, 255, cv2.FILLED)

    soft_mask = clean_mask.astype(np.float32) / 255.0
    soft_mask = cv2.GaussianBlur(soft_mask, (0, 0), sigmaX=5)
    return soft_mask


def enhance_garment_region(
    result: Image.Image,
    mask: np.ndarray,
    sharpen_strength: float = 1.5,
) -> Image.Image:
    """Sharpen garment texture only (inside mask)."""
    arr = np.array(result, dtype=np.float32)
    low_freq = cv2.GaussianBlur(arr, (0, 0), sigmaX=3)
    high_freq = arr - low_freq
    mask_3ch = mask[:, :, np.newaxis]
    sharpened = arr + high_freq * (sharpen_strength - 1.0) * mask_3ch
    return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))


def transfer_lighting(
    result: Image.Image,
    original: Image.Image,
    mask: np.ndarray,
    strength: float = 0.6,
) -> tuple[Image.Image, dict]:
    """Transfer scene lighting from original photo onto the garment only.

    Estimates light direction from the original photo's garment region
    (where the old garment was — same torso area), then applies a matching
    luminance gradient to the new garment inside the mask.
    """
    if result.size != original.size:
        original = original.resize(result.size, Image.LANCZOS)

    res_arr = np.array(result, dtype=np.float32)
    orig_arr = np.array(original, dtype=np.float32)
    h, w = mask.shape

    orig_lab = cv2.cvtColor(orig_arr.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    orig_L = orig_lab[:, :, 0]

    hard_mask = mask > 0.5
    if np.sum(hard_mask) < 1000:
        return result, {"lighting_transferred": False, "reason": "mask_too_small"}

    # Estimate light direction from original's torso region (same area)
    ys, xs = np.where(hard_mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    y_mid = (y_min + y_max) // 2
    x_mid = (x_min + x_max) // 2

    left_mask = hard_mask.copy(); left_mask[:, x_mid:] = False
    right_mask = hard_mask.copy(); right_mask[:, :x_mid] = False
    top_mask = hard_mask.copy(); top_mask[y_mid:, :] = False
    bottom_mask = hard_mask.copy(); bottom_mask[:y_mid, :] = False

    left_L = np.mean(orig_L[left_mask]) if np.sum(left_mask) > 100 else 128
    right_L = np.mean(orig_L[right_mask]) if np.sum(right_mask) > 100 else 128
    top_L = np.mean(orig_L[top_mask]) if np.sum(top_mask) > 100 else 128
    bottom_L = np.mean(orig_L[bottom_mask]) if np.sum(bottom_mask) > 100 else 128

    lr_diff = float(right_L - left_L)
    tb_diff = float(bottom_L - top_L)

    # Create directional gradient
    y_coords = np.arange(h, dtype=np.float64)
    x_coords = np.arange(w, dtype=np.float64)
    xx, yy = np.meshgrid(x_coords, y_coords)

    x_norm = (xx - x_mid) / max((x_max - x_min) / 2.0, 1)
    y_norm = (yy - y_mid) / max((y_max - y_min) / 2.0, 1)
    gradient = np.clip(x_norm * lr_diff * 0.5 + y_norm * tb_diff * 0.5, -25, 25)

    # Contrast matching
    res_lab = cv2.cvtColor(res_arr.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    orig_garment_L_std = np.std(orig_L[hard_mask])
    res_garment_L = res_lab[:, :, 0]
    res_garment_L_mean = np.mean(res_garment_L[hard_mask])
    res_garment_L_std = np.std(res_garment_L[hard_mask])

    std_ratio = 1.0
    if res_garment_L_std > 1e-6 and orig_garment_L_std > 1e-6:
        std_ratio = max(0.7, min(1.4, orig_garment_L_std / res_garment_L_std))

    # Apply contrast + gradient inside mask only
    mask_float = mask.astype(np.float64)
    adjusted_L = res_lab[:, :, 0].copy()
    garment_L = adjusted_L[hard_mask]
    adjusted_L[hard_mask] = (garment_L - res_garment_L_mean) * std_ratio + res_garment_L_mean
    adjusted_L += gradient * mask_float * strength

    orig_result_L = cv2.cvtColor(res_arr.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)[:, :, 0]
    res_lab[:, :, 0] = np.clip(
        orig_result_L * (1.0 - mask_float) + adjusted_L * mask_float,
        0, 255
    )

    result_rgb = cv2.cvtColor(res_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(result_rgb), {
        "lighting_transferred": True,
        "light_direction_lr": round(lr_diff, 1),
        "light_direction_tb": round(tb_diff, 1),
        "contrast_ratio": round(float(std_ratio), 3),
        "strength": strength,
    }


def composite_garment(
    result: Image.Image,
    original: Image.Image,
    mask: np.ndarray | None = None,
    sharpen_strength: float = 1.5,
    lighting_strength: float = 0.6,
    category: str = "tops",
) -> tuple[Image.Image, dict]:
    """Composite VTON garment onto original photo using two masks.

    Two-mask approach:
    - **Garment mask** (semantic parser): where the NEW garment is in the result.
      Used for enhancement (lighting transfer + sharpening) — only the garment.
    - **Changed mask** (diff): everything VTON modified (garment + surrounding skin/bg).
      Used for compositing — prevents old garment from bleeding through.

    Result:
    - Inside garment mask: enhanced VTON (sharpened + relit garment)
    - Inside changed mask but outside garment: raw VTON pixels (transition zone)
    - Outside changed mask: original photo pixels (untouched face, hair, bg)

    Args:
        result: VTON output.
        original: Original person photo.
        mask: Pre-computed garment mask, or None to auto-detect via parser.
        sharpen_strength: Garment sharpening (1.0=none, 1.5=moderate, 2.0=strong).
        lighting_strength: Scene lighting transfer (0=none, 0.6=default, 1.0=full).
        category: Garment category for semantic parsing ("tops"/"bottoms"/"one-pieces").

    Returns:
        (composited_image, details_dict)
    """
    # Ensure same size
    if result.size != original.size:
        result = result.resize(original.size, Image.LANCZOS)

    # Normalize category name
    category = _TYPE_TO_CATEGORY.get(category, category)

    # ── Mask 1: Semantic garment mask on RESULT (for enhancement) ─────
    # Where the NEW garment is → apply lighting + sharpening here
    if mask is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            garment_mask = detect_garment_mask(result, category=category, device=device)
            mask_method = "semantic_parser"
        except Exception as e:
            logger.warning("Semantic parser failed, falling back to diff: %s", e)
            garment_mask = detect_garment_mask_by_diff(result, original)
            mask_method = "diff_fallback"
    else:
        garment_mask = mask
        mask_method = "provided"

    # ── Mask 2: Compositing mask (union of all changed regions) ──────
    # Must cover: old garment + new garment + diff transition zone
    # This prevents ANY old garment pixels from bleeding through
    diff_mask = detect_garment_mask_by_diff(result, original)

    # Also parse the ORIGINAL to find where the old garment was
    try:
        old_garment_mask = detect_garment_mask(original, category=category, device=device)
    except Exception:
        old_garment_mask = np.zeros_like(diff_mask)

    # Dilate old garment mask to cover hem/edges the parser might miss
    # The parser often stops a few pixels short of the actual garment edge
    old_binary = (old_garment_mask > 0.3).astype(np.uint8) * 255
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    old_binary = cv2.dilate(old_binary, dilate_kernel)
    old_garment_expanded = old_binary.astype(np.float32) / 255.0

    # Union: expanded old garment ∪ new garment ∪ diff
    changed_mask = np.maximum(np.maximum(diff_mask, garment_mask), old_garment_expanded)
    # Re-feather the union mask for smooth edges
    changed_binary = (changed_mask > 0.3).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    changed_binary = cv2.morphologyEx(changed_binary, cv2.MORPH_CLOSE, kernel)
    changed_mask = cv2.GaussianBlur(
        changed_binary.astype(np.float32) / 255.0, (0, 0), sigmaX=5
    )

    garment_pixels = int(np.sum(garment_mask > 0.5))
    changed_pixels = int(np.sum(changed_mask > 0.5))
    total_pixels = garment_mask.shape[0] * garment_mask.shape[1]

    if garment_pixels < 1000:
        return result, {
            "composited": False,
            "reason": "no_garment_region_detected",
            "mask_method": mask_method,
        }

    details = {
        "composited": True,
        "mask_method": mask_method,
        "garment_pixels": garment_pixels,
        "garment_ratio": round(garment_pixels / total_pixels, 3),
        "changed_pixels": changed_pixels,
        "changed_ratio": round(changed_pixels / total_pixels, 3),
        "category": category,
    }

    enhanced_result = result

    # 1. Transfer lighting onto garment only (semantic mask)
    if lighting_strength > 0:
        enhanced_result, light_details = transfer_lighting(
            enhanced_result, original, garment_mask, lighting_strength
        )
        details["lighting"] = light_details

    # 2. Sharpen garment texture only (semantic mask)
    if sharpen_strength > 1.0:
        enhanced_result = enhance_garment_region(enhanced_result, garment_mask, sharpen_strength)
        details["sharpen_strength"] = sharpen_strength

    # 3. Composite using CHANGED mask (not garment mask)
    #    This ensures old garment doesn't bleed through in transition zones
    orig_arr = np.array(original, dtype=np.float32)
    res_arr = np.array(enhanced_result, dtype=np.float32)
    changed_3ch = changed_mask[:, :, np.newaxis]

    composited = orig_arr * (1.0 - changed_3ch) + res_arr * changed_3ch

    return Image.fromarray(np.clip(composited, 0, 255).astype(np.uint8)), details
