"""Skin tone correction for try-on results.

Fixes VTON skin tone shift:
- Detect skin pixels using HSV thresholds
- Match LAB histogram of skin regions: result skin → original skin
- Apply correction only to skin pixels, leave garment/background untouched
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# HSV ranges for skin detection (covers diverse skin tones)
_SKIN_HSV_LOWER = np.array([0, 20, 70], dtype=np.uint8)
_SKIN_HSV_UPPER = np.array([35, 255, 255], dtype=np.uint8)

# Secondary range for reddish skin tones
_SKIN_HSV_LOWER2 = np.array([340 // 2, 20, 70], dtype=np.uint8)
_SKIN_HSV_UPPER2 = np.array([360 // 2, 255, 255], dtype=np.uint8)


def detect_skin_pixels(image: Image.Image) -> np.ndarray:
    """Detect skin pixels using HSV color space thresholds.

    Returns:
        Boolean mask of shape (H, W) where True = skin pixel.
    """
    img_arr = np.array(image)
    hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)

    # Primary skin range (yellowish-brown tones)
    mask1 = cv2.inRange(hsv, _SKIN_HSV_LOWER, _SKIN_HSV_UPPER)

    # Secondary range (reddish tones)
    mask2 = cv2.inRange(hsv, _SKIN_HSV_LOWER2, _SKIN_HSV_UPPER2)

    combined = cv2.bitwise_or(mask1, mask2)

    # Clean up: morphological open to remove noise, close to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    return combined > 0


def _match_lab_histogram(
    result_pixels: np.ndarray,
    original_pixels: np.ndarray,
) -> np.ndarray:
    """Match LAB color distribution of result skin to original skin.

    Only adjusts L (lightness) and A/B (color) channels to preserve structure.

    Args:
        result_pixels: (N, 3) RGB pixels from result skin region
        original_pixels: (N, 3) RGB pixels from original skin region

    Returns:
        Corrected (N, 3) RGB pixels
    """
    if len(result_pixels) < 100 or len(original_pixels) < 100:
        return result_pixels  # Not enough pixels for reliable matching

    # Convert to LAB
    result_lab = cv2.cvtColor(
        result_pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float64)

    original_lab = cv2.cvtColor(
        original_pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float64)

    # Per-channel mean/std matching
    for c in range(3):
        orig_mean = np.mean(original_lab[:, c])
        orig_std = np.std(original_lab[:, c])
        res_mean = np.mean(result_lab[:, c])
        res_std = np.std(result_lab[:, c])

        if res_std < 1e-6 or orig_std < 1e-6:
            continue

        # Normalize, then transform to match original distribution
        result_lab[:, c] = (result_lab[:, c] - res_mean) * (orig_std / res_std) + orig_mean

    # Clip to valid LAB range and convert back to RGB
    result_lab[:, 0] = np.clip(result_lab[:, 0], 0, 255)
    result_lab[:, 1] = np.clip(result_lab[:, 1], 0, 255)
    result_lab[:, 2] = np.clip(result_lab[:, 2], 0, 255)

    corrected_rgb = cv2.cvtColor(
        result_lab.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB
    ).reshape(-1, 3)

    return corrected_rgb


def match_skin_tone(
    result: Image.Image,
    original: Image.Image,
    blend_factor: float = 0.70,
) -> tuple[Image.Image, dict]:
    """Match skin tone of try-on result to the original photo.

    Detects skin regions in both images, computes LAB histogram correction,
    and applies it only to skin pixels in the result.

    Args:
        result: Try-on result image
        original: Original person photo
        blend_factor: How much correction to apply (0=none, 1=full). Default 0.70.

    Returns:
        (corrected_image, details_dict)
    """
    details = {"corrected": False}

    # Ensure same size
    if result.size != original.size:
        original_resized = original.resize(result.size, Image.LANCZOS)
    else:
        original_resized = original

    # Detect skin in both images
    result_skin_mask = detect_skin_pixels(result)
    original_skin_mask = detect_skin_pixels(original_resized)

    result_skin_count = int(np.sum(result_skin_mask))
    original_skin_count = int(np.sum(original_skin_mask))

    details["result_skin_pixels"] = result_skin_count
    details["original_skin_pixels"] = original_skin_count

    # Need sufficient skin pixels in both images
    min_pixels = 500
    if result_skin_count < min_pixels or original_skin_count < min_pixels:
        details["reason"] = "insufficient_skin_pixels"
        return result, details

    result_arr = np.array(result)
    original_arr = np.array(original_resized)

    # Extract skin pixels
    result_skin_px = result_arr[result_skin_mask]
    original_skin_px = original_arr[original_skin_mask]

    # Match LAB histograms
    corrected_skin_px = _match_lab_histogram(result_skin_px, original_skin_px)

    # Apply correction with blend factor
    corrected_arr = result_arr.copy()
    blended = (
        result_skin_px.astype(np.float64) * (1 - blend_factor)
        + corrected_skin_px.astype(np.float64) * blend_factor
    )
    corrected_arr[result_skin_mask] = np.clip(blended, 0, 255).astype(np.uint8)

    # Compute delta for reporting
    delta_lab = _compute_skin_delta(result_arr, corrected_arr, result_skin_mask)

    details["corrected"] = True
    details["blend_factor"] = blend_factor
    details["skin_delta_lab"] = round(delta_lab, 2)

    return Image.fromarray(corrected_arr), details


def _compute_skin_delta(
    before: np.ndarray,
    after: np.ndarray,
    skin_mask: np.ndarray,
) -> float:
    """Compute mean LAB distance between before and after for skin pixels."""
    before_lab = cv2.cvtColor(before, cv2.COLOR_RGB2LAB).astype(np.float64)
    after_lab = cv2.cvtColor(after, cv2.COLOR_RGB2LAB).astype(np.float64)

    diff = before_lab[skin_mask] - after_lab[skin_mask]
    return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=1))))
