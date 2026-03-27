"""Detail enhancement for try-on results.

Recovers fine detail lost during diffusion:
- Frequency-separated sharpening (sharpen high-freq, preserve low-freq)
- Edge-aware upscaling back to original input resolution
- Garment boundary softening via guided filter
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def frequency_sharpen(
    image: Image.Image,
    strength: float = 1.5,
    radius: int = 3,
) -> Image.Image:
    """Sharpen via high-frequency boost instead of naive unsharp mask.

    Decomposes image into low-freq (structure) and high-freq (detail),
    then amplifies only the detail layer. Preserves colors and avoids
    the halo artifacts that unsharp mask produces.

    Args:
        image: Input RGB image.
        strength: High-frequency amplification factor (1.0 = no change, 2.0 = strong).
        radius: Gaussian blur radius for frequency separation.

    Returns:
        Sharpened image.
    """
    arr = np.array(image, dtype=np.float32)

    # Separate into low-frequency (structure) and high-frequency (detail)
    low_freq = cv2.GaussianBlur(arr, (0, 0), sigmaX=radius)
    high_freq = arr - low_freq

    # Amplify detail layer
    result = low_freq + high_freq * strength

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def upscale_to_original(
    result: Image.Image,
    original: Image.Image,
) -> tuple[Image.Image, dict]:
    """Upscale try-on result back to the original photo's resolution.

    Uses Lanczos for base upscale, then transfers high-frequency detail
    from the original (non-garment regions: face, hair, background) to
    recover sharpness lost during downscale.

    Args:
        result: VTON output (typically 576x768).
        original: Original person photo (typically 768x1024).

    Returns:
        (upscaled_image, details_dict)
    """
    orig_w, orig_h = original.size
    res_w, res_h = result.size

    if res_w >= orig_w and res_h >= orig_h:
        return result, {"upscaled": False, "reason": "already_sufficient_resolution"}

    # Base upscale with Lanczos
    upscaled = result.resize((orig_w, orig_h), Image.LANCZOS)

    # Transfer high-frequency detail from original for non-garment regions
    # This recovers face detail, hair texture, background sharpness
    up_arr = np.array(upscaled, dtype=np.float32)
    orig_arr = np.array(original, dtype=np.float32)

    # Extract high-freq from original
    orig_low = cv2.GaussianBlur(orig_arr, (0, 0), sigmaX=2.0)
    orig_high = orig_arr - orig_low

    # Detect which regions changed (garment area) vs preserved (face/bg)
    diff = np.mean(np.abs(up_arr - orig_arr), axis=2)
    # Regions with small diff = preserved (face, bg) → transfer detail
    # Regions with large diff = garment → keep VTON output
    # Use high percentile so only clearly-unchanged regions get detail
    threshold = np.percentile(diff, 70)
    weight = 1.0 / (1.0 + np.exp((diff - threshold) / 8.0))
    weight = weight[:, :, np.newaxis]

    # Blend: add original high-freq detail to preserved regions only
    # Use 0.3 blend to keep it subtle — avoids darkening artifacts
    enhanced = up_arr + orig_high * weight * 0.3

    result_img = Image.fromarray(np.clip(enhanced, 0, 255).astype(np.uint8))

    scale_factor = orig_w / res_w
    return result_img, {
        "upscaled": True,
        "from": f"{res_w}x{res_h}",
        "to": f"{orig_w}x{orig_h}",
        "scale_factor": round(scale_factor, 2),
    }


def soften_garment_boundaries(
    result: Image.Image,
    original: Image.Image,
    blur_radius: int = 3,
) -> tuple[Image.Image, dict]:
    """Soften hard edges at garment boundaries within the result image.

    VTON models often produce hard/jagged edges where the new garment
    meets skin or background. This applies selective Gaussian smoothing
    along the garment boundary ring only — does NOT blend the original
    image back (that would ghost the old garment through).

    Args:
        result: Try-on result.
        original: Original person photo (used to detect garment region).
        blur_radius: Gaussian blur radius for edge smoothing.

    Returns:
        (softened_image, details_dict)
    """
    if result.size != original.size:
        original = original.resize(result.size, Image.LANCZOS)

    res_arr = np.array(result, dtype=np.float32)
    orig_arr = np.array(original, dtype=np.float32)

    # Detect garment region: where result differs from original
    diff = np.mean(np.abs(res_arr - orig_arr), axis=2)
    _, garment_mask = cv2.threshold(
        diff.astype(np.uint8), 30, 255, cv2.THRESH_BINARY
    )

    # Clean up mask
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    garment_mask = cv2.morphologyEx(garment_mask, cv2.MORPH_CLOSE, kernel_clean)

    # Extract boundary ring (dilate - erode)
    kernel_boundary = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blur_radius * 2 + 1, blur_radius * 2 + 1))
    dilated = cv2.dilate(garment_mask, kernel_boundary)
    eroded = cv2.erode(garment_mask, kernel_boundary)
    boundary = cv2.subtract(dilated, eroded)

    boundary_pixels = int(np.sum(boundary > 0))
    if boundary_pixels < 100:
        return result, {"softened": False, "reason": "no_significant_boundary"}

    # Create a smoothed version of the result
    smoothed = cv2.GaussianBlur(res_arr, (0, 0), sigmaX=blur_radius)

    # Only apply smoothing at the boundary ring
    boundary_mask = (boundary > 0).astype(np.float32)
    # Feather the boundary mask itself for gradual transition
    boundary_mask = cv2.GaussianBlur(boundary_mask, (0, 0), sigmaX=blur_radius * 0.5)
    boundary_mask = boundary_mask[:, :, np.newaxis]

    blended = res_arr * (1.0 - boundary_mask) + smoothed * boundary_mask

    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)), {
        "softened": True,
        "boundary_pixels": boundary_pixels,
        "blur_radius": blur_radius,
    }


def harmonize_lighting(
    result: Image.Image,
    original: Image.Image,
    strength: float = 0.4,
) -> tuple[Image.Image, dict]:
    """Match the garment's lighting to the scene's ambient lighting.

    Uses the NON-garment regions (skin, background) from the original
    photo to estimate scene lighting, then adjusts the garment's
    luminance variance to match. This makes the garment look like it
    was photographed in the same conditions — not flat or overlit.

    Does NOT match to the old garment's luminance (that would force a
    blue shirt to have pink-shirt brightness).

    Args:
        result: Try-on result.
        original: Original person photo.
        strength: How much to adjust (0=none, 1=full).

    Returns:
        (harmonized_image, details_dict)
    """
    if result.size != original.size:
        original = original.resize(result.size, Image.LANCZOS)

    res_arr = np.array(result)
    orig_arr = np.array(original)

    # Convert to LAB
    res_lab = cv2.cvtColor(res_arr, cv2.COLOR_RGB2LAB).astype(np.float64)
    orig_lab = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2LAB).astype(np.float64)

    # Detect garment region (where result differs from original)
    diff = np.mean(np.abs(res_arr.astype(np.float32) - orig_arr.astype(np.float32)), axis=2)
    garment_mask = diff > 30
    non_garment_mask = ~garment_mask

    garment_pixels = int(np.sum(garment_mask))
    non_garment_pixels = int(np.sum(non_garment_mask))

    if garment_pixels < 500 or non_garment_pixels < 500:
        return result, {"harmonized": False, "reason": "insufficient_regions"}

    # Estimate scene lighting from NON-garment regions (skin, hair, bg)
    # These regions should be consistent between original and result
    scene_L = orig_lab[:, :, 0]
    scene_mean = np.mean(scene_L[non_garment_mask])
    scene_std = np.std(scene_L[non_garment_mask])

    # Get garment luminance stats from result
    garment_L = res_lab[:, :, 0]
    garment_mean = np.mean(garment_L[garment_mask])
    garment_std = np.std(garment_L[garment_mask])

    if garment_std < 1e-6 or scene_std < 1e-6:
        return result, {"harmonized": False, "reason": "zero_variance"}

    # Match garment's luminance VARIANCE to the scene (not the mean —
    # the garment can be darker/lighter than skin, but should have
    # similar contrast/dynamic range as the scene lighting allows)
    std_ratio = scene_std / garment_std
    # Clamp to prevent extreme corrections
    std_ratio = max(0.7, min(1.4, std_ratio))

    corrected_L = (garment_L[garment_mask] - garment_mean) * std_ratio + garment_mean

    # Apply with strength blending
    res_lab[garment_mask, 0] = (
        garment_L[garment_mask] * (1 - strength)
        + corrected_L * strength
    )
    res_lab[:, :, 0] = np.clip(res_lab[:, :, 0], 0, 255)

    corrected = cv2.cvtColor(res_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    contrast_adjustment = round(float(std_ratio), 3)
    return Image.fromarray(corrected), {
        "harmonized": True,
        "garment_pixels": garment_pixels,
        "scene_luminance_mean": round(float(scene_mean), 1),
        "garment_luminance_mean": round(float(garment_mean), 1),
        "contrast_adjustment": contrast_adjustment,
        "strength": strength,
    }
