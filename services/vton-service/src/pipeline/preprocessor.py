"""Preprocessing stages for person photo and garment image.

Person preprocessing:
- Auto-crop to upper body (detect face/skin region, crop torso frame)
- Resize to model-expected dimensions (768x1024)
- Normalize brightness/contrast

Garment preprocessing:
- Download garment image
- Remove background (edge-based segmentation)
- Generate binary garment mask
- Resize to standard dimensions
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

PREPROCESS_DIR = Path("/tmp/fitview_preprocess")
PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
#  Person preprocessing
# ---------------------------------------------------------------------------

def preprocess_person(photo_path: str, config) -> dict:
    """Preprocess a person photo for try-on.

    Steps:
    1. Load and auto-orient (EXIF)
    2. Detect approximate person region using skin/edge detection
    3. Crop to upper body with padding
    4. Resize to model input size (768x1024)
    5. Normalize brightness and contrast
    """
    img = Image.open(photo_path)
    img = ImageOps.exif_transpose(img)  # Fix mobile rotation
    original_size = img.size
    details = {"original_size": list(original_size)}

    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Step 1: Auto-crop to person region
    if config.auto_crop_person:
        img, crop_info = _auto_crop_upper_body(img)
        details["crop"] = crop_info

    # Step 2: Resize to target dimensions maintaining aspect ratio
    target_w, target_h = config.normalize_person_size
    img = _resize_and_pad(img, target_w, target_h)
    details["resized_to"] = [target_w, target_h]

    # Step 3: Normalize brightness/contrast
    img, norm_info = _normalize_exposure(img)
    details["normalization"] = norm_info

    # Save
    out_id = str(uuid4())
    out_path = str(PREPROCESS_DIR / f"person_{out_id}.jpg")
    img.save(out_path, "JPEG", quality=95)

    return {"path": out_path, "details": details}


def _auto_crop_upper_body(img: Image.Image) -> tuple[Image.Image, dict]:
    """Crop image to approximate upper body region.

    Uses a combination of:
    - Center-weighted crop (assume person is roughly centered)
    - Skin tone detection to find face region
    - Crop from above face to below chest
    """
    w, h = img.size
    arr = np.array(img)

    # Detect skin-tone pixels to find face region
    # HSV-based skin detection
    from skimage.color import rgb2hsv
    hsv = rgb2hsv(arr)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Skin tone: hue 0-0.1 or 0.9-1.0, saturation 0.1-0.7, value 0.2-0.9
    skin_mask = (
        ((hue < 0.1) | (hue > 0.9))
        & (sat > 0.08)
        & (sat < 0.75)
        & (val > 0.15)
        & (val < 0.95)
    )

    # Find the topmost significant skin region (likely face)
    skin_rows = np.any(skin_mask, axis=1)
    skin_indices = np.where(skin_rows)[0]

    if len(skin_indices) > 10:
        # Face is roughly at the top cluster of skin pixels
        face_top = int(skin_indices[0])
        # Estimate face height as ~15% of image height
        face_height = max(int(h * 0.15), 50)
        face_center_y = face_top + face_height // 2

        # Crop: from slightly above face to ~70% below
        crop_top = max(0, face_top - int(face_height * 0.5))
        crop_bottom = min(h, face_center_y + int(h * 0.55))
    else:
        # Fallback: use top 80% of image
        crop_top = 0
        crop_bottom = int(h * 0.85)

    # Horizontal: center crop with 85% width
    margin_x = int(w * 0.075)
    crop_left = margin_x
    crop_right = w - margin_x

    img_cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    return img_cropped, {
        "face_detected": len(skin_indices) > 10,
        "crop_box": [crop_left, crop_top, crop_right, crop_bottom],
    }


def _resize_and_pad(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize image to target dimensions, padding with neutral color if needed."""
    img_ratio = img.width / img.height
    target_ratio = target_w / target_h

    if img_ratio > target_ratio:
        # Wider than target — fit to width
        new_w = target_w
        new_h = int(target_w / img_ratio)
    else:
        # Taller than target — fit to height
        new_h = target_h
        new_w = int(target_h * img_ratio)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Pad to exact target size
    canvas = Image.new("RGB", (target_w, target_h), (240, 240, 240))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))

    return canvas


def _normalize_exposure(img: Image.Image) -> tuple[Image.Image, dict]:
    """Normalize brightness and contrast to improve model input quality."""
    arr = np.array(img, dtype=np.float32)
    mean_brightness = float(np.mean(arr))

    adjustments = {}

    # Brightness correction: target mean ~130
    if mean_brightness < 90:
        factor = min(130 / max(mean_brightness, 1), 1.6)
        img = ImageEnhance.Brightness(img).enhance(factor)
        adjustments["brightness_factor"] = round(factor, 2)
    elif mean_brightness > 190:
        factor = max(130 / mean_brightness, 0.7)
        img = ImageEnhance.Brightness(img).enhance(factor)
        adjustments["brightness_factor"] = round(factor, 2)

    # Slight contrast boost
    img = ImageEnhance.Contrast(img).enhance(1.05)
    adjustments["contrast_factor"] = 1.05

    return img, adjustments


# ---------------------------------------------------------------------------
#  Garment preprocessing
# ---------------------------------------------------------------------------

def preprocess_garment(garment_image_url: str, config) -> dict:
    """Preprocess a garment image for try-on.

    Steps:
    1. Download garment image
    2. Remove background (edge-based segmentation)
    3. Generate binary mask
    4. Resize to standard dimensions
    """
    details = {}

    # Load garment from local path or URL
    garment_path = Path(garment_image_url)
    if garment_path.exists():
        img = Image.open(garment_path).convert("RGBA")
    else:
        with httpx.Client(timeout=30) as client:
            resp = client.get(garment_image_url)
            resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    details["original_size"] = list(img.size)

    if config.remove_garment_bg:
        img_nobg, mask = _remove_background(img)
        details["bg_removed"] = True
    else:
        img_nobg = img
        mask = Image.new("L", img.size, 255)
        details["bg_removed"] = False

    # Resize garment to standard size
    garment_resized = _resize_garment(img_nobg, 768, 1024)
    mask_resized = _resize_garment(mask.convert("RGBA"), 768, 1024).convert("L")

    # Save garment (with transparency)
    garment_id = str(uuid4())
    garment_path = str(PREPROCESS_DIR / f"garment_{garment_id}.png")
    garment_resized.save(garment_path, "PNG")

    # Save mask
    mask_path = str(PREPROCESS_DIR / f"mask_{garment_id}.png")
    mask_resized.save(mask_path, "PNG")

    return {
        "path": garment_path,
        "url": garment_image_url,
        "mask_path": mask_path,
        "details": details,
    }


def _remove_background(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Remove background from garment image using edge-based segmentation.

    Uses a multi-step approach:
    1. Edge detection (Canny-like using Pillow)
    2. Flood fill from corners (likely background)
    3. Morphological cleanup
    4. Apply as alpha mask
    """
    # Convert to RGB for processing
    rgb = img.convert("RGB")
    arr = np.array(rgb)

    # Detect edges
    gray = np.mean(arr, axis=2).astype(np.uint8)
    from skimage.filters import sobel
    from skimage.morphology import dilation, erosion, disk

    edges = sobel(gray / 255.0)
    edge_mask = edges > 0.05  # Threshold for edges

    # Dilate edges to create a boundary
    edge_boundary = dilation(edge_mask, disk(3))

    # Flood fill from corners to find background
    from skimage.segmentation import flood_fill
    h, w = gray.shape

    # Create a seed image (0 = unknown, 1 = likely background)
    bg_seeds = np.zeros_like(gray, dtype=bool)
    # Sample corners
    corner_size = min(h, w) // 8
    bg_seeds[:corner_size, :corner_size] = True
    bg_seeds[:corner_size, -corner_size:] = True
    bg_seeds[-corner_size:, :corner_size] = True
    bg_seeds[-corner_size:, -corner_size:] = True

    # Use color similarity from corners to detect background
    # Get average corner colors
    corners = []
    for sy, sx in [(0, 0), (0, w - corner_size), (h - corner_size, 0), (h - corner_size, w - corner_size)]:
        region = arr[sy : sy + corner_size, sx : sx + corner_size]
        corners.append(np.mean(region, axis=(0, 1)))
    avg_bg_color = np.mean(corners, axis=0)

    # Pixels similar to background color
    color_diff = np.sqrt(np.sum((arr.astype(float) - avg_bg_color) ** 2, axis=2))
    bg_by_color = color_diff < 60  # Threshold

    # Combine: background = similar to corner colors AND not near strong edges
    background = bg_by_color & ~edge_boundary
    foreground = ~background

    # Morphological cleanup
    foreground = erosion(foreground, disk(2))
    foreground = dilation(foreground, disk(4))

    # Create mask (255 = foreground, 0 = background)
    mask_arr = (foreground * 255).astype(np.uint8)
    mask = Image.fromarray(mask_arr, mode="L")

    # Smooth mask edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2))

    # Apply mask to image
    result = img.copy()
    if result.mode != "RGBA":
        result = result.convert("RGBA")
    result.putalpha(mask)

    return result, mask


def _resize_garment(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize garment image, fitting within target with transparent padding."""
    img_ratio = img.width / img.height
    target_ratio = target_w / target_h

    if img_ratio > target_ratio:
        new_w = target_w
        new_h = int(target_w / img_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * img_ratio)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))

    return canvas
