"""Photo quality assessment service for uploaded images."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image, ImageFilter

StatusLevel = Literal["pass", "warn", "fail"]


def check_brightness(image: Image.Image) -> tuple[StatusLevel, float]:
    """Check mean pixel brightness of an image.

    Returns:
        Tuple of (status, mean_brightness).
        - < 50  -> fail (too dark)
        - < 80  -> warn (somewhat dark)
        - >= 80 -> pass
    """
    grayscale = image.convert("L")
    mean_brightness = float(np.mean(np.array(grayscale)))

    if mean_brightness < 50:
        return "fail", mean_brightness
    if mean_brightness < 80:
        return "warn", mean_brightness
    return "pass", mean_brightness


def check_blur(image: Image.Image) -> tuple[StatusLevel, float]:
    """Check image sharpness using Laplacian variance.

    A low variance indicates a blurry image.

    Returns:
        Tuple of (status, laplacian_variance).
        - < 50  -> fail (very blurry)
        - < 100 -> warn (somewhat blurry)
        - >= 100 -> pass
    """
    grayscale = image.convert("L")
    # Approximate Laplacian using PIL kernel (3x3)
    laplacian = grayscale.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
        scale=1,
        offset=128,
    ))
    arr = np.array(laplacian, dtype=np.float64) - 128.0
    variance = float(np.var(arr))

    if variance < 50:
        return "fail", variance
    if variance < 100:
        return "warn", variance
    return "pass", variance


def check_dimensions(image: Image.Image) -> tuple[StatusLevel, tuple[int, int]]:
    """Check image dimensions meet minimum requirements.

    Returns:
        Tuple of (status, (width, height)).
        - min 512x512 -> pass
        - min 256x256 -> warn
        - below 256   -> fail
    """
    w, h = image.size

    if w >= 512 and h >= 512:
        return "pass", (w, h)
    if w >= 256 and h >= 256:
        return "warn", (w, h)
    return "fail", (w, h)


def assess_quality(image: Image.Image) -> dict:
    """Run all quality checks and return a combined assessment.

    Returns:
        Dict with brightness, blur, dimensions, and overall status.
    """
    brightness_status, brightness_val = check_brightness(image)
    blur_status, blur_val = check_blur(image)
    dim_status, _ = check_dimensions(image)

    statuses = [brightness_status, blur_status, dim_status]

    if "fail" in statuses:
        overall = "fail"
    elif "warn" in statuses:
        overall = "warn"
    else:
        overall = "pass"

    return {
        "brightness": {"status": brightness_status, "value": round(brightness_val, 1)},
        "blur": {"status": blur_status, "value": round(blur_val, 1)},
        "dimensions": {"status": dim_status, "value": None},
        "overall": overall,
    }
