"""Replicate IDM-VTON backend.

Calls the cuuupid/idm-vton model on Replicate for real AI virtual try-on.
Supports:
- Base64 person image upload
- Garment image via URL or base64
- Garment mask for better results
- Configurable denoise steps and seed
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("/tmp/fitview_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def run_replicate_inference(
    person_path: str,
    garment_path: str | None,
    garment_url: str,
    garment_mask_path: str | None,
    config,
    token: str,
) -> dict:
    """Run IDM-VTON on Replicate.

    Args:
        person_path: Path to preprocessed person photo
        garment_path: Path to preprocessed garment image (with bg removed)
        garment_url: Original garment image URL (fallback)
        garment_mask_path: Path to binary garment mask
        config: PipelineConfig
        token: Replicate API token

    Returns:
        dict with result_path, method, raw_confidence
    """
    import replicate

    # Prepare person image as base64
    with open(person_path, "rb") as f:
        person_bytes = f.read()
    ext = "jpeg" if person_path.endswith((".jpg", ".jpeg")) else "png"
    person_b64 = f"data:image/{ext};base64,{base64.b64encode(person_bytes).decode()}"

    # Prepare garment image — prefer preprocessed version (bg removed)
    if garment_path and Path(garment_path).exists():
        with open(garment_path, "rb") as f:
            garment_bytes = f.read()
        garment_ext = "png" if garment_path.endswith(".png") else "jpeg"
        garment_input = f"data:image/{garment_ext};base64,{base64.b64encode(garment_bytes).decode()}"
    else:
        garment_input = garment_url

    # Build model input
    model_input = {
        "human_img": person_b64,
        "garm_img": garment_input,
        "garment_des": "upper body garment",
        "is_checked": True,
        "is_checked_crop": False,
        "denoise_steps": config.denoise_steps,
        "seed": config.seed,
    }

    # Add garment mask if available
    if garment_mask_path and Path(garment_mask_path).exists():
        with open(garment_mask_path, "rb") as f:
            mask_bytes = f.read()
        model_input["mask_img"] = f"data:image/png;base64,{base64.b64encode(mask_bytes).decode()}"

    logger.info("Calling Replicate IDM-VTON (denoise_steps=%d)", config.denoise_steps)

    # Run model
    output = await asyncio.to_thread(
        replicate.run,
        config.replicate_model,
        input=model_input,
    )

    # output can be a string URL or a FileOutput object
    result_url = str(output)

    logger.info("Replicate returned result URL: %s", result_url[:100])

    # Download result
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(result_url)
        resp.raise_for_status()

    result_id = str(uuid4())
    result_path = str(RESULTS_DIR / f"{result_id}.png")
    with open(result_path, "wb") as f:
        f.write(resp.content)

    return {
        "result_path": result_path,
        "method": "replicate",
        "raw_confidence": 0.82,  # Base confidence for real AI results
    }
