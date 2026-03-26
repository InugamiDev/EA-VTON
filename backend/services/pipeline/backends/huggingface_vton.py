"""FASHN VTON v1.5 local inference backend.

Runs fashn-ai/fashn-vton-1.5 locally:
- Apache 2.0 licensed (no NC restrictions)
- 972M param MMDiT architecture, pixel-space
- Maskless — no DensePose, OpenPose, or segmentation masks needed
- ~8GB VRAM, ~5s on H100/A100, bf16 on Ampere+
- Output resolution: 576×864
- Supports CUDA and CPU only (no MPS — DWPose uses ONNX runtime)

Replaces the previous IDM-VTON backend which required:
- CC-NC license, ~27GB model, DensePose preprocessing, custom hacked UNets
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from uuid import uuid4

from PIL import Image

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("/tmp/fitview_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "weights" / "fashn-vton"

# Singleton model state
_model_lock = threading.Lock()
_pipeline = None
_device = None


def _detect_device(preference: str = "auto") -> str:
    """Detect best available device.

    FASHN VTON supports cuda and cpu only. MPS is not supported because
    DWPose uses ONNX runtime which doesn't have an MPS execution provider.
    On Mac, inference falls back to CPU.
    """
    import torch

    if preference not in ("auto", "mps"):
        return preference
    # FASHN only supports cuda or cpu — MPS falls back to cpu
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_model_status() -> dict:
    """Return current model loading status."""
    import torch

    return {
        "loaded": _pipeline is not None,
        "device": _device or "none",
        "model_id": "fashn-ai/fashn-vton-1.5",
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "vram_allocated_mb": (
            round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
            if torch.cuda.is_available()
            else None
        ),
        "vram_reserved_mb": (
            round(torch.cuda.memory_reserved() / 1024 / 1024, 1)
            if torch.cuda.is_available()
            else None
        ),
        "weights_downloaded": (WEIGHTS_DIR / "model.safetensors").exists(),
    }


def _download_weights():
    """Download FASHN VTON weights if not present.

    Downloads:
    - model.safetensors (~2GB) from fashn-ai/fashn-vton-1.5
    - DWPose ONNX models from fashn-ai/DWPose
    - FashnHumanParser auto-downloads to ~/.cache/huggingface on first use
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dwpose_dir = WEIGHTS_DIR / "dwpose"
    dwpose_dir.mkdir(exist_ok=True)

    model_path = WEIGHTS_DIR / "model.safetensors"
    yolox_path = dwpose_dir / "yolox_l.onnx"
    dwpose_path = dwpose_dir / "dw-ll_ucoco_384.onnx"

    if model_path.exists() and yolox_path.exists() and dwpose_path.exists():
        logger.info("FASHN VTON weights already present at %s", WEIGHTS_DIR)
        return

    from huggingface_hub import hf_hub_download

    # Main model (~2GB)
    if not model_path.exists():
        logger.info("Downloading FASHN VTON model.safetensors (~2GB)...")
        hf_hub_download(
            repo_id="fashn-ai/fashn-vton-1.5",
            filename="model.safetensors",
            local_dir=str(WEIGHTS_DIR),
        )

    # DWPose ONNX models for pose extraction
    for fname, target in [("yolox_l.onnx", yolox_path), ("dw-ll_ucoco_384.onnx", dwpose_path)]:
        if not target.exists():
            logger.info("Downloading DWPose %s...", fname)
            hf_hub_download(
                repo_id="fashn-ai/DWPose",
                filename=fname,
                local_dir=str(dwpose_dir),
            )

    # Verify all files landed correctly
    for path in (model_path, yolox_path, dwpose_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Weight file missing after download: {path}\n"
                f"Try manually: python -c \"from huggingface_hub import hf_hub_download; "
                f"hf_hub_download('fashn-ai/fashn-vton-1.5', 'model.safetensors', local_dir='{WEIGHTS_DIR}')\""
            )

    logger.info("FASHN VTON weights ready at %s", WEIGHTS_DIR)


def _load_pipeline(device: str):
    """Load FASHN VTON pipeline (singleton, thread-safe).

    First call downloads weights (~2GB). Subsequent calls reuse.
    """
    global _pipeline, _device

    with _model_lock:
        if _pipeline is not None and _device == device:
            return _pipeline

        _download_weights()

        logger.info("Loading FASHN VTON v1.5 on %s...", device)

        from fashn_vton import TryOnPipeline

        # FASHN auto-detects device if None, but we pass explicitly
        # to ensure consistent behavior. Only "cuda" or "cpu" are valid.
        pipe = TryOnPipeline(
            weights_dir=str(WEIGHTS_DIR),
            device=device,
        )

        _pipeline = pipe
        _device = device

        logger.info("FASHN VTON v1.5 ready on %s", device)
        return pipe


# intent: map garment categories from our data to FASHN's 3 categories
# status: done
# confidence: high
_CATEGORY_MAP = {
    "upper_body": "tops",
    "lower_body": "bottoms",
    "full_body": "one-pieces",
    # Direct mappings
    "tops": "tops",
    "bottoms": "bottoms",
    "one-pieces": "one-pieces",
    # Garment-level categories
    "t-shirt": "tops",
    "shirt": "tops",
    "blouse": "tops",
    "jacket": "tops",
    "sweater": "tops",
    "hoodie": "tops",
    "cardigan": "tops",
    "vest": "tops",
    "coat": "tops",
    "pants": "bottoms",
    "trousers": "bottoms",
    "skirt": "bottoms",
    "shorts": "bottoms",
    "dress": "one-pieces",
    "jumpsuit": "one-pieces",
    "romper": "one-pieces",
}


def _run_inference_sync(
    person_path: str,
    garment_path: str | None,
    garment_url: str,
    garment_mask_path: str | None,
    config,
) -> dict:
    """Synchronous inference using FASHN VTON v1.5.

    FASHN is maskless — garment_mask_path is ignored.
    """
    device = _detect_device(config.hf_device)
    pipe = _load_pipeline(device)

    # Load person image
    person_img = Image.open(person_path).convert("RGB")

    # Load garment image
    if garment_path and Path(garment_path).exists():
        garment_img = Image.open(garment_path).convert("RGB")
    else:
        import httpx
        from io import BytesIO

        resp = httpx.get(garment_url, timeout=30)
        resp.raise_for_status()
        garment_img = Image.open(BytesIO(resp.content)).convert("RGB")

    # Map garment type to FASHN's 3 categories
    garment_type = getattr(config, "garment_type", "upper_body")
    category = _CATEGORY_MAP.get(garment_type, "tops")

    num_steps = config.denoise_steps
    guidance_scale = 1.5

    logger.info(
        "Running FASHN VTON v1.5: category=%s, steps=%d, device=%s",
        category, num_steps, device,
    )

    # FASHN handles all preprocessing internally:
    # - DWPose for body keypoints
    # - FashnHumanParser for segmentation
    # - Resizing/padding to model input shape
    # No masks, no DensePose, no agnostic images needed.
    result = pipe(
        person_image=person_img,
        garment_image=garment_img,
        category=category,
        garment_photo_type="flat-lay",
        num_samples=1,
        num_timesteps=num_steps,
        guidance_scale=guidance_scale,
        seed=config.seed,
    )

    # Save result
    result_id = str(uuid4())
    result_path = str(RESULTS_DIR / f"{result_id}.png")
    result.images[0].save(result_path, "PNG")

    logger.info("FASHN VTON inference complete: %s", result_path)

    return {
        "result_path": result_path,
        "method": "huggingface",
        "raw_confidence": 0.88,
        "details": {
            "model_id": "fashn-ai/fashn-vton-1.5",
            "device": device,
            "steps": num_steps,
            "guidance_scale": guidance_scale,
            "category": category,
        },
    }


async def run_huggingface_inference(
    person_path: str,
    garment_path: str | None,
    garment_url: str,
    garment_mask_path: str | None,
    config,
) -> dict:
    """Run FASHN VTON v1.5 inference locally.

    Args:
        person_path: Path to preprocessed person photo
        garment_path: Path to preprocessed garment image
        garment_url: Original garment image URL (fallback)
        garment_mask_path: Ignored — FASHN is maskless
        config: PipelineConfig

    Returns:
        dict with result_path, method, raw_confidence, details
    """
    return await asyncio.to_thread(
        _run_inference_sync,
        person_path,
        garment_path,
        garment_url,
        garment_mask_path,
        config,
    )
