"""VTON inference service — warping-based virtual try-on.

Endpoints:
  POST /infer        — single-image try-on
  POST /infer/video  — video try-on (frame sequence)
  GET  /health       — service health check
"""

# intent: FastAPI entry point for VTON inference service
# status: done
# next: implement custom TPS warping model in warping/
# confidence: high

from __future__ import annotations

import io
import logging
import time
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

logger = logging.getLogger(__name__)

app = FastAPI(
    title="VTON Service",
    description="Virtual try-on inference with TPS warping + composition U-Net",
    version="0.1.0",
)


@app.get("/health")
def health():
    """Service health check."""
    return {
        "status": "healthy",
        "service": "vton-service",
        "backends": _get_available_backends(),
    }


@app.post("/infer")
async def infer(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    pose: UploadFile | None = File(None),
    parsing: UploadFile | None = File(None),
    garment_type: str = Form("upper_body"),
    backend: str = Form("auto"),
):
    """Run virtual try-on inference.

    Accepts person image + garment image, optionally with pre-computed
    pose heatmaps and parsing map. Returns the try-on result image.
    """
    t0 = time.perf_counter()

    person_bytes = await person_image.read()
    garment_bytes = await garment_image.read()

    pose_data = None
    parsing_data = None

    if pose is not None:
        pose_data = np.load(io.BytesIO(await pose.read()))
    if parsing is not None:
        parsing_data = np.load(io.BytesIO(await parsing.read()))

    person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
    garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")

    selected_backend = _resolve_backend(backend)

    try:
        result_img, metadata = _run_inference(
            person_img=person_img,
            garment_img=garment_img,
            pose=pose_data,
            parsing=parsing_data,
            garment_type=garment_type,
            backend=selected_backend,
        )
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Processing-Time-Ms": str(elapsed_ms),
            "X-Backend": selected_backend,
            "X-Confidence": str(metadata.get("confidence", 0.0)),
        },
    )


@app.post("/infer/video")
async def infer_video(
    video: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    garment_type: str = Form("upper_body"),
    backend: str = Form("auto"),
    max_frames: int = Form(300),
):
    """Run video try-on with frame reuse + optical flow optimization.

    Returns processed video as MP4.
    """
    # intent: video try-on endpoint with 3-tier frame processing
    # status: partially_done
    # next: integrate 3-tier frame processing from video/ module
    # confidence: medium

    raise HTTPException(
        status_code=501,
        detail="Video try-on endpoint not yet implemented. Use single-frame /infer for now.",
    )


# ── Internal helpers ──


def _get_available_backends() -> list[str]:
    """Detect available inference backends."""
    backends = ["local_composite"]
    try:
        import torch  # noqa: F401
        backends.append("custom_warping")
        backends.append("huggingface")
    except ImportError:
        pass
    import os
    if os.environ.get("REPLICATE_API_TOKEN"):
        backends.append("replicate")
    return backends


def _resolve_backend(requested: str) -> str:
    """Resolve 'auto' to best available backend."""
    available = _get_available_backends()
    if requested != "auto" and requested in available:
        return requested

    # Priority: custom_warping > huggingface > replicate > local_composite
    for pref in ["custom_warping", "huggingface", "replicate", "local_composite"]:
        if pref in available:
            return pref
    return "local_composite"


def _run_inference(
    person_img: Image.Image,
    garment_img: Image.Image,
    pose: np.ndarray | None,
    parsing: np.ndarray | None,
    garment_type: str,
    backend: str,
) -> tuple[Image.Image, dict]:
    """Dispatch inference to the selected backend.

    Returns (result_image, metadata_dict).
    """
    if backend == "custom_warping":
        return _infer_custom_warping(person_img, garment_img, pose, parsing, garment_type)
    elif backend == "huggingface":
        from src.pipeline.backends.huggingface_vton import run_inference
        return run_inference(person_img, garment_img, garment_type)
    elif backend == "replicate":
        from src.pipeline.backends.replicate_vton import run_inference
        return run_inference(person_img, garment_img, garment_type)
    else:
        from src.pipeline.backends.local_composite import run_inference
        return run_inference(person_img, garment_img, garment_type)


def _infer_custom_warping(
    person_img: Image.Image,
    garment_img: Image.Image,
    pose: np.ndarray | None,
    parsing: np.ndarray | None,
    garment_type: str,
) -> tuple[Image.Image, dict]:
    """Run custom TPS warping + composition U-Net inference.

    Falls back to local_composite if model weights are not loaded.
    """
    # intent: custom warping-based VTON inference
    # status: partially_done
    # next: load trained TPS + composition model weights
    # confidence: medium

    try:
        from src.warping.model import VTONModel
        model = VTONModel.get_instance()
        result_img, metadata = model.infer(person_img, garment_img, pose, parsing, garment_type)
        metadata["backend"] = "custom_warping"
        return result_img, metadata
    except Exception as e:
        logger.warning("Custom warping failed (%s), falling back to local_composite", e)
        from src.pipeline.backends.local_composite import run_inference
        return run_inference(person_img, garment_img, garment_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
