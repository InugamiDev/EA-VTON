"""FitView Virtual Try-On Demo — FastAPI backend.

Run with:
    uvicorn main:app --reload --port 8000

For real AI try-on, set:
    export REPLICATE_API_TOKEN=r8_your_token_here
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers import garments, tryon, tryon_simple, size_recommendation, upload, video_tryon, body_measurement

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = FastAPI(
    title="FitView API",
    description="Virtual try-on backend with real AI inference, photo quality assessment, and size recommendation.",
    version="0.3.0",
)

# CORS — allow all origins for demo purposes.
# Note: allow_credentials cannot be True when allow_origins is ["*"] —
# browsers reject Access-Control-Allow-Origin: * with credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ───────────────────────────────────────────────────────────
app.include_router(garments.router)
app.include_router(tryon.router)
app.include_router(tryon_simple.router)
app.include_router(size_recommendation.router)
app.include_router(upload.router)
app.include_router(video_tryon.router)
app.include_router(body_measurement.router)

# ── Serve uploaded photos ───────────────────────────────────────────────────
_UPLOAD_DIR = "/tmp/fitview_uploads"
_RESULTS_DIR = "/tmp/fitview_results"
_VIDEO_RESULTS_DIR = "/tmp/fitview_video_results"
os.makedirs(_UPLOAD_DIR, exist_ok=True)
os.makedirs(_RESULTS_DIR, exist_ok=True)
os.makedirs(_VIDEO_RESULTS_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=_UPLOAD_DIR), name="uploads")
app.mount("/results", StaticFiles(directory=_RESULTS_DIR), name="results")
app.mount("/video-results", StaticFiles(directory=_VIDEO_RESULTS_DIR), name="video-results")


# ── Root endpoints ──────────────────────────────────────────────────────────

def _detect_inference_mode() -> str:
    """Detect best available inference mode."""
    if os.environ.get("REPLICATE_API_TOKEN"):
        return "replicate"
    try:
        import torch  # noqa: F401
        return "huggingface"
    except ImportError:
        return "local_composite"


@app.get("/")
def root() -> dict:
    """Application info."""
    mode = _detect_inference_mode()
    mode_label = {
        "replicate": "replicate (IDM-VTON)",
        "huggingface": "huggingface (FASHN VTON v1.5)",
        "local_composite": "local_composite",
    }
    return {
        "app": "FitView API",
        "version": "0.3.0",
        "inference_mode": mode_label.get(mode, mode),
        "docs": "/docs",
    }


@app.get("/api/health")
def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "inference_mode": _detect_inference_mode(),
    }


@app.get("/api/model-status")
def model_status() -> dict:
    """Report HuggingFace model loading status, device, and VRAM usage."""
    try:
        from services.pipeline.backends.huggingface_vton import _get_model_status
        return _get_model_status()
    except ImportError:
        return {
            "loaded": False,
            "device": "none",
            "model_id": None,
            "error": "torch not installed",
        }
