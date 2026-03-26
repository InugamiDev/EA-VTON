"""Virtual try-on endpoints.

Real pipeline:
1. POST /api/tryon — create job, start background inference pipeline
2. GET /api/tryon/{job_id} — poll status with pipeline stage details
3. GET /api/tryon/{job_id}/image — serve result image
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from uuid import uuid4

import numpy as np

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import FileResponse

from data import get_garment_by_id
from services.tryon_inference import run_tryon
from services.pipeline import PipelineConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tryon", tags=["tryon"])

# In-memory job store (demo — production would use Redis/DB)
_jobs: dict[str, dict] = {}
_MAX_JOBS = 200  # Evict oldest when exceeded


@router.post("")
async def create_tryon_job(
    garment_id: str = Form(...),
    photo_id: str = Form(...),
    backend: str = Form("auto"),
) -> dict:
    """Create a try-on job and start the full pipeline in background.

    Pipeline stages: preprocess_person → preprocess_garment → inference → postprocess
    """
    garment = get_garment_by_id(garment_id)
    if garment is None:
        raise HTTPException(status_code=404, detail=f"Garment '{garment_id}' not found")

    photo_path = _find_photo(photo_id)
    if photo_path is None:
        raise HTTPException(status_code=404, detail=f"Photo '{photo_id}' not found. Upload it first.")

    if backend not in _ALLOWED_BACKENDS:
        raise HTTPException(status_code=400, detail=f"Invalid backend '{backend}'. Must be one of: {', '.join(sorted(_ALLOWED_BACKENDS))}")

    tryon_img = next(
        (img for img in garment["images"] if img["type"] == "tryon_input"),
        garment["images"][0],
    )

    # Evict oldest jobs if at capacity
    if len(_jobs) >= _MAX_JOBS:
        oldest_key = min(_jobs, key=lambda k: _jobs[k].get("created_at", 0))
        del _jobs[oldest_key]

    job_id = str(uuid4())
    _jobs[job_id] = {
        "status": "processing",
        "garment_id": garment_id,
        "photo_path": photo_path,
        "garment_image_url": tryon_img["url"],
        "backend": backend,
        "created_at": time.time(),
        "result_path": None,
        "confidence_score": None,
        "confidence_label": None,
        "processing_time_ms": None,
        "method": None,
        "error": None,
        "stages": [],
    }

    asyncio.create_task(_process_job(job_id))

    return {"job_id": job_id, "status": "processing"}


@router.get("/{job_id}")
def get_tryon_status(job_id: str) -> dict:
    """Poll the status of a try-on job.

    Returns pipeline stage details alongside the overall status.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    result_url = None
    if job["result_path"]:
        result_url = f"/api/tryon/{job_id}/image"

    return {
        "job_id": job_id,
        "status": job["status"],
        "garment_id": job["garment_id"],
        "result_url": result_url,
        "confidence_score": job.get("confidence_score"),
        "confidence_label": job.get("confidence_label"),
        "processing_time_ms": job.get("processing_time_ms"),
        "method": job.get("method"),
        "error": job.get("error"),
        "stages": job.get("stages", []),
    }


@router.get("/{job_id}/image")
def get_tryon_image(job_id: str):
    """Serve the generated try-on result image."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    result_path = job.get("result_path")
    if not result_path:
        raise HTTPException(status_code=404, detail="Result not ready yet")

    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result image file not found")

    media_type = "image/jpeg" if result_path.endswith(".jpg") else "image/png"
    return FileResponse(result_path, media_type=media_type)


# ---------------------------------------------------------------------------
#  Background processing
# ---------------------------------------------------------------------------

async def _process_job(job_id: str) -> None:
    """Run the full try-on pipeline in background."""
    job = _jobs[job_id]

    try:
        config = PipelineConfig(backend=job.get("backend", "auto"))

        result = await run_tryon(
            user_photo_path=job["photo_path"],
            garment_image_url=job["garment_image_url"],
            garment_id=job["garment_id"],
            config=config,
        )

        job["status"] = "completed"
        job["result_path"] = result.result_path
        job["confidence_score"] = _sanitize(result.confidence_score)
        job["confidence_label"] = result.confidence_label
        job["processing_time_ms"] = _sanitize(result.processing_time_ms)
        job["method"] = result.method
        job["stages"] = _sanitize([
            {
                "name": s.name,
                "status": s.status,
                "duration_ms": s.duration_ms,
                "details": s.details,
            }
            for s in result.stages
        ])

        logger.info(
            "Job %s completed: method=%s confidence=%.2f time=%dms stages=%d",
            job_id,
            result.method,
            result.confidence_score,
            result.processing_time_ms,
            len(result.stages),
        )

    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e, exc_info=True)
        job["status"] = "failed"
        job["error"] = str(e)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _sanitize(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


_ALLOWED_BACKENDS = {"auto", "replicate", "huggingface", "local_composite"}


def _find_photo(photo_id: str) -> str | None:
    """Find an uploaded photo by its ID (path-traversal safe)."""
    upload_dir = "/tmp/fitview_uploads"
    # Reject path components that could escape the upload directory
    if "/" in photo_id or "\\" in photo_id or ".." in photo_id:
        return None
    for ext in ("jpg", "png"):
        path = os.path.join(upload_dir, f"{photo_id}.{ext}")
        # Resolve symlinks and verify the path stays within upload_dir
        real = os.path.realpath(path)
        if not real.startswith(os.path.realpath(upload_dir) + os.sep):
            return None
        if os.path.exists(real):
            return real
    return None
