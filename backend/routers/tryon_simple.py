"""Simplified try-on endpoint: upload person + garment → get result.

No catalog lookup, no photo_id — just two image files in, one image out.

POST /api/tryon/simple — upload two images, start pipeline
GET  /api/tryon/simple/{job_id} — poll status
GET  /api/tryon/simple/{job_id}/image — serve result
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from uuid import uuid4

import numpy as np

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from services.tryon_inference import run_tryon
from services.pipeline import PipelineConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tryon/simple", tags=["tryon-simple"])

_UPLOAD_DIR = "/tmp/fitview_uploads"
_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# In-memory job store
_jobs: dict[str, dict] = {}
_MAX_JOBS = 200


@router.post("")
async def create_simple_tryon(
    person: UploadFile = File(...),
    garment: UploadFile = File(...),
    garment_type: str = Form("upper_body"),
) -> dict:
    """Upload person + garment images, start try-on pipeline.

    Returns job_id to poll for status.
    """
    # Validate files
    for label, f in [("person", person), ("garment", garment)]:
        if f.content_type not in _ALLOWED_TYPES:
            raise HTTPException(400, f"{label}: unsupported type '{f.content_type}'. Use JPEG, PNG, or WebP.")
        content = await f.read()
        if len(content) > _MAX_FILE_SIZE:
            raise HTTPException(400, f"{label}: file too large (max 10MB)")
        await f.seek(0)

    # Save person image
    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    person_id = str(uuid4())
    person_ext = _ext_from_type(person.content_type)
    person_path = os.path.join(_UPLOAD_DIR, f"{person_id}.{person_ext}")
    with open(person_path, "wb") as fp:
        fp.write(await person.read())

    # Save garment image
    garment_id = str(uuid4())
    garment_ext = _ext_from_type(garment.content_type)
    garment_path = os.path.join(_UPLOAD_DIR, f"{garment_id}.{garment_ext}")
    with open(garment_path, "wb") as fp:
        fp.write(await garment.read())

    # Evict oldest jobs if at capacity
    if len(_jobs) >= _MAX_JOBS:
        oldest_key = min(_jobs, key=lambda k: _jobs[k].get("created_at", 0))
        del _jobs[oldest_key]

    job_id = str(uuid4())
    _jobs[job_id] = {
        "status": "processing",
        "person_path": person_path,
        "garment_path": garment_path,
        "garment_type": garment_type,
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
def get_simple_status(job_id: str) -> dict:
    """Poll job status."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found")

    result_url = f"/api/tryon/simple/{job_id}/image" if job["result_path"] else None

    return {
        "job_id": job_id,
        "status": job["status"],
        "result_url": result_url,
        "confidence_score": job.get("confidence_score"),
        "confidence_label": job.get("confidence_label"),
        "processing_time_ms": job.get("processing_time_ms"),
        "method": job.get("method"),
        "error": job.get("error"),
        "stages": job.get("stages", []),
    }


@router.get("/{job_id}/image")
def get_simple_image(job_id: str):
    """Serve the result image."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found")

    result_path = job.get("result_path")
    if not result_path:
        raise HTTPException(404, "Result not ready yet")
    if not os.path.exists(result_path):
        raise HTTPException(404, "Result image file not found")

    media_type = "image/jpeg" if result_path.endswith(".jpg") else "image/png"
    return FileResponse(result_path, media_type=media_type)


# ---------------------------------------------------------------------------
#  Background processing
# ---------------------------------------------------------------------------

async def _process_job(job_id: str) -> None:
    """Run the pipeline in background."""
    job = _jobs[job_id]

    try:
        config = PipelineConfig(
            backend="auto",
            garment_type=job["garment_type"],
        )

        result = await run_tryon(
            user_photo_path=job["person_path"],
            garment_image_url=job["garment_path"],  # local path — preprocessor handles both
            garment_id="direct_upload",
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
            "Simple job %s completed: method=%s confidence=%.2f time=%dms",
            job_id, result.method, result.confidence_score, result.processing_time_ms,
        )

    except Exception as e:
        logger.error("Simple job %s failed: %s", job_id, e, exc_info=True)
        job["status"] = "failed"
        job["error"] = str(e)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _ext_from_type(content_type: str | None) -> str:
    return {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}.get(
        content_type or "", "jpg"
    )


def _sanitize(obj):
    """Convert numpy types to native Python for JSON serialization."""
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
