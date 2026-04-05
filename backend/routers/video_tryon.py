"""Video try-on endpoints.

// intent: REST API for video virtual try-on (upload, poll, download)
// status: done
// confidence: high

Endpoints:
- POST /api/video-tryon      — Upload video + garment → start job
- GET  /api/video-tryon/{id}  — Poll job status
- GET  /api/video-tryon/{id}/video — Download result video
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from data import get_garment_by_id
from models import VideoTryOnStatusResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/video-tryon", tags=["video-tryon"])

# In-memory job store (demo)
_video_jobs: dict[str, dict] = {}
_MAX_JOBS = 50

_UPLOAD_DIR = Path("/tmp/fitview_video_uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


@router.post("")
async def create_video_tryon_job(
    garment_id: str = Form(...),
    video: UploadFile = File(...),
    target_fps: int = Form(15),
    max_duration_sec: float = Form(30.0),
) -> dict:
    """Create a video try-on job.

    Upload a video + select a garment → starts background processing.
    Returns a job_id to poll for status.
    """
    # Validate garment
    garment = get_garment_by_id(garment_id)
    if garment is None:
        raise HTTPException(status_code=404, detail=f"Garment '{garment_id}' not found")

    # Validate video file
    filename = video.filename or "upload.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    # Validate params
    target_fps = max(1, min(30, target_fps))
    max_duration_sec = max(1.0, min(60.0, max_duration_sec))

    # Save uploaded video
    video_id = uuid4().hex[:12]
    video_path = _UPLOAD_DIR / f"{video_id}{ext}"
    content = await video.read()

    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Video file too large (max 100MB)")

    video_path.write_bytes(content)

    # Get garment image URL
    tryon_img = next(
        (img for img in garment["images"] if img["type"] == "tryon_input"),
        garment["images"][0],
    )

    # Evict oldest jobs
    if len(_video_jobs) >= _MAX_JOBS:
        oldest_key = min(_video_jobs, key=lambda k: _video_jobs[k].get("created_at", 0))
        del _video_jobs[oldest_key]

    job_id = str(uuid4())
    _video_jobs[job_id] = {
        "status": "processing",
        "stage": "extract_frames",
        "garment_id": garment_id,
        "garment_image_url": tryon_img["url"],
        "video_path": str(video_path),
        "target_fps": target_fps,
        "max_duration_sec": max_duration_sec,
        "created_at": time.time(),
        "result_path": None,
        "total_frames": None,
        "keyframe_count": None,
        "processing_time_ms": None,
        "avg_fps": None,
        "error": None,
        "stages": [],
    }

    asyncio.create_task(_process_video_job(job_id))

    return {"job_id": job_id, "status": "processing"}


@router.get("/{job_id}")
def get_video_tryon_status(job_id: str) -> dict:
    """Poll the status of a video try-on job."""
    job = _video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    result_url = None
    if job["result_path"]:
        result_url = f"/api/video-tryon/{job_id}/video"

    return {
        "job_id": job_id,
        "status": job["status"],
        "stage": job.get("stage", "unknown"),
        "garment_id": job["garment_id"],
        "result_url": result_url,
        "total_frames": job.get("total_frames"),
        "keyframe_count": job.get("keyframe_count"),
        "processing_time_ms": job.get("processing_time_ms"),
        "avg_fps": job.get("avg_fps"),
        "error": job.get("error"),
        "stages": _sanitize(job.get("stages", [])),
    }


@router.get("/{job_id}/video")
def get_video_tryon_result(job_id: str):
    """Serve the generated try-on video."""
    job = _video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    result_path = job.get("result_path")
    if not result_path:
        raise HTTPException(status_code=404, detail="Result not ready yet")

    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result video file not found")

    return FileResponse(result_path, media_type="video/mp4", filename=f"tryon_{job_id}.mp4")


# ---------------------------------------------------------------------------
#  Background processing
# ---------------------------------------------------------------------------

async def _process_video_job(job_id: str) -> None:
    """Run the video try-on pipeline in background."""
    from PIL import Image
    from services.video_pipeline import VideoPipelineConfig, run_video_pipeline
    from services.pipeline import PipelineConfig

    job = _video_jobs[job_id]

    try:
        # Load garment image
        garment_image = await _load_garment_image(job["garment_image_url"])

        config = VideoPipelineConfig(
            vton_config=PipelineConfig(backend="huggingface"),
            target_fps=job["target_fps"],
            max_duration_sec=job["max_duration_sec"],
        )

        async def progress_cb(stage: str, pct: float):
            job["stage"] = stage

        result = await run_video_pipeline(
            video_path=job["video_path"],
            garment_image=garment_image,
            config=config,
            progress_callback=progress_cb,
        )

        job["status"] = "completed"
        job["stage"] = "completed"
        job["result_path"] = result.video_path
        job["total_frames"] = result.total_frames
        job["keyframe_count"] = result.keyframe_count
        job["processing_time_ms"] = result.processing_time_ms
        job["avg_fps"] = result.avg_fps
        job["stages"] = [
            {
                "name": s.name,
                "status": s.status,
                "duration_ms": s.duration_ms,
                "details": s.details,
            }
            for s in result.stages
        ]

        logger.info(
            "Video job %s completed: %d frames, %d keyframes, %dms",
            job_id, result.total_frames, result.keyframe_count, result.processing_time_ms,
        )

    except Exception as e:
        logger.error("Video job %s failed: %s", job_id, e, exc_info=True)
        job["status"] = "failed"
        job["stage"] = "failed"
        job["error"] = str(e)


async def _load_garment_image(url: str) -> Image:
    """Load garment image from URL or local path."""
    from PIL import Image
    from pathlib import Path

    # Check if it's a local path first
    if Path(url).exists():
        return Image.open(url).convert("RGB")

    # Download from URL
    import httpx
    from io import BytesIO

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _sanitize(obj):
    """Recursively convert numpy types to native Python for JSON."""
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
