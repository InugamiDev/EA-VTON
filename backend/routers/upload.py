"""Photo upload endpoint with quality assessment."""

from __future__ import annotations

import os
from io import BytesIO
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File
from PIL import Image

from models import PhotoUploadResponse
from services.quality import assess_quality

router = APIRouter(prefix="/api/photos", tags=["photos"])

_UPLOAD_DIR = "/tmp/fitview_uploads"
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
_MIN_DIMENSION = 256  # Relaxed for demo; quality service will warn


@router.post("/upload", response_model=PhotoUploadResponse)
async def upload_photo(file: UploadFile = File(...)) -> dict:
    """Upload a user photo for virtual try-on.

    Validates the file type, size, and minimum dimensions, then saves it
    locally and returns a real quality assessment (brightness, blur, dimensions).
    """
    # Validate content type
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG.",
        )

    # Read and validate file size
    contents = await file.read()
    if len(contents) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents)} bytes). Maximum is 10 MB.",
        )

    # Open and validate image
    try:
        img = Image.open(BytesIO(contents))
        img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    w, h = img.size
    if w < _MIN_DIMENSION or h < _MIN_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small ({w}x{h}). Minimum {_MIN_DIMENSION}x{_MIN_DIMENSION}.",
        )

    # Run quality assessment (real: brightness, blur via Laplacian, dimensions)
    quality = assess_quality(img)

    # Save to disk
    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    photo_id = str(uuid4())
    ext = "jpg" if file.content_type == "image/jpeg" else "png"
    filename = f"{photo_id}.{ext}"
    filepath = os.path.join(_UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(contents)

    return {
        "photo_id": photo_id,
        "filename": filename,
        "url": f"/uploads/{filename}",
        "width": w,
        "height": h,
        "quality": quality,
    }
