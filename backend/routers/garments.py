"""Garment catalog endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from data import GARMENTS
from models import GarmentListResponse, GarmentResponse

router = APIRouter(prefix="/api/garments", tags=["garments"])


@router.get("", response_model=GarmentListResponse)
def list_garments(
    type: str | None = Query(None, description="Filter by garment type (e.g. t-shirt, shirt, hoodie, jacket)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
) -> dict:
    """List all garments with optional type filter and pagination."""
    filtered = GARMENTS
    if type is not None:
        filtered = [g for g in filtered if g["type"] == type]

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = filtered[start:end]

    return {
        "garments": page_items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/{garment_id}", response_model=GarmentResponse)
def get_garment(garment_id: str) -> dict:
    """Get a single garment by ID."""
    for g in GARMENTS:
        if g["id"] == garment_id:
            return g
    raise HTTPException(status_code=404, detail=f"Garment '{garment_id}' not found")
