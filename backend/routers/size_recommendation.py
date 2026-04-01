"""Size recommendation endpoints — baseline and EA-VTON comparison."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from data import get_garment_by_id
from models import SizeRecommendationRequest, SizeRecommendationResponse
from services.ea_size_engine import (
    compare_recommendations,
    estimate_chest_cm,
    find_best_size,
)

router = APIRouter(prefix="/api/size-recommendation", tags=["size-recommendation"])


@router.post("", response_model=SizeRecommendationResponse)
def recommend_size(req: SizeRecommendationRequest) -> dict:
    """Recommend a garment size (backward-compatible, uses universal model)."""
    garment = get_garment_by_id(req.garment_id)
    if garment is None:
        raise HTTPException(status_code=404, detail=f"Garment '{req.garment_id}' not found")

    size_chart = garment.get("sizeChart", [])
    if not size_chart:
        raise HTTPException(status_code=400, detail="Garment has no size chart data")

    chest = estimate_chest_cm(req.height_cm, req.weight_kg, req.fit_preference, "universal")
    result = find_best_size(chest, size_chart)

    return {
        "recommended_size": result["recommended_size"],
        "confidence": result["confidence"],
        "explanation": (
            f"Based on your measurements, your estimated chest size is "
            f"{chest:.1f} cm. Size {result['recommended_size']} is recommended."
        ),
        "estimated_chest_cm": chest,
        "alternatives": result["alternatives"],
    }


class CompareRequest(BaseModel):
    garment_id: str
    height_cm: float = Field(gt=100, lt=250)
    weight_kg: float = Field(gt=30, lt=250)
    fit_preference: Literal["slim", "regular", "relaxed"] = "regular"


@router.post("/compare")
def compare_size(req: CompareRequest) -> dict:
    """Compare baseline (US) vs EA-VTON (Vietnamese) size recommendations.

    Returns both predictions side-by-side with:
    - Estimated body measurements per population model
    - Recommended size + confidence
    - Difference explanation
    - Nearest Vietnamese body type cluster
    """
    garment = get_garment_by_id(req.garment_id)
    if garment is None:
        raise HTTPException(status_code=404, detail=f"Garment '{req.garment_id}' not found")

    size_chart = garment.get("sizeChart", [])
    if not size_chart:
        raise HTTPException(status_code=400, detail="Garment has no size chart data")

    return compare_recommendations(
        req.height_cm,
        req.weight_kg,
        size_chart,
        req.fit_preference,
    )
