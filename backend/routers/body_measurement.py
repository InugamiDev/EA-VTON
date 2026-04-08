"""Body measurement estimation API — upload photo + height/weight → measurements + size.

Endpoints:
  POST /api/body-measurement          — upload photo + height + weight → full measurements
  POST /api/body-measurement/quick    — height + weight only (no photo, regression fallback)
"""

# intent: API endpoint for photo-based body measurement estimation
# status: done
# confidence: high

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

from services.body_estimator import estimate_from_photo, BodyMeasurements, _fallback_estimation
from services.ea_size_engine import (
    estimate_chest_cm,
    estimate_waist_cm,
    estimate_hip_cm,
    find_best_size,
    compare_recommendations,
    Population,
    FitPreference,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/body-measurement", tags=["body-measurement"])

# ── Default size chart (generic women's tops) ──

DEFAULT_SIZE_CHART = [
    {"size": "XS", "chestMin": 76, "chestMax": 80},
    {"size": "S", "chestMin": 80, "chestMax": 84},
    {"size": "M", "chestMin": 84, "chestMax": 88},
    {"size": "L", "chestMin": 88, "chestMax": 92},
    {"size": "XL", "chestMin": 92, "chestMax": 96},
    {"size": "XXL", "chestMin": 96, "chestMax": 100},
]


class MeasurementResponse(BaseModel):
    # Estimated measurements (cm)
    chest_circumference: float
    waist_circumference: float
    hip_circumference: float
    shoulder_width: float
    arm_length: float
    leg_length: float
    torso_length: float

    # Metadata
    bmi: float
    method: str
    landmark_confidence: float
    warnings: list[str]

    # Size recommendation
    recommended_size: str
    size_confidence: str
    alternatives: list[dict]

    # Population comparison
    population_comparison: dict | None = None


class QuickMeasurementRequest(BaseModel):
    height_cm: float
    weight_kg: float
    population: Population = "universal"
    fit_preference: FitPreference = "regular"


@router.post("", response_model=MeasurementResponse)
async def estimate_body_measurements(
    photo: UploadFile = File(..., description="Front-facing full-body photo"),
    height_cm: float = Form(..., description="Height in cm", ge=100, le=250),
    weight_kg: float = Form(..., description="Weight in kg", ge=25, le=200),
    population: str = Form("universal", description="Population model: universal or vietnamese"),
    fit_preference: str = Form("regular", description="Fit preference: slim, regular, relaxed"),
) -> MeasurementResponse:
    """Estimate body measurements from a front-facing photo + known height/weight."""

    if photo.content_type and not photo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, WebP)")

    image_bytes = await photo.read()
    if len(image_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Image file too small — may be corrupted")

    logger.info(
        "Body measurement request: height=%.1f, weight=%.1f, population=%s",
        height_cm, weight_kg, population,
    )

    # Run photo-based estimation
    measurements = estimate_from_photo(image_bytes, height_cm, weight_kg)

    # Get size recommendation using the photo-estimated chest
    size_result = find_best_size(measurements.chest_circumference, DEFAULT_SIZE_CHART)

    # Population comparison (photo-based vs regression-only)
    comparison = compare_recommendations(
        height_cm, weight_kg, DEFAULT_SIZE_CHART, fit_preference,
    )

    return MeasurementResponse(
        chest_circumference=measurements.chest_circumference,
        waist_circumference=measurements.waist_circumference,
        hip_circumference=measurements.hip_circumference,
        shoulder_width=measurements.shoulder_width,
        arm_length=measurements.arm_length,
        leg_length=measurements.leg_length,
        torso_length=measurements.torso_length,
        bmi=round(measurements.bmi, 1),
        method=measurements.method,
        landmark_confidence=measurements.landmark_confidence,
        warnings=measurements.warnings,
        recommended_size=size_result["recommended_size"],
        size_confidence=size_result["confidence"],
        alternatives=size_result["alternatives"],
        population_comparison=comparison,
    )


@router.post("/quick", response_model=MeasurementResponse)
async def quick_measurement(body: QuickMeasurementRequest) -> MeasurementResponse:
    """Quick measurement using height/weight only (no photo needed)."""

    measurements = BodyMeasurements()
    measurements.bmi = body.weight_kg / ((body.height_cm / 100.0) ** 2)
    measurements = _fallback_estimation(measurements, body.height_cm, body.weight_kg)

    # Use population-specific formulas from ea_size_engine
    chest = estimate_chest_cm(body.height_cm, body.weight_kg, body.fit_preference, body.population)
    waist = estimate_waist_cm(body.height_cm, body.weight_kg, body.population)
    hip = estimate_hip_cm(body.height_cm, body.weight_kg, body.population)

    size_result = find_best_size(chest, DEFAULT_SIZE_CHART)

    comparison = compare_recommendations(
        body.height_cm, body.weight_kg, DEFAULT_SIZE_CHART, body.fit_preference,
    )

    return MeasurementResponse(
        chest_circumference=chest,
        waist_circumference=waist,
        hip_circumference=hip,
        shoulder_width=measurements.shoulder_width,
        arm_length=measurements.arm_length,
        leg_length=measurements.leg_length,
        torso_length=measurements.torso_length,
        bmi=round(measurements.bmi, 1),
        method="height_weight_regression",
        landmark_confidence=0.0,
        warnings=["No photo provided — using regression estimates only"],
        recommended_size=size_result["recommended_size"],
        size_confidence=size_result["confidence"],
        alternatives=size_result["alternatives"],
        population_comparison=comparison,
    )
