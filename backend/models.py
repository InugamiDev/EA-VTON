"""Pydantic models for FitView API request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Garment models ──────────────────────────────────────────────────────────

class GarmentImage(BaseModel):
    id: str
    url: str
    type: Literal["display", "tryon_input"]


class SizeChartEntry(BaseModel):
    size: str
    chest_min: int = Field(alias="chestMin")
    chest_max: int = Field(alias="chestMax")
    length_cm: int = Field(alias="lengthCm")

    model_config = {"populate_by_name": True}


class GarmentResponse(BaseModel):
    id: str
    name: str
    type: str
    brand: str
    description: str
    price: float
    colors: list[str]
    sizes: list[str]
    images: list[GarmentImage]
    size_chart: list[SizeChartEntry] = Field(alias="sizeChart")

    model_config = {"populate_by_name": True}


class GarmentListResponse(BaseModel):
    garments: list[GarmentResponse]
    total: int
    page: int
    page_size: int


# ── Try-on models ───────────────────────────────────────────────────────────

class TryOnCreateResponse(BaseModel):
    job_id: str
    status: Literal["processing", "completed", "failed"]


class TryOnResultResponse(BaseModel):
    job_id: str
    status: Literal["processing", "completed", "failed"]
    result_url: str | None = None
    garment_id: str | None = None
    confidence_score: float | None = None
    confidence_label: Literal["high", "medium", "low"] | None = None
    processing_time_ms: int | None = None
    method: str | None = None
    error: str | None = None


# ── Size recommendation models ──────────────────────────────────────────────

class SizeRecommendationRequest(BaseModel):
    garment_id: str
    height_cm: float = Field(gt=100, lt=250)
    weight_kg: float = Field(gt=30, lt=250)
    fit_preference: Literal["slim", "regular", "relaxed"] = "regular"


class SizeAlternative(BaseModel):
    size: str
    note: str


class SizeRecommendationResponse(BaseModel):
    recommended_size: str
    confidence: Literal["high", "medium", "low"]
    explanation: str
    estimated_chest_cm: float
    alternatives: list[SizeAlternative]


# ── Photo upload models ─────────────────────────────────────────────────────

class QualityCheck(BaseModel):
    status: Literal["pass", "warn", "fail"]
    value: float | None = None


class QualityAssessment(BaseModel):
    brightness: QualityCheck
    blur: QualityCheck
    dimensions: QualityCheck
    overall: Literal["pass", "warn", "fail"]


class PhotoUploadResponse(BaseModel):
    photo_id: str
    filename: str
    url: str
    width: int
    height: int
    quality: QualityAssessment


# ── Video try-on models ────────────────────────────────────────────

class VideoTryOnCreateResponse(BaseModel):
    job_id: str
    status: Literal["processing", "completed", "failed"]


class VideoTryOnStageDetail(BaseModel):
    name: str
    status: str
    duration_ms: int = 0
    details: dict = {}


class VideoTryOnStatusResponse(BaseModel):
    job_id: str
    status: Literal["processing", "completed", "failed"]
    stage: str | None = None
    garment_id: str | None = None
    result_url: str | None = None
    total_frames: int | None = None
    keyframe_count: int | None = None
    processing_time_ms: int | None = None
    avg_fps: float | None = None
    error: str | None = None
    stages: list[VideoTryOnStageDetail] = []
