"""Recommendation service — size prediction + item recommendation.

Endpoints:
  POST /recommend-size           — predict best size (rule-based, existing)
  POST /recommend-size/research  — predict best size using trained GBM model
  POST /compare                  — compare all available research model variants
  GET  /recommend                — get personalized garment recommendations
  GET  /health                   — service health check
"""

# intent: FastAPI entry point for recommendation service with research model integration
# status: done
# next: add two-tower recommendation model
# confidence: high

from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Recommendation Service",
    description="Size recommendation (rule-based + trained research models) + item recommendation (two-tower)",
    version="0.2.0",
)


# ── Request/Response models ──


class SizeRequest(BaseModel):
    height_cm: float = Field(gt=100, lt=250)
    weight_kg: float = Field(gt=30, lt=250)
    fit_preference: Literal["slim", "regular", "relaxed"] = "regular"
    population: Literal["universal", "vietnamese"] = "universal"
    size_chart: list[dict]
    body_embedding: list[float] | None = None


class SizeResponse(BaseModel):
    recommended_size: str
    confidence: Literal["high", "medium", "low"]
    estimated_chest_cm: float
    estimated_waist_cm: float
    estimated_hip_cm: float
    alternatives: list[dict]
    population: str
    nearest_body_type: str | None = None
    nearest_body_type_pct: float | None = None


class ResearchSizeRequest(BaseModel):
    """Request for research model-based size prediction."""
    height_cm: float = Field(gt=100, lt=250)
    weight_kg: float = Field(gt=30, lt=250)
    age: float = Field(default=30.0, ge=10, le=100)
    body_type: str = "unknown"
    category: str = "dress"
    population: Literal["universal", "vietnamese"] = "vietnamese"
    size_chart: list[dict] | None = None


class ResearchSizeResponse(BaseModel):
    """Response from research model-based size prediction."""
    research_model: dict
    rule_based: dict
    input_summary: dict


class VisualSizeRequest(BaseModel):
    """Request for visual (webcam) size prediction — height + body ratios, no weight needed."""
    height_cm: float = Field(gt=100, lt=250)
    shoulder_to_hip: float = Field(ge=0.5, le=2.0, description="Shoulder width / hip width")
    waist_to_hip: float = Field(ge=0.3, le=1.5, description="Waist width / hip width")
    shoulder_to_torso: float = Field(default=0.0, ge=0.0, le=3.0)
    torso_to_leg: float = Field(default=0.0, ge=0.0, le=2.0)
    arm_to_torso: float = Field(default=0.0, ge=0.0, le=3.0)
    age: float = Field(default=30.0, ge=10, le=100)
    category: str = "dress"
    population: Literal["universal", "vietnamese"] = "vietnamese"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Landmark detection confidence")
    source: Literal["world_3d", "image_2d"] = "image_2d"


class StyleUpperRatios(BaseModel):
    """Body ratios for upper-body style recommendation."""
    sh: float = Field(ge=0.5, le=2.0, description="Shoulder/Hip ratio")
    wh: float = Field(ge=0.3, le=1.5, description="Waist/Hip ratio")
    st: float = Field(default=0.66, ge=0.0, le=3.0, description="Shoulder/Torso ratio")
    tl: float = Field(default=0.45, ge=0.0, le=2.0, description="Torso/Leg ratio")
    at: float = Field(default=1.10, ge=0.0, le=3.0, description="Arm/Torso ratio")


class StyleUpperContext(BaseModel):
    """Context: occasion + steerable sliders."""
    occasion: Literal["work", "casual", "date", "formal", "party"] = "casual"
    sliders: dict[str, float] = Field(default_factory=dict, description="Slider axes ∈ [-1, 1]: bold, loose, warm, cover")


class StyleUpperRequest(BaseModel):
    """Request for upper-body style recommendation (VN women, v1)."""
    height_cm: float = Field(gt=100, lt=250)
    predicted_size: Literal["XS", "S", "M", "L", "XL", "XXL"]
    ratios: StyleUpperRatios
    context: StyleUpperContext = Field(default_factory=StyleUpperContext)
    user_palette_lab: list[float] | None = Field(default=None, description="User skin-tone palette anchor in CIE LAB (optional)")
    top_k: int = Field(default=10, ge=1, le=50)
    weights: dict[str, float] | None = None


class CompareRequest(BaseModel):
    """Request for comparing all 6 model variants."""
    height_cm: float = Field(gt=100, lt=250)
    weight_kg: float = Field(gt=30, lt=250)
    age: float = Field(default=30.0, ge=10, le=100)
    body_type: str = "unknown"
    category: str = "dress"
    population: Literal["universal", "vietnamese"] = "vietnamese"


class CompareResponse(BaseModel):
    """Response from model comparison endpoint."""
    models: dict
    rule_based: dict
    recommended: str
    input_summary: dict


class RecommendRequest(BaseModel):
    body_embedding: list[float]
    behavior_embedding: list[float] | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    category_filter: str | None = None


class RecommendResponse(BaseModel):
    items: list[dict]
    model_version: str


# ── Endpoints ──


@app.get("/health")
def health():
    """Service health check."""
    research_variants = _get_research_variants()
    return {
        "status": "healthy",
        "service": "recommendation",
        "models": {
            "size_mlp": _is_mlp_loaded(),
            "two_tower": _is_two_tower_loaded(),
            "research_variants": research_variants,
            "fallback": "regression",
        },
    }


@app.post("/recommend-size", response_model=SizeResponse)
def recommend_size(req: SizeRequest):
    """Predict best size for a garment given body measurements.

    Uses PyTorch MLP if available, falls back to regression formulas.
    """
    from src.size_engine import (
        estimate_chest_cm,
        estimate_waist_cm,
        estimate_hip_cm,
        find_best_size,
        VN_CLUSTERS,
    )

    # Try MLP first
    if req.body_embedding and _is_mlp_loaded():
        try:
            return _predict_size_mlp(req)
        except Exception as e:
            logger.warning("MLP prediction failed, falling back: %s", e)

    # Regression fallback
    chest = estimate_chest_cm(req.height_cm, req.weight_kg, req.fit_preference, req.population)
    waist = estimate_waist_cm(req.height_cm, req.weight_kg, req.population)
    hip = estimate_hip_cm(req.height_cm, req.weight_kg, req.population)

    result = find_best_size(chest, req.size_chart)

    # Find nearest Vietnamese cluster
    nearest_cluster = None
    if req.population == "vietnamese":
        best_dist = float("inf")
        for c in VN_CLUSTERS:
            dist = ((req.height_cm - c["h"]) ** 2 + (req.weight_kg - c["w"]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                nearest_cluster = c

    return SizeResponse(
        recommended_size=result["recommended_size"],
        confidence=result["confidence"],
        estimated_chest_cm=chest,
        estimated_waist_cm=waist,
        estimated_hip_cm=hip,
        alternatives=result["alternatives"],
        population="Vietnamese" if req.population == "vietnamese" else "US (Universal)",
        nearest_body_type=nearest_cluster["label"] if nearest_cluster else None,
        nearest_body_type_pct=nearest_cluster["pct"] if nearest_cluster else None,
    )


# intent: research model endpoint — runs GBM (best practical) + rule-based for comparison
# status: done
# next: add size_chart-aware mapping for research predictions
# confidence: high


@app.post("/recommend-size/research", response_model=ResearchSizeResponse)
def recommend_size_research(req: ResearchSizeRequest):
    """Predict size using the tuned GBM model (best current practical variant).

    Runs gbm_copula_tempered_a075 as the primary model and also returns the rule-based
    prediction for comparison. Gracefully falls back to rule-based only
    if the GBM model file is missing.
    """
    from src.size_engine import (
        estimate_chest_cm,
        estimate_waist_cm,
        estimate_hip_cm,
        find_best_size,
    )

    height_m = req.height_cm / 100.0
    bmi = round(req.weight_kg / (height_m ** 2), 1) if height_m > 0 else 22.0

    input_summary = {
        "height_cm": req.height_cm,
        "weight_kg": req.weight_kg,
        "bmi": bmi,
        "age": req.age,
        "body_type": req.body_type,
        "category": req.category,
        "population": req.population,
    }

    # Rule-based prediction
    pop = "vietnamese" if req.population == "vietnamese" else "universal"
    chest = estimate_chest_cm(req.height_cm, req.weight_kg, "regular", pop)
    waist = estimate_waist_cm(req.height_cm, req.weight_kg, pop)
    hip = estimate_hip_cm(req.height_cm, req.weight_kg, pop)

    rule_based_result: dict = {
        "predicted_size": None,
        "confidence": "low",
        "estimated_chest_cm": chest,
        "estimated_waist_cm": waist,
        "estimated_hip_cm": hip,
    }

    if req.size_chart:
        size_result = find_best_size(chest, req.size_chart)
        rule_based_result["predicted_size"] = size_result["recommended_size"]
        rule_based_result["confidence"] = size_result["confidence"]
        rule_based_result["alternatives"] = size_result["alternatives"]
    else:
        # Without a size chart, estimate using BMI-based heuristic
        rule_based_result["predicted_size"] = _bmi_to_size_heuristic(bmi)
        rule_based_result["confidence"] = "low"

    # Research model prediction (tuned GBM copula — best current practical model)
    research_result = _predict_research_model(
        "gbm_copula_tempered_a075", req.height_cm, req.weight_kg, req.age, req.body_type, req.category
    )

    if research_result is None:
        research_result = {
            "predicted_size": rule_based_result["predicted_size"],
            "confidence": "low",
            "model": "gbm_copula_tempered_a075",
            "error": "Model not loaded — falling back to rule-based",
        }
    else:
        research_result["model"] = "gbm_copula_tempered_a075"

    return ResearchSizeResponse(
        research_model=research_result,
        rule_based=rule_based_result,
        input_summary=input_summary,
    )


# intent: visual size endpoint — webcam body ratios → weight estimation → GBM prediction
# status: done
# next: train a dedicated model on ratios directly (skip weight estimation)
# confidence: high


@app.post("/recommend-size/visual")
def recommend_size_visual(req: VisualSizeRequest):
    """Predict size from webcam body ratios + height (no weight needed).

    Pipeline:
      1. Estimate weight from height + shoulder-to-hip + waist-to-hip ratios
      2. Feed estimated weight into existing GBM model
      3. Return prediction + estimated weight + ratio analysis

    This makes the webcam actually contribute data instead of just decoration.
    """
    from src.size_engine import (
        estimate_chest_cm,
        estimate_waist_cm,
        estimate_hip_cm,
    )

    # ── Step 1: Estimate weight from body ratios ──
    # Baselines differ by landmark source:
    #   world_3d: real-body proportions in meters (matches anthropometric literature)
    #   image_2d: inflated by 2D camera projection (needs dampened coefficients)
    height_m = req.height_cm / 100.0
    base_weight = 21.5 * height_m * height_m  # BMI ~21.5 baseline

    if req.source == "world_3d":
        # World landmarks: ratios match ANSUR/CAESAR anthropometric data
        sth_delta = req.shoulder_to_hip - 1.1   # real S/H ~1.05–1.15 women
        base_weight += sth_delta * 20
        wth_delta = req.waist_to_hip - 0.8      # real W/H ~0.75–0.85 women
        base_weight += wth_delta * 25
    else:
        # 2D fallback: camera perspective inflates ratios
        sth_delta = req.shoulder_to_hip - 1.5   # 2D S/H ~1.4–1.8
        base_weight += sth_delta * 12
        wth_delta = req.waist_to_hip - 0.95     # 2D W/H ~0.90–1.10
        base_weight += wth_delta * 15

    estimated_weight = round(max(35, min(150, base_weight)), 1)
    estimated_bmi = round(estimated_weight / (height_m ** 2), 1)

    # ── Step 2: Run GBM with estimated weight ──
    research_result = _predict_research_model(
        "gbm_copula_tempered_a075",
        req.height_cm,
        estimated_weight,
        req.age,
        "unknown",
        req.category,
    )

    if research_result is None:
        research_result = {
            "predicted_size": _bmi_to_size_heuristic(estimated_bmi),
            "confidence": "low",
            "model": "gbm_copula_tempered_a075",
            "error": "Model not loaded — using BMI heuristic",
        }
    else:
        research_result["model"] = "gbm_copula_tempered_a075"

    # ── Step 3: Rule-based with estimated weight ──
    pop = "vietnamese" if req.population == "vietnamese" else "universal"
    chest = estimate_chest_cm(req.height_cm, estimated_weight, "regular", pop)
    waist = estimate_waist_cm(req.height_cm, estimated_weight, pop)
    hip = estimate_hip_cm(req.height_cm, estimated_weight, pop)

    # ── Body shape analysis from ratios ──
    if req.shoulder_to_hip > 1.15:
        body_shape = "inverted_triangle"
    elif req.waist_to_hip < 0.7:
        body_shape = "hourglass"
    elif req.waist_to_hip > 0.9:
        body_shape = "rectangle"
    elif req.shoulder_to_hip < 0.95:
        body_shape = "pear"
    else:
        body_shape = "balanced"

    return {
        "research_model": research_result,
        "rule_based": {
            "predicted_size": _bmi_to_size_heuristic(estimated_bmi),
            "confidence": "low",
            "estimated_chest_cm": chest,
            "estimated_waist_cm": waist,
            "estimated_hip_cm": hip,
        },
        "weight_estimation": {
            "estimated_weight_kg": estimated_weight,
            "estimated_bmi": estimated_bmi,
            "method": "ratio_regression",
            "note": "Weight derived from height + webcam body proportions",
        },
        "body_shape": {
            "type": body_shape,
            "shoulder_to_hip": req.shoulder_to_hip,
            "waist_to_hip": req.waist_to_hip,
            "shoulder_to_torso": req.shoulder_to_torso,
            "torso_to_leg": req.torso_to_leg,
            "arm_to_torso": req.arm_to_torso,
        },
        "input_summary": {
            "height_cm": req.height_cm,
            "weight_kg": estimated_weight,
            "bmi": estimated_bmi,
            "age": req.age,
            "category": req.category,
            "population": req.population,
            "landmark_confidence": req.confidence,
            "source": "webcam_visual",
        },
    }


# intent: upper-body style recommendation (VN women v1) — Fit-Flatter-Match decomposition
# status: done
# next: add reranking with diversity (MMR), accept skin tone, learn weights from feedback
# confidence: high


@app.post("/recommend-style/upper")
def recommend_style_upper(req: StyleUpperRequest):
    """Upper-body style recommendation for Vietnamese women (v1).

    Decomposed scoring:
        s = α·Fit + β·Flatter + γ·Match
    with hard constraints on size availability, occasion, and minimum fit score.

    Returns a ranked list of items, each with a full explanation DAG containing
    the per-component scores and the rules/sections that produced them.
    """
    from src.style_upper import rank_upper_body, build_response

    body, ranked = rank_upper_body(
        height_cm=req.height_cm,
        predicted_size=req.predicted_size,
        ratios=req.ratios.model_dump(),
        occasion=req.context.occasion,
        sliders=req.context.sliders,
        user_palette_lab=req.user_palette_lab,
        top_k=req.top_k,
        weights=req.weights,
    )
    return build_response(body, req.predicted_size, ranked, req.weights)


# intent: compare trained research variants side-by-side + rule-based baseline
# status: done
# next: add latency tracking per model
# confidence: high


@app.post("/compare", response_model=CompareResponse)
def compare_models(req: CompareRequest):
    """Compare all available research model variants + rule-based baseline.

    Returns predictions from every available variant and recommends the
    best model (gbm_copula_tempered_a075 by default — best tuned within-1 trade-off).
    """
    from src.size_engine import (
        estimate_chest_cm,
        estimate_waist_cm,
        estimate_hip_cm,
    )

    height_m = req.height_cm / 100.0
    bmi = round(req.weight_kg / (height_m ** 2), 1) if height_m > 0 else 22.0

    input_summary = {
        "height_cm": req.height_cm,
        "weight_kg": req.weight_kg,
        "bmi": bmi,
        "age": req.age,
        "body_type": req.body_type,
        "category": req.category,
        "population": req.population,
    }

    # Run all research variants
    registry = _get_research_registry()
    all_results: dict = {}

    if registry is not None:
        all_results = registry.predict_all(
            height_cm=req.height_cm,
            weight_kg=req.weight_kg,
            age=req.age,
            body_type=req.body_type,
            category=req.category,
        )

    # Rule-based prediction
    pop = "vietnamese" if req.population == "vietnamese" else "universal"
    chest = estimate_chest_cm(req.height_cm, req.weight_kg, "regular", pop)
    waist = estimate_waist_cm(req.height_cm, req.weight_kg, pop)
    hip = estimate_hip_cm(req.height_cm, req.weight_kg, pop)

    rule_based = {
        "predicted_size": _bmi_to_size_heuristic(bmi),
        "confidence": "medium",
        "confidence_score": None,
        "estimated_chest_cm": chest,
        "estimated_waist_cm": waist,
        "estimated_hip_cm": hip,
    }

    # Recommended model: prefer tuned copula candidate, else other GBM baselines
    recommended = "rule_based"
    if "gbm_copula_tempered_a075" in all_results:
        recommended = "gbm_copula_tempered_a075"
    elif all_results:
        # Pick any available GBM variant first, then MLP
        for name in [
            "gbm_indep_tempered_a05",
            "gbm_copula",
            "gbm_indep",
            "gbm_uniform",
            "mlp_ce_uniform",
            "mlp_ce_copula",
            "mlp_corn_copula",
        ]:
            if name in all_results:
                recommended = name
                break

    return CompareResponse(
        models=all_results,
        rule_based=rule_based,
        recommended=recommended,
        input_summary=input_summary,
    )


@app.get("/recommend")
def recommend_items(
    body_embedding: str | None = None,
    top_k: int = 10,
    category: str | None = None,
):
    """Get personalized garment recommendations using two-tower model.

    Returns top-K items ranked by embedding similarity.
    """
    # intent: two-tower recommendation endpoint
    # status: partially_done
    # next: integrate FAISS index + trained two-tower model
    # confidence: medium

    if not _is_two_tower_loaded():
        raise HTTPException(
            status_code=501,
            detail="Two-tower recommendation model not yet trained. Use /recommend-size for size prediction.",
        )

    raise HTTPException(status_code=501, detail="Not yet implemented")


# ── Internal helpers ──


def _is_mlp_loaded() -> bool:
    """Check if PyTorch size MLP is available."""
    try:
        from src.size_mlp import SizeMLP
        return SizeMLP.is_loaded()
    except ImportError:
        return False


def _is_two_tower_loaded() -> bool:
    """Check if two-tower recommendation model is available."""
    try:
        from src.item_recommender import TwoTowerModel
        return TwoTowerModel.is_loaded()
    except ImportError:
        return False


def _predict_size_mlp(req: SizeRequest) -> SizeResponse:
    """Run size prediction through PyTorch MLP."""
    from src.size_mlp import SizeMLP
    model = SizeMLP.get_instance()
    return model.predict(
        height_cm=req.height_cm,
        weight_kg=req.weight_kg,
        body_embedding=req.body_embedding,
        size_chart=req.size_chart,
        fit_preference=req.fit_preference,
    )


def _get_research_registry():
    """Get the research model registry singleton. Returns None if unavailable."""
    try:
        from src.research_models import ResearchModelRegistry
        return ResearchModelRegistry.get_instance()
    except Exception as e:
        logger.warning("Research model registry unavailable: %s", e)
        return None


def _get_research_variants() -> list[str]:
    """Return list of available research model variant names."""
    registry = _get_research_registry()
    if registry is None:
        return []
    return registry.available_variants()


def _predict_research_model(
    name: str,
    height_cm: float,
    weight_kg: float,
    age: float = 30.0,
    body_type: str = "unknown",
    category: str = "dress",
) -> dict | None:
    """Run prediction through a named research variant. Returns None on failure."""
    registry = _get_research_registry()
    if registry is None:
        return None
    try:
        return registry.predict(name, height_cm, weight_kg, age, body_type, category)
    except Exception as e:
        logger.warning("Research model '%s' prediction failed: %s", name, e)
        return None


def _bmi_to_size_heuristic(bmi: float) -> str:
    """Quick BMI-to-size heuristic when no size chart is provided."""
    if bmi < 16.5:
        return "XXS"
    elif bmi < 18.5:
        return "XS"
    elif bmi < 21.0:
        return "S"
    elif bmi < 24.0:
        return "M"
    elif bmi < 27.0:
        return "L"
    elif bmi < 30.0:
        return "XL"
    return "XXL"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
