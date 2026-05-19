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

PopulationLiteral = Literal["us_women", "universal", "vietnamese"]
SIZE_ORDER = ["XXS", "XS", "S", "M", "L", "XL", "XXL"]
SIZE_INDEX = {size: index for index, size in enumerate(SIZE_ORDER)}

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
    population: PopulationLiteral = "us_women"
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
    category: str = "shirt"
    population: PopulationLiteral = "us_women"
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
    category: str = "shirt"
    population: PopulationLiteral = "us_women"
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
    """Context for upper-body style recommendation.

    Default UX is occasion-only — recommendations work cleanly without sliders.
    Sliders are an advanced/optional knob retained for power users; passing an
    empty dict (the default) yields recommendations driven purely by body type,
    occasion, and color palette.
    """
    occasion: Literal["work", "casual", "date", "formal", "party"] = "casual"
    sliders: dict[str, float] = Field(
        default_factory=dict,
        description="ADVANCED — optional slider axes in [-1, 1]: bold, loose, warm, cover. "
                    "Leave empty for default UX.",
    )


SeasonLiteral = Literal[
    "light_spring", "true_spring", "bright_spring",
    "light_summer", "true_summer", "soft_summer",
    "soft_autumn", "true_autumn", "deep_autumn",
    "deep_winter", "true_winter", "bright_winter",
]


class StyleUpperRequest(BaseModel):
    """Request for upper-body style recommendation."""
    height_cm: float = Field(gt=100, lt=250)
    predicted_size: Literal["XS", "S", "M", "L", "XL", "XXL"]
    population: PopulationLiteral = "us_women"
    ratios: StyleUpperRatios
    context: StyleUpperContext = Field(default_factory=StyleUpperContext)
    # Color match — three ways to specify (priority: anchors > season > single lab):
    season: SeasonLiteral | None = Field(
        default=None,
        description="Pick a 12-season palette (preferred UX). Resolves to 8 LAB anchors internally.",
    )
    user_palette_anchors: list[list[float]] | None = Field(
        default=None,
        description="Explicit list of LAB anchors. Score = max(color_score over anchors).",
    )
    user_palette_lab: list[float] | None = Field(
        default=None,
        description="Single LAB point (back-compat). Use 'season' for typical UX.",
    )
    top_k: int = Field(default=10, ge=1, le=50)
    weights: dict[str, float] | None = None
    catalog_source: Literal["app_garments_v1", "deepfashion2_upper", "merged"] = "app_garments_v1"
    candidate_pool_size: int = Field(default=300, ge=50, le=2000, description="DF2 sample pool size before scoring")


SeasonLiteralForCal = Literal[
    "light_spring", "true_spring", "bright_spring",
    "light_summer", "true_summer", "soft_summer",
    "soft_autumn", "true_autumn", "deep_autumn",
    "bright_winter", "true_winter", "deep_winter",
]


class SeedItemsRequest(BaseModel):
    """Stratified-diverse calibration set request.

    Returns ~18 visually-distinct items spanning the (neckline × silhouette ×
    color_temperature) attribute space so the user can express style preference
    by picking a few favorites — a cold-start signal that bypasses the need for
    historical user-item interactions.
    """
    predicted_size: Literal["XS", "S", "M", "L", "XL", "XXL"] = "M"
    season: SeasonLiteralForCal | None = None
    max_items: int = Field(default=18, ge=6, le=36)


class CalibrateRequest(BaseModel):
    """Re-rank candidates using the user's liked-item CLIP centroid as a preference vector."""
    height_cm: float = Field(gt=100, lt=250)
    predicted_size: Literal["XS", "S", "M", "L", "XL", "XXL"]
    population: PopulationLiteral = "vietnamese"
    ratios: StyleUpperRatios
    context: StyleUpperContext = Field(default_factory=StyleUpperContext)
    liked_garment_ids: list[str] = Field(min_length=1, max_length=20)
    season: SeasonLiteralForCal | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    candidate_pool_size: int = Field(default=300, ge=50, le=2000)
    preference_weight: float = Field(default=0.30, ge=0.0, le=1.0,
                                     description="δ in score = (1-δ)·FFM + δ·sim(item, user_centroid)")


class CompareRequest(BaseModel):
    """Request for comparing all 6 model variants."""
    height_cm: float = Field(gt=100, lt=250)
    weight_kg: float = Field(gt=30, lt=250)
    age: float = Field(default=30.0, ge=10, le=100)
    body_type: str = "unknown"
    category: str = "dress"
    population: PopulationLiteral = "us_women"


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


@app.get("/dataset/stats")
def dataset_stats():
    """Aggregate catalog statistics for the UI visualizer.

    Reads the DuckDB view live and returns per-source counts, labeling
    coverage, person-cluster stats, and source citations.
    """
    from src.deepfashion2_catalog import DeepFashion2Catalog, is_available
    if not is_available():
        raise HTTPException(status_code=503, detail="Catalog DB not built")
    cat = DeepFashion2Catalog.get_instance()
    con = cat.con

    total = int(con.execute("SELECT COUNT(*) FROM items").fetchone()[0])
    by_source = [
        {"source": r[0], "n": int(r[1])}
        for r in con.execute(
            "SELECT source, COUNT(*) FROM items GROUP BY source ORDER BY 2 DESC"
        ).fetchall()
    ]

    # Labeling coverage
    n_clip = int(con.execute(
        "SELECT COUNT(*) FROM items WHERE clip_embedding IS NOT NULL"
    ).fetchone()[0])
    n_redacted = int(con.execute(
        "SELECT COUNT(*) FROM items WHERE redaction_status = 'face_detected_redacted'"
    ).fetchone()[0])
    n_no_face = int(con.execute(
        "SELECT COUNT(*) FROM items WHERE redaction_status = 'no_face_detected'"
    ).fetchone()[0])
    n_not_processed = int(con.execute(
        "SELECT COUNT(*) FROM items WHERE redaction_status = 'not_processed'"
    ).fetchone()[0])

    # Person clustering: requires the parquet to have person_id column
    try:
        n_person_assigned = int(con.execute(
            "SELECT COUNT(*) FROM items WHERE person_id IS NOT NULL AND person_id != -1"
        ).fetchone()[0])
        cluster_top = [
            {"person_id": int(r[0]), "size": int(r[1])}
            for r in con.execute(
                """SELECT person_id, COUNT(*) AS n FROM items
                   WHERE person_id IS NOT NULL AND person_id != -1
                   GROUP BY person_id ORDER BY n DESC LIMIT 10"""
            ).fetchall()
        ]
    except Exception:
        n_person_assigned = 0
        cluster_top = []

    # Attribute distributions
    def dist(col: str, limit: int = 8) -> list[dict]:
        rows = con.execute(
            f"SELECT {col}, COUNT(*) FROM items GROUP BY {col} ORDER BY 2 DESC LIMIT {limit}"
        ).fetchall()
        return [{"value": r[0], "n": int(r[1])} for r in rows]

    return {
        "total_rows": total,
        "by_source": by_source,
        "labeling": {
            "with_clip_embedding": n_clip,
            "redacted_face_detected": n_redacted,
            "no_face_detected": n_no_face,
            "not_processed": n_not_processed,
        },
        "person_clustering": {
            "rows_assigned": n_person_assigned,
            "top_clusters_by_size": cluster_top,
        },
        "attributes": {
            "neckline": dist("neckline"),
            "silhouette": dist("silhouette"),
            "color_temperature": dist("color_temperature"),
            "best_season": dist("best_season"),
            "style_personality": dist("style_personality"),
            "best_suits_cluster": dist("best_suits_cluster"),
        },
        "sources": [
            {
                "id": "deepfashion2",
                "name": "DeepFashion2",
                "citation": "Ge, Y., Zhang, R., Wang, X., Tang, X., Luo, P. (2019). DeepFashion2: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images. CVPR.",
                "url": "https://github.com/switchablenorms/DeepFashion2",
                "license": "CUHK research-only (images not redistributable)",
            },
            {
                "id": "fashionpedia",
                "name": "Fashionpedia",
                "citation": "Jia, M., Shi, M., Sirotenko, M., Cui, Y., Cardie, C., Hariharan, B., Adam, H., Belongie, S. (2020). Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset. ECCV.",
                "url": "https://fashionpedia.github.io/home/",
                "license": "CC BY 4.0",
            },
            {
                "id": "deepfashion_with_masks",
                "name": "DeepFashion (with masks)",
                "citation": "Liu, Z., Luo, P., Qiu, S., Wang, X., Tang, X. (2016). DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations. CVPR.",
                "url": "https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html",
                "license": "Apache 2.0 (masks); original images per CUHK terms",
            },
            {
                "id": "fashion_products_small",
                "name": "Fashion Products Small (Myntra)",
                "citation": "Myntra/Param Aggarwal catalog snapshot, Kaggle 2019.",
                "url": "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small",
                "license": "Per Kaggle terms; verify per use",
            },
            {
                "id": "vn_dottie",
                "name": "Dottie (VN brand)",
                "citation": "Scraped from dottie.vn 2026-04 with EAVTONResearchBot. Research-fair-use only.",
                "url": "https://dottie.vn/",
                "license": "Research-fair-use; not redistributable",
            },
            {
                "id": "vn_ivy_moda",
                "name": "IvyModa (VN brand)",
                "citation": "Scraped from ivymoda.com 2026-04 with EAVTONResearchBot. Research-fair-use only.",
                "url": "https://ivymoda.com/",
                "license": "Research-fair-use; not redistributable",
            },
            {
                "id": "vn_yody",
                "name": "Yody (VN brand)",
                "citation": "Scraped from yody.vn 2026-05-19 via sitemap-image schema with EAVTONResearchBot.",
                "url": "https://yody.vn/",
                "license": "Research-fair-use; not redistributable",
            },
            {
                "id": "vn_coupletx",
                "name": "Couple TX (VN brand)",
                "citation": "Scraped from coupletx.com 2026-05-19 via sitemap-image schema (Haravan).",
                "url": "https://coupletx.com/",
                "license": "Research-fair-use; not redistributable",
            },
        ],
        "anthropometric_sources": [
            {
                "id": "vn_trieu_2024",
                "name": "VN Body Taxonomy (Trieu et al. 2024)",
                "citation": "Trieu, Tang, Nguyen, Le (2024). Analysis of Vietnamese Women's Body Shape from Anthropometric Data. Vlákna a Textil 2024(1). n=480 3D body scans.",
                "url": "http://vat.ft.tul.cz/2024/1/VaT_2024_1_1.pdf",
            },
            {
                "id": "us_lee_2007",
                "name": "US Body Taxonomy (Lee et al. 2007 / FFIT)",
                "citation": "Lee, Istook, Nam, Park (2007). Comparison of body shape between USA and Korean women. IJCST 19(5). FFIT 5-shape on n=222.",
                "url": "https://www.emerald.com/insight/content/doi/10.1108/09556220710819492/full/html",
            },
            {
                "id": "kr_size_korea",
                "name": "KR Body Taxonomy (Size Korea + Lee 2007)",
                "citation": "Size Korea 8th KATS anthropometric survey, 2020. n=259 KR women FFIT clusters.",
                "url": "https://sizekorea.kr/page/about/1",
            },
            {
                "id": "polarity_methodology",
                "name": "Body-rule Polarity Methodology (Hidayati et al. 2018)",
                "citation": "Hidayati, S. C., Hsu, C.-C., Chang, Y.-T., Hua, K.-L., Fu, J., Cheng, W.-H. (2018). What Dress Fits Me Best? Fashion Recommendation on the Clothing Style for Personal Body Shape. ACM MM 2018.",
                "url": "https://dl.acm.org/doi/10.1145/3240508.3240546",
            },
        ],
        "size_model": {
            "primary": "gbm_female_upper_vn",
            "exact_accuracy": 0.522,
            "within1_accuracy": 0.707,
            "training_data": "RTR female upper-body, fit='fit' filter, n_train=8,750",
            "reweighting": "copula+PSIS, α=0.75 tempering, target=VN_FEMALE prior (Tran 2024: h=156.2±5.5, w=53.9±8.0)",
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
    from src.population_profiles import get_population_profile

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
        population=get_population_profile(req.population).label,
        nearest_body_type=nearest_cluster["label"] if nearest_cluster else None,
        nearest_body_type_pct=nearest_cluster["pct"] if nearest_cluster else None,
    )


# intent: research model endpoint — runs population-appropriate GBM + rule-based for comparison
# status: done
# next: add size_chart-aware mapping for research predictions
# confidence: high


@app.post("/recommend-size/research", response_model=ResearchSizeResponse)
def recommend_size_research(req: ResearchSizeRequest):
    """Predict size using the population-appropriate tuned GBM variant.

    Vietnamese requests use the target-weighted Copula model. US/default requests use
    the unweighted current model to avoid applying Vietnamese density ratios to the
    US-women objective.
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
    pop = req.population
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
        # Without a garment chart, use population bust bands instead of BMI alone.
        rule_based_result["predicted_size"] = _chest_to_size_heuristic(chest, req.population)
        rule_based_result["confidence"] = "medium"

    # Research model prediction: choose the model for the requested population objective.
    selected_model, research_result = _predict_population_research_model(
        req.population,
        req.height_cm,
        req.weight_kg,
        req.age,
        req.body_type,
        req.category,
    )

    if research_result is None:
        research_result = {
            "predicted_size": rule_based_result["predicted_size"],
            "confidence": "low",
            "model": selected_model,
            "error": "Model not loaded — falling back to rule-based",
        }
    else:
        research_result["model"] = selected_model

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
    from src.population_profiles import estimate_visual_weight_kg, get_population_profile

    # intent: use explicit population calibration instead of hidden visual-ratio constants
    # status: done
    # next: train direct ratio-to-size model so visual sizing can avoid weight estimation
    # blockers: no labeled webcam-ratio-to-garment-size dataset yet
    # confidence: medium
    height_m = req.height_cm / 100.0
    profile = get_population_profile(req.population)
    estimated_weight = estimate_visual_weight_kg(
        height_cm=req.height_cm,
        shoulder_to_hip=req.shoulder_to_hip,
        waist_to_hip=req.waist_to_hip,
        source=req.source,
        population=req.population,
    )
    estimated_bmi = round(estimated_weight / (height_m ** 2), 1)

    # ── Step 2: Run GBM with estimated weight ──
    selected_model, research_result = _predict_population_research_model(
        req.population,
        req.height_cm,
        estimated_weight,
        req.age,
        "unknown",
        req.category,
    )

    if research_result is None:
        research_result = {
            "predicted_size": _chest_to_size_heuristic(
                estimate_chest_cm(req.height_cm, estimated_weight, "regular", req.population),
                req.population,
            ),
            "confidence": "low",
            "model": selected_model,
            "error": "Model not loaded - using calibrated chest heuristic",
        }
    else:
        research_result["model"] = selected_model

    # ── Step 3: Rule-based with estimated weight ──
    pop = req.population
    chest = estimate_chest_cm(req.height_cm, estimated_weight, "regular", pop)
    waist = estimate_waist_cm(req.height_cm, estimated_weight, pop)
    hip = estimate_hip_cm(req.height_cm, estimated_weight, pop)

    # ── Body shape analysis from ratios ──
    if req.source == "image_2d":
        shape_shoulder_to_hip = (
            req.shoulder_to_hip
            / profile.image_2d.shoulder_to_hip
            * profile.world_3d.shoulder_to_hip
        )
        shape_waist_to_hip = (
            req.waist_to_hip
            / profile.image_2d.waist_to_hip
            * profile.world_3d.waist_to_hip
        )
    else:
        shape_shoulder_to_hip = req.shoulder_to_hip
        shape_waist_to_hip = req.waist_to_hip

    if shape_shoulder_to_hip > 1.12 and shape_waist_to_hip >= 0.78:
        body_shape = "inverted_triangle"
    elif shape_waist_to_hip < 0.74:
        body_shape = "hourglass"
    elif shape_waist_to_hip > 0.88:
        body_shape = "rectangle"
    elif shape_shoulder_to_hip < 0.98:
        body_shape = "pear"
    else:
        body_shape = "balanced"

    rule_based_size = _chest_to_size_heuristic(chest, req.population)
    research_result = _stabilize_visual_research_result(research_result, rule_based_size)

    return {
        "research_model": research_result,
        "rule_based": {
            "predicted_size": rule_based_size,
            "confidence": "medium",
            "estimated_chest_cm": chest,
            "estimated_waist_cm": waist,
            "estimated_hip_cm": hip,
        },
        "weight_estimation": {
            "estimated_weight_kg": estimated_weight,
            "estimated_bmi": estimated_bmi,
            "method": "population_ratio_regression",
            "note": f"Weight derived from height + webcam body proportions using {profile.label} priors",
        },
        "body_shape": {
            "type": body_shape,
            "shoulder_to_hip": req.shoulder_to_hip,
            "waist_to_hip": req.waist_to_hip,
            "shoulder_to_torso": req.shoulder_to_torso,
            "torso_to_leg": req.torso_to_leg,
            "arm_to_torso": req.arm_to_torso,
            "normalized_shoulder_to_hip": round(shape_shoulder_to_hip, 3),
            "normalized_waist_to_hip": round(shape_waist_to_hip, 3),
        },
        "input_summary": {
            "height_cm": req.height_cm,
            "weight_kg": estimated_weight,
            "bmi": estimated_bmi,
            "age": req.age,
            "category": req.category,
            "population": req.population,
            "landmark_confidence": req.confidence,
            "source": req.source,
        },
    }


# intent: upper-body style recommendation — Fit-Flatter-Match decomposition
# status: done
# next: add reranking with diversity (MMR), accept skin tone, learn weights from feedback
# confidence: high


@app.get("/style/image/{garment_id:path}")
def get_style_image(garment_id: str):
    """Serve a cropped garment thumbnail from the DeepFashion2 catalog.

    garment_id format: 'split:image_id:item_idx' (matches DeepFashion2Catalog).
    """
    from fastapi.responses import Response
    from PIL import Image
    from io import BytesIO
    from src.deepfashion2_catalog import DeepFashion2Catalog, is_available

    if not is_available():
        raise HTTPException(status_code=503, detail="DeepFashion2 catalog not built yet")

    catalog = DeepFashion2Catalog.get_instance()
    item = catalog.get_by_garment_id(garment_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Garment {garment_id} not found")

    local_path = item.get("image_local_path")
    if not local_path:
        raise HTTPException(status_code=404, detail="Image path missing")

    try:
        img = Image.open(local_path).convert("RGB")
        bbox = item.get("bbox") or []
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            margin_x = (x2 - x1) * 0.05
            margin_y = (y2 - y1) * 0.05
            w, h = img.size
            crop = img.crop((
                max(0, int(x1 - margin_x)),
                max(0, int(y1 - margin_y)),
                min(w, int(x2 + margin_x)),
                min(h, int(y2 + margin_y)),
            ))
        else:
            crop = img

        # Standardize size for the rec card
        crop.thumbnail((512, 512))
        buf = BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        return Response(content=buf.getvalue(), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image serve failed: {e}")


@app.post("/recommend-style/upper")
def recommend_style_upper(req: StyleUpperRequest):
    """Upper-body style recommendation.

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
        population=req.population,
        ratios=req.ratios.model_dump(),
        occasion=req.context.occasion,
        sliders=req.context.sliders,
        user_palette_lab=req.user_palette_lab,
        user_palette_anchors=req.user_palette_anchors,
        season=req.season,
        top_k=req.top_k,
        weights=req.weights,
        catalog_source=req.catalog_source,
        candidate_pool_size=req.candidate_pool_size,
    )
    return build_response(body, req.predicted_size, ranked, req.weights, req.population)


# ── Calibration UX (cold-start preference elicitation) ──
# Two endpoints power the camera → size → style flow without requiring any
# historical user data:
#   POST /recommend-style/upper/seed-items
#       → 18 attribute-diverse seed items the user picks favorites from
#   POST /recommend-style/upper/calibrate
#       → re-ranks recommendations using the CLIP-centroid of those picks
#
# intent: cold-start preference signal via stratified-diverse seed → CLIP centroid → re-rank
# status: done
# confidence: high


@app.post("/recommend-style/upper/seed-items")
def recommend_style_upper_seeds(req: SeedItemsRequest):
    """Return ~18 attribute-diverse items spanning (neckline × silhouette × color_temp).

    These are SEED items for cold-start preference elicitation — the user picks
    3+ they like, the calibrate endpoint uses those picks to personalize.
    """
    from src.deepfashion2_catalog import DeepFashion2Catalog, is_available

    if not is_available():
        raise HTTPException(status_code=503, detail="Style catalog not built")
    cat = DeepFashion2Catalog.get_instance()
    items = cat.seed_diverse_items(
        size=req.predicted_size,
        season=req.season,
        max_items=req.max_items,
    )
    return {
        "items": items,
        "total": len(items),
        "stratification": "neckline × silhouette × color_temperature",
        "instructions": "Pick at least 3 items you find appealing.",
    }


@app.post("/recommend-style/upper/calibrate")
def recommend_style_upper_calibrate(req: CalibrateRequest):
    """Re-rank top items with a CLIP-similarity preference boost.

    Pipeline:
      1. Fetch CLIP embeddings for liked_garment_ids → mean → L2-normalize = preference vector v*
      2. Call rank_upper_body() with a wide top_k to get base FFM scores
      3. For each base item, compute sim = cos(item_emb, v*) ∈ [-1, 1]; normalize to [0, 1]
      4. new_score = (1 - δ) * FFM_score + δ * preference_score, with δ = req.preference_weight
      5. Re-sort and truncate to req.top_k.
    """
    import math
    from src.deepfashion2_catalog import DeepFashion2Catalog, is_available
    from src.style_upper import rank_upper_body, build_response

    if not is_available():
        raise HTTPException(status_code=503, detail="Style catalog not built")
    cat = DeepFashion2Catalog.get_instance()

    # 1. Preference vector
    liked_embs = cat.get_clip_embeddings(req.liked_garment_ids)
    if not liked_embs:
        raise HTTPException(
            status_code=400,
            detail=f"None of the liked garment_ids resolved to embeddings: {req.liked_garment_ids}",
        )
    dim = len(next(iter(liked_embs.values())))
    centroid = [0.0] * dim
    for emb in liked_embs.values():
        for i, v in enumerate(emb):
            centroid[i] += v
    n = len(liked_embs)
    centroid = [v / n for v in centroid]
    norm = math.sqrt(sum(v * v for v in centroid)) or 1.0
    pref_vec = [v / norm for v in centroid]

    # 2. Wide base ranking — ask for more so the re-rank has room to reshuffle
    wide_k = max(req.top_k * 5, 50)
    body, ranked = rank_upper_body(
        height_cm=req.height_cm,
        predicted_size=req.predicted_size,
        population=req.population,
        ratios=req.ratios.model_dump(),
        occasion=req.context.occasion,
        sliders=req.context.sliders,
        season=req.season,
        top_k=wide_k,
        catalog_source="deepfashion2_upper",
        candidate_pool_size=req.candidate_pool_size,
    )

    # 3-4. CLIP-similarity boost
    candidate_ids = [r.item["item_id"] for r in ranked]
    cand_embs = cat.get_clip_embeddings(candidate_ids)
    delta = req.preference_weight
    pref_by_id: dict[str, float] = {}
    base_by_id: dict[str, float] = {r.item["item_id"]: r.total_score for r in ranked}
    for r in ranked:
        emb = cand_embs.get(r.item["item_id"])
        if emb is None:
            pref_score = 0.5  # neutral when embedding missing
        else:
            inorm = math.sqrt(sum(v * v for v in emb)) or 1.0
            cos = sum(a * b for a, b in zip(emb, pref_vec)) / inorm
            pref_score = (cos + 1.0) / 2.0  # → [0, 1]
        pref_by_id[r.item["item_id"]] = round(pref_score, 4)
        r.total_score = round((1.0 - delta) * r.total_score + delta * pref_score, 4)

    ranked.sort(key=lambda r: r.total_score, reverse=True)
    ranked = ranked[:req.top_k]
    for i, r in enumerate(ranked, start=1):
        r.rank = i

    response = build_response(body, req.predicted_size, ranked, None, req.population)
    # Attach per-item preference_score and base FFM score for transparency
    for entry in response["items"]:
        gid = entry["item_id"]
        entry["preference_score"] = pref_by_id.get(gid)
        entry["base_ffm_score"] = base_by_id.get(gid)
    response["calibration"] = {
        "n_liked_resolved": len(liked_embs),
        "n_liked_requested": len(req.liked_garment_ids),
        "preference_weight": delta,
        "base_pool_size": wide_k,
    }
    return response


# intent: compare trained research variants side-by-side + rule-based baseline
# status: done
# next: add latency tracking per model
# confidence: high


@app.post("/compare", response_model=CompareResponse)
def compare_models(req: CompareRequest):
    """Compare all available research model variants + rule-based baseline.

    Returns predictions from every available variant and recommends the model that
    matches the requested population objective.
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
    pop = req.population
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

    recommended = _select_available_research_model(req.population, all_results) or "rule_based"

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


def _research_model_candidates(population: str | None) -> list[str]:
    """Return model preference order for a population objective."""
    # intent: prevent Vietnamese density-ratio models from becoming the US-women default
    # status: done
    # next: tune direct population-specific models when target labels exist
    # blockers: no labeled Vietnamese garment-fit feedback dataset yet
    # confidence: high
    # Female upper-body specialized model is preferred when we know the user
    # is female and shopping for upper-body — that's our paper's primary model.
    if population == "vietnamese":
        return [
            "gbm_female_upper_vn",          # paper's primary model
            "gbm_indep_tempered_a05",
            "gbm_copula_tempered_a075",
            "gbm_copula",
            "gbm_uniform_current",
            "gbm_uniform",
        ]
    return [
        "gbm_female_upper_vn",              # still best for female upper-body
        "gbm_uniform_current",
        "gbm_uniform",
        "gbm_indep_tempered_a05",
        "gbm_copula_tempered_a075",
        "gbm_copula",
    ]


def _select_available_research_model(population: str | None, results: dict) -> str | None:
    """Select the first population-preferred model present in a result map."""
    for name in _research_model_candidates(population):
        if name in results:
            return name
    return next(iter(results), None)


def _predict_population_research_model(
    population: str | None,
    height_cm: float,
    weight_kg: float,
    age: float = 30.0,
    body_type: str = "unknown",
    category: str = "dress",
) -> tuple[str, dict | None]:
    """Run the first loadable research variant for the requested population."""
    candidates = _research_model_candidates(population)
    for name in candidates:
        result = _predict_research_model(name, height_cm, weight_kg, age, body_type, category)
        if result is not None:
            return name, result
    return candidates[0], None


def _stabilize_visual_research_result(research_result: dict, rule_based_size: str) -> dict:
    """Keep visual-only predictions from jumping far beyond calibrated chest size."""
    # intent: prevent upper-body-only webcam ratios from pumping estimates to XL/XXL
    # status: done
    # next: replace this guard with a direct webcam-ratio-to-size model when labeled data exists
    # blockers: current GBM expects full body weight, but visual flow estimates weight from chest-up ratios
    # confidence: high
    predicted_size = research_result.get("predicted_size")
    predicted_idx = SIZE_INDEX.get(str(predicted_size))
    rule_idx = SIZE_INDEX.get(str(rule_based_size))
    if predicted_idx is None or rule_idx is None:
        return research_result

    if predicted_idx == rule_idx:
        return research_result

    stabilized = dict(research_result)
    stabilized["raw_predicted_size"] = predicted_size
    stabilized["raw_confidence_score"] = research_result.get("confidence_score")
    stabilized["raw_probabilities"] = research_result.get("probabilities")
    stabilized["predicted_size"] = rule_based_size
    stabilized["confidence"] = "medium"
    stabilized["confidence_score"] = 0.5
    stabilized["probabilities"] = {rule_based_size: 1.0}
    stabilized["stabilized_by"] = "visual_chest_heuristic"
    stabilized["note"] = "Visual mode uses upper-body ratios and estimated weight only; displayed size follows calibrated chest sizing."
    return stabilized


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


def _chest_to_size_heuristic(chest_cm: float, population: str | None = "us_women") -> str:
    """Map an estimated bust/chest circumference to the nearest population size band."""
    from src.population_profiles import get_population_profile

    profile = get_population_profile(population)
    best_size = "M"
    best_distance = float("inf")
    for size, bands in profile.size_bands_cm.items():
        bust_band = bands.get("bust")
        if bust_band is None:
            continue
        lo, hi = bust_band
        if lo <= chest_cm <= hi:
            return size
        distance = min(abs(chest_cm - lo), abs(chest_cm - hi))
        if distance < best_distance:
            best_distance = distance
            best_size = size
    return best_size


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
