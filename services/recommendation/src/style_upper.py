"""Upper-body style recommender: composition layer + explanation DAG.

Implements the three-component scoring from Section 12 of the design doc:

    s(u, i, c) = α · Fit(b_u, i) + β · Flatter(b_u, a_i) + γ · Match(a_i, c)

Each component is computed independently and exposed in the response so the
user (and we) can see why each item ranked where it did.

Fit:     fit_upper.compute_fit (parametric, no training)
Flatter: rule lookup table by VN body type cluster (Trieu et al. 2024)
Match:   occasion hard filter + LAB color distance + slider projection
"""

# intent: upper-body style recommendation (Vietnamese women, v1)
# status: done
# next: replace static rules with FM correction once feedback data exists
# confidence: high

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from .body_type_vn import VNBodyTypeClassifier, BodyTypeResult
from .fit_upper import compute_fit, fit_to_dict, FitResult

# ── Data paths ──
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_RULES_PATH = _DATA_DIR / "upper_body_rules_vn_v1.json"
_CATALOG_PATH = _DATA_DIR / "upper_catalog_seed_v1.json"

# ── Default scoring weights (no feedback data yet) ──
DEFAULT_WEIGHTS = {"alpha": 0.40, "beta": 0.30, "gamma": 0.30}

# ── Hard-filter thresholds ──
MIN_FIT_SCORE = 0.40            # below this, item is hidden (won't fit)

# ── Match component config ──
COLOR_DELTA_E_SIGMA = 18.0      # ΔE76 distance at which color score = exp(-1)
SLIDER_AXES = ("bold", "loose", "warm", "cover")  # canonical user-facing axes


# ── VN size pivot table (target body bands per VN size, from reports/size_rosetta_stone) ──
# Used to map predicted_size → user body circumference midpoint.
VN_SIZE_BANDS_CM: dict[str, dict[str, tuple[float, float]]] = {
    "XS": {"shoulder": (33, 35), "bust": (80, 83), "waist": (61, 64)},
    "S":  {"shoulder": (35, 37), "bust": (84, 87), "waist": (66, 69)},
    "M":  {"shoulder": (37, 39), "bust": (90, 93), "waist": (71, 74)},
    "L":  {"shoulder": (39, 41), "bust": (96, 105), "waist": (76, 86)},
    "XL": {"shoulder": (41, 43), "bust": (110, 115), "waist": (91, 96)},
    "XXL":{"shoulder": (43, 45), "bust": (120, 125), "waist": (101, 106)},
}


def size_to_user_cm(predicted_size: str) -> dict[str, float]:
    """Return midpoint body circumferences for a VN size."""
    bands = VN_SIZE_BANDS_CM.get(predicted_size.upper())
    if not bands:
        bands = VN_SIZE_BANDS_CM["M"]
    return {sec: (lo + hi) / 2 for sec, (lo, hi) in bands.items()}


# ── Loaders ──

class _DataCache:
    """Lazy singleton loader for rules + catalog."""
    _lock = Lock()
    _rules: dict | None = None
    _catalog: dict | None = None

    @classmethod
    def get_rules(cls) -> dict:
        if cls._rules is None:
            with cls._lock:
                if cls._rules is None:
                    with open(_RULES_PATH, "r", encoding="utf-8") as f:
                        cls._rules = json.load(f)
        return cls._rules

    @classmethod
    def get_catalog(cls, name: str = "seed_v1") -> list[dict]:
        if cls._catalog is None:
            with cls._lock:
                if cls._catalog is None:
                    with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
                        cls._catalog = json.load(f)
        return cls._catalog["items"]


# ── Component: Flatter ──

@dataclass
class FlatterResult:
    score: float                       # ∈ [0, 1]
    rules_fired: list[dict]            # [{attribute, value, polarity, source}, ...]
    cluster_id: int
    raw_sum: int                       # sum of polarities before normalization
    n_attrs: int                       # count of scored attributes


def compute_flatter(
    body_type: BodyTypeResult,
    item_attributes: dict[str, str],
) -> FlatterResult:
    """Score how well an item's attributes flatter the user's VN body cluster.

    Logic:
      Σ polarity(cluster, attr_dim, attr_value)  over all attribute dimensions
      normalized to [0, 1] via shifted sigmoid (raw_sum / n_attrs):
        score = 0.5 + 0.5 · tanh(mean_polarity)
    """
    rules = _DataCache.get_rules()["rules"]
    cluster_rules = rules.get(str(body_type.cluster_id), {})

    rules_fired: list[dict] = []
    raw_sum = 0
    n_scored = 0

    for dim, value in item_attributes.items():
        dim_rules = cluster_rules.get(dim)
        if not dim_rules or not isinstance(dim_rules, dict):
            continue
        polarity = dim_rules.get(value)
        if polarity is None:
            continue
        raw_sum += polarity
        n_scored += 1
        if polarity != 0:
            rules_fired.append({
                "attribute_dim": dim,
                "attribute_value": value,
                "polarity": polarity,
                "verdict": "recommend" if polarity > 0 else "avoid",
            })

    if n_scored == 0:
        score = 0.5
    else:
        mean_p = raw_sum / n_scored
        score = 0.5 + 0.5 * math.tanh(mean_p * 1.2)

    return FlatterResult(
        score=round(score, 4),
        rules_fired=rules_fired,
        cluster_id=body_type.cluster_id,
        raw_sum=raw_sum,
        n_attrs=n_scored,
    )


def flatter_to_dict(result: FlatterResult) -> dict[str, Any]:
    return {
        "score": result.score,
        "cluster_id": result.cluster_id,
        "n_attributes_scored": result.n_attrs,
        "raw_polarity_sum": result.raw_sum,
        "rules_fired": result.rules_fired,
    }


# ── Component: Match ──

@dataclass
class MatchResult:
    score: float
    occasion_ok: bool
    color_score: float
    color_delta_e: float | None
    slider_score: float
    slider_projection: dict[str, float]


def _delta_e76(lab_a: list[float], lab_b: list[float]) -> float:
    """CIE76 color distance in LAB space."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab_a, lab_b)))


def _attribute_to_axes(attrs: dict[str, str]) -> dict[str, float]:
    """Project item attributes onto user-facing slider axes ∈ [-1, 1].

    bold: +1 = statement (puff sleeve, off-shoulder, deep V, large print, bright)
    loose: +1 = oversized/loose, -1 = bodycon
    warm: +1 = warm color, -1 = cool color (0 = neutral)
    cover: +1 = high-neck/long-sleeve/tunic, -1 = sleeveless/crop/deep-v
    """
    sil = attrs.get("silhouette", "")
    nck = attrs.get("neckline", "")
    slv = attrs.get("sleeve", "")
    lng = attrs.get("length", "")
    pat = attrs.get("pattern", "")
    ct = attrs.get("color_temperature", "")
    cv = attrs.get("color_value", "")

    silhouette_loose = {"bodycon": -1.0, "fitted": -0.5, "semi_fitted": 0.0,
                        "a_line": 0.3, "loose": 0.7, "oversized": 1.0}.get(sil, 0.0)

    bold_pieces = {
        "neckline_bold": {"deep_v_neck": +1, "off_shoulder": +1, "sweetheart": +0.5,
                          "v_neck": 0, "scoop": 0, "boat": +0.3, "crew": -0.5,
                          "high_neck": -0.5}.get(nck, 0),
        "sleeve_bold":   {"puff": +1, "drop_shoulder": +0.5, "sleeveless": +0.3,
                          "cap": 0, "short": 0, "three_quarter": -0.2, "long": -0.3}.get(slv, 0),
        "pattern_bold":  {"large_print": +1, "horizontal_stripe": +0.5,
                          "vertical_stripe": +0.2, "small_print": 0, "solid": -0.3}.get(pat, 0),
        "color_bold":    {"light": +0.3, "medium": 0, "dark": -0.3}.get(cv, 0),
    }
    bold = sum(bold_pieces.values()) / 4.0

    warm = {"warm": +1.0, "cool": -1.0, "neutral": 0.0}.get(ct, 0.0)

    cover_pieces = {
        "neck": {"deep_v_neck": -1.0, "v_neck": -0.3, "scoop": -0.3, "off_shoulder": -1.0,
                 "sweetheart": -0.5, "boat": -0.2, "crew": +0.3, "high_neck": +1.0}.get(nck, 0),
        "sleeve": {"sleeveless": -1.0, "cap": -0.5, "short": -0.2, "three_quarter": +0.5,
                   "long": +1.0, "puff": 0, "drop_shoulder": -0.2}.get(slv, 0),
        "length": {"crop": -1.0, "regular": 0.0, "tunic": +1.0}.get(lng, 0),
    }
    cover = sum(cover_pieces.values()) / 3.0

    return {
        "bold": max(-1.0, min(1.0, bold)),
        "loose": silhouette_loose,
        "warm": warm,
        "cover": max(-1.0, min(1.0, cover)),
    }


def compute_match(
    item: dict,
    occasion: str,
    sliders: dict[str, float] | None = None,
    user_palette_lab: list[float] | None = None,
) -> MatchResult:
    """Compute match score combining occasion, color, and slider projection."""
    sliders = sliders or {}
    occ_tags = item.get("occasion_tags", [])
    occasion_ok = bool(occ_tags) and occasion in occ_tags

    # Color score
    delta_e = None
    color_score = 0.5
    if user_palette_lab is not None and "primary_color_lab" in item:
        delta_e = _delta_e76(item["primary_color_lab"], user_palette_lab)
        color_score = math.exp(-(delta_e / COLOR_DELTA_E_SIGMA) ** 2)

    # Slider projection — only count axes the user actually moved, so a single
    # active slider isn't diluted by averaging over four neutral axes.
    item_axes = _attribute_to_axes(item.get("attributes", {}))
    aligned_axes = {axis: round(item_axes.get(axis, 0.0), 3) for axis in SLIDER_AXES}
    active = [a for a in SLIDER_AXES if abs(float(sliders.get(a, 0.0))) > 0.05]
    if not active:
        slider_score = 0.5  # neutral when no slider engagement
    else:
        dot = sum(float(sliders.get(a, 0.0)) * item_axes.get(a, 0.0) for a in active)
        slider_score = 0.5 + 0.5 * (dot / len(active))

    # Combined match: gated by occasion (hard filter), then linear combo
    if not occasion_ok:
        score = 0.0
    else:
        score = 0.5 * color_score + 0.5 * slider_score

    return MatchResult(
        score=round(score, 4),
        occasion_ok=occasion_ok,
        color_score=round(color_score, 4),
        color_delta_e=round(delta_e, 2) if delta_e is not None else None,
        slider_score=round(slider_score, 4),
        slider_projection=aligned_axes,
    )


def match_to_dict(result: MatchResult) -> dict[str, Any]:
    return {
        "score": result.score,
        "occasion_ok": result.occasion_ok,
        "color": {
            "score": result.color_score,
            "delta_e76": result.color_delta_e,
        },
        "sliders": {
            "score": result.slider_score,
            "item_projection": result.slider_projection,
        },
    }


# ── Top-level orchestration ──

@dataclass
class RankedItem:
    item: dict
    rank: int
    total_score: float
    fit: FitResult
    flatter: FlatterResult
    match: MatchResult


def rank_upper_body(
    height_cm: float,
    predicted_size: str,
    ratios: dict[str, float],
    occasion: str,
    sliders: dict[str, float] | None = None,
    user_palette_lab: list[float] | None = None,
    top_k: int = 10,
    weights: dict[str, float] | None = None,
    catalog: list[dict] | None = None,
) -> tuple[BodyTypeResult, list[RankedItem]]:
    """Rank upper-body items for a user.

    Returns (body_type, ranked_items[:top_k]).
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    catalog = catalog if catalog is not None else _DataCache.get_catalog()

    # Classify body type
    body_type = VNBodyTypeClassifier.get_instance().predict(
        height_cm=height_cm,
        shoulder_to_hip=ratios.get("sh", 1.08),
        waist_to_hip=ratios.get("wh", 0.78),
        shoulder_to_torso=ratios.get("st", 0.66),
        torso_to_leg=ratios.get("tl", 0.45),
        arm_to_torso=ratios.get("at", 1.10),
    )

    user_cm = size_to_user_cm(predicted_size)
    pred_size_upper = predicted_size.upper()

    scored: list[RankedItem] = []
    for item in catalog:
        # Hard filter: size availability
        if pred_size_upper not in [s.upper() for s in item.get("available_sizes", [])]:
            continue

        garm_chart = item["size_chart_cm"].get(pred_size_upper)
        if garm_chart is None:
            continue

        fit = compute_fit(
            user_circumferences=user_cm,
            garment_size_chart_cm=garm_chart,
            ease_cm=item.get("ease_cm", {}),
            stretch=float(item.get("stretch", 0.0)),
            drape_class=item.get("drape_class", "semi_fitted"),
        )
        if fit.score < MIN_FIT_SCORE:
            continue

        flatter = compute_flatter(body_type, item.get("attributes", {}))
        match = compute_match(item, occasion, sliders, user_palette_lab)

        if not match.occasion_ok:
            # Hard filter on occasion
            continue

        total = (
            weights["alpha"] * fit.score
            + weights["beta"] * flatter.score
            + weights["gamma"] * match.score
        )

        scored.append(RankedItem(
            item=item, rank=0, total_score=round(total, 4),
            fit=fit, flatter=flatter, match=match,
        ))

    scored.sort(key=lambda r: r.total_score, reverse=True)
    for i, r in enumerate(scored[:top_k], start=1):
        r.rank = i

    return body_type, scored[:top_k]


def build_response(
    body_type: BodyTypeResult,
    predicted_size: str,
    ranked: list[RankedItem],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Assemble the API response with full explanation DAGs."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    user_cm = size_to_user_cm(predicted_size)

    return {
        "user_state": {
            "body_type_vn": {
                "cluster_id": body_type.cluster_id,
                "label": body_type.label,
                "label_vi": body_type.label_vi,
                "confidence": body_type.confidence,
                "distances": body_type.distances,
            },
            "predicted_size": predicted_size,
            "size_band_midpoints_cm": user_cm,
        },
        "weights": weights,
        "items": [
            {
                "item_id": r.item["item_id"],
                "title": r.item["title"],
                "brand": r.item.get("brand"),
                "image_url": r.item.get("image_url"),
                "price_vnd": r.item.get("price_vnd"),
                "category": r.item.get("category"),
                "rank": r.rank,
                "total_score": r.total_score,
                "explanation": {
                    "fit": fit_to_dict(r.fit),
                    "flatter": flatter_to_dict(r.flatter),
                    "match": match_to_dict(r.match),
                    "weights": weights,
                },
            }
            for r in ranked
        ],
    }


if __name__ == "__main__":
    # Smoke test — Group 3 user, work occasion, neutral sliders
    body, ranked = rank_upper_body(
        height_cm=158,
        predicted_size="M",
        ratios={"sh": 1.10, "wh": 0.78, "st": 0.66, "tl": 0.45, "at": 1.10},
        occasion="work",
        sliders={"bold": 0.0, "loose": 0.0, "warm": 0.0, "cover": 0.0},
        user_palette_lab=[70.0, 5.0, 15.0],  # warm-medium palette
        top_k=10,
    )
    print(f"Body type: Group {body.cluster_id} ({body.label}) conf={body.confidence}")
    print(f"\nTop {len(ranked)} for work occasion:")
    for r in ranked:
        print(f"\n  #{r.rank} score={r.total_score} | {r.item['title']}")
        print(f"     Fit={r.fit.score} ({r.fit.overall_verdict})")
        print(f"     Flatter={r.flatter.score} ({len(r.flatter.rules_fired)} rules fired)")
        print(f"     Match={r.match.score} (occasion_ok={r.match.occasion_ok}, ΔE={r.match.color_delta_e})")
