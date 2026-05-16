"""Upper-body style recommender: composition layer + explanation DAG.

Implements the three-component scoring from Section 12 of the design doc:

    s(u, i, c) = α · Fit(b_u, i) + β · Flatter(b_u, a_i) + γ · Match(a_i, c)

Each component is computed independently and exposed in the response so the
user (and we) can see why each item ranked where it did.

Fit:     fit_upper.compute_fit (parametric, no training)
Flatter: body-shape rule lookup table, population-aware when available
Match:   occasion hard filter + LAB color distance + slider projection
"""

# intent: upper-body style recommendation with population-aware fit/flattery priors
# status: done
# next: replace heuristic real-catalog metadata with merchandised attributes once available
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
from .population_profiles import (
    canonical_population,
    get_population_profile,
    size_band_midpoint as profile_size_band_midpoint,
    size_band_midpoints,
)

# ── Data paths ──
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RULES_PATH = _DATA_DIR / "upper_body_rules_vn_v1.json"
_APP_GARMENTS_PATH = _REPO_ROOT / "apps" / "api" / "src" / "data" / "garments.json"

# ── Default scoring weights (no feedback data yet) ──
DEFAULT_WEIGHTS = {"alpha": 0.40, "beta": 0.30, "gamma": 0.30}

# ── Hard-filter thresholds ──
MIN_FIT_SCORE = 0.40            # below this, item is hidden (won't fit)

# ── Match component config ──
COLOR_DELTA_E_SIGMA = 18.0      # ΔE76 distance at which color score = exp(-1)
SLIDER_AXES = ("bold", "loose", "warm", "cover")  # canonical user-facing axes
USD_TO_VND = 25_000


# intent: convert the app's real women's top catalog into Fit-Flatter-Match item metadata
# status: done
# next: replace inferred attributes with catalog-authored merchandising fields
# blockers: app catalog currently has product basics but no neckline/silhouette/color LAB fields
# confidence: medium
WOMENS_TOP_GARMENT_TYPES = {"t-shirt", "shirt", "hoodie", "sweater", "polo"}

TYPE_STYLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "t-shirt": {
        "attributes": {"neckline": "crew", "silhouette": "semi_fitted", "sleeve": "short", "length": "regular"},
        "ease_cm": {"shoulder": 0, "bust": 4, "waist": 4},
        "stretch": 0.12,
        "drape_class": "semi_fitted",
        "occasion_tags": ["casual", "date"],
    },
    "shirt": {
        "attributes": {"neckline": "crew", "silhouette": "semi_fitted", "sleeve": "long", "length": "regular"},
        "ease_cm": {"shoulder": 0, "bust": 6, "waist": 8},
        "stretch": 0.03,
        "drape_class": "semi_fitted",
        "occasion_tags": ["work", "casual", "date", "formal"],
    },
    "hoodie": {
        "attributes": {"neckline": "crew", "silhouette": "loose", "sleeve": "long", "length": "regular"},
        "ease_cm": {"shoulder": 2, "bust": 14, "waist": 14},
        "stretch": 0.08,
        "drape_class": "loose",
        "occasion_tags": ["casual", "work"],
    },
    "jacket": {
        "attributes": {"neckline": "high_neck", "silhouette": "loose", "sleeve": "long", "length": "tunic"},
        "ease_cm": {"shoulder": 2, "bust": 12, "waist": 16},
        "stretch": 0.0,
        "drape_class": "loose",
        "occasion_tags": ["casual", "work", "formal"],
    },
    "sweater": {
        "attributes": {"neckline": "crew", "silhouette": "semi_fitted", "sleeve": "long", "length": "regular"},
        "ease_cm": {"shoulder": 0, "bust": 6, "waist": 6},
        "stretch": 0.08,
        "drape_class": "semi_fitted",
        "occasion_tags": ["casual", "work", "date", "formal"],
    },
    "polo": {
        "attributes": {"neckline": "high_neck", "silhouette": "semi_fitted", "sleeve": "short", "length": "regular"},
        "ease_cm": {"shoulder": 0, "bust": 6, "waist": 6},
        "stretch": 0.08,
        "drape_class": "semi_fitted",
        "occasion_tags": ["casual", "work", "date"],
    },
    "vest": {
        "attributes": {"neckline": "high_neck", "silhouette": "loose", "sleeve": "sleeveless", "length": "regular"},
        "ease_cm": {"shoulder": 1, "bust": 10, "waist": 12},
        "stretch": 0.0,
        "drape_class": "loose",
        "occasion_tags": ["casual", "work"],
    },
}

COLOR_STYLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "white": {"hex": "#fafafa", "lab": [98.0, 0.0, 0.0], "temperature": "neutral", "value": "light"},
    "off white": {"hex": "#f5f0e8", "lab": [94.0, 1.0, 6.0], "temperature": "warm", "value": "light"},
    "natural": {"hex": "#d8c8b4", "lab": [80.0, 3.0, 14.0], "temperature": "warm", "value": "light"},
    "beige": {"hex": "#c8ad8d", "lab": [72.0, 5.0, 18.0], "temperature": "warm", "value": "light"},
    "light brown": {"hex": "#b98f64", "lab": [62.0, 10.0, 25.0], "temperature": "warm", "value": "medium"},
    "brown": {"hex": "#7a5238", "lab": [39.0, 12.0, 19.0], "temperature": "warm", "value": "medium"},
    "yellow": {"hex": "#d8b23f", "lab": [74.0, 3.0, 58.0], "temperature": "warm", "value": "light"},
    "pink": {"hex": "#e9a9b6", "lab": [76.0, 24.0, 4.0], "temperature": "warm", "value": "light"},
    "red": {"hex": "#a51f2f", "lab": [36.0, 47.0, 24.0], "temperature": "warm", "value": "medium"},
    "green": {"hex": "#3f7d58", "lab": [46.0, -28.0, 19.0], "temperature": "cool", "value": "medium"},
    "olive": {"hex": "#5f6a3d", "lab": [43.0, -9.0, 23.0], "temperature": "warm", "value": "medium"},
    "dark olive": {"hex": "#3f4a30", "lab": [30.0, -9.0, 17.0], "temperature": "warm", "value": "dark"},
    "blue": {"hex": "#4b79b7", "lab": [50.0, 2.0, -38.0], "temperature": "cool", "value": "medium"},
    "navy": {"hex": "#1f2a44", "lab": [16.5, 4.0, -22.0], "temperature": "cool", "value": "dark"},
    "black": {"hex": "#1a1a1a", "lab": [10.0, 0.0, 0.0], "temperature": "neutral", "value": "dark"},
    "gray": {"hex": "#8b8f94", "lab": [59.0, 0.0, -2.0], "temperature": "neutral", "value": "medium"},
}


def size_to_user_cm(predicted_size: str, population: str | None = "us_women") -> dict[str, float]:
    """Return midpoint body circumferences for a population-specific size."""
    return size_band_midpoints(predicted_size, population)


def _size_band_midpoint(
    size: str,
    section: str,
    fallback: float,
    population: str | None = "us_women",
) -> float:
    return profile_size_band_midpoint(size, section, fallback, population)


def _display_image_url(garment: dict) -> str | None:
    images = garment.get("images") or []
    display = next((img for img in images if img.get("type") == "display"), None)
    image = display or (images[0] if images else None)
    return image.get("url") if image else None


def _primary_color_style(garment: dict) -> dict[str, Any]:
    colors = garment.get("colors") or []
    for color in colors:
        style = COLOR_STYLE_DEFAULTS.get(str(color).strip().lower())
        if style:
            return style
    return {"hex": "#8b8f94", "lab": [59.0, 0.0, -2.0], "temperature": "neutral", "value": "medium"}


def _type_defaults(garment_type: str) -> dict[str, Any]:
    defaults = TYPE_STYLE_DEFAULTS.get(garment_type, TYPE_STYLE_DEFAULTS["t-shirt"])
    return {
        "attributes": dict(defaults["attributes"]),
        "ease_cm": dict(defaults["ease_cm"]),
        "stretch": float(defaults["stretch"]),
        "drape_class": defaults["drape_class"],
        "occasion_tags": list(defaults["occasion_tags"]),
    }


def _infer_style_metadata(garment: dict) -> dict[str, Any]:
    garment_type = str(garment.get("type", "t-shirt"))
    meta = _type_defaults(garment_type)
    name = str(garment.get("name", "")).lower()
    description = str(garment.get("description", "")).lower()
    text = f"{name} {description}"
    attrs = meta["attributes"]
    ease = meta["ease_cm"]

    material_tags: list[str] = []
    for needle, tag in [
        ("cotton", "cotton"),
        ("supima", "supima_cotton"),
        ("airism", "airism"),
        ("denim", "denim"),
        ("wool", "wool"),
        ("merino", "merino_wool"),
        ("puffer", "insulated"),
        ("pufftech", "insulated"),
        ("parka", "shell"),
        ("water-repellent", "water_repellent"),
        ("uv", "uv_protection"),
    ]:
        if needle in text and tag not in material_tags:
            material_tags.append(tag)

    if ("sweat " in text or name.startswith("sweat")) and "sweat_fleece" not in material_tags:
        material_tags.append("sweat_fleece")

    fabric_weight = "medium"
    if any(token in text for token in ("lightweight", "ultra-light", "airism")):
        fabric_weight = "light"
    elif any(token in text for token in ("heavyweight", "puffer", "pufftech", "insulation")):
        fabric_weight = "heavy"

    structure = "soft"
    if any(token in text for token in ("crisp", "denim", "parka", "windproof", "puffer", "structure")):
        structure = "structured"

    closure = "pullover"
    if "button" in text or "placket" in text:
        closure = "button"
    elif "zip" in text or "zipper" in text:
        closure = "zip"

    layering = "base_layer"
    if garment_type in {"hoodie", "jacket", "vest"}:
        layering = "outer_layer"
    elif garment_type in {"sweater", "polo", "shirt"}:
        layering = "mid_layer"

    weather_tags: list[str] = []
    if "uv" in text:
        weather_tags.append("sun")
    if "water-repellent" in text or "rain" in text:
        weather_tags.append("rain")
    if any(token in text for token in ("warm", "insulation", "puffer", "pufftech", "wool")):
        weather_tags.append("cool_weather")

    if "v-neck" in name or "v neck" in name or "skipper" in name:
        attrs["neckline"] = "v_neck"
    elif "polo" in name or "parka" in name or "vest" in name:
        attrs["neckline"] = "high_neck"
    elif "crew" in name:
        attrs["neckline"] = "crew"

    if "oversized" in name or "boxy" in name:
        attrs["silhouette"] = "oversized"
        meta["drape_class"] = "oversized"
        ease["shoulder"] = max(ease.get("shoulder", 0), 3)
        ease["bust"] = max(ease.get("bust", 0), 18)
        ease["waist"] = max(ease.get("waist", 0), 18)
    elif "slim" in name:
        attrs["silhouette"] = "fitted"
        meta["drape_class"] = "fitted"
        ease["bust"] = min(ease.get("bust", 4), 4)
        ease["waist"] = min(ease.get("waist", 4), 4)

    if "sleeveless" in name or "vest" in name:
        attrs["sleeve"] = "sleeveless"
    elif "half sleeve" in name or "short" in name or "t-shirt" in name or "polo" in name:
        attrs["sleeve"] = "short"
    elif "long" in name or garment_type in {"hoodie", "jacket", "sweater"}:
        attrs["sleeve"] = "long"

    if "parka" in name or "jacket" in name:
        attrs["length"] = "tunic"
    elif "crop" in name:
        attrs["length"] = "crop"

    attrs["pattern"] = "horizontal_stripe" if "stripe" in name or "striped" in name else "solid"
    attrs["fabric_weight"] = fabric_weight
    attrs["structure"] = structure
    attrs["closure"] = closure
    attrs["layering"] = layering

    color = _primary_color_style(garment)
    attrs["color_temperature"] = color["temperature"]
    attrs["color_value"] = color["value"]

    is_evening_friendly = attrs["color_value"] == "dark" or attrs["neckline"] == "v_neck"
    if is_evening_friendly and "date" not in meta["occasion_tags"]:
        meta["occasion_tags"].append("date")
    if is_evening_friendly and "party" not in meta["occasion_tags"]:
        meta["occasion_tags"].append("party")

    return {
        **meta,
        "primary_color_hex": color["hex"],
        "primary_color_lab": color["lab"],
        "garment_parameters": {
            "material_tags": material_tags,
            "fabric_weight": fabric_weight,
            "structure": structure,
            "closure": closure,
            "layering": layering,
            "weather_tags": weather_tags,
        },
    }


def _chart_to_upper_fit_chart(garment: dict, ease_cm: dict[str, float]) -> dict[str, dict[str, float]]:
    chart: dict[str, dict[str, float]] = {}
    for entry in garment.get("sizeChart", []):
        size = str(entry.get("size", "")).upper()
        if not size:
            continue

        chest_min = float(entry.get("chestMin", 0.0))
        chest_max = float(entry.get("chestMax", chest_min))
        if chest_min <= 0:
            continue

        bust_midpoint = (chest_min + chest_max) / 2
        target_bust = max(60.0, bust_midpoint - float(ease_cm.get("bust", 0.0)))

        waist_min = entry.get("waistMin")
        waist_max = entry.get("waistMax", waist_min)
        if waist_min is not None:
            target_waist = (float(waist_min) + float(waist_max)) / 2
        else:
            target_waist = _size_band_midpoint(size, "waist", target_bust * 0.78)

        chart[size] = {
            "shoulder": round(_size_band_midpoint(size, "shoulder", target_bust * 0.42), 1),
            "bust": round(target_bust, 1),
            "waist": round(target_waist, 1),
        }
    return chart


def _app_garment_to_style_item(garment: dict) -> dict[str, Any] | None:
    garment_type = str(garment.get("type", ""))
    if garment_type not in WOMENS_TOP_GARMENT_TYPES:
        return None

    style = _infer_style_metadata(garment)
    size_chart = _chart_to_upper_fit_chart(garment, style["ease_cm"])
    if not size_chart:
        return None

    price = float(garment.get("price", 0.0))
    price_vnd = int(round(price * USD_TO_VND)) if price else None

    return {
        "item_id": garment["id"],
        "title": garment.get("name", garment["id"]),
        "category": garment_type,
        "target_gender": "female",
        "garment_family": "tops",
        "brand": garment.get("brand"),
        "image_url": _display_image_url(garment),
        "price_vnd": price_vnd,
        "available_sizes": [str(size).upper() for size in garment.get("sizes", [])],
        "size_chart_cm": size_chart,
        "ease_cm": style["ease_cm"],
        "stretch": style["stretch"],
        "drape_class": style["drape_class"],
        "attributes": style["attributes"],
        "garment_parameters": style["garment_parameters"],
        "primary_color_hex": style["primary_color_hex"],
        "primary_color_lab": style["primary_color_lab"],
        "occasion_tags": style["occasion_tags"],
    }


def _load_app_catalog_items() -> list[dict]:
    with open(_APP_GARMENTS_PATH, "r", encoding="utf-8") as f:
        garments = json.load(f)

    items = [_app_garment_to_style_item(garment) for garment in garments]
    return [item for item in items if item is not None]


# ── Loaders ──

class _DataCache:
    """Lazy singleton loader for rules + real app garment catalog."""
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
    def get_catalog(cls, name: str = "app_garments_v1") -> list[dict]:
        """Return the cached app catalog (14 hand-curated items).

        For DeepFashion2 (140k items) use _resolve_catalog → query_candidates
        at request time so we don't load 140k rows into Python memory.
        """
        if cls._catalog is None:
            with cls._lock:
                if cls._catalog is None:
                    cls._catalog = {"items": _load_app_catalog_items()}
        return cls._catalog["items"]


def _resolve_catalog(source: str, pool_size: int) -> list[dict]:
    """Resolve the catalog dict-list given a named source.

    source ∈ {"app_garments_v1", "deepfashion2_upper", "merged"}
    """
    if source == "app_garments_v1":
        return _DataCache.get_catalog()

    if source in ("deepfashion2_upper", "merged"):
        try:
            from .deepfashion2_catalog import DeepFashion2Catalog, is_available
        except Exception:
            return _DataCache.get_catalog()

        if not is_available():
            # Fall back to app catalog if DB hasn't been built yet
            return _DataCache.get_catalog()

        df2 = DeepFashion2Catalog.get_instance()
        df2_items = df2.query_candidates(sample_size=pool_size)

        if source == "merged":
            return _DataCache.get_catalog() + df2_items
        return df2_items

    return _DataCache.get_catalog()


# ── Component: Flatter ──

# intent: provide US/global upper-body shape flattery without using VN cluster rules
# status: done
# next: validate rules with explicit user feedback or return-rate data
# blockers: no labeled US body-shape-to-style preference dataset yet
# confidence: medium

GLOBAL_SHAPE_PROTOTYPES: dict[int, tuple[str, str, tuple[float, float]]] = {
    1: ("hourglass", "Hourglass", (1.04, 0.72)),
    2: ("pear", "Pear", (0.94, 0.82)),
    3: ("inverted_triangle", "Inverted triangle", (1.16, 0.84)),
    4: ("rectangle", "Rectangle", (1.04, 0.91)),
    5: ("balanced", "Balanced", (1.04, 0.82)),
}

GLOBAL_FLATTER_RULES: dict[str, dict[str, dict[str, int]]] = {
    "1": {
        "silhouette": {"fitted": 1, "semi_fitted": 1, "loose": 0, "oversized": -1},
        "neckline": {"v_neck": 1, "crew": 0, "high_neck": 0},
        "length": {"regular": 1, "tunic": 0, "crop": 0},
    },
    "2": {
        "silhouette": {"fitted": 0, "semi_fitted": 1, "loose": 1, "oversized": 0},
        "neckline": {"v_neck": 0, "crew": 1, "high_neck": 1},
        "sleeve": {"short": 0, "long": 1, "sleeveless": -1},
        "pattern": {"solid": 0, "horizontal_stripe": 1},
    },
    "3": {
        "silhouette": {"fitted": 1, "semi_fitted": 1, "loose": 0, "oversized": -1},
        "neckline": {"v_neck": 1, "crew": 0, "high_neck": -1},
        "pattern": {"solid": 1, "horizontal_stripe": -1},
        "color_value": {"dark": 1, "medium": 0, "light": -1},
    },
    "4": {
        "silhouette": {"fitted": 1, "semi_fitted": 1, "loose": 0, "oversized": 1},
        "neckline": {"v_neck": 1, "crew": 0, "high_neck": 0},
        "pattern": {"solid": 0, "horizontal_stripe": 1},
        "length": {"regular": 1, "tunic": 0, "crop": 1},
    },
    "5": {
        "silhouette": {"fitted": 0, "semi_fitted": 1, "loose": 0, "oversized": 0},
        "neckline": {"v_neck": 0, "crew": 1, "high_neck": 0},
        "length": {"regular": 1, "tunic": 0, "crop": 0},
    },
}


def _normalize_shape_ratios(ratios: dict[str, float]) -> tuple[float, float]:
    shoulder_to_hip = float(ratios.get("sh", 1.04))
    waist_to_hip = float(ratios.get("wh", 0.84))

    # 2D fallback landmarks inflate S/H; map the neutral 2D priors back to
    # world-like proportions before classifying body shape.
    if shoulder_to_hip > 1.30:
        shoulder_to_hip = shoulder_to_hip / 1.50 * 1.04
        waist_to_hip = waist_to_hip / 0.95 * 0.84

    return shoulder_to_hip, waist_to_hip


def _classify_global_body_shape(height_cm: float, ratios: dict[str, float]) -> BodyTypeResult:
    shoulder_to_hip, waist_to_hip = _normalize_shape_ratios(ratios)

    if shoulder_to_hip > 1.12 and waist_to_hip >= 0.78:
        shape_id = 3
    elif waist_to_hip < 0.74:
        shape_id = 1
    elif waist_to_hip > 0.88:
        shape_id = 4
    elif shoulder_to_hip < 0.98:
        shape_id = 2
    else:
        shape_id = 5

    distances: dict[int, float] = {}
    for proto_id, (_, _, (proto_sh, proto_wh)) in GLOBAL_SHAPE_PROTOTYPES.items():
        distances[proto_id] = math.sqrt(
            ((shoulder_to_hip - proto_sh) / 0.08) ** 2
            + ((waist_to_hip - proto_wh) / 0.08) ** 2
        )

    label, display_label, _ = GLOBAL_SHAPE_PROTOTYPES[shape_id]
    nearest_two = sorted(distances.values())[:2]
    margin = nearest_two[1] - nearest_two[0] if len(nearest_two) > 1 else 0.0
    confidence = max(0.52, min(0.92, 0.62 + margin * 0.12))

    return BodyTypeResult(
        cluster_id=shape_id,
        label=label,
        label_vi=display_label,
        confidence=round(confidence, 4),
        distances={k: round(v, 4) for k, v in distances.items()},
        feature_vector=[round(height_cm, 1), round(shoulder_to_hip, 3), round(waist_to_hip, 3)],
    )


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
    population: str | None = "us_women",
) -> FlatterResult:
    """Score how well an item's attributes flatter the user's body shape.

    Logic:
      Σ polarity(cluster, attr_dim, attr_value)  over all attribute dimensions
      normalized to [0, 1] via shifted sigmoid (raw_sum / n_attrs):
        score = 0.5 + 0.5 · tanh(mean_polarity)
    """
    if canonical_population(population) == "vietnamese":
        rules = _DataCache.get_rules()["rules"]
    else:
        rules = GLOBAL_FLATTER_RULES
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
    color_matched_anchor_idx: int | None  # which palette anchor matched best
    slider_score: float | None  # None when no slider was engaged
    slider_projection: dict[str, float]


def _delta_e76(lab_a: list[float], lab_b: list[float]) -> float:
    """CIE76 color distance in LAB space (kept for back-compat)."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab_a, lab_b)))


def _delta_e2000(lab_a: list[float], lab_b: list[float]) -> float:
    """CIEDE2000 — the modern perceptual color difference standard.

    Better than ΔE76 near pure blue/red boundaries; corrects for hue rotation
    and chroma weighting. Reference: Sharma et al. 2005.
    """
    L1, a1, b1 = lab_a
    L2, a2, b2 = lab_b
    kL = kC = kH = 1.0

    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    Cbar = (C1 + C2) / 2.0
    G = 0.5 * (1 - math.sqrt(Cbar ** 7 / (Cbar ** 7 + 25 ** 7))) if Cbar > 0 else 0
    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)
    C1p = math.sqrt(a1p * a1p + b1 * b1)
    C2p = math.sqrt(a2p * a2p + b2 * b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if (a1p or b1) else 0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if (a2p or b2) else 0

    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0
    else:
        diff = h2p - h1p
        if diff > 180: diff -= 360
        elif diff < -180: diff += 360
        dhp = diff
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))

    Lbarp = (L1 + L2) / 2
    Cbarp = (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbarp = h1p + h2p
    else:
        if abs(h1p - h2p) <= 180:
            hbarp = (h1p + h2p) / 2
        else:
            hbarp = (h1p + h2p + 360) / 2 if (h1p + h2p) < 360 else (h1p + h2p - 360) / 2

    T = (1 - 0.17 * math.cos(math.radians(hbarp - 30))
           + 0.24 * math.cos(math.radians(2 * hbarp))
           + 0.32 * math.cos(math.radians(3 * hbarp + 6))
           - 0.20 * math.cos(math.radians(4 * hbarp - 63)))
    SL = 1 + (0.015 * (Lbarp - 50) ** 2) / math.sqrt(20 + (Lbarp - 50) ** 2)
    SC = 1 + 0.045 * Cbarp
    SH = 1 + 0.015 * Cbarp * T
    RT = (-2 * math.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25 ** 7))
          * math.sin(math.radians(60 * math.exp(-((hbarp - 275) / 25) ** 2)))) if Cbarp > 0 else 0

    return math.sqrt(
        (dLp / (kL * SL)) ** 2
        + (dCp / (kC * SC)) ** 2
        + (dHp / (kH * SH)) ** 2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )


def _load_seasonal_palette(season: str) -> list[list[float]] | None:
    """Resolve a season key (e.g. 'soft_autumn') to its LAB anchor list."""
    path = _DATA_DIR / "seasonal_palettes.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            palettes = json.load(f).get("palettes", {})
    except Exception:
        return None
    entry = palettes.get(season.lower())
    return entry["anchors"] if entry else None


def _resolve_palette_anchors(
    season: str | None,
    palette_anchors: list[list[float]] | None,
    palette_lab: list[float] | None,
) -> list[list[float]] | None:
    """Decide which anchor list to score against. Priority:
       1. explicit list of anchors
       2. season name → preset anchors
       3. single LAB point (back-compat) → list of one
    """
    if palette_anchors:
        return [a for a in palette_anchors if len(a) == 3]
    if season:
        anchors = _load_seasonal_palette(season)
        if anchors:
            return anchors
    if palette_lab and len(palette_lab) == 3:
        return [palette_lab]
    return None


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
    user_palette_anchors: list[list[float]] | None = None,
    season: str | None = None,
) -> MatchResult:
    """Compute match score combining occasion, color, and slider projection.

    Color matching priority:
      1. user_palette_anchors (explicit list of LAB anchors)
      2. season (e.g. 'soft_autumn' → load 8 anchors from preset palette)
      3. user_palette_lab (single LAB point — back-compat)

    Item color score = max over anchors of exp(-(ΔE2000 / σ)²).
    """
    sliders = sliders or {}
    occ_tags = item.get("occasion_tags", [])
    occasion_ok = bool(occ_tags) and occasion in occ_tags

    # Color score — multi-anchor max with CIEDE2000
    delta_e = None
    color_score = 0.5
    matched_anchor_idx = None
    anchors = _resolve_palette_anchors(season, user_palette_anchors, user_palette_lab)
    item_lab = item.get("primary_color_lab")
    if anchors and item_lab and len(item_lab) == 3:
        per_anchor = [_delta_e2000(item_lab, anchor) for anchor in anchors]
        delta_e = min(per_anchor)
        matched_anchor_idx = per_anchor.index(delta_e)
        color_score = math.exp(-(delta_e / COLOR_DELTA_E_SIGMA) ** 2)

    # Slider projection — only count axes the user actually moved, so a single
    # active slider isn't diluted by averaging over four neutral axes.
    item_axes = _attribute_to_axes(item.get("attributes", {}))
    aligned_axes = {axis: round(item_axes.get(axis, 0.0), 3) for axis in SLIDER_AXES}
    active = [a for a in SLIDER_AXES if abs(float(sliders.get(a, 0.0))) > 0.05]
    if not active:
        slider_score = None  # signal: no slider engagement; treat as "no preference"
    else:
        dot = sum(float(sliders.get(a, 0.0)) * item_axes.get(a, 0.0) for a in active)
        slider_score = 0.5 + 0.5 * (dot / len(active))

    # Combined match:
    #   - Hard-filtered by occasion
    #   - When user did NOT touch sliders, match is purely color match (1.0 if
    #     no palette provided, else exp(-ΔE/σ)). This avoids artificially
    #     capping match at 0.75 when sliders are absent.
    #   - When user DID move sliders, weight color + slider 50/50.
    if not occasion_ok:
        score = 0.0
    elif slider_score is None:
        score = color_score
    else:
        score = 0.5 * color_score + 0.5 * slider_score

    return MatchResult(
        score=round(score, 4),
        occasion_ok=occasion_ok,
        color_score=round(color_score, 4),
        color_delta_e=round(delta_e, 2) if delta_e is not None else None,
        color_matched_anchor_idx=matched_anchor_idx,
        slider_score=round(slider_score, 4) if slider_score is not None else None,
        slider_projection=aligned_axes,
    )


def match_to_dict(result: MatchResult) -> dict[str, Any]:
    out: dict[str, Any] = {
        "score": result.score,
        "occasion_ok": result.occasion_ok,
        "color": {
            "score": result.color_score,
            "delta_e2000": result.color_delta_e,
            "matched_anchor_idx": result.color_matched_anchor_idx,
        },
    }
    # Only surface sliders block when the user actually moved a slider.
    # Keeps the explanation DAG clean for the default (no-slider) UX.
    if result.slider_score is not None:
        out["sliders"] = {
            "score": result.slider_score,
            "item_projection": result.slider_projection,
        }
    return out


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
    user_palette_anchors: list[list[float]] | None = None,
    season: str | None = None,
    top_k: int = 10,
    weights: dict[str, float] | None = None,
    catalog: list[dict] | None = None,
    population: str | None = "us_women",
    catalog_source: str = "app_garments_v1",
    candidate_pool_size: int = 300,
) -> tuple[BodyTypeResult, list[RankedItem]]:
    """Rank upper-body items for a user.

    catalog_source ∈ {"app_garments_v1", "deepfashion2_upper", "merged"}
        - app_garments_v1: 14 hand-curated branded items
        - deepfashion2_upper: pull ~candidate_pool_size pre-filtered items
          from the 140k labeled DeepFashion2 catalog (DuckDB-backed)
        - merged: app catalog + DF2 sample concatenated

    Returns (body_type, ranked_items[:top_k]).
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    # Build catalog from source selection
    if catalog is None:
        catalog = _resolve_catalog(catalog_source, candidate_pool_size)

    if canonical_population(population) == "vietnamese":
        body_type = VNBodyTypeClassifier.get_instance().predict(
            height_cm=height_cm,
            shoulder_to_hip=ratios.get("sh", 1.08),
            waist_to_hip=ratios.get("wh", 0.78),
            shoulder_to_torso=ratios.get("st", 0.66),
            torso_to_leg=ratios.get("tl", 0.45),
            arm_to_torso=ratios.get("at", 1.10),
        )
    else:
        body_type = _classify_global_body_shape(height_cm, ratios)

    user_cm = size_to_user_cm(predicted_size, population)
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

        flatter = compute_flatter(body_type, item.get("attributes", {}), population)
        match = compute_match(
            item, occasion, sliders,
            user_palette_lab=user_palette_lab,
            user_palette_anchors=user_palette_anchors,
            season=season,
        )

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
    population: str | None = "us_women",
) -> dict[str, Any]:
    """Assemble the API response with full explanation DAGs."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    user_cm = size_to_user_cm(predicted_size, population)
    profile = get_population_profile(population)

    body_shape = {
        "id": body_type.cluster_id,
        "label": body_type.label,
        "display_label": body_type.label_vi,
        "confidence": body_type.confidence,
        "distances": body_type.distances,
    }
    user_state: dict[str, Any] = {
        "body_shape": body_shape,
        "predicted_size": predicted_size,
        "population": profile.label,
        "size_band_midpoints_cm": user_cm,
    }
    if canonical_population(population) == "vietnamese":
        user_state["body_type_vn"] = {
            "cluster_id": body_type.cluster_id,
            "label": body_type.label,
            "label_vi": body_type.label_vi,
            "confidence": body_type.confidence,
            "distances": body_type.distances,
        }

    return {
        "user_state": user_state,
        "weights": weights,
        "items": [
            {
                "item_id": r.item["item_id"],
                "title": r.item["title"],
                "brand": r.item.get("brand"),
                "image_url": r.item.get("image_url"),
                "price_vnd": r.item.get("price_vnd"),
                "category": r.item.get("category"),
                "garment_parameters": r.item.get("garment_parameters"),
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
        population="us_women",
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
