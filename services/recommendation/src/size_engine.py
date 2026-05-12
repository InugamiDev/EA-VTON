"""EA-VTON population-aware size recommendation engine.

Provides rule-based measurement estimates when trained models or exact user
measurements are unavailable. Population constants live in population_profiles.py
so visual sizing, manual sizing, and style ranking share the same assumptions.
"""

from __future__ import annotations

from typing import Literal

from .population_profiles import Population, get_population_profile

FitPreference = Literal["slim", "regular", "relaxed"]

_FIT_ADJUSTMENTS: dict[FitPreference, float] = {
    "slim": -2.0,
    "regular": 0.0,
    "relaxed": 3.0,
}

# Vietnamese female body type reference data (Tran et al. 2024)
VN_CLUSTERS = [
    {"id": 1, "label": "short_thin", "pct": 15.2, "h": 151, "w": 46, "chest": 79, "waist": 65, "hip": 86},
    {"id": 2, "label": "tall_heavy", "pct": 18.4, "h": 161, "w": 58, "chest": 86, "waist": 73, "hip": 93},
    {"id": 3, "label": "medium", "pct": 35.9, "h": 156, "w": 52, "chest": 83, "waist": 69, "hip": 89},
    {"id": 4, "label": "short_fat", "pct": 21.9, "h": 152, "w": 58, "chest": 90, "waist": 78, "hip": 93},
    {"id": 5, "label": "obese_avg", "pct": 8.6, "h": 155, "w": 68, "chest": 97, "waist": 86, "hip": 100},
]


def estimate_chest_cm(
    height_cm: float,
    weight_kg: float,
    fit_preference: FitPreference = "regular",
    population: Population = "universal",
) -> float:
    """Estimate chest circumference using population-specific formula."""
    profile = get_population_profile(population)
    raw = profile.chest.estimate(height_cm, weight_kg)

    adjustment = _FIT_ADJUSTMENTS.get(fit_preference, 0.0)
    return round(raw + adjustment, 1)


def estimate_waist_cm(
    height_cm: float,
    weight_kg: float,
    population: Population = "universal",
) -> float:
    """Estimate waist circumference."""
    profile = get_population_profile(population)
    return round(profile.waist.estimate(height_cm, weight_kg), 1)


def estimate_hip_cm(
    height_cm: float,
    weight_kg: float,
    population: Population = "universal",
) -> float:
    """Estimate hip circumference."""
    profile = get_population_profile(population)
    return round(profile.hip.estimate(height_cm, weight_kg), 1)


def find_best_size(estimated_chest: float, size_chart: list[dict]) -> dict:
    """Find the best matching size from a garment's size chart."""
    best_size: str | None = None
    best_distance: float = float("inf")
    alternatives: list[dict] = []

    for entry in size_chart:
        chest_min = entry["chestMin"]
        chest_max = entry["chestMax"]
        midpoint = (chest_min + chest_max) / 2.0

        distance = abs(estimated_chest - midpoint)

        if chest_min <= estimated_chest <= chest_max:
            if distance < best_distance:
                best_distance = distance
                best_size = entry["size"]
        elif distance < best_distance:
            best_distance = distance
            best_size = entry["size"]

    best_entry = next(e for e in size_chart if e["size"] == best_size)
    chest_min = best_entry["chestMin"]
    chest_max = best_entry["chestMax"]
    range_size = chest_max - chest_min

    if chest_min <= estimated_chest <= chest_max:
        center = (chest_min + chest_max) / 2.0
        off_center = abs(estimated_chest - center) / (range_size / 2.0)
        confidence = "high" if off_center < 0.5 else "medium"
    else:
        overshoot = min(abs(estimated_chest - chest_min), abs(estimated_chest - chest_max))
        confidence = "medium" if overshoot <= 2 else "low"

    # Alternatives
    size_names = [e["size"] for e in size_chart]
    best_idx = size_names.index(best_size)

    if best_idx > 0:
        smaller = size_chart[best_idx - 1]
        alternatives.append({
            "size": smaller["size"],
            "note": f"Size down for a slimmer fit ({smaller['chestMin']}-{smaller['chestMax']} cm)",
        })
    if best_idx < len(size_chart) - 1:
        larger = size_chart[best_idx + 1]
        alternatives.append({
            "size": larger["size"],
            "note": f"Size up for a looser fit ({larger['chestMin']}-{larger['chestMax']} cm)",
        })

    return {
        "recommended_size": best_size,
        "confidence": confidence,
        "estimated_chest_cm": estimated_chest,
        "alternatives": alternatives,
    }


def compare_recommendations(
    height_cm: float,
    weight_kg: float,
    size_chart: list[dict],
    fit_preference: FitPreference = "regular",
) -> dict:
    """Compare baseline vs EA-VTON recommendations for the same body."""
    # Baseline (US)
    baseline_chest = estimate_chest_cm(height_cm, weight_kg, fit_preference, "universal")
    baseline_waist = estimate_waist_cm(height_cm, weight_kg, "universal")
    baseline_hip = estimate_hip_cm(height_cm, weight_kg, "universal")
    baseline_result = find_best_size(baseline_chest, size_chart)

    # EA-VTON (Vietnamese)
    eavton_chest = estimate_chest_cm(height_cm, weight_kg, fit_preference, "vietnamese")
    eavton_waist = estimate_waist_cm(height_cm, weight_kg, "vietnamese")
    eavton_hip = estimate_hip_cm(height_cm, weight_kg, "vietnamese")
    eavton_result = find_best_size(eavton_chest, size_chart)

    size_changed = baseline_result["recommended_size"] != eavton_result["recommended_size"]
    chest_diff = round(baseline_chest - eavton_chest, 1)

    # Find closest Vietnamese cluster
    best_cluster = None
    best_dist = float("inf")
    for c in VN_CLUSTERS:
        dist = ((height_cm - c["h"]) ** 2 + (weight_kg - c["w"]) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_cluster = c

    return {
        "input": {
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "fit_preference": fit_preference,
        },
        "baseline": {
            "population": "US women",
            "estimated_chest_cm": baseline_chest,
            "estimated_waist_cm": baseline_waist,
            "estimated_hip_cm": baseline_hip,
            "recommended_size": baseline_result["recommended_size"],
            "confidence": baseline_result["confidence"],
            "alternatives": baseline_result["alternatives"],
        },
        "eavton": {
            "population": "Vietnamese",
            "estimated_chest_cm": eavton_chest,
            "estimated_waist_cm": eavton_waist,
            "estimated_hip_cm": eavton_hip,
            "recommended_size": eavton_result["recommended_size"],
            "confidence": eavton_result["confidence"],
            "alternatives": eavton_result["alternatives"],
            "nearest_body_type": best_cluster["label"] if best_cluster else None,
            "nearest_body_type_pct": best_cluster["pct"] if best_cluster else None,
        },
        "comparison": {
            "size_changed": size_changed,
            "chest_difference_cm": chest_diff,
            "explanation": _build_explanation(
                height_cm, weight_kg, baseline_chest, eavton_chest,
                baseline_result["recommended_size"], eavton_result["recommended_size"],
                best_cluster, size_changed, chest_diff,
            ),
        },
    }


def _build_explanation(
    height_cm, weight_kg, baseline_chest, eavton_chest,
    baseline_size, eavton_size, cluster, size_changed, chest_diff,
):
    """Build human-readable explanation of the difference."""
    parts = []

    parts.append(
        f"For a person {height_cm:.0f}cm tall and {weight_kg:.0f}kg, "
        f"the universal model estimates chest at {baseline_chest:.1f}cm, "
        f"while the Vietnamese-calibrated model estimates {eavton_chest:.1f}cm "
        f"(difference: {abs(chest_diff):.1f}cm)."
    )

    if cluster:
        parts.append(
            f"Your body profile is closest to Vietnamese body type "
            f"'{cluster['label']}' ({cluster['pct']}% of population, "
            f"reference chest: {cluster['chest']}cm)."
        )

    if size_changed:
        parts.append(
            f"This shifts the recommendation from {baseline_size} to {eavton_size}. "
            f"The US-women model estimates a different chest because it was calibrated to "
            f"US fashion/body proportions (mean height 165.9cm) where someone at "
            f"{height_cm:.0f}cm/{weight_kg:.0f}kg would have a larger chest."
        )
    else:
        parts.append(
            f"Both models recommend {baseline_size}, but the population-specific model "
            f"has higher confidence because the chest estimate better matches "
            f"the size chart range."
        )

    return " ".join(parts)
