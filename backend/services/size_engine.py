"""Size recommendation engine.

Uses statistical body measurement estimation from height and weight
to recommend garment sizes based on a garment's size chart.
"""

from __future__ import annotations

from typing import Literal


FitPreference = Literal["slim", "regular", "relaxed"]

# Adjustment applied to estimated chest measurement based on fit preference.
_FIT_ADJUSTMENTS: dict[FitPreference, float] = {
    "slim": -2.0,
    "regular": 0.0,
    "relaxed": 3.0,
}


def estimate_chest_cm(
    height_cm: float,
    weight_kg: float,
    fit_preference: FitPreference = "regular",
) -> float:
    """Estimate chest circumference from height and weight.

    Formula (unisex demo approximation):
        chest_cm = height_cm * 0.52 + weight_kg * 0.18

    The result is then adjusted by fit preference:
        slim    -> -2 cm
        regular ->  0 cm
        relaxed -> +3 cm
    """
    raw = height_cm * 0.52 + weight_kg * 0.18
    adjustment = _FIT_ADJUSTMENTS.get(fit_preference, 0.0)
    return round(raw + adjustment, 1)


def find_best_size(
    estimated_chest: float,
    size_chart: list[dict],
) -> dict:
    """Find the best matching size from a garment's size chart.

    Args:
        estimated_chest: The estimated chest measurement in cm.
        size_chart: List of dicts with keys: size, chestMin, chestMax.

    Returns:
        Dict with recommended_size, confidence, explanation, and alternatives.
    """
    best_size: str | None = None
    best_distance: float = float("inf")
    alternatives: list[dict] = []

    for entry in size_chart:
        chest_min = entry["chestMin"]
        chest_max = entry["chestMax"]
        midpoint = (chest_min + chest_max) / 2.0
        half_range = (chest_max - chest_min) / 2.0

        distance = abs(estimated_chest - midpoint)

        if chest_min <= estimated_chest <= chest_max:
            # Perfect fit within range
            if distance < best_distance:
                best_distance = distance
                best_size = entry["size"]
        elif distance < best_distance:
            best_distance = distance
            best_size = entry["size"]

    # Determine confidence based on how well the measurement fits
    best_entry = next(e for e in size_chart if e["size"] == best_size)
    chest_min = best_entry["chestMin"]
    chest_max = best_entry["chestMax"]
    range_size = chest_max - chest_min

    if chest_min <= estimated_chest <= chest_max:
        # Within the range — how centered are we?
        center = (chest_min + chest_max) / 2.0
        off_center = abs(estimated_chest - center) / (range_size / 2.0)
        if off_center < 0.5:
            confidence = "high"
        else:
            confidence = "medium"
    else:
        # Outside the range
        overshoot = min(
            abs(estimated_chest - chest_min),
            abs(estimated_chest - chest_max),
        )
        if overshoot <= 2:
            confidence = "medium"
        else:
            confidence = "low"

    # Build explanation
    explanation = (
        f"Based on your measurements, your estimated chest size is "
        f"{estimated_chest:.1f} cm. Size {best_size} covers "
        f"{chest_min}-{chest_max} cm."
    )

    if confidence == "high":
        explanation += " This is an excellent fit."
    elif confidence == "medium":
        explanation += " This should be a good fit, but you may want to consider adjacent sizes."
    else:
        explanation += " Your measurements fall outside typical ranges; we recommend trying on in-store."

    # Collect alternatives (adjacent sizes)
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
        "explanation": explanation,
        "alternatives": alternatives,
    }
