"""Geometric fit predictor for upper-body garments.

Models the fit of an upper-body garment on a user's body as the relative
deviation between the user's body circumferences and the garment's published
size-chart circumferences for the user's predicted size:

    deviation_k = (user_k − chart_k) / user_k

Positive deviation = user is bigger than the size chart's target body → tighter
Negative deviation = user is smaller than the size chart's target body → looser

We do NOT include design ease in the deviation formula because ease cancels
out when the user matches the size chart (the chart's target body is by
construction the body the designer's ease was specified for).

Stretch attenuates positive deviation only (a stretchy garment forgives a
bigger body, but does not magically shrink onto a smaller one).

Drape class affects asymmetric tolerance:
  - bodycon: looseness penalized more than tightness
  - oversized: tightness penalized more than looseness

Per-section deviations are aggregated as weighted RMS, then squashed to a
[0, 1] fit score via Gaussian decay with tolerance σ.

Reference: see Section 12.2 of reports/style_recommendation_design.md.
"""

# intent: parametric upper-body fit math, no training data required
# status: done
# next: validate against any return-data we accumulate
# confidence: high

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Body sections we score for upper-body garments.
SECTIONS = ("shoulder", "bust", "waist")

# Per-section base weight in the aggregate score. Bust is the canonical
# upper-body fit signal; waist matters less for non-fitted styles.
SECTION_WEIGHTS: dict[str, float] = {
    "shoulder": 0.30,
    "bust":     0.45,
    "waist":    0.25,
}

# Drape class → asymmetric penalty weights for (tight_side, loose_side).
# Higher number means deviation in that direction is penalized harder.
# Convention: deviation > 0 → user bigger than chart → garment tight on user.
DRAPE_PENALTY: dict[str, tuple[float, float]] = {
    "bodycon":     (1.0, 2.0),  # snug intentional, loose looks bad
    "fitted":      (1.3, 1.3),
    "semi_fitted": (1.0, 1.0),
    "a_line":      (1.0, 0.8),  # designed to flare, looseness fine
    "loose":       (1.5, 0.6),  # tightness contradicts design intent
    "oversized":   (2.0, 0.4),  # tightness very bad, looseness expected
}

# Tolerance σ — body-circumference deviation at which fit score drops to
# exp(-1) ≈ 0.37. 6% body tolerance is roughly one apparel grading step.
SIGMA = 0.06

# Verdict thresholds on effective deviation (after stretch relief).
# Tuple of (lower_bound, label); first match wins.
VERDICT_THRESHOLDS = (
    (-0.10, "too_loose"),
    (-0.04, "loose"),
    (+0.04, "good"),
    (+0.08, "snug"),
)


@dataclass
class SectionFit:
    section: str
    user_cm: float
    garment_cm: float
    ease_cm: float
    stretch: float
    deviation: float           # (user - chart) / user, before stretch
    effective_deviation: float # after stretch relief on positive side
    verdict: str               # too_loose | loose | good | snug | too_tight


@dataclass
class FitResult:
    score: float                       # ∈ (0, 1]
    sections: list[SectionFit]
    drape_class: str
    overall_verdict: str               # human-readable summary


def _verdict(deviation: float) -> str:
    """Map effective deviation to a 5-level verdict."""
    for threshold, label in VERDICT_THRESHOLDS:
        if deviation < threshold:
            return label
    return "too_tight"


def _overall_verdict(sections: list[SectionFit]) -> str:
    """Collapse per-section verdicts into a single line."""
    bad = [s for s in sections if s.verdict in ("too_tight", "too_loose")]
    if bad:
        names = ", ".join(f"{s.section} {s.verdict.replace('_', ' ')}" for s in bad)
        return f"Issues: {names}"
    snug = [s.section for s in sections if s.verdict == "snug"]
    loose = [s.section for s in sections if s.verdict == "loose"]
    if snug and not loose:
        return f"Slightly snug at {', '.join(snug)} — true to size or size up"
    if loose and not snug:
        return f"Slightly loose at {', '.join(loose)} — true to size or size down"
    if snug and loose:
        return "Mixed fit — true to size, no clear up/down"
    return "Good fit across all sections"


def compute_fit(
    user_circumferences: dict[str, float],
    garment_size_chart_cm: dict[str, float],
    ease_cm: dict[str, float] | None = None,
    stretch: float = 0.0,
    drape_class: str = "semi_fitted",
) -> FitResult:
    """Compute fit score and per-section breakdown.

    Args:
        user_circumferences: e.g. {"shoulder": 38, "bust": 91, "waist": 72}
                            (typically band-midpoints from VN size pivot)
        garment_size_chart_cm: garment chart at user's predicted size
                              (the body circumference the size is designed for)
        ease_cm: designed ease per section (kept for explanation only — does
                 NOT enter the deviation formula because it cancels with the
                 chart's design intent)
        stretch: fabric stretch coefficient ∈ [0, 0.3]; reduces positive
                 deviation (tight side) only
        drape_class: one of DRAPE_PENALTY keys
    """
    if drape_class not in DRAPE_PENALTY:
        drape_class = "semi_fitted"
    ease_cm = ease_cm or {}
    stretch = max(0.0, min(stretch, 0.3))
    tight_w, loose_w = DRAPE_PENALTY[drape_class]

    sections: list[SectionFit] = []
    weighted_sq_dev = 0.0
    total_weight = 0.0

    for sec in SECTIONS:
        if sec not in user_circumferences or sec not in garment_size_chart_cm:
            continue
        user_c = float(user_circumferences[sec])
        garm_c = float(garment_size_chart_cm[sec])
        if user_c <= 0:
            continue

        # Core deviation: how much bigger is the user than the size chart's
        # target body? Positive → tight; negative → loose.
        dev = (user_c - garm_c) / user_c

        # Stretch relieves only positive deviation (tightness).
        # A 25% stretch fabric absorbs 25% of the tight-side deviation.
        if dev > 0:
            effective = dev * (1.0 - stretch)
        else:
            effective = dev

        verdict = _verdict(effective)

        sections.append(
            SectionFit(
                section=sec,
                user_cm=round(user_c, 1),
                garment_cm=round(garm_c, 1),
                ease_cm=round(float(ease_cm.get(sec, 0.0)), 1),
                stretch=round(stretch, 3),
                deviation=round(dev, 4),
                effective_deviation=round(effective, 4),
                verdict=verdict,
            )
        )

        # Asymmetric penalty by drape class
        asym = tight_w if effective > 0 else loose_w
        w = SECTION_WEIGHTS[sec] * asym
        weighted_sq_dev += w * effective * effective
        total_weight += SECTION_WEIGHTS[sec]  # normalize by base weight only

    if total_weight == 0:
        return FitResult(
            score=0.5,
            sections=[],
            drape_class=drape_class,
            overall_verdict="No measurable sections",
        )

    rms = math.sqrt(weighted_sq_dev / total_weight)
    score = math.exp(-((rms / SIGMA) ** 2))

    return FitResult(
        score=round(score, 4),
        sections=sections,
        drape_class=drape_class,
        overall_verdict=_overall_verdict(sections),
    )


def fit_to_dict(result: FitResult) -> dict[str, Any]:
    """Serialize FitResult for JSON response (used by explanation DAG)."""
    return {
        "score": result.score,
        "drape_class": result.drape_class,
        "overall_verdict": result.overall_verdict,
        "sections": [
            {
                "section": s.section,
                "user_cm": s.user_cm,
                "garment_cm": s.garment_cm,
                "ease_cm": s.ease_cm,
                "stretch": s.stretch,
                "deviation": s.deviation,
                "effective_deviation": s.effective_deviation,
                "verdict": s.verdict,
            }
            for s in result.sections
        ],
    }


if __name__ == "__main__":
    # Smoke tests — user matching size M band midpoint (bust 91.5 from VN pivot)
    user = {"shoulder": 38, "bust": 91.5, "waist": 72}

    test_cases = [
        # (description, garment_chart, ease, stretch, drape)
        ("True-to-size semi-fitted M",     {"shoulder": 38, "bust": 90, "waist": 72},  {"bust": 6, "waist": 6}, 0.05, "semi_fitted"),
        ("Brand runs small M",             {"shoulder": 36, "bust": 86, "waist": 68},  {"bust": 4},             0.05, "semi_fitted"),
        ("Brand runs large M",             {"shoulder": 40, "bust": 96, "waist": 80},  {"bust": 8},             0.05, "semi_fitted"),
        ("Bodycon true-to-size",           {"shoulder": 38, "bust": 91, "waist": 72},  {},                      0.20, "bodycon"),
        ("Bodycon runs large (bad)",       {"shoulder": 40, "bust": 96, "waist": 80},  {},                      0.10, "bodycon"),
        ("Oversized M (intentional)",      {"shoulder": 44, "bust": 108, "waist": 106}, {"bust": 16},           0.0,  "oversized"),
        ("Oversized but tight (bad)",      {"shoulder": 36, "bust": 86, "waist": 70},  {"bust": 16},            0.0,  "oversized"),
        ("Stretchy fitted, slightly small",{"shoulder": 36, "bust": 86, "waist": 70},  {},                      0.25, "fitted"),
    ]

    for desc, garm, ease, stretch, drape in test_cases:
        r = compute_fit(user, garm, ease, stretch, drape)
        print(f"\n{desc}: score={r.score} drape={r.drape_class}")
        print(f"  {r.overall_verdict}")
        for s in r.sections:
            print(f"  {s.section:8s}: user={s.user_cm:5.1f} chart={s.garment_cm:5.1f} "
                  f"dev={s.deviation:+.3f} eff={s.effective_deviation:+.3f} -> {s.verdict}")
