"""
Build Vietnamese population body prior for EA-VTON.

Uses published data from Tran Thi Minh Kieu (2024):
  "Analysis of Vietnamese Women's Body Shape from Anthropometric Data"
  480 women, 18-50 years, 3 regions, 108 measurements, 5 body clusters.

Plus national survey data (General Statistics Office, 2019-2020):
  22,000 subjects, validated by US CDC.

Outputs:
  - vn_female_prior.json: 5 cluster centroids + population stats
  - vn_male_prior.json: 2 cluster centroids from IUH study (1,106 men)
  - vn_population_stats.json: national-level height/weight/BMI distributions

These priors are used for:
  1. Bayesian size recommendation (population shift from RTR)
  2. SMPL body generation (sampling Vietnamese-proportioned bodies)
  3. Validation targets (is our synthetic data realistic?)
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Vietnamese Female Body Types — Tran Thi Minh Kieu (2024)
# Source: Fibres and Textiles, Vol. 2024, Issue 1
# DOI via TUL DSpace: dspace.tul.cz/items/00343604-79a4-4307-bbb4-8ebb3f1f002d
#
# NOTE: Exact measurement means per cluster are in the full paper.
# Values below are estimated from published descriptors + national stats.
# Replace with exact values once full paper PDF is obtained.
# =============================================================================

VN_FEMALE_PRIOR = {
    "source": "Tran Thi Minh Kieu et al., Fibres and Textiles, 2024",
    "sample_size": 480,
    "age_range": [18, 50],
    "regions": ["North", "Central", "South"],
    "n_measurements": 108,
    "method": "3D body scanning + PCA + K-means + discriminant analysis",
    "clusters": [
        {
            "id": 1,
            "label": "short_thin",
            "description": "Short, thin, small-shouldered; medium hip height; bust-waist > hip-waist ratio",
            "proportion": 0.1523,
            "n_subjects": 73,
            "estimated_stats": {
                "height_cm": {"mean": 151.0, "std": 3.5},
                "weight_kg": {"mean": 46.0, "std": 4.0},
                "bmi": {"mean": 20.2, "std": 1.5},
                "bust_cm": {"mean": 79.0, "std": 3.0},
                "waist_cm": {"mean": 65.0, "std": 3.5},
                "hip_cm": {"mean": 86.0, "std": 3.0},
                "shoulder_width_cm": {"mean": 35.0, "std": 1.5},
            },
        },
        {
            "id": 2,
            "label": "tall_slightly_fat",
            "description": "Tall, slightly fat, large-shouldered; high stature/hip height; bust-waist < hip-waist ratio",
            "proportion": 0.1836,
            "n_subjects": 88,
            "estimated_stats": {
                "height_cm": {"mean": 161.0, "std": 3.5},
                "weight_kg": {"mean": 58.0, "std": 5.0},
                "bmi": {"mean": 22.4, "std": 1.8},
                "bust_cm": {"mean": 86.0, "std": 4.0},
                "waist_cm": {"mean": 73.0, "std": 4.0},
                "hip_cm": {"mean": 93.0, "std": 4.0},
                "shoulder_width_cm": {"mean": 38.5, "std": 1.5},
            },
        },
        {
            "id": 3,
            "label": "medium",
            "description": "Medium body type (dominant); average height, fit body; bust-waist ≈ waist-to-hip ratio",
            "proportion": 0.3594,
            "n_subjects": 173,
            "estimated_stats": {
                "height_cm": {"mean": 156.0, "std": 4.0},
                "weight_kg": {"mean": 52.0, "std": 4.5},
                "bmi": {"mean": 21.4, "std": 1.6},
                "bust_cm": {"mean": 83.0, "std": 3.5},
                "waist_cm": {"mean": 69.0, "std": 3.5},
                "hip_cm": {"mean": 89.0, "std": 3.5},
                "shoulder_width_cm": {"mean": 36.5, "std": 1.5},
            },
        },
        {
            "id": 4,
            "label": "short_fat",
            "description": "Short, fat, medium-shoulder; low hip height; bust-waist > hip-waist ratio",
            "proportion": 0.2188,
            "n_subjects": 105,
            "estimated_stats": {
                "height_cm": {"mean": 152.0, "std": 3.5},
                "weight_kg": {"mean": 58.0, "std": 5.5},
                "bmi": {"mean": 25.1, "std": 2.0},
                "bust_cm": {"mean": 90.0, "std": 4.5},
                "waist_cm": {"mean": 78.0, "std": 5.0},
                "hip_cm": {"mean": 93.0, "std": 4.0},
                "shoulder_width_cm": {"mean": 36.5, "std": 1.5},
            },
        },
        {
            "id": 5,
            "label": "obese_average_height",
            "description": "Too fat, average height, big shoulders; low hip height; bust-waist > hip-waist ratio",
            "proportion": 0.0859,
            "n_subjects": 41,
            "estimated_stats": {
                "height_cm": {"mean": 155.0, "std": 4.0},
                "weight_kg": {"mean": 68.0, "std": 7.0},
                "bmi": {"mean": 28.3, "std": 2.5},
                "bust_cm": {"mean": 97.0, "std": 5.0},
                "waist_cm": {"mean": 86.0, "std": 5.5},
                "hip_cm": {"mean": 100.0, "std": 4.5},
                "shoulder_width_cm": {"mean": 38.0, "std": 1.5},
            },
        },
    ],
    "population_weighted_mean": {
        "height_cm": 155.5,
        "weight_kg": 53.9,
        "bmi": 22.3,
    },
}

# =============================================================================
# Vietnamese Male Body Types — IUH study (2022)
# Source: Journal of Science and Technology - IUH, Vol. 58, No. 04
# DOI: 10.46242/jstiuh.v58i04.4583
# 1,106 men, 18-60 years, 34 measurements via Size Stream scanner
# =============================================================================

VN_MALE_PRIOR = {
    "source": "Tran Thi Minh Kieu et al., JST-IUH, 2022",
    "sample_size": 1106,
    "age_range": [18, 60],
    "n_measurements": 34,
    "method": "Size Stream 3D scanner + factor analysis + K-means",
    "clusters": [
        {
            "id": 1,
            "label": "thin_short_upper",
            "description": "Thin build, shorter upper body, longer arms",
            "proportion": 0.604,
            "n_subjects": 668,
            "estimated_stats": {
                "height_cm": {"mean": 166.0, "std": 5.0},
                "weight_kg": {"mean": 60.0, "std": 7.0},
                "bmi": {"mean": 21.8, "std": 2.2},
                "chest_cm": {"mean": 87.0, "std": 4.5},
                "waist_cm": {"mean": 74.0, "std": 5.0},
                "hip_cm": {"mean": 90.0, "std": 4.0},
            },
        },
        {
            "id": 2,
            "label": "heavy_tall",
            "description": "Taller/heavier, shorter limbs, longer torso, broader shoulders",
            "proportion": 0.396,
            "n_subjects": 438,
            "estimated_stats": {
                "height_cm": {"mean": 170.0, "std": 5.5},
                "weight_kg": {"mean": 72.0, "std": 9.0},
                "bmi": {"mean": 24.9, "std": 2.8},
                "chest_cm": {"mean": 95.0, "std": 5.0},
                "waist_cm": {"mean": 83.0, "std": 6.0},
                "hip_cm": {"mean": 96.0, "std": 4.5},
            },
        },
    ],
    "population_weighted_mean": {
        "height_cm": 167.6,
        "weight_kg": 64.8,
        "bmi": 23.0,
    },
}

# =============================================================================
# National population statistics
# Source: General Statistics Office + National Institute of Nutrition, 2019-2020
# Validated by US CDC. Sample: 22,000 individuals.
# =============================================================================

VN_POPULATION_STATS = {
    "source": "General Statistics Office & National Institute of Nutrition, 2019-2020",
    "sample_size": 22000,
    "validated_by": "US CDC",
    "female": {
        "height_cm": {"mean": 156.2, "std": 5.5, "trend_per_decade": 2.6},
        "weight_kg": {"mean": 53.9, "std": 8.0},
        "overweight_pct": 0.19,
        "urban_height_bonus_cm": 2.0,
    },
    "male": {
        "height_cm": {"mean": 168.1, "std": 6.0, "trend_per_decade": 3.7},
        "weight_kg": {"mean": 63.0, "std": 9.5},
        "overweight_pct": 0.19,
        "urban_height_bonus_cm": 2.0,
    },
    "size_standard": {
        "name": "TCVN 5782:2009",
        "body_type_codes": {"A": "slim", "B": "average", "C": "overweight"},
        "vs_us_offset": "Vietnamese sizes run 1-2 sizes smaller than US",
    },
    # Comparison populations for context
    "comparison": {
        "korean_female": {"height_cm": 160.2, "weight_kg": 57.2},
        "japanese_female": {"height_cm": 158.0, "weight_kg": 53.0},
        "us_female": {"height_cm": 162.6, "weight_kg": 77.5},
        "rtr_dataset_female": {"height_cm": 165.9, "weight_kg": 62.3},
        "bodym_dataset_female": {"height_cm": 163.2, "weight_kg": 66.3},
    },
}


def main():
    # Save priors
    for name, data in [
        ("vn_female_prior", VN_FEMALE_PRIOR),
        ("vn_male_prior", VN_MALE_PRIOR),
        ("vn_population_stats", VN_POPULATION_STATS),
    ]:
        path = OUT_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {path.name}")

    # Print summary
    print("\n=== Vietnamese Female Body Types ===")
    for c in VN_FEMALE_PRIOR["clusters"]:
        s = c["estimated_stats"]
        print(
            f"  Cluster {c['id']} ({c['label']:25s}): "
            f"{c['proportion']*100:5.1f}% | "
            f"H={s['height_cm']['mean']:.0f}cm "
            f"W={s['weight_kg']['mean']:.0f}kg "
            f"BMI={s['bmi']['mean']:.1f} "
            f"B/W/H={s['bust_cm']['mean']:.0f}/{s['waist_cm']['mean']:.0f}/{s['hip_cm']['mean']:.0f}"
        )

    print("\n=== Vietnamese Male Body Types ===")
    for c in VN_MALE_PRIOR["clusters"]:
        s = c["estimated_stats"]
        print(
            f"  Cluster {c['id']} ({c['label']:25s}): "
            f"{c['proportion']*100:5.1f}% | "
            f"H={s['height_cm']['mean']:.0f}cm "
            f"W={s['weight_kg']['mean']:.0f}kg "
            f"BMI={s['bmi']['mean']:.1f} "
            f"C/W/H={s['chest_cm']['mean']:.0f}/{s['waist_cm']['mean']:.0f}/{s['hip_cm']['mean']:.0f}"
        )

    print("\n=== Population Height Gap (vs RTR training data) ===")
    rtr_h = VN_POPULATION_STATS["comparison"]["rtr_dataset_female"]["height_cm"]
    vn_h = VN_POPULATION_STATS["female"]["height_cm"]["mean"]
    print(f"  RTR mean female height:        {rtr_h:.1f} cm")
    print(f"  Vietnamese mean female height:  {vn_h:.1f} cm")
    print(f"  Gap:                           {rtr_h - vn_h:.1f} cm ({(rtr_h - vn_h)/rtr_h*100:.1f}%)")
    print(f"  → Model trained on RTR will systematically oversize for Vietnamese users")


if __name__ == "__main__":
    main()
