"""Population-specific anthropometric calibration for size and style ranking.

The visual flow has no user-entered weight, so it needs an explicit population
prior instead of hidden constants inside an endpoint. US women use a fashion-fit
BMI prior from the Rent the Runway cohort already documented in this repo, while
the civilian measurement anchors come from NHANES body-measures files.
"""

# intent: centralize population priors used by visual sizing and upper-body fit
# status: done
# next: replace formula coefficients with a trained measurement model when labeled data exists
# blockers: public NHANES does not include chest/bust circumference
# confidence: medium

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Population = Literal["us_women", "universal", "vietnamese"]
CanonicalPopulation = Literal["us_women", "vietnamese"]
VisualSource = Literal["world_3d", "image_2d"]


@dataclass(frozen=True)
class VisualRatioCalibration:
    shoulder_to_hip: float
    waist_to_hip: float
    shoulder_weight_kg: float
    waist_weight_kg: float


@dataclass(frozen=True)
class MeasurementFormula:
    height_coef: float
    weight_coef: float
    intercept: float

    def estimate(self, height_cm: float, weight_kg: float) -> float:
        return height_cm * self.height_coef + weight_kg * self.weight_coef + self.intercept


@dataclass(frozen=True)
class PopulationProfile:
    key: CanonicalPopulation
    label: str
    visual_base_bmi: float
    reference_height_cm: float
    reference_weight_kg: float
    reference_bmi: float
    chest: MeasurementFormula
    waist: MeasurementFormula
    hip: MeasurementFormula
    world_3d: VisualRatioCalibration
    image_2d: VisualRatioCalibration
    size_bands_cm: dict[str, dict[str, tuple[float, float]]]


# Sources:
# - CDC NHANES 2017-2018 Body Measures data file (BMX_J):
#   https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.htm
# - CDC NHANES 2017-2018 Demographics data file (DEMO_J):
#   https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.htm
# - ANSUR II female public data, used only for shoulder/torso ratio geometry:
#   https://tools.openlab.psu.edu/publicData/ANSUR_II_FEMALE_Public.csv
# - Local RTR notes for fashion-fit prior: docs/07-datasets.md
US_WOMEN_SIZE_BANDS_CM: dict[str, dict[str, tuple[float, float]]] = {
    "XS": {"shoulder": (34.0, 36.0), "bust": (80.0, 85.0), "waist": (62.0, 67.0)},
    "S": {"shoulder": (35.5, 37.5), "bust": (86.0, 91.0), "waist": (68.0, 73.0)},
    "M": {"shoulder": (37.0, 39.5), "bust": (92.0, 98.0), "waist": (74.0, 80.0)},
    "L": {"shoulder": (39.0, 42.0), "bust": (99.0, 106.0), "waist": (81.0, 89.0)},
    "XL": {"shoulder": (41.5, 44.5), "bust": (107.0, 116.0), "waist": (90.0, 100.0)},
    "XXL": {"shoulder": (44.0, 47.0), "bust": (117.0, 126.0), "waist": (101.0, 112.0)},
}


VN_SIZE_BANDS_CM: dict[str, dict[str, tuple[float, float]]] = {
    "XS": {"shoulder": (33.0, 35.0), "bust": (80.0, 83.0), "waist": (61.0, 64.0)},
    "S": {"shoulder": (35.0, 37.0), "bust": (84.0, 87.0), "waist": (66.0, 69.0)},
    "M": {"shoulder": (37.0, 39.0), "bust": (90.0, 93.0), "waist": (71.0, 74.0)},
    "L": {"shoulder": (39.0, 41.0), "bust": (96.0, 105.0), "waist": (76.0, 86.0)},
    "XL": {"shoulder": (41.0, 43.0), "bust": (110.0, 115.0), "waist": (91.0, 96.0)},
    "XXL": {"shoulder": (43.0, 45.0), "bust": (120.0, 125.0), "waist": (101.0, 106.0)},
}


_US_WOMEN_PROFILE = PopulationProfile(
    key="us_women",
    label="US women",
    visual_base_bmi=22.7,
    reference_height_cm=165.9,
    reference_weight_kg=62.3,
    reference_bmi=22.7,
    chest=MeasurementFormula(height_coef=0.24, weight_coef=0.42, intercept=24.0),
    waist=MeasurementFormula(height_coef=0.20, weight_coef=0.50, intercept=10.0),
    hip=MeasurementFormula(height_coef=0.35, weight_coef=0.55, intercept=6.0),
    world_3d=VisualRatioCalibration(
        shoulder_to_hip=1.04,
        waist_to_hip=0.84,
        shoulder_weight_kg=18.0,
        waist_weight_kg=36.0,
    ),
    image_2d=VisualRatioCalibration(
        shoulder_to_hip=1.50,
        waist_to_hip=0.95,
        shoulder_weight_kg=12.0,
        waist_weight_kg=22.0,
    ),
    size_bands_cm=US_WOMEN_SIZE_BANDS_CM,
)


_VN_PROFILE = PopulationProfile(
    key="vietnamese",
    label="Vietnamese women",
    visual_base_bmi=21.4,
    reference_height_cm=156.0,
    reference_weight_kg=52.0,
    reference_bmi=21.4,
    chest=MeasurementFormula(height_coef=0.485, weight_coef=0.22, intercept=-2.5),
    waist=MeasurementFormula(height_coef=0.30, weight_coef=0.38, intercept=-4.0),
    hip=MeasurementFormula(height_coef=0.42, weight_coef=0.24, intercept=3.0),
    world_3d=VisualRatioCalibration(
        shoulder_to_hip=1.08,
        waist_to_hip=0.78,
        shoulder_weight_kg=20.0,
        waist_weight_kg=25.0,
    ),
    image_2d=VisualRatioCalibration(
        shoulder_to_hip=1.50,
        waist_to_hip=0.95,
        shoulder_weight_kg=12.0,
        waist_weight_kg=15.0,
    ),
    size_bands_cm=VN_SIZE_BANDS_CM,
)


POPULATION_PROFILES: dict[Population, PopulationProfile] = {
    "us_women": _US_WOMEN_PROFILE,
    "universal": _US_WOMEN_PROFILE,
    "vietnamese": _VN_PROFILE,
}


def canonical_population(population: str | None) -> CanonicalPopulation:
    if population == "vietnamese":
        return "vietnamese"
    return "us_women"


def get_population_profile(population: str | None) -> PopulationProfile:
    return POPULATION_PROFILES.get(population or "us_women", _US_WOMEN_PROFILE)


def size_band_midpoints(size: str, population: str | None = "us_women") -> dict[str, float]:
    profile = get_population_profile(population)
    bands = profile.size_bands_cm.get(size.upper()) or profile.size_bands_cm["M"]
    return {section: (lo + hi) / 2.0 for section, (lo, hi) in bands.items()}


def size_band_midpoint(
    size: str,
    section: str,
    fallback: float,
    population: str | None = "us_women",
) -> float:
    profile = get_population_profile(population)
    band = profile.size_bands_cm.get(size.upper(), {}).get(section)
    if band is None:
        return fallback
    lo, hi = band
    return (lo + hi) / 2.0


def estimate_visual_weight_kg(
    height_cm: float,
    shoulder_to_hip: float,
    waist_to_hip: float,
    source: VisualSource,
    population: str | None,
) -> float:
    profile = get_population_profile(population)
    calibration = profile.world_3d if source == "world_3d" else profile.image_2d
    height_m = height_cm / 100.0
    estimate = profile.visual_base_bmi * height_m * height_m

    # Chest-up webcam frames can produce unstable hip/waist landmarks. Treat visual
    # ratios as small corrections around the population prior, not full-body weight.
    max_shoulder_delta = 0.35 if source == "world_3d" else 0.45
    max_waist_delta = 0.22 if source == "world_3d" else 0.25
    shoulder_delta = max(-max_shoulder_delta, min(max_shoulder_delta, shoulder_to_hip - calibration.shoulder_to_hip))
    waist_delta = max(-max_waist_delta, min(max_waist_delta, waist_to_hip - calibration.waist_to_hip))

    estimate += shoulder_delta * calibration.shoulder_weight_kg
    estimate += waist_delta * calibration.waist_weight_kg

    prior = profile.visual_base_bmi * height_m * height_m
    return round(max(prior - 16.0, min(prior + 18.0, estimate)), 1)
