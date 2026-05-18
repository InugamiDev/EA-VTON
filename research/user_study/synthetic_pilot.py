"""Generate a synthetic N=30 dataset to smoke-test analysis.py.

Distributions calibrated so the analysis script's output is well-defined:
  - 60% of participants prefer calibrated (mean diff = +0.5 likert)
  - 25% prefer baseline (mean diff = -0.3 likert)
  - 15% indifferent

Run this once to validate analysis.py runs end-to-end. DELETE the synthetic
data before running on real participants. Real data drops into the same path.
"""

# intent: smoke-test the analysis script before real data arrives
# status: done
# next: delete synthetic_results.csv before real data collection begins
# confidence: high

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/study_results.csv"
N = 30
RNG = np.random.default_rng(42)


def main() -> None:
    rows = []
    for pid in range(1, N + 1):
        order = RNG.choice(["AB", "BA"])
        # 60% calibration-favoring, 25% baseline-favoring, 15% indifferent
        archetype = RNG.choice(["pro_cal", "pro_bas", "indiff"], p=[0.60, 0.25, 0.15])
        true_base = float(np.clip(RNG.normal(3.4, 0.7), 1, 5))
        if archetype == "pro_cal":
            cal = float(np.clip(true_base + RNG.normal(0.5, 0.5), 1, 5))
        elif archetype == "pro_bas":
            cal = float(np.clip(true_base + RNG.normal(-0.3, 0.5), 1, 5))
        else:
            cal = float(np.clip(true_base + RNG.normal(0.0, 0.3), 1, 5))

        cal_label = RNG.choice(["A", "B"])
        if cal_label == "A":
            c1, c2 = round(cal), round(true_base)
        else:
            c1, c2 = round(true_base), round(cal)

        # Forced choice — biased by true_diff but noisy
        true_diff = cal - true_base
        choice_probs = {"calibrated": 0.5 + 0.15 * true_diff,
                        "baseline":   0.5 - 0.15 * true_diff,
                        "neither":    0.10}
        # renormalize and clip
        total = sum(max(v, 0) for v in choice_probs.values())
        choice = RNG.choice(
            [cal_label, "B" if cal_label == "A" else "A", "neither"],
            p=[max(choice_probs["calibrated"], 0)/total,
               max(choice_probs["baseline"], 0)/total,
               max(choice_probs["neither"], 0)/total],
        )

        rows.append({
            "participant_id":         pid,
            "age_band":               RNG.choice(["18-24","25-34","35-45"], p=[0.4,0.4,0.2]),
            "height_cm":              round(float(np.clip(RNG.normal(157, 5.5), 145, 175)), 1),
            "body_cluster_self":      int(RNG.choice([1,2,3,4,5], p=[0.15,0.10,0.50,0.15,0.10])),
            "order_assignment":       order,
            "condition_1_likert":     int(c1),
            "condition_1_comment":    "ok",
            "condition_2_likert":     int(c2),
            "condition_2_comment":    "ok",
            "condition_calibrated":   cal_label,
            "forced_choice":          choice,
            "forced_choice_reason":   "felt more like my style" if choice == cal_label else "easier to wear",
            "seeds_picked_ids":       "deepfashion2:000001:0;deepfashion2:000010:1;deepfashion2:000100:2",
            "time_in_seconds":        int(RNG.normal(900, 200)),
            "device":                 RNG.choice(["smartphone","laptop"], p=[0.8,0.2]),
            "dropout":                False,
        })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUT, index=False)
    print(f"  wrote synthetic N={N} dataset → {OUT}")
    print(f"  archetypes: 60% pro-cal, 25% pro-bas, 15% indifferent")
    print(f"  → DELETE {OUT} before real data collection.")


if __name__ == "__main__":
    main()
