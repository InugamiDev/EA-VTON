"""Pre-registered analysis for the PRISM-UB calibration user study.

Reads `data/study_results.csv` (columns per PROTOCOL.md §5) and produces:
  - RESULTS.md with the pre-registered tests and confidence intervals
  - data/summary_stats.json

PRE-REGISTERED TESTS (committed in advance):
    H1 — Calibrated likert > baseline likert: paired t-test, one-sided, α=0.05
    H3 — Proportion choosing calibrated > 0.5: one-proportion z-test, one-sided, α=0.05

Exploratory analyses are reported separately, clearly labelled.

Usage:
    .venv/bin/python3 research/user_study/analysis.py
"""

# intent: produce the user-study results table directly from pre-registered tests
# status: implementation complete; awaits real data
# next: run after pilot (~3 participants) to validate format, then full N=30
# confidence: high

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/study_results.csv"
SUMMARY = ROOT / "data/summary_stats.json"
RESULTS = ROOT / "RESULTS.md"


def cohens_dz(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's dz for paired samples."""
    diffs = x - y
    return float(diffs.mean() / (diffs.std(ddof=1) + 1e-12))


def main() -> None:
    if not DATA.exists():
        sys.exit(
            f"!! {DATA} not found.\n"
            f"   To run analysis on real data, populate {DATA} with columns "
            f"matching PROTOCOL.md §5. To verify the script works, create "
            f"a synthetic dataset first."
        )

    df = pd.read_csv(DATA)
    n_total = len(df)
    print(f"  loaded {n_total} rows from {DATA.name}")

    # Pre-registered exclusions
    excl_dropout = df["dropout"] == True  # noqa: E712
    excl_engage = (
        (df["condition_1_likert"] == df["condition_2_likert"])
        & (df["forced_choice"] == "neither")
        & (df["forced_choice_reason"].fillna("").str.strip().str.len() < 5)
    )
    excluded = excl_dropout | excl_engage
    n_excluded = int(excluded.sum())
    df_a = df[~excluded].copy()
    n_analysis = len(df_a)
    print(f"  pre-registered exclusions: {n_excluded} ({n_excluded/n_total*100:.1f}%)")
    print(f"  analysis sample: {n_analysis}")

    # Recover the calibrated / baseline likert columns per participant
    is_a_cal = df_a["condition_calibrated"] == "A"
    cal_likert = np.where(is_a_cal,
                          df_a["condition_1_likert"],
                          df_a["condition_2_likert"]).astype(float)
    bas_likert = np.where(is_a_cal,
                          df_a["condition_2_likert"],
                          df_a["condition_1_likert"]).astype(float)
    diffs = cal_likert - bas_likert

    # ── H1: paired t-test, one-sided ──
    t_stat, p_two_sided = stats.ttest_rel(cal_likert, bas_likert)
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    diff_mean = float(diffs.mean())
    diff_se = float(diffs.std(ddof=1) / np.sqrt(n_analysis))
    ci_low = diff_mean - 1.96 * diff_se
    ci_high = diff_mean + 1.96 * diff_se
    dz = cohens_dz(cal_likert, bas_likert)

    h1 = {
        "test": "paired t-test, one-sided (calibrated > baseline)",
        "n": n_analysis,
        "mean_diff_likert": diff_mean,
        "ci95": [ci_low, ci_high],
        "t_statistic": float(t_stat),
        "df": n_analysis - 1,
        "p_one_sided": float(p_one_sided),
        "cohens_dz": dz,
        "result": "accept H1" if p_one_sided < 0.05 and diff_mean > 0 else "fail to reject H0",
    }

    # ── H3: one-proportion z-test, one-sided ──
    cal_choice = (df_a["forced_choice"] == df_a["condition_calibrated"])
    bas_choice = (df_a["forced_choice"] != df_a["condition_calibrated"]) & \
                 (df_a["forced_choice"] != "neither")
    n_decisive = int(cal_choice.sum() + bas_choice.sum())
    n_cal = int(cal_choice.sum())
    if n_decisive > 0:
        p_hat = n_cal / n_decisive
        se = np.sqrt(0.5 * 0.5 / n_decisive)
        z = (p_hat - 0.5) / se
        p_one = float(1 - stats.norm.cdf(z))
    else:
        p_hat, p_one = None, None

    h3 = {
        "test": "one-proportion z-test, one-sided (P(choose calibrated) > 0.5)",
        "n_decisive": n_decisive,
        "n_chose_calibrated": n_cal,
        "p_hat": p_hat,
        "z": float(z) if n_decisive > 0 else None,
        "p_one_sided": p_one,
        "result": "accept H3" if (p_one is not None and p_one < 0.05) else "fail to reject H0",
    }

    # ── Exploratory: subgroup by body cluster ──
    by_cluster: dict[int, dict] = {}
    for cl, g in df_a.groupby("body_cluster_self"):
        if len(g) < 3:
            continue
        is_a_cal_g = g["condition_calibrated"] == "A"
        cal_g = np.where(is_a_cal_g, g["condition_1_likert"], g["condition_2_likert"]).astype(float)
        bas_g = np.where(is_a_cal_g, g["condition_2_likert"], g["condition_1_likert"]).astype(float)
        by_cluster[int(cl)] = {
            "n": len(g),
            "mean_calibrated": float(np.mean(cal_g)),
            "mean_baseline": float(np.mean(bas_g)),
            "diff_mean": float(np.mean(cal_g - bas_g)),
        }

    # ── Order-effect exploratory ──
    by_order = {}
    for o, g in df_a.groupby("order_assignment"):
        if len(g) < 3:
            continue
        is_a_cal_g = g["condition_calibrated"] == "A"
        cal_g = np.where(is_a_cal_g, g["condition_1_likert"], g["condition_2_likert"]).astype(float)
        bas_g = np.where(is_a_cal_g, g["condition_2_likert"], g["condition_1_likert"]).astype(float)
        by_order[o] = {
            "n": len(g),
            "mean_diff": float(np.mean(cal_g - bas_g)),
        }

    summary = {
        "study": "PRISM-UB Calibration Effect (pre-registered)",
        "n_total": n_total,
        "n_excluded": n_excluded,
        "n_analysis": n_analysis,
        "H1_primary": h1,
        "H3_secondary": h3,
        "exploratory": {
            "by_body_cluster": by_cluster,
            "by_order_assignment": by_order,
        },
    }
    DATA.parent.mkdir(exist_ok=True, parents=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  wrote {SUMMARY}")

    # ── RESULTS.md ──
    md = []
    md.append(f"# User Study Results — PRISM-UB Calibration (N={n_analysis})\n")
    md.append(f"Auto-generated by `analysis.py` from `data/study_results.csv`.\n")
    md.append(f"\nPre-registration: see `PROTOCOL.md` §6.\n")
    md.append("\n## H1 (primary) — paired t-test\n")
    md.append(f"- N analysis = {n_analysis} (excluded {n_excluded})\n")
    md.append(f"- Mean difference (calibrated − baseline) = **{diff_mean:+.3f} Likert points** "
              f"(95% CI [{ci_low:+.3f}, {ci_high:+.3f}])\n")
    md.append(f"- t({n_analysis-1}) = {t_stat:+.3f}, one-sided p = **{p_one_sided:.4f}**\n")
    md.append(f"- Cohen's dz = {dz:+.3f}\n")
    md.append(f"- **Result:** {h1['result']}\n")
    md.append("\n## H3 (secondary) — forced-choice z-test\n")
    md.append(f"- Decisive votes: {n_cal} chose calibrated of {n_decisive} non-tied\n")
    if h3['p_hat'] is not None:
        md.append(f"- Sample proportion = {h3['p_hat']:.3f}, z = {h3['z']:.3f}, one-sided p = **{h3['p_one_sided']:.4f}**\n")
    md.append(f"- **Result:** {h3['result']}\n")
    md.append("\n## Exploratory analyses (NOT pre-registered)\n")
    md.append("\n### Per body-cluster\n")
    md.append("\n| Cluster | N | Mean calibrated | Mean baseline | Diff |\n|---|---:|---:|---:|---:|\n")
    for cl, v in by_cluster.items():
        md.append(f"| {cl} | {v['n']} | {v['mean_calibrated']:.2f} | "
                  f"{v['mean_baseline']:.2f} | {v['diff_mean']:+.2f} |\n")
    md.append("\n### Order effect\n")
    md.append("\n| Order | N | Mean diff |\n|---|---:|---:|\n")
    for o, v in by_order.items():
        md.append(f"| {o} | {v['n']} | {v['mean_diff']:+.2f} |\n")
    md.append("\n---\n*This report is auto-generated from raw data; do not edit by hand.*\n")
    RESULTS.write_text("".join(md))
    print(f"  wrote {RESULTS}")


if __name__ == "__main__":
    main()
