"""
Adapt RTR dataset for Vietnamese population using distribution shift.

Problem: RTR has 192K rows with fit feedback but NO ethnicity.
         Mean height is 165.9cm (US women), Vietnamese is 156.2cm.
         A model trained naively on RTR will oversize for Vietnamese users.

Solution: Importance-weighted sampling.
  - Compute P(body | Vietnamese) / P(body | RTR) for each RTR row
  - Rows with body measurements close to Vietnamese distribution get higher weight
  - This simulates "what if RTR had Vietnamese body proportions?"

Output:
  - rtr_vn_adapted.parquet: RTR rows with importance weights for VN population
  - rtr_vn_sampled.parquet: Resampled subset (~20K rows) matching VN distributions
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROCESSED_DIR = Path(__file__).parent.parent / "processed"


def gaussian_pdf(x, mean, std):
    """Univariate Gaussian PDF."""
    return stats.norm.pdf(x, loc=mean, scale=std)


def compute_importance_weights(df, vn_stats, rtr_stats):
    """
    Compute importance weight = P(body | Vietnamese) / P(body | RTR)
    for each row, using height and weight as the key features.
    """
    weights = np.ones(len(df))

    # Height weight
    mask_h = df["height_cm"].notna()
    if mask_h.any():
        h = df.loc[mask_h, "height_cm"].values
        p_vn_h = gaussian_pdf(h, vn_stats["height_cm"]["mean"], vn_stats["height_cm"]["std"])
        p_rtr_h = gaussian_pdf(h, rtr_stats["height_cm"]["mean"], rtr_stats["height_cm"]["std"])
        # Clip to avoid extreme weights
        ratio_h = np.clip(p_vn_h / (p_rtr_h + 1e-10), 0.01, 100)
        weights[mask_h] *= ratio_h

    # Weight weight (heh)
    mask_w = df["weight_kg"].notna()
    if mask_w.any():
        w = df.loc[mask_w, "weight_kg"].values
        p_vn_w = gaussian_pdf(w, vn_stats["weight_kg"]["mean"], vn_stats["weight_kg"]["std"])
        p_rtr_w = gaussian_pdf(w, rtr_stats["weight_kg"]["mean"], rtr_stats["weight_kg"]["std"])
        ratio_w = np.clip(p_vn_w / (p_rtr_w + 1e-10), 0.01, 100)
        weights[mask_w] *= ratio_w

    # BMI weight
    mask_b = df["bmi"].notna()
    if mask_b.any():
        b = df.loc[mask_b, "bmi"].values
        # Vietnamese BMI: mean ~21.4 (from cluster 3, dominant group)
        p_vn_b = gaussian_pdf(b, 21.4, 2.5)
        # RTR BMI: computed from their height/weight
        p_rtr_b = gaussian_pdf(b, rtr_stats["bmi"]["mean"], rtr_stats["bmi"]["std"])
        ratio_b = np.clip(p_vn_b / (p_rtr_b + 1e-10), 0.01, 100)
        weights[mask_b] *= ratio_b

    # Normalize to mean=1
    weights = weights / weights.mean()
    return weights


def main():
    # Load processed RTR
    rtr_path = PROCESSED_DIR / "rtr_body_fit.parquet"
    print(f"Loading {rtr_path.name}...")
    df = pd.read_parquet(rtr_path)
    print(f"  Rows: {len(df):,}")

    # Load Vietnamese prior
    vn_path = PROCESSED_DIR / "vn_female_prior.json"
    with open(vn_path) as f:
        vn_prior = json.load(f)

    # Vietnamese target stats (population-level)
    vn_stats = {
        "height_cm": {"mean": 156.2, "std": 5.5},
        "weight_kg": {"mean": 53.9, "std": 8.0},
    }

    # RTR source stats (computed from data)
    rtr_stats = {
        "height_cm": {"mean": df["height_cm"].mean(), "std": df["height_cm"].std()},
        "weight_kg": {"mean": df["weight_kg"].mean(), "std": df["weight_kg"].std()},
        "bmi": {"mean": df["bmi"].mean(), "std": df["bmi"].std()},
    }

    print(f"\n  RTR distribution:  H={rtr_stats['height_cm']['mean']:.1f}±{rtr_stats['height_cm']['std']:.1f}cm, "
          f"W={rtr_stats['weight_kg']['mean']:.1f}±{rtr_stats['weight_kg']['std']:.1f}kg")
    print(f"  VN target:         H={vn_stats['height_cm']['mean']:.1f}±{vn_stats['height_cm']['std']:.1f}cm, "
          f"W={vn_stats['weight_kg']['mean']:.1f}±{vn_stats['weight_kg']['std']:.1f}kg")

    # Compute importance weights
    print("\nComputing importance weights...")
    df["vn_weight"] = compute_importance_weights(df, vn_stats, rtr_stats)

    # Stats on weights
    print(f"  Weight range: {df['vn_weight'].min():.4f} – {df['vn_weight'].max():.4f}")
    print(f"  Weight mean:  {df['vn_weight'].mean():.4f}")
    print(f"  Weight median: {df['vn_weight'].median():.4f}")
    print(f"  Effective sample size: {(df['vn_weight'].sum()**2 / (df['vn_weight']**2).sum()):.0f} "
          f"(of {len(df):,})")

    # Save full dataset with weights
    adapted_path = PROCESSED_DIR / "rtr_vn_adapted.parquet"
    df.to_parquet(adapted_path, index=False)
    print(f"\n  Saved: {adapted_path.name} ({adapted_path.stat().st_size / 1e6:.1f} MB)")

    # Create resampled subset matching VN distribution
    print("\nResampling to match Vietnamese distribution...")
    n_resample = 20000

    # Normalize weights to probabilities
    probs = df["vn_weight"].values / df["vn_weight"].sum()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=n_resample, replace=True, p=probs)
    df_resampled = df.iloc[idx].copy().reset_index(drop=True)

    # Verify resampled distribution
    print(f"\n  Resampled distribution (n={n_resample:,}):")
    print(f"    Height: {df_resampled['height_cm'].mean():.1f}±{df_resampled['height_cm'].std():.1f} cm "
          f"(target: {vn_stats['height_cm']['mean']}±{vn_stats['height_cm']['std']})")
    print(f"    Weight: {df_resampled['weight_kg'].mean():.1f}±{df_resampled['weight_kg'].std():.1f} kg "
          f"(target: {vn_stats['weight_kg']['mean']}±{vn_stats['weight_kg']['std']})")
    print(f"    BMI:    {df_resampled['bmi'].mean():.1f}±{df_resampled['bmi'].std():.1f}")

    print(f"\n  Fit distribution (resampled):")
    print(df_resampled["fit"].value_counts().to_string(header=False))

    print(f"\n  Size distribution shift:")
    for size in sorted(df["size"].dropna().unique()):
        orig_pct = (df["size"] == size).mean() * 100
        resamp_pct = (df_resampled["size"] == size).mean() * 100
        bar = "█" * int(resamp_pct * 2)
        print(f"    Size {size:>4}: {orig_pct:5.1f}% → {resamp_pct:5.1f}% {bar}")

    sampled_path = PROCESSED_DIR / "rtr_vn_sampled.parquet"
    df_resampled.to_parquet(sampled_path, index=False)
    print(f"\n  Saved: {sampled_path.name} ({sampled_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
