"""
Copula-based density ratio estimation with Pareto Smoothed Importance Sampling.

intent: replace independent Gaussian weighting with joint distribution modeling
status: done
next: feed weights into model training (train_size_corn.py)
confidence: high

Method:
  1. Fit bivariate Gaussian to US (RTR) data: P_US(height, weight)
  2. Construct bivariate Gaussian for VN target: P_VN(height, weight)
     - Marginals from Tran et al. 2024
     - Correlation assumed same as US (no raw VN data available)
  3. Density ratio: w(x) = P_VN(x) / P_US(x) per sample
  4. PSIS: fit GPD to tail weights, smooth extremes (Vehtari et al. JMLR 2024)
  5. Normalize to mean=1

References:
  - Shimodaira (2000) JSPI — covariate shift foundation
  - Vehtari et al. (2024) JMLR — Pareto Smoothed Importance Sampling
  - Tran et al. (2024) Fibres & Textiles — Vietnamese body clusters

Usage:
    from research.weighting.copula_psis import compute_weights
    weights, diagnostics = compute_weights(heights, weights_kg)
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal


ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "research" / "datasets" / "processed"


# ── Vietnamese population parameters (Tran et al. 2024) ──

VN_FEMALE = {
    "height_mean": 156.2,
    "height_std": 5.5,
    "weight_mean": 53.9,
    "weight_std": 8.0,
}


def _build_cov_matrix(std_h, std_w, rho):
    """Build 2x2 covariance matrix from marginal stds and correlation."""
    return np.array([
        [std_h ** 2, rho * std_h * std_w],
        [rho * std_h * std_w, std_w ** 2],
    ])


def _psis_smooth(log_weights, min_tail_samples=20):
    """
    Pareto Smoothed Importance Sampling (Vehtari et al. 2024).

    Fits a Generalized Pareto Distribution to the largest importance weights
    and replaces them with smoothed values. Returns smoothed log-weights
    and the shape parameter k_hat (diagnostic).

    k_hat interpretation:
      k < 0.5: good — finite variance, reliable estimates
      0.5 < k < 0.7: moderate — finite variance but slow convergence
      k > 0.7: bad — infinite variance, results unreliable
    """
    n = len(log_weights)
    # Number of tail samples: min(n/5, 3*sqrt(n)) per Vehtari et al.
    M = max(min_tail_samples, min(n // 5, int(3 * np.sqrt(n))))

    lw = log_weights.copy()
    sorted_indices = np.argsort(lw)
    # Tail = top M values
    tail_indices = sorted_indices[-M:]
    tail_cutoff = lw[sorted_indices[-(M + 1)]]  # value just below tail

    tail_values = lw[tail_indices] - tail_cutoff  # shift to start at 0

    # Fit GPD to tail via method of moments (Zhang & Stephens 2009 approx)
    # For simplicity, use scipy's GPD fit
    if np.all(tail_values == tail_values[0]):
        # All tail values identical — no smoothing needed
        return lw, 0.0

    try:
        # scipy genpareto: F(x) = 1 - (1 + c*x/scale)^(-1/c)
        # c = shape parameter (our k_hat)
        c, loc, scale = stats.genpareto.fit(tail_values, floc=0)
        k_hat = c
    except Exception:
        # Fit failed — return unsmoothed
        return lw, np.inf

    if k_hat < -0.5 or scale <= 0:
        # Bad fit — return unsmoothed
        return lw, k_hat

    # Replace tail values with expected order statistics from fitted GPD
    probs = (np.arange(1, M + 1) - 0.5) / M
    smoothed_tail = stats.genpareto.ppf(probs, c=k_hat, loc=0, scale=scale)
    smoothed_tail += tail_cutoff  # shift back

    # Sort tail indices to match sorted smoothed values
    tail_sorted_order = np.argsort(lw[tail_indices])
    lw[tail_indices[tail_sorted_order]] = np.sort(smoothed_tail)

    return lw, k_hat


def compute_weights_independent(heights, weights_kg):
    """
    Baseline: independent Gaussian density ratio (what we had before).
    Returns weights and diagnostics for comparison.
    """
    h, w = np.asarray(heights), np.asarray(weights_kg)

    us_h_mu, us_h_sig = h.mean(), h.std()
    us_w_mu, us_w_sig = w.mean(), w.std()

    vn_h_mu, vn_h_sig = VN_FEMALE["height_mean"], VN_FEMALE["height_std"]
    vn_w_mu, vn_w_sig = VN_FEMALE["weight_mean"], VN_FEMALE["weight_std"]

    w_h = stats.norm.pdf(h, vn_h_mu, vn_h_sig) / (stats.norm.pdf(h, us_h_mu, us_h_sig) + 1e-10)
    w_w = stats.norm.pdf(w, vn_w_mu, vn_w_sig) / (stats.norm.pdf(w, us_w_mu, us_w_sig) + 1e-10)

    raw_weights = w_h * w_w
    # Hard clipping (old method)
    clipped = np.clip(raw_weights, 0.01, 100)
    clipped = clipped / clipped.mean()

    ess = (clipped.sum() ** 2) / (clipped ** 2).sum()

    return clipped, {
        "method": "independent_gaussian",
        "ess": float(ess),
        "ess_pct": float(ess / len(h) * 100),
        "k_hat": None,
        "weight_stats": {
            "mean": float(clipped.mean()),
            "std": float(clipped.std()),
            "max": float(clipped.max()),
            "min": float(clipped.min()),
            "pct_99": float(np.percentile(clipped, 99)),
        },
    }


def compute_weights_copula(heights, weights_kg, apply_psis=True):
    """
    Copula-based density ratio with optional PSIS smoothing.

    Models joint P(height, weight) as bivariate Gaussian for both
    US and VN populations. Assumes same correlation structure (Gaussian copula
    assumption — we only have VN cluster centroids, not raw paired data).

    Args:
        heights: array of height values (cm)
        weights_kg: array of weight values (kg)
        apply_psis: whether to apply Pareto smoothing to tail weights

    Returns:
        weights: normalized importance weights (mean=1)
        diagnostics: dict with ESS, k_hat, weight stats
    """
    h, w = np.asarray(heights, dtype=np.float64), np.asarray(weights_kg, dtype=np.float64)
    X = np.column_stack([h, w])
    n = len(h)

    # ── US (source) distribution: fit from data ──
    us_mean = np.array([h.mean(), w.mean()])
    us_cov = np.cov(h, w)  # full 2x2 covariance (captures correlation)
    us_rho = us_cov[0, 1] / (np.sqrt(us_cov[0, 0]) * np.sqrt(us_cov[1, 1]))

    # ── VN (target) distribution: from Tran et al. 2024 ──
    vn_mean = np.array([VN_FEMALE["height_mean"], VN_FEMALE["weight_mean"]])
    # Assume same correlation as US (Gaussian copula assumption)
    vn_cov = _build_cov_matrix(VN_FEMALE["height_std"], VN_FEMALE["weight_std"], us_rho)

    # ── Density ratio in log space (numerical stability) ──
    us_dist = multivariate_normal(mean=us_mean, cov=us_cov)
    vn_dist = multivariate_normal(mean=vn_mean, cov=vn_cov)

    log_p_vn = vn_dist.logpdf(X)
    log_p_us = us_dist.logpdf(X)
    log_ratios = log_p_vn - log_p_us

    # ── PSIS smoothing ──
    k_hat = None
    if apply_psis:
        log_ratios, k_hat = _psis_smooth(log_ratios)

    # ── Convert to weights and normalize ──
    # Subtract max for numerical stability before exp
    log_ratios_shifted = log_ratios - log_ratios.max()
    raw_weights = np.exp(log_ratios_shifted)
    weights = raw_weights / raw_weights.mean()

    # ── Diagnostics ──
    ess = (weights.sum() ** 2) / (weights ** 2).sum()

    return weights, {
        "method": "copula_psis" if apply_psis else "copula_raw",
        "ess": float(ess),
        "ess_pct": float(ess / n * 100),
        "k_hat": float(k_hat) if k_hat is not None else None,
        "us_rho": float(us_rho),
        "us_mean_h": float(us_mean[0]),
        "us_mean_w": float(us_mean[1]),
        "vn_mean_h": float(vn_mean[0]),
        "vn_mean_w": float(vn_mean[1]),
        "height_gap": float(us_mean[0] - vn_mean[0]),
        "weight_gap": float(us_mean[1] - vn_mean[1]),
        "weight_stats": {
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "max": float(weights.max()),
            "min": float(weights.min()),
            "pct_99": float(np.percentile(weights, 99)),
        },
    }


def compute_all_weight_variants(heights, weights_kg):
    """
    Compute all weight variants for comparison:
      1. No weighting (uniform)
      2. Independent Gaussian (baseline, old method)
      3. Copula raw (no PSIS)
      4. Copula + PSIS (proposed)

    Returns dict of {method_name: (weights, diagnostics)}
    """
    n = len(heights)

    # Uniform (no weighting)
    uniform = np.ones(n)
    uniform_diag = {"method": "uniform", "ess": float(n), "ess_pct": 100.0, "k_hat": None,
                    "weight_stats": {"mean": 1.0, "std": 0.0, "max": 1.0, "min": 1.0, "pct_99": 1.0}}

    # Independent Gaussian
    indep_w, indep_d = compute_weights_independent(heights, weights_kg)

    # Copula without PSIS
    copula_raw_w, copula_raw_d = compute_weights_copula(heights, weights_kg, apply_psis=False)

    # Copula with PSIS
    copula_psis_w, copula_psis_d = compute_weights_copula(heights, weights_kg, apply_psis=True)

    return {
        "uniform": (uniform, uniform_diag),
        "independent_gaussian": (indep_w, indep_d),
        "copula_raw": (copula_raw_w, copula_raw_d),
        "copula_psis": (copula_psis_w, copula_psis_d),
    }


# ── CLI entry point for testing ──

def main():
    import pandas as pd

    print("=" * 60)
    print("  Copula + PSIS Density Ratio Estimation")
    print("=" * 60)

    # Load RTR data
    df = pd.read_parquet(DATA_DIR / "rtr_body_fit.parquet")
    df = df[
        df["height_cm"].notna()
        & df["weight_kg"].notna()
        & df["fit"].isin(["fit", "small", "large"])
    ].copy()
    df_fit = df[df["fit"] == "fit"].copy()

    heights = df_fit["height_cm"].values
    weights = df_fit["weight_kg"].values

    print(f"\nData: {len(heights):,} samples (fit='fit' only)")
    print(f"Height: {heights.mean():.1f} ± {heights.std():.1f} cm")
    print(f"Weight: {weights.mean():.1f} ± {weights.std():.1f} kg")
    print(f"Correlation(h,w): {np.corrcoef(heights, weights)[0,1]:.3f}")

    # Compute all variants
    variants = compute_all_weight_variants(heights, weights)

    print(f"\n{'Method':<25} {'ESS':>8} {'ESS%':>7} {'k_hat':>7} {'w_max':>8} {'w_99%':>8}")
    print("-" * 70)
    for name, (w, d) in variants.items():
        k = f"{d['k_hat']:.3f}" if d['k_hat'] is not None else "N/A"
        ws = d['weight_stats']
        print(f"{name:<25} {d['ess']:>8.0f} {d['ess_pct']:>6.1f}% {k:>7} {ws['max']:>8.2f} {ws['pct_99']:>8.2f}")

    # Print copula diagnostics
    _, copula_d = variants["copula_psis"]
    print(f"\nCopula diagnostics:")
    print(f"  US-VN height gap: {copula_d['height_gap']:.1f} cm")
    print(f"  US-VN weight gap: {copula_d['weight_gap']:.1f} cm")
    print(f"  US h-w correlation (rho): {copula_d['us_rho']:.3f}")
    print(f"  PSIS k_hat: {copula_d['k_hat']:.3f}")

    if copula_d['k_hat'] is not None:
        k = copula_d['k_hat']
        if k < 0.5:
            print(f"  Diagnostic: GOOD (k < 0.5, finite variance)")
        elif k < 0.7:
            print(f"  Diagnostic: MODERATE (0.5 < k < 0.7, slow convergence)")
        else:
            print(f"  Diagnostic: BAD (k > 0.7, infinite variance — shift too extreme)")

    # Compare weight distributions
    print(f"\n{'='*60}")
    print(f"  Weight Distribution Comparison")
    print(f"{'='*60}")
    for name, (w, d) in variants.items():
        ws = d['weight_stats']
        print(f"\n  {name}:")
        print(f"    mean={ws['mean']:.3f}  std={ws['std']:.3f}  min={ws['min']:.4f}  max={ws['max']:.2f}  p99={ws['pct_99']:.2f}")

    # Save weights for downstream use
    out_dir = ROOT / "research" / "weighting"
    for name, (w, d) in variants.items():
        np.save(out_dir / f"weights_{name}.npy", w)

    # Save diagnostics
    diagnostics = {name: d for name, (_, d) in variants.items()}
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\nWeights saved to {out_dir}/")
    print(f"  weights_uniform.npy")
    print(f"  weights_independent_gaussian.npy")
    print(f"  weights_copula_raw.npy")
    print(f"  weights_copula_psis.npy")
    print(f"  diagnostics.json")


if __name__ == "__main__":
    main()
