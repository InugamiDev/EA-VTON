"""
GO/NO-GO Gate: Evaluate baseline vs EA-VTON on Vietnamese-range body subset.

intent: verify importance weighting helps on the target population before proceeding with paper
status: done
next: interpret results — if EA-VTON within-1 is NOT ≥2pp better on VN subset, rethink method
confidence: high

The key question: does importance weighting actually improve predictions for
Vietnamese-range bodies (height < 160cm, weight < 55kg)?

Current results on FULL RTR test set (US population):
  - Baseline: 49.04% top-1, 83.37% within-1
  - EA-VTON:  47.96% top-1, 82.96% within-1 (WORSE)

This is EXPECTED — the EA-VTON model sacrifices US accuracy for Vietnamese accuracy.
But we need to VERIFY it actually gains on Vietnamese-range bodies.

Usage:
    python research/evaluation/eval_vn_subset.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "research" / "datasets" / "processed"
MODEL_DIR = ROOT / "research" / "models"

SIZE_MAP = {
    0: "XXS", 1: "XXS", 2: "XS", 3: "XS", 4: "S",
    5: "S", 6: "S", 7: "M", 8: "M",
    9: "M", 10: "L", 11: "L", 12: "L",
    13: "L", 14: "XL", 15: "XL", 16: "XL",
    17: "XXL", 18: "XXL", 19: "XXL", 20: "XXL",
}

SIZE_ORDER = {"XXS": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5, "XXL": 6}


def within_k_accuracy(y_true_labels, y_pred_labels, k=1):
    """Fraction of predictions within k ordinal steps of truth."""
    return np.mean([
        abs(SIZE_ORDER.get(t, 3) - SIZE_ORDER.get(p, 3)) <= k
        for t, p in zip(y_true_labels, y_pred_labels)
    ])


def mean_ordinal_error(y_true_labels, y_pred_labels):
    """Average ordinal distance between prediction and truth."""
    return np.mean([
        abs(SIZE_ORDER.get(t, 3) - SIZE_ORDER.get(p, 3))
        for t, p in zip(y_true_labels, y_pred_labels)
    ])


def directional_bias(y_true_labels, y_pred_labels):
    """Positive = predicts too large, negative = predicts too small."""
    return np.mean([
        SIZE_ORDER.get(p, 3) - SIZE_ORDER.get(t, 3)
        for t, p in zip(y_true_labels, y_pred_labels)
    ])


def eval_on_subset(model, X, y, size_enc, subset_mask, label):
    """Evaluate model on a subset of the test data."""
    X_sub = X[subset_mask]
    y_sub = y[subset_mask]
    n = len(X_sub)

    if n == 0:
        print(f"  {label}: NO SAMPLES in subset")
        return None

    y_pred = model.predict(X_sub)
    y_true_labels = size_enc.inverse_transform(y_sub)
    y_pred_labels = size_enc.inverse_transform(y_pred)

    acc = accuracy_score(y_sub, y_pred)
    w1 = within_k_accuracy(y_true_labels, y_pred_labels, k=1)
    w2 = within_k_accuracy(y_true_labels, y_pred_labels, k=2)
    moe = mean_ordinal_error(y_true_labels, y_pred_labels)
    bias = directional_bias(y_true_labels, y_pred_labels)

    # Size distribution
    true_dist = pd.Series(y_true_labels).value_counts(normalize=True).sort_index()
    pred_dist = pd.Series(y_pred_labels).value_counts(normalize=True).sort_index()

    print(f"\n  {'='*55}")
    print(f"  {label} (n={n:,})")
    print(f"  {'='*55}")
    print(f"  Top-1 Accuracy:      {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Within-1 Accuracy:   {w1:.4f} ({w1*100:.1f}%)")
    print(f"  Within-2 Accuracy:   {w2:.4f} ({w2*100:.1f}%)")
    print(f"  Mean Ordinal Error:  {moe:.3f} sizes")
    print(f"  Directional Bias:    {bias:+.3f} (+ = oversizes, - = undersizes)")
    print(f"  Size distribution:")
    for s in ["XXS", "XS", "S", "M", "L", "XL", "XXL"]:
        tp = true_dist.get(s, 0) * 100
        pp = pred_dist.get(s, 0) * 100
        print(f"    {s:>3s}: true {tp:5.1f}% | pred {pp:5.1f}%")

    return {
        "n": n, "acc": acc, "w1": w1, "w2": w2,
        "moe": moe, "bias": bias,
    }


def main():
    print("=" * 60)
    print("  GO/NO-GO GATE: Vietnamese Subset Evaluation")
    print("=" * 60)

    # ── Load and prepare data (same as train_size_rec.py) ──
    df = pd.read_parquet(DATA_DIR / "rtr_body_fit.parquet")
    df = df[
        df["height_cm"].notna()
        & df["weight_kg"].notna()
        & df["fit"].isin(["fit", "small", "large"])
    ].copy()
    df_fit = df[df["fit"] == "fit"].copy()
    df_fit["size_label"] = df_fit["size"].map(SIZE_MAP)
    df_fit = df_fit[df_fit["size_label"].notna()].copy()

    # ── Build features (same pipeline) ──
    body_enc = LabelEncoder()
    cat_enc = LabelEncoder()
    size_enc = LabelEncoder()

    df_fit["body_type_clean"] = df_fit["body_type_norm"].fillna("unknown")
    body_encoded = body_enc.fit_transform(df_fit["body_type_clean"])
    cat_encoded = cat_enc.fit_transform(df_fit["category"])
    size_labels = size_enc.fit_transform(df_fit["size_label"])

    X = np.column_stack([
        df_fit["height_cm"].values,
        df_fit["weight_kg"].values,
        df_fit["bmi"].fillna(22.0).values,
        df_fit["age"].fillna(30).values,
        body_encoded,
        cat_encoded,
    ])

    # ── Same split as training (seed=42) ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, size_labels, test_size=0.2, random_state=42, stratify=size_labels
    )

    # Heights and weights in test set (column 0 and 1)
    test_heights = X_test[:, 0]
    test_weights = X_test[:, 1]

    print(f"\nFull test set: {len(X_test):,} samples")
    print(f"Height range: {test_heights.min():.1f} - {test_heights.max():.1f} cm")
    print(f"  Mean: {test_heights.mean():.1f} ± {test_heights.std():.1f} cm")
    print(f"Weight range: {test_weights.min():.1f} - {test_weights.max():.1f} cm")
    print(f"  Mean: {test_weights.mean():.1f} ± {test_weights.std():.1f} kg")

    # ── Train both models (reproduce exactly) ──
    print("\nTraining baseline model...")
    baseline = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=20, random_state=42,
    )
    baseline.fit(X_train, y_train)

    print("Training EA-VTON model (VN importance weighting)...")
    train_h = X_train[:, 0]
    train_w = X_train[:, 1]
    train_b = X_train[:, 2]

    rtr_h_mu, rtr_h_sig = train_h.mean(), train_h.std()
    rtr_w_mu, rtr_w_sig = train_w.mean(), train_w.std()
    rtr_b_mu, rtr_b_sig = train_b.mean(), train_b.std()

    vn_h_mu, vn_h_sig = 156.2, 5.5
    vn_w_mu, vn_w_sig = 53.9, 8.0

    w_h = np.clip(
        stats.norm.pdf(train_h, vn_h_mu, vn_h_sig) /
        (stats.norm.pdf(train_h, rtr_h_mu, rtr_h_sig) + 1e-10),
        0.01, 100
    )
    w_w = np.clip(
        stats.norm.pdf(train_w, vn_w_mu, vn_w_sig) /
        (stats.norm.pdf(train_w, rtr_w_mu, rtr_w_sig) + 1e-10),
        0.01, 100
    )
    w_b = np.clip(
        stats.norm.pdf(train_b, 21.4, 2.5) /
        (stats.norm.pdf(train_b, rtr_b_mu, rtr_b_sig) + 1e-10),
        0.01, 100
    )
    vn_weights = w_h * w_w * w_b
    vn_weights = vn_weights / vn_weights.mean()

    ess = (vn_weights.sum() ** 2) / (vn_weights ** 2).sum()
    print(f"  ESS: {ess:.0f} / {len(X_train):,} ({ess/len(X_train)*100:.1f}%)")

    eavton = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=20, random_state=42,
    )
    eavton.fit(X_train, y_train, sample_weight=vn_weights)

    # ══════════════════════════════════════════════════════════
    #  EVALUATION ON SUBSETS
    # ══════════════════════════════════════════════════════════

    # Define subsets by height/weight ranges
    subsets = {
        "FULL TEST SET": np.ones(len(X_test), dtype=bool),
        "VN-RANGE: h<160 AND w<55": (test_heights < 160) & (test_weights < 55),
        "VN-RANGE: h<160": test_heights < 160,
        "VN-RANGE: w<55": test_weights < 55,
        "VN CLUSTER 1 (short/thin): h<155 AND w<50": (test_heights < 155) & (test_weights < 50),
        "VN CLUSTER 3 (medium): 153<h<160 AND 48<w<56": (
            (test_heights > 153) & (test_heights < 160) &
            (test_weights > 48) & (test_weights < 56)
        ),
        "VN CLUSTER 4 (short/heavy): h<155 AND w>55": (test_heights < 155) & (test_weights > 55),
        "US-TYPICAL: h>163 AND w>58": (test_heights > 163) & (test_weights > 58),
    }

    results = {}
    for subset_name, mask in subsets.items():
        n_in_subset = mask.sum()
        print(f"\n\n{'#'*60}")
        print(f"  SUBSET: {subset_name} (n={n_in_subset})")
        print(f"{'#'*60}")

        if n_in_subset < 10:
            print(f"  ⚠ Too few samples ({n_in_subset}), skipping")
            continue

        r_base = eval_on_subset(baseline, X_test, y_test, size_enc, mask, "BASELINE (US)")
        r_ea = eval_on_subset(eavton, X_test, y_test, size_enc, mask, "EA-VTON (VN-weighted)")

        if r_base and r_ea:
            delta_acc = (r_ea["acc"] - r_base["acc"]) * 100
            delta_w1 = (r_ea["w1"] - r_base["w1"]) * 100
            delta_bias = r_ea["bias"] - r_base["bias"]

            print(f"\n  {'─'*55}")
            print(f"  DELTA (EA-VTON − Baseline):")
            print(f"    Top-1:    {delta_acc:+.1f} pp")
            print(f"    Within-1: {delta_w1:+.1f} pp")
            print(f"    Bias:     {delta_bias:+.3f} (negative = EA-VTON sizes down more)")
            print(f"  {'─'*55}")

            results[subset_name] = {
                "n": n_in_subset,
                "baseline": r_base,
                "eavton": r_ea,
                "delta_acc": delta_acc,
                "delta_w1": delta_w1,
            }

    # ══════════════════════════════════════════════════════════
    #  GO/NO-GO VERDICT
    # ══════════════════════════════════════════════════════════
    print("\n\n" + "=" * 60)
    print("  GO/NO-GO VERDICT")
    print("=" * 60)

    vn_key = "VN-RANGE: h<160 AND w<55"
    if vn_key in results:
        r = results[vn_key]
        delta = r["delta_w1"]
        n = r["n"]

        print(f"\n  Target subset: {vn_key}")
        print(f"  Samples: {n}")
        print(f"  Baseline within-1: {r['baseline']['w1']*100:.1f}%")
        print(f"  EA-VTON  within-1: {r['eavton']['w1']*100:.1f}%")
        print(f"  Delta: {delta:+.1f} pp")

        if delta >= 2.0:
            print(f"\n  ✅ GO — EA-VTON gains ≥2pp on Vietnamese-range bodies")
            print(f"  Proceed with paper. Importance weighting demonstrably helps.")
        elif delta >= 0:
            print(f"\n  ⚠️  MARGINAL — EA-VTON gains {delta:+.1f}pp (< 2pp threshold)")
            print(f"  Consider: copula weighting may push this over the threshold.")
            print(f"  Proceed cautiously — need stronger method (Step 2: copula).")
        else:
            print(f"\n  ❌ NO-GO — EA-VTON is WORSE by {abs(delta):.1f}pp on target subset")
            print(f"  Current importance weighting is broken.")
            print(f"  Must fix methodology before proceeding with paper.")

        # Check if bias correction is happening
        base_bias = r["baseline"]["bias"]
        ea_bias = r["eavton"]["bias"]
        print(f"\n  Bias check:")
        print(f"    Baseline bias: {base_bias:+.3f} (+ = oversizes for small bodies)")
        print(f"    EA-VTON  bias: {ea_bias:+.3f}")
        if ea_bias < base_bias:
            print(f"    ✅ EA-VTON reduces oversizing tendency ({base_bias - ea_bias:.3f} improvement)")
        else:
            print(f"    ❌ EA-VTON does NOT reduce oversizing")
    else:
        print(f"\n  ❌ Could not evaluate — insufficient samples in VN range")

    # ── Summary table ──
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"  {'Subset':<45} {'N':>5} {'Δ Top-1':>8} {'Δ W-1':>8}")
    print(f"  {'─'*45} {'─'*5} {'─'*8} {'─'*8}")
    for name, r in results.items():
        print(f"  {name:<45} {r['n']:>5} {r['delta_acc']:>+7.1f}% {r['delta_w1']:>+7.1f}%")
    print(f"  {'='*80}")


if __name__ == "__main__":
    main()
