"""
Evaluate all 6 model variants across population subsets.

intent: answer the critical question — does copula+PSIS+CORN help on Vietnamese-range bodies?
status: done
next: run ablation studies, download ModCloth for replication
confidence: high

Population subsets (from execution guide):
  1. Full test set       — overall accuracy (should not degrade much)
  2. VN-range            — h<160 AND w<55 (target population proxy)
  3. VN Cluster 1        — h<155 AND w<50 (extreme small bodies)
  4. VN Cluster 3        — 153<h<160 AND 48<w<56 (dominant VN body type, 35.9%)
  5. US-typical          — h>163 AND w>58 (sanity check, should stay similar)

Success criteria:
  - Within-1 on VN-range: ≥ +2pp over unweighted baseline
  - Within-1 on full test: ≤ 1pp degradation
  - Directional bias on VN-range: closer to 0 than baseline
  - ESS with copula+PSIS: higher than independent Gaussian

Usage:
    python research/evaluation/eval_all_variants.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from research.models.train_size_corn import (
    SIZE_ORDER,
    NUM_CLASSES,
    SizeMLP,
    corn_predict,
)

VARIANTS_DIR = ROOT / "research" / "models" / "variants"
OUT_DIR = ROOT / "research" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
#  Population Subsets
# ══════════════════════════════════════════════════════════

SUBSETS = {
    "full_test": lambda h, w: np.ones(len(h), dtype=bool),
    "vn_range": lambda h, w: (h < 160) & (w < 55),
    "vn_cluster1": lambda h, w: (h < 155) & (w < 50),
    "vn_cluster3": lambda h, w: (h > 153) & (h < 160) & (w > 48) & (w < 56),
    "us_typical": lambda h, w: (h > 163) & (w > 58),
}


# ══════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, size_enc):
    """Compute all evaluation metrics."""
    y_true_labels = size_enc.inverse_transform(y_true)
    y_pred_labels = size_enc.inverse_transform(y_pred)

    true_ord = np.array([SIZE_ORDER.get(t, 3) for t in y_true_labels])
    pred_ord = np.array([SIZE_ORDER.get(p, 3) for p in y_pred_labels])

    diff = pred_ord - true_ord

    return {
        "n": len(y_true),
        "acc": float(np.mean(y_true == y_pred)),
        "within_1": float(np.mean(np.abs(diff) <= 1)),
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "over_size_pct": float(np.mean(diff > 0) * 100),
        "under_size_pct": float(np.mean(diff < 0) * 100),
    }


# ══════════════════════════════════════════════════════════
#  Load Models and Predict
# ══════════════════════════════════════════════════════════

def load_and_predict_all(X_test, eval_data):
    """Load all 6 model variants and generate predictions on X_test."""
    size_enc = eval_data["size_enc"]
    predictions = {}

    # GBM models
    for name in ["gbm_uniform", "gbm_indep", "gbm_copula"]:
        path = VARIANTS_DIR / f"{name}.pkl"
        if not path.exists():
            print(f"  Skip {name}: not found")
            continue
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        predictions[name] = bundle["model"].predict(X_test)

    # MLP models
    input_dim = X_test.shape[1]
    for name in ["mlp_ce_uniform", "mlp_ce_copula", "mlp_corn_copula"]:
        path = VARIANTS_DIR / f"{name}.pt"
        if not path.exists():
            print(f"  Skip {name}: not found")
            continue

        ckpt = torch.load(path, weights_only=False)
        loss_type = ckpt["loss_type"]
        model = SizeMLP(input_dim, hidden_dim=128, num_classes=NUM_CLASSES, loss_type=loss_type)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        X_mean, X_std = ckpt["X_mean"], ckpt["X_std"]
        X_norm = (X_test - X_mean) / X_std
        X_t = torch.FloatTensor(X_norm)

        with torch.no_grad():
            logits = model(X_t)
            if loss_type == "ce":
                preds = logits.argmax(dim=1).numpy()
            else:
                preds = corn_predict(logits).numpy()

        predictions[name] = preds

    return predictions


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  Population Subset Evaluation: 6 Model Variants × 5 Subsets")
    print("=" * 80)

    # Load eval data
    with open(VARIANTS_DIR / "eval_data.pkl", "rb") as f:
        eval_data = pickle.load(f)

    X_test = eval_data["X_test"]
    y_test = eval_data["y_test"]
    size_enc = eval_data["size_enc"]

    heights = X_test[:, 0]
    weights_kg = X_test[:, 1]

    print(f"Test set: {len(X_test):,} samples")
    print(f"Height range: {heights.min():.0f}–{heights.max():.0f} cm (mean={heights.mean():.1f})")
    print(f"Weight range: {weights_kg.min():.0f}–{weights_kg.max():.0f} kg (mean={weights_kg.mean():.1f})")

    # Subset sizes
    print(f"\nSubset sizes:")
    for subset_name, mask_fn in SUBSETS.items():
        mask = mask_fn(heights, weights_kg)
        print(f"  {subset_name:<20} {mask.sum():>6,} samples ({mask.sum()/len(mask)*100:.1f}%)")

    # Load all models and predict
    print(f"\nLoading models...")
    predictions = load_and_predict_all(X_test, eval_data)
    print(f"  Loaded {len(predictions)} model variants")

    # Evaluate each variant on each subset
    all_results = {}
    for subset_name, mask_fn in SUBSETS.items():
        mask = mask_fn(heights, weights_kg)
        n_subset = mask.sum()
        if n_subset == 0:
            print(f"\n  WARNING: subset '{subset_name}' has 0 samples, skipping")
            continue

        print(f"\n{'─'*80}")
        print(f"  Subset: {subset_name} (n={n_subset:,})")
        print(f"{'─'*80}")
        print(f"  {'Variant':<30} {'Top-1':>7} {'W-1':>7} {'MAE':>7} {'Bias':>7} {'Over%':>7} {'Under%':>7}")
        print(f"  {'─'*30} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

        subset_results = {}
        for model_name, preds in predictions.items():
            metrics = compute_metrics(y_test[mask], preds[mask], size_enc)
            subset_results[model_name] = metrics
            print(
                f"  {model_name:<30} "
                f"{metrics['acc']:>6.1%} "
                f"{metrics['within_1']:>6.1%} "
                f"{metrics['mae']:>7.3f} "
                f"{metrics['bias']:>+6.3f} "
                f"{metrics['over_size_pct']:>6.1f}% "
                f"{metrics['under_size_pct']:>6.1f}%"
            )

        all_results[subset_name] = subset_results

    # ── Delta Analysis: Compare against GBM uniform baseline ──
    print(f"\n{'='*80}")
    print(f"  DELTA ANALYSIS: Improvement over GBM Uniform Baseline")
    print(f"{'='*80}")

    baseline_name = "gbm_uniform"
    if baseline_name not in predictions:
        print("  ERROR: baseline model not found")
        return

    for subset_name in SUBSETS:
        if subset_name not in all_results:
            continue
        subset_r = all_results[subset_name]
        if baseline_name not in subset_r:
            continue

        base = subset_r[baseline_name]
        print(f"\n  {subset_name} (n={base['n']:,}):")
        print(f"  {'Variant':<30} {'Δ Top-1':>8} {'Δ W-1':>8} {'Δ MAE':>8} {'Δ Bias':>8}")
        print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

        for model_name, metrics in subset_r.items():
            if model_name == baseline_name:
                continue
            d_acc = metrics["acc"] - base["acc"]
            d_w1 = metrics["within_1"] - base["within_1"]
            d_mae = metrics["mae"] - base["mae"]
            # For bias, improvement = closer to 0
            d_bias = abs(metrics["bias"]) - abs(base["bias"])
            print(
                f"  {model_name:<30} "
                f"{d_acc:>+7.1%} "
                f"{d_w1:>+7.1%} "
                f"{d_mae:>+7.3f} "
                f"{d_bias:>+7.3f} {'✓' if d_bias < 0 else '✗'}"
            )

    # ── GO/NO-GO Assessment ──
    print(f"\n{'='*80}")
    print(f"  GO/NO-GO ASSESSMENT")
    print(f"{'='*80}")

    proposed = "mlp_corn_copula"
    checks = []

    if "vn_range" in all_results and proposed in all_results["vn_range"]:
        vn_r = all_results["vn_range"]
        base_w1 = vn_r[baseline_name]["within_1"]
        prop_w1 = vn_r[proposed]["within_1"]
        delta = prop_w1 - base_w1
        passed = delta >= 0.02
        checks.append(("VN-range within-1 ≥ +2pp", delta, passed))
        print(f"  [{'PASS' if passed else 'FAIL'}] VN-range within-1: {base_w1:.1%} → {prop_w1:.1%} (Δ={delta:+.1%}, need ≥+2.0%)")

        # Bias check
        base_bias = abs(vn_r[baseline_name]["bias"])
        prop_bias = abs(vn_r[proposed]["bias"])
        passed = prop_bias < base_bias
        checks.append(("VN-range bias closer to 0", prop_bias - base_bias, passed))
        print(f"  [{'PASS' if passed else 'FAIL'}] VN-range |bias|: {base_bias:.3f} → {prop_bias:.3f} ({'closer' if passed else 'worse'})")

    if "full_test" in all_results and proposed in all_results["full_test"]:
        full_r = all_results["full_test"]
        base_w1 = full_r[baseline_name]["within_1"]
        prop_w1 = full_r[proposed]["within_1"]
        delta = prop_w1 - base_w1
        passed = delta >= -0.01
        checks.append(("Full test within-1 ≤ 1pp degradation", delta, passed))
        print(f"  [{'PASS' if passed else 'FAIL'}] Full test within-1: {base_w1:.1%} → {prop_w1:.1%} (Δ={delta:+.1%}, need ≥-1.0%)")

    n_pass = sum(1 for _, _, p in checks if p)
    n_total = len(checks)
    verdict = "GO" if n_pass == n_total else "CONDITIONAL" if n_pass >= n_total / 2 else "NO-GO"
    print(f"\n  Verdict: {verdict} ({n_pass}/{n_total} checks passed)")

    # ── Save results ──
    with open(OUT_DIR / "subset_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {OUT_DIR / 'subset_results.json'}")


if __name__ == "__main__":
    main()
