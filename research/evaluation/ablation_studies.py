"""
Ablation studies for EA-VTON size recommendation.

intent: isolate the contribution of each method component (PSIS, copula, ordinal loss, feature dims)
status: done
next: incorporate results into paper tables
confidence: high
blockers: none

Ablations:
  1. PSIS ablation:           copula_raw vs copula+PSIS
  2. Copula vs Independent:   delta from main results
  3. CORAL vs CORN:           ordinal loss comparison (Cao 2020 vs Shi 2023)
  4. CE vs CORAL vs CORN:     all three losses with copula weights
  5. Single-feature ablation: height-only vs weight-only vs joint copula
  6. VN param sensitivity:    vary VN target mean +/-1 sigma

Usage:
    .venv/bin/python research/evaluation/ablation_studies.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from scipy.stats import multivariate_normal

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from research.models.train_size_corn import (
    SIZE_ORDER,
    NUM_CLASSES,
    SizeMLP,
    CORNLoss,
    corn_predict,
    train_gbm,
    train_mlp,
    predict_mlp,
    evaluate,
)
from research.weighting.copula_psis import (
    compute_weights_independent,
    compute_weights_copula,
)

VARIANTS_DIR = ROOT / "research" / "models" / "variants"
OUT_DIR = ROOT / "research" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
#  CORAL Loss (Cao et al. 2020 PRL)
# ══════════════════════════════════════════════════════════

# intent: implement CORAL as ablation comparison for CORN
# status: done
# confidence: high

class CORALLoss(nn.Module):
    """
    Consistent Rank Logits (CORAL) ordinal regression.

    Key difference from CORN: CORAL uses ALL samples for each threshold task,
    while CORN conditions on Y >= k (only samples where Y is at or above rank k).
    CORAL also shares the penultimate layer weights across all K-1 binary tasks.

    For each threshold k in 0..K-2:
        binary target = (Y > k)
        loss += BCE(logit_k, binary_target)
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets, sample_weights=None):
        # logits: (batch, K-1)
        # For each threshold k: binary target = (Y > k)
        # Use ALL samples (not conditional like CORN)
        batch_size = logits.size(0)
        num_tasks = self.num_classes - 1

        total_loss = torch.zeros(batch_size, device=logits.device)
        for k in range(num_tasks):
            binary_target = (targets > k).float()
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits[:, k], binary_target, reduction='none'
            )
            total_loss += bce

        if sample_weights is not None:
            return (total_loss * sample_weights).mean()
        return total_loss.mean()


def coral_predict(logits):
    """
    Convert CORAL logits to class predictions.

    P(Y > k) = sigmoid(logit_k)
    P(Y = 0) = 1 - P(Y > 0)
    P(Y = k) = P(Y > k-1) - P(Y > k) for k = 1..K-2
    P(Y = K-1) = P(Y > K-2)
    """
    probs_gt = torch.sigmoid(logits)  # (batch, K-1)
    K = logits.size(1) + 1

    class_probs = torch.zeros(logits.size(0), K, device=logits.device)
    class_probs[:, 0] = 1.0 - probs_gt[:, 0]
    for k in range(1, K - 1):
        class_probs[:, k] = probs_gt[:, k - 1] - probs_gt[:, k]
    class_probs[:, -1] = probs_gt[:, -1]

    # Clamp to avoid negative probabilities from numerical issues
    class_probs = torch.clamp(class_probs, min=0.0)

    return class_probs.argmax(dim=1)


def train_mlp_coral(X_train, y_train, sample_weight=None, epochs=80, lr=1e-3, label=""):
    """Train MLP with CORAL loss. Mirrors train_mlp but uses CORALLoss."""
    from torch.utils.data import DataLoader, TensorDataset

    print(f"\n  Training MLP (CORAL): {label}...")

    input_dim = X_train.shape[1]
    # CORAL uses K-1 outputs with shared weights (same architecture as CORN for fair comparison)
    model = SizeMLP(input_dim, hidden_dim=128, num_classes=NUM_CLASSES, loss_type="corn")

    # Normalize features
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_norm = (X_train - X_mean) / X_std

    X_t = torch.FloatTensor(X_norm)
    y_t = torch.LongTensor(y_train)
    w_t = torch.FloatTensor(sample_weight) if sample_weight is not None else None

    dataset = TensorDataset(X_t, y_t) if w_t is None else TensorDataset(X_t, y_t, w_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    criterion = CORALLoss(NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            if w_t is not None:
                xb, yb, wb = batch
            else:
                xb, yb = batch
                wb = None

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb, sample_weights=wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}")

    return model, X_mean, X_std


def predict_mlp_coral(model, X, X_mean, X_std):
    """Get predictions from CORAL-trained MLP."""
    model.eval()
    X_norm = (X - X_mean) / X_std
    X_t = torch.FloatTensor(X_norm)
    with torch.no_grad():
        logits = model(X_t)
        preds = coral_predict(logits).numpy()
    return preds


# ══════════════════════════════════════════════════════════
#  Single-Feature Weighting
# ══════════════════════════════════════════════════════════

# intent: isolate contribution of each feature dimension to importance weighting
# status: done
# confidence: high

def compute_weights_height_only(heights):
    """Density ratio using only height marginals (1D Gaussian ratio)."""
    h = np.asarray(heights, dtype=np.float64)
    us_mu, us_sig = h.mean(), h.std()
    vn_mu, vn_sig = 156.2, 5.5
    log_ratio = stats.norm.logpdf(h, vn_mu, vn_sig) - stats.norm.logpdf(h, us_mu, us_sig)
    w = np.exp(log_ratio - log_ratio.max())
    return w / w.mean()


def compute_weights_weight_only(weights_kg):
    """Density ratio using only weight marginals (1D Gaussian ratio)."""
    w = np.asarray(weights_kg, dtype=np.float64)
    us_mu, us_sig = w.mean(), w.std()
    vn_mu, vn_sig = 53.9, 8.0
    log_ratio = stats.norm.logpdf(w, vn_mu, vn_sig) - stats.norm.logpdf(w, us_mu, us_sig)
    wt = np.exp(log_ratio - log_ratio.max())
    return wt / wt.mean()


# ══════════════════════════════════════════════════════════
#  VN Parameter Sensitivity
# ══════════════════════════════════════════════════════════

# intent: check stability of results under uncertainty in VN population parameters
# status: done
# confidence: high

def compute_weights_copula_custom(heights, weights_kg, vn_h_mean, vn_h_std,
                                   vn_w_mean, vn_w_std, apply_psis=True):
    """
    Copula weighting with custom VN target parameters.
    Inline version of compute_weights_copula for sensitivity analysis.
    """
    from research.weighting.copula_psis import _build_cov_matrix, _psis_smooth

    h = np.asarray(heights, dtype=np.float64)
    w = np.asarray(weights_kg, dtype=np.float64)
    X = np.column_stack([h, w])

    # US (source) distribution: fit from data
    us_mean = np.array([h.mean(), w.mean()])
    us_cov = np.cov(h, w)
    us_rho = us_cov[0, 1] / (np.sqrt(us_cov[0, 0]) * np.sqrt(us_cov[1, 1]))

    # VN (target) with custom params
    vn_mean = np.array([vn_h_mean, vn_w_mean])
    vn_cov = _build_cov_matrix(vn_h_std, vn_w_std, us_rho)

    us_dist = multivariate_normal(mean=us_mean, cov=us_cov)
    vn_dist = multivariate_normal(mean=vn_mean, cov=vn_cov)

    log_ratios = vn_dist.logpdf(X) - us_dist.logpdf(X)

    k_hat = None
    if apply_psis:
        log_ratios, k_hat = _psis_smooth(log_ratios)

    log_ratios_shifted = log_ratios - log_ratios.max()
    raw_weights = np.exp(log_ratios_shifted)
    weights = raw_weights / raw_weights.mean()

    ess = (weights.sum() ** 2) / (weights ** 2).sum()

    return weights, {
        "ess": float(ess),
        "ess_pct": float(ess / len(h) * 100),
        "k_hat": float(k_hat) if k_hat is not None else None,
    }


# ══════════════════════════════════════════════════════════
#  Evaluation Helpers
# ══════════════════════════════════════════════════════════

def eval_full_and_vn(y_test, y_pred, X_test, size_enc, label=""):
    """Evaluate on full test set and VN-range subset, return dict of metrics."""
    heights = X_test[:, 0]
    weights_kg = X_test[:, 1]

    # Full test
    full = evaluate(y_test, y_pred, size_enc, label=f"{label} [full]")

    # VN-range subset: h < 160 AND w < 55
    vn_mask = (heights < 160) & (weights_kg < 55)
    n_vn = vn_mask.sum()
    if n_vn > 0:
        vn = evaluate(y_test[vn_mask], y_pred[vn_mask], size_enc, label=f"{label} [VN-range n={n_vn}]")
    else:
        vn = {"acc": 0.0, "within_1": 0.0, "mae": 0.0, "bias": 0.0}

    return {"full": full, "vn_range": vn, "vn_n": int(n_vn)}


def ess_from_weights(w):
    """Effective sample size."""
    w = np.asarray(w)
    return float((w.sum() ** 2) / (w ** 2).sum())


def print_table(title, rows, columns):
    """Print a formatted table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    # Column widths
    col_widths = [max(len(str(c)), 8) for c in columns]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    # Header
    header = "  ".join(f"{c:<{col_widths[i]}}" for i, c in enumerate(columns))
    print(f"  {header}")
    print(f"  {'  '.join('-' * w for w in col_widths)}")

    # Rows
    for row in rows:
        line = "  ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(row))
        print(f"  {line}")


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  ABLATION STUDIES: EA-VTON Size Recommendation")
    print("=" * 80)

    # Load eval data
    with open(VARIANTS_DIR / "eval_data.pkl", "rb") as f:
        eval_data = pickle.load(f)

    X_train = eval_data["X_train"]
    X_test = eval_data["X_test"]
    y_train = eval_data["y_train"]
    y_test = eval_data["y_test"]
    size_enc = eval_data["size_enc"]
    train_weights = eval_data["train_weights"]

    train_heights = X_train[:, 0]
    train_weights_kg = X_train[:, 1]

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    all_results = {}

    # ──────────────────────────────────────────────────────
    #  Ablation 1: PSIS Effect (copula_raw vs copula+PSIS)
    # ──────────────────────────────────────────────────────

    print(f"\n\n{'#'*80}")
    print(f"  ABLATION 1: PSIS Smoothing Effect")
    print(f"{'#'*80}")

    # Compute copula without PSIS
    w_copula_raw, d_copula_raw = compute_weights_copula(
        train_heights, train_weights_kg, apply_psis=False
    )
    w_copula_psis = train_weights["copula_psis"]

    ess_raw = ess_from_weights(w_copula_raw)
    ess_psis = ess_from_weights(w_copula_psis)

    print(f"\n  Weight diagnostics:")
    print(f"    Copula raw:  ESS={ess_raw:.0f} ({ess_raw/len(X_train)*100:.1f}%)  "
          f"max={w_copula_raw.max():.2f}  p99={np.percentile(w_copula_raw, 99):.2f}")
    print(f"    Copula+PSIS: ESS={ess_psis:.0f} ({ess_psis/len(X_train)*100:.1f}%)  "
          f"max={w_copula_psis.max():.2f}  p99={np.percentile(w_copula_psis, 99):.2f}")
    print(f"    k_hat: {d_copula_raw.get('k_hat', 'N/A')}")

    # Train GBM with copula_raw
    gbm_raw = train_gbm(X_train, y_train, sample_weight=w_copula_raw, label="GBM copula_raw (no PSIS)")
    y_pred_raw = gbm_raw.predict(X_test)
    res_raw = eval_full_and_vn(y_test, y_pred_raw, X_test, size_enc, "GBM copula_raw")

    # GBM with copula+PSIS (retrain for fair comparison)
    gbm_psis = train_gbm(X_train, y_train, sample_weight=w_copula_psis, label="GBM copula+PSIS")
    y_pred_psis = gbm_psis.predict(X_test)
    res_psis = eval_full_and_vn(y_test, y_pred_psis, X_test, size_enc, "GBM copula+PSIS")

    abl1 = {
        "copula_raw": {
            "ess": ess_raw, "ess_pct": ess_raw / len(X_train) * 100,
            "full": res_raw["full"], "vn_range": res_raw["vn_range"],
        },
        "copula_psis": {
            "ess": ess_psis, "ess_pct": ess_psis / len(X_train) * 100,
            "full": res_psis["full"], "vn_range": res_psis["vn_range"],
        },
    }
    all_results["ablation_1_psis"] = abl1

    print_table("Ablation 1: PSIS Effect (GBM)", [
        ["copula_raw",  f"{ess_raw:.0f}",
         f"{res_raw['full']['acc']:.3f}", f"{res_raw['full']['within_1']:.3f}", f"{res_raw['full']['bias']:+.3f}",
         f"{res_raw['vn_range']['acc']:.3f}", f"{res_raw['vn_range']['within_1']:.3f}", f"{res_raw['vn_range']['bias']:+.3f}"],
        ["copula+PSIS", f"{ess_psis:.0f}",
         f"{res_psis['full']['acc']:.3f}", f"{res_psis['full']['within_1']:.3f}", f"{res_psis['full']['bias']:+.3f}",
         f"{res_psis['vn_range']['acc']:.3f}", f"{res_psis['vn_range']['within_1']:.3f}", f"{res_psis['vn_range']['bias']:+.3f}"],
    ], ["Method", "ESS", "Full Acc", "Full W1", "Full Bias", "VN Acc", "VN W1", "VN Bias"])

    # ──────────────────────────────────────────────────────
    #  Ablation 2: Copula vs Independent
    # ──────────────────────────────────────────────────────

    print(f"\n\n{'#'*80}")
    print(f"  ABLATION 2: Copula vs Independent Gaussian")
    print(f"{'#'*80}")

    w_indep = train_weights["independent_gaussian"]
    ess_indep = ess_from_weights(w_indep)

    gbm_indep = train_gbm(X_train, y_train, sample_weight=w_indep, label="GBM independent")
    y_pred_indep = gbm_indep.predict(X_test)
    res_indep = eval_full_and_vn(y_test, y_pred_indep, X_test, size_enc, "GBM indep")

    # Reuse copula+PSIS from ablation 1
    abl2 = {
        "independent_gaussian": {
            "ess": ess_indep, "ess_pct": ess_indep / len(X_train) * 100,
            "full": res_indep["full"], "vn_range": res_indep["vn_range"],
        },
        "copula_psis": {
            "ess": ess_psis, "ess_pct": ess_psis / len(X_train) * 100,
            "full": res_psis["full"], "vn_range": res_psis["vn_range"],
        },
    }
    all_results["ablation_2_copula_vs_indep"] = abl2

    print_table("Ablation 2: Copula vs Independent (GBM)", [
        ["indep_gaussian", f"{ess_indep:.0f}",
         f"{res_indep['full']['acc']:.3f}", f"{res_indep['full']['within_1']:.3f}", f"{res_indep['full']['bias']:+.3f}",
         f"{res_indep['vn_range']['acc']:.3f}", f"{res_indep['vn_range']['within_1']:.3f}", f"{res_indep['vn_range']['bias']:+.3f}"],
        ["copula+PSIS",   f"{ess_psis:.0f}",
         f"{res_psis['full']['acc']:.3f}", f"{res_psis['full']['within_1']:.3f}", f"{res_psis['full']['bias']:+.3f}",
         f"{res_psis['vn_range']['acc']:.3f}", f"{res_psis['vn_range']['within_1']:.3f}", f"{res_psis['vn_range']['bias']:+.3f}"],
    ], ["Method", "ESS", "Full Acc", "Full W1", "Full Bias", "VN Acc", "VN W1", "VN Bias"])

    # ──────────────────────────────────────────────────────
    #  Ablation 3 & 4: CE vs CORAL vs CORN (all with copula weights)
    # ──────────────────────────────────────────────────────

    print(f"\n\n{'#'*80}")
    print(f"  ABLATION 3-4: Loss Function Comparison (CE vs CORAL vs CORN)")
    print(f"{'#'*80}")

    w_copula = train_weights["copula_psis"]

    # MLP + CE + copula (retrain)
    mlp_ce, mu_ce, sig_ce = train_mlp(
        X_train, y_train, sample_weight=w_copula, loss_type="ce",
        epochs=50, label="MLP+CE (copula)"
    )
    y_pred_ce = predict_mlp(mlp_ce, X_test, mu_ce, sig_ce, "ce")
    res_ce = eval_full_and_vn(y_test, y_pred_ce, X_test, size_enc, "MLP+CE+copula")

    # MLP + CORAL + copula
    mlp_coral, mu_coral, sig_coral = train_mlp_coral(
        X_train, y_train, sample_weight=w_copula,
        epochs=80, label="MLP+CORAL (copula)"
    )
    y_pred_coral = predict_mlp_coral(mlp_coral, X_test, mu_coral, sig_coral)
    res_coral = eval_full_and_vn(y_test, y_pred_coral, X_test, size_enc, "MLP+CORAL+copula")

    # MLP + CORN + copula (retrain)
    mlp_corn, mu_corn, sig_corn = train_mlp(
        X_train, y_train, sample_weight=w_copula, loss_type="corn",
        epochs=80, label="MLP+CORN (copula)"
    )
    y_pred_corn = predict_mlp(mlp_corn, X_test, mu_corn, sig_corn, "corn")
    res_corn = eval_full_and_vn(y_test, y_pred_corn, X_test, size_enc, "MLP+CORN+copula")

    abl34 = {
        "mlp_ce_copula": {"full": res_ce["full"], "vn_range": res_ce["vn_range"]},
        "mlp_coral_copula": {"full": res_coral["full"], "vn_range": res_coral["vn_range"]},
        "mlp_corn_copula": {"full": res_corn["full"], "vn_range": res_corn["vn_range"]},
    }
    all_results["ablation_3_4_loss_comparison"] = abl34

    print_table("Ablation 3-4: Loss Functions (MLP + copula+PSIS weights)", [
        ["CE (baseline)",
         f"{res_ce['full']['acc']:.3f}", f"{res_ce['full']['within_1']:.3f}",
         f"{res_ce['full']['mae']:.3f}", f"{res_ce['full']['bias']:+.3f}",
         f"{res_ce['vn_range']['acc']:.3f}", f"{res_ce['vn_range']['within_1']:.3f}",
         f"{res_ce['vn_range']['mae']:.3f}", f"{res_ce['vn_range']['bias']:+.3f}"],
        ["CORAL (Cao20)",
         f"{res_coral['full']['acc']:.3f}", f"{res_coral['full']['within_1']:.3f}",
         f"{res_coral['full']['mae']:.3f}", f"{res_coral['full']['bias']:+.3f}",
         f"{res_coral['vn_range']['acc']:.3f}", f"{res_coral['vn_range']['within_1']:.3f}",
         f"{res_coral['vn_range']['mae']:.3f}", f"{res_coral['vn_range']['bias']:+.3f}"],
        ["CORN (Shi23)",
         f"{res_corn['full']['acc']:.3f}", f"{res_corn['full']['within_1']:.3f}",
         f"{res_corn['full']['mae']:.3f}", f"{res_corn['full']['bias']:+.3f}",
         f"{res_corn['vn_range']['acc']:.3f}", f"{res_corn['vn_range']['within_1']:.3f}",
         f"{res_corn['vn_range']['mae']:.3f}", f"{res_corn['vn_range']['bias']:+.3f}"],
    ], ["Loss", "Full Acc", "Full W1", "Full MAE", "Full Bias",
        "VN Acc", "VN W1", "VN MAE", "VN Bias"])

    # ──────────────────────────────────────────────────────
    #  Ablation 5: Single-Feature Weighting
    # ──────────────────────────────────────────────────────

    print(f"\n\n{'#'*80}")
    print(f"  ABLATION 5: Single-Feature Weighting")
    print(f"{'#'*80}")

    w_height = compute_weights_height_only(train_heights)
    w_weight = compute_weights_weight_only(train_weights_kg)
    w_joint = train_weights["copula_psis"]

    ess_h = ess_from_weights(w_height)
    ess_w = ess_from_weights(w_weight)
    ess_j = ess_from_weights(w_joint)

    print(f"\n  Weight ESS:")
    print(f"    Height-only: {ess_h:.0f} ({ess_h/len(X_train)*100:.1f}%)")
    print(f"    Weight-only: {ess_w:.0f} ({ess_w/len(X_train)*100:.1f}%)")
    print(f"    Joint copula: {ess_j:.0f} ({ess_j/len(X_train)*100:.1f}%)")

    # Train GBM with each
    gbm_h = train_gbm(X_train, y_train, sample_weight=w_height, label="GBM height-only")
    y_pred_h = gbm_h.predict(X_test)
    res_h = eval_full_and_vn(y_test, y_pred_h, X_test, size_enc, "GBM height-only")

    gbm_w = train_gbm(X_train, y_train, sample_weight=w_weight, label="GBM weight-only")
    y_pred_w = gbm_w.predict(X_test)
    res_w = eval_full_and_vn(y_test, y_pred_w, X_test, size_enc, "GBM weight-only")

    # Joint already computed
    abl5 = {
        "height_only": {
            "ess": ess_h, "full": res_h["full"], "vn_range": res_h["vn_range"],
        },
        "weight_only": {
            "ess": ess_w, "full": res_w["full"], "vn_range": res_w["vn_range"],
        },
        "joint_copula": {
            "ess": ess_j, "full": res_psis["full"], "vn_range": res_psis["vn_range"],
        },
    }
    all_results["ablation_5_single_feature"] = abl5

    print_table("Ablation 5: Feature Dimension Weighting (GBM)", [
        ["Height only", f"{ess_h:.0f}",
         f"{res_h['full']['acc']:.3f}", f"{res_h['full']['within_1']:.3f}", f"{res_h['full']['bias']:+.3f}",
         f"{res_h['vn_range']['acc']:.3f}", f"{res_h['vn_range']['within_1']:.3f}", f"{res_h['vn_range']['bias']:+.3f}"],
        ["Weight only", f"{ess_w:.0f}",
         f"{res_w['full']['acc']:.3f}", f"{res_w['full']['within_1']:.3f}", f"{res_w['full']['bias']:+.3f}",
         f"{res_w['vn_range']['acc']:.3f}", f"{res_w['vn_range']['within_1']:.3f}", f"{res_w['vn_range']['bias']:+.3f}"],
        ["Joint copula", f"{ess_j:.0f}",
         f"{res_psis['full']['acc']:.3f}", f"{res_psis['full']['within_1']:.3f}", f"{res_psis['full']['bias']:+.3f}",
         f"{res_psis['vn_range']['acc']:.3f}", f"{res_psis['vn_range']['within_1']:.3f}", f"{res_psis['vn_range']['bias']:+.3f}"],
    ], ["Weighting", "ESS", "Full Acc", "Full W1", "Full Bias", "VN Acc", "VN W1", "VN Bias"])

    # ──────────────────────────────────────────────────────
    #  Ablation 6: VN Parameter Sensitivity
    # ──────────────────────────────────────────────────────

    print(f"\n\n{'#'*80}")
    print(f"  ABLATION 6: VN Target Parameter Sensitivity")
    print(f"{'#'*80}")

    # Baseline VN params: height=156.2 +/- 5.5, weight=53.9 +/- 8.0
    sensitivity_configs = {
        "baseline":      {"h_mean": 156.2, "h_std": 5.5, "w_mean": 53.9, "w_std": 8.0},
        "height_-1sig":  {"h_mean": 150.7, "h_std": 5.5, "w_mean": 53.9, "w_std": 8.0},
        "height_+1sig":  {"h_mean": 161.7, "h_std": 5.5, "w_mean": 53.9, "w_std": 8.0},
        "weight_-1sig":  {"h_mean": 156.2, "h_std": 5.5, "w_mean": 45.9, "w_std": 8.0},
        "weight_+1sig":  {"h_mean": 156.2, "h_std": 5.5, "w_mean": 61.9, "w_std": 8.0},
    }

    abl6 = {}
    sens_rows = []

    for config_name, params in sensitivity_configs.items():
        print(f"\n  Config: {config_name} (h={params['h_mean']}, w={params['w_mean']})")

        w_sens, d_sens = compute_weights_copula_custom(
            train_heights, train_weights_kg,
            vn_h_mean=params["h_mean"], vn_h_std=params["h_std"],
            vn_w_mean=params["w_mean"], vn_w_std=params["w_std"],
            apply_psis=True,
        )

        ess_sens = ess_from_weights(w_sens)
        print(f"    ESS={ess_sens:.0f} ({ess_sens/len(X_train)*100:.1f}%) k_hat={d_sens['k_hat']}")

        gbm_sens = train_gbm(X_train, y_train, sample_weight=w_sens, label=f"GBM {config_name}")
        y_pred_sens = gbm_sens.predict(X_test)
        res_sens = eval_full_and_vn(y_test, y_pred_sens, X_test, size_enc, f"GBM {config_name}")

        abl6[config_name] = {
            "params": params,
            "ess": ess_sens,
            "k_hat": d_sens["k_hat"],
            "full": res_sens["full"],
            "vn_range": res_sens["vn_range"],
        }

        sens_rows.append([
            config_name,
            f"{params['h_mean']:.1f}", f"{params['w_mean']:.1f}",
            f"{ess_sens:.0f}",
            f"{d_sens['k_hat']:.3f}" if d_sens["k_hat"] is not None else "N/A",
            f"{res_sens['full']['acc']:.3f}", f"{res_sens['full']['within_1']:.3f}",
            f"{res_sens['vn_range']['acc']:.3f}", f"{res_sens['vn_range']['within_1']:.3f}",
            f"{res_sens['vn_range']['bias']:+.3f}",
        ])

    all_results["ablation_6_vn_sensitivity"] = abl6

    print_table("Ablation 6: VN Parameter Sensitivity (GBM + copula+PSIS)", sens_rows,
                ["Config", "h_mean", "w_mean", "ESS", "k_hat",
                 "Full Acc", "Full W1", "VN Acc", "VN W1", "VN Bias"])

    # ──────────────────────────────────────────────────────
    #  Summary
    # ──────────────────────────────────────────────────────

    print(f"\n\n{'='*80}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*80}")

    print(f"\n  1. PSIS: ESS {ess_raw:.0f} -> {ess_psis:.0f} "
          f"(+{(ess_psis-ess_raw)/ess_raw*100:.1f}%)")
    print(f"     VN W1: {res_raw['vn_range']['within_1']:.3f} -> {res_psis['vn_range']['within_1']:.3f}")

    print(f"\n  2. Copula vs Independent:")
    print(f"     ESS: indep={ess_indep:.0f} copula={ess_psis:.0f}")
    print(f"     VN W1: indep={res_indep['vn_range']['within_1']:.3f} "
          f"copula={res_psis['vn_range']['within_1']:.3f}")

    print(f"\n  3-4. Loss comparison (VN-range W1):")
    print(f"     CE={res_ce['vn_range']['within_1']:.3f}  "
          f"CORAL={res_coral['vn_range']['within_1']:.3f}  "
          f"CORN={res_corn['vn_range']['within_1']:.3f}")

    print(f"\n  5. Single-feature (VN-range W1):")
    print(f"     Height={res_h['vn_range']['within_1']:.3f}  "
          f"Weight={res_w['vn_range']['within_1']:.3f}  "
          f"Joint={res_psis['vn_range']['within_1']:.3f}")

    # Sensitivity range
    vn_w1_values = [v["vn_range"]["within_1"] for v in abl6.values()]
    print(f"\n  6. VN sensitivity range: W1 in [{min(vn_w1_values):.3f}, {max(vn_w1_values):.3f}] "
          f"(spread={max(vn_w1_values)-min(vn_w1_values):.3f})")

    # ── Save JSON ──

    # Convert all float values for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = OUT_DIR / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)

    print(f"\n  Results saved to {results_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
