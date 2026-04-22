"""
Train 6 size recommendation model variants for cross-population comparison.

intent: compare all weighting × loss combinations to isolate each contribution
status: done
next: run eval_all_variants.py to evaluate on population subsets
confidence: high

Variants:
  1. GBM + no weights          (baseline)
  2. GBM + indep. Gaussian     (old method)
  3. GBM + copula+PSIS         (improved weighting)
  4. MLP + CE + no weights     (neural baseline)
  5. MLP + CE + copula+PSIS    (neural + improved weighting)
  6. MLP + CORN + copula+PSIS  (proposed: neural + improved weighting + ordinal loss)

References:
  - Shi & Raschka (2023) PAA — CORN ordinal regression
  - Cao et al. (2020) PRL — CORAL ordinal regression
  - Ferrari Dacrema et al. (2019) RecSys — proper baseline comparison

Usage:
    python research/models/train_size_corn.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "research" / "datasets" / "processed"
WEIGHT_DIR = ROOT / "research" / "weighting"
OUT_DIR = ROOT / "research" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE_MAP = {
    0: "XXS", 1: "XXS", 2: "XS", 3: "XS", 4: "S",
    5: "S", 6: "S", 7: "M", 8: "M", 9: "M",
    10: "L", 11: "L", 12: "L", 13: "L",
    14: "XL", 15: "XL", 16: "XL",
    17: "XXL", 18: "XXL", 19: "XXL", 20: "XXL",
}
SIZE_ORDER = {"XXS": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5, "XXL": 6}
NUM_CLASSES = 7


# ══════════════════════════════════════════════════════════
#  CORN Loss (Shi & Raschka, PAA 2023)
# ══════════════════════════════════════════════════════════

class CORNLoss(nn.Module):
    """
    Conditional Ordinal Regression for Neural Networks.

    Decomposes K-class ordinal problem into K-1 conditional binary tasks:
      P(Y > k | Y > k-1) for k = 1, ..., K-1

    Each task has its own output head (NO weight sharing — key improvement
    over CORAL which shares penultimate layer weights).

    Final class probability:
      P(Y = k) = P(Y > k-1) * prod_{j<k} P(Y > j | Y > j-1) - P(Y > k)
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets, sample_weights=None):
        """
        Args:
            logits: (batch, num_classes - 1) — raw logits for each conditional task
            targets: (batch,) — integer class labels 0..K-1
            sample_weights: (batch,) — importance weights (optional)

        Returns:
            Weighted mean of conditional binary cross-entropy losses
        """
        batch_size = logits.size(0)
        num_tasks = self.num_classes - 1
        total_loss = torch.zeros(batch_size, device=logits.device)

        for task_idx in range(num_tasks):
            # For task k: only use samples where Y >= k (conditional on Y > k-1)
            mask = targets >= task_idx
            if mask.sum() == 0:
                continue

            # Binary target: 1 if Y > task_idx (i.e., Y >= task_idx + 1)
            binary_target = (targets[mask] > task_idx).float()
            task_logit = logits[mask, task_idx]

            # Binary cross-entropy
            bce = nn.functional.binary_cross_entropy_with_logits(
                task_logit, binary_target, reduction='none'
            )

            # Accumulate into total_loss for masked samples
            total_loss[mask] = total_loss[mask] + bce

        if sample_weights is not None:
            return (total_loss * sample_weights).mean()
        return total_loss.mean()


def corn_predict(logits):
    """
    Convert CORN logits to class predictions.

    P(Y > k) = prod_{j=0}^{k} sigmoid(logit_j)
    P(Y = k) = P(Y > k-1) - P(Y > k)
    """
    probs_gt = torch.sigmoid(logits)  # (batch, K-1)

    # P(Y > k) = cumulative product of conditional probs
    cum_probs = torch.cumprod(probs_gt, dim=1)  # (batch, K-1)

    # P(Y = 0) = 1 - P(Y > 0)
    # P(Y = k) = P(Y > k-1) - P(Y > k) for k = 1..K-2
    # P(Y = K-1) = P(Y > K-2)
    class_probs = torch.zeros(logits.size(0), logits.size(1) + 1, device=logits.device)
    class_probs[:, 0] = 1.0 - cum_probs[:, 0]
    for k in range(1, logits.size(1)):
        class_probs[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
    class_probs[:, -1] = cum_probs[:, -1]

    # Clamp to avoid negative probabilities from numerical issues
    class_probs = torch.clamp(class_probs, min=0.0)

    return class_probs.argmax(dim=1)


# ══════════════════════════════════════════════════════════
#  MLP Model
# ══════════════════════════════════════════════════════════

class SizeMLP(nn.Module):
    """Simple MLP for size prediction. Output depends on loss type."""

    def __init__(self, input_dim, hidden_dim=128, num_classes=7, loss_type="ce"):
        super().__init__()
        self.loss_type = loss_type
        out_dim = num_classes if loss_type == "ce" else num_classes - 1

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════

def load_data():
    """Load and prepare RTR data. Returns X, y, encoders, and the fit dataframe."""
    df = pd.read_parquet(DATA_DIR / "rtr_body_fit.parquet")
    df = df[
        df["height_cm"].notna()
        & df["weight_kg"].notna()
        & df["fit"].isin(["fit", "small", "large"])
    ].copy()
    df_fit = df[df["fit"] == "fit"].copy()
    df_fit["size_label"] = df_fit["size"].map(SIZE_MAP)
    df_fit = df_fit[df_fit["size_label"].notna()].copy()

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

    return X, size_labels, body_enc, cat_enc, size_enc, df_fit


def load_weights(n_train):
    """Load precomputed importance weights. Must match training set size."""
    variants = {}
    for name in ["uniform", "independent_gaussian", "copula_psis"]:
        path = WEIGHT_DIR / f"weights_{name}.npy"
        if path.exists():
            w = np.load(path)
            # Weights are computed on full fit dataset — need to subset to training split
            variants[name] = w
        else:
            print(f"  Warning: {path} not found, skipping {name}")
    return variants


# ══════════════════════════════════════════════════════════
#  Training Functions
# ══════════════════════════════════════════════════════════

def train_gbm(X_train, y_train, sample_weight=None, label=""):
    """Train GradientBoosting classifier."""
    print(f"\n  Training GBM: {label}...")
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=20, random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_mlp(X_train, y_train, sample_weight=None, loss_type="ce",
              epochs=50, lr=1e-3, label=""):
    """Train MLP with CE or CORN loss."""
    print(f"\n  Training MLP ({loss_type}): {label}...")

    input_dim = X_train.shape[1]
    model = SizeMLP(input_dim, hidden_dim=128, num_classes=NUM_CLASSES, loss_type=loss_type)

    # Normalize features
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_norm = (X_train - X_mean) / X_std

    X_t = torch.FloatTensor(X_norm)
    y_t = torch.LongTensor(y_train)
    w_t = torch.FloatTensor(sample_weight) if sample_weight is not None else None

    dataset = TensorDataset(X_t, y_t) if w_t is None else TensorDataset(X_t, y_t, w_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = CORNLoss(NUM_CLASSES)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

            if loss_type == "ce":
                loss_per_sample = criterion(logits, yb)
                if wb is not None:
                    loss = (loss_per_sample * wb).mean()
                else:
                    loss = loss_per_sample.mean()
            else:
                loss = criterion(logits, yb, sample_weights=wb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}")

    return model, X_mean, X_std


def predict_mlp(model, X, X_mean, X_std, loss_type="ce"):
    """Get predictions from trained MLP."""
    model.eval()
    X_norm = (X - X_mean) / X_std
    X_t = torch.FloatTensor(X_norm)
    with torch.no_grad():
        logits = model(X_t)
        if loss_type == "ce":
            preds = logits.argmax(dim=1).numpy()
        else:
            preds = corn_predict(logits).numpy()
    return preds


# ══════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, size_enc, label=""):
    """Quick evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    y_true_labels = size_enc.inverse_transform(y_true)
    y_pred_labels = size_enc.inverse_transform(y_pred)
    within_1 = np.mean([
        abs(SIZE_ORDER.get(t, 3) - SIZE_ORDER.get(p, 3)) <= 1
        for t, p in zip(y_true_labels, y_pred_labels)
    ])
    mae = np.mean([
        abs(SIZE_ORDER.get(t, 3) - SIZE_ORDER.get(p, 3))
        for t, p in zip(y_true_labels, y_pred_labels)
    ])
    bias = np.mean([
        SIZE_ORDER.get(p, 3) - SIZE_ORDER.get(t, 3)
        for t, p in zip(y_true_labels, y_pred_labels)
    ])
    print(f"  {label}: top1={acc:.3f} w1={within_1:.3f} mae={mae:.3f} bias={bias:+.3f}")
    return {"acc": acc, "within_1": within_1, "mae": mae, "bias": bias}


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Training 6 Size Recommendation Variants")
    print("=" * 60)

    X, y, body_enc, cat_enc, size_enc, df_fit = load_data()
    print(f"Total fit samples: {len(X):,}")

    # Split — same seed as original
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Compute weights directly on training data (not from precomputed files,
    # because dataset size changes after filtering valid size labels)
    sys.path.insert(0, str(ROOT))
    from research.weighting.copula_psis import (
        compute_weights_independent,
        compute_weights_copula,
    )

    train_heights = X_train[:, 0]
    train_weights_kg = X_train[:, 1]

    print("\nComputing importance weights on training set...")
    w_indep, d_indep = compute_weights_independent(train_heights, train_weights_kg)
    print(f"  Independent Gaussian: ESS={d_indep['ess']:.0f} ({d_indep['ess_pct']:.1f}%)")

    w_copula, d_copula = compute_weights_copula(train_heights, train_weights_kg, apply_psis=True)
    print(f"  Copula+PSIS: ESS={d_copula['ess']:.0f} ({d_copula['ess_pct']:.1f}%) k_hat={d_copula['k_hat']:.3f}")

    train_weights = {
        "uniform": np.ones(len(X_train)),
        "independent_gaussian": w_indep,
        "copula_psis": w_copula,
    }

    # ── Train all 6 variants ──

    models = {}
    results = {}

    # 1. GBM + no weights
    models["gbm_uniform"] = train_gbm(X_train, y_train, label="GBM (no weights)")
    y_pred = models["gbm_uniform"].predict(X_test)
    results["gbm_uniform"] = evaluate(y_test, y_pred, size_enc, "GBM (no weights)")

    # 2. GBM + independent Gaussian
    if "independent_gaussian" in train_weights:
        w = train_weights["independent_gaussian"]
        models["gbm_indep"] = train_gbm(X_train, y_train, sample_weight=w, label="GBM (indep gaussian)")
        y_pred = models["gbm_indep"].predict(X_test)
        results["gbm_indep"] = evaluate(y_test, y_pred, size_enc, "GBM (indep gaussian)")

    # 3. GBM + copula+PSIS
    if "copula_psis" in train_weights:
        w = train_weights["copula_psis"]
        models["gbm_copula"] = train_gbm(X_train, y_train, sample_weight=w, label="GBM (copula+PSIS)")
        y_pred = models["gbm_copula"].predict(X_test)
        results["gbm_copula"] = evaluate(y_test, y_pred, size_enc, "GBM (copula+PSIS)")

    # 4. MLP + CE + no weights
    mlp_ce, mu4, sig4 = train_mlp(X_train, y_train, loss_type="ce", label="MLP+CE (no weights)")
    y_pred = predict_mlp(mlp_ce, X_test, mu4, sig4, "ce")
    results["mlp_ce_uniform"] = evaluate(y_test, y_pred, size_enc, "MLP+CE (no weights)")
    models["mlp_ce_uniform"] = (mlp_ce, mu4, sig4)

    # 5. MLP + CE + copula+PSIS
    if "copula_psis" in train_weights:
        w = train_weights["copula_psis"]
        mlp_ce_w, mu5, sig5 = train_mlp(X_train, y_train, sample_weight=w, loss_type="ce",
                                         label="MLP+CE (copula+PSIS)")
        y_pred = predict_mlp(mlp_ce_w, X_test, mu5, sig5, "ce")
        results["mlp_ce_copula"] = evaluate(y_test, y_pred, size_enc, "MLP+CE (copula+PSIS)")
        models["mlp_ce_copula"] = (mlp_ce_w, mu5, sig5)

    # 6. MLP + CORN + copula+PSIS (PROPOSED)
    if "copula_psis" in train_weights:
        w = train_weights["copula_psis"]
        mlp_corn, mu6, sig6 = train_mlp(X_train, y_train, sample_weight=w, loss_type="corn",
                                         epochs=80, label="MLP+CORN (copula+PSIS)")
        y_pred = predict_mlp(mlp_corn, X_test, mu6, sig6, "corn")
        results["mlp_corn_copula"] = evaluate(y_test, y_pred, size_enc, "MLP+CORN (copula+PSIS)")
        models["mlp_corn_copula"] = (mlp_corn, mu6, sig6)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Full Test Set Results")
    print(f"{'='*70}")
    print(f"  {'Variant':<30} {'Top-1':>7} {'W-1':>7} {'MAE':>7} {'Bias':>7}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for name, r in results.items():
        print(f"  {name:<30} {r['acc']:>6.1%} {r['within_1']:>6.1%} {r['mae']:>7.3f} {r['bias']:>+6.3f}")

    # ── Save everything ──
    save_dir = OUT_DIR / "variants"
    save_dir.mkdir(exist_ok=True)

    # Save GBM models
    for name in ["gbm_uniform", "gbm_indep", "gbm_copula"]:
        if name in models:
            bundle = {
                "model": models[name],
                "body_enc": body_enc,
                "cat_enc": cat_enc,
                "size_enc": size_enc,
                "results": results.get(name),
            }
            with open(save_dir / f"{name}.pkl", "wb") as f:
                pickle.dump(bundle, f)

    # Save MLP models
    for name in ["mlp_ce_uniform", "mlp_ce_copula", "mlp_corn_copula"]:
        if name in models:
            model_tuple = models[name]
            torch.save({
                "state_dict": model_tuple[0].state_dict(),
                "X_mean": model_tuple[1],
                "X_std": model_tuple[2],
                "loss_type": "corn" if "corn" in name else "ce",
                "results": results.get(name),
            }, save_dir / f"{name}.pt")

    # Save results
    with open(save_dir / "results_summary.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)

    # Save encoders and split indices for downstream evaluation
    with open(save_dir / "eval_data.pkl", "wb") as f:
        pickle.dump({
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "idx_train": idx_train, "idx_test": idx_test,
            "body_enc": body_enc, "cat_enc": cat_enc, "size_enc": size_enc,
            "train_weights": train_weights,
        }, f)

    print(f"\nModels and results saved to {save_dir}/")


if __name__ == "__main__":
    main()
