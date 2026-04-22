"""Train the PyTorch MLP for size recommendation.

Replaces the sklearn GradientBoosting model with a PyTorch MLP
that accepts body embedding features for more accurate prediction.

Usage:
    python training/scripts/train_size_mlp.py --config training/configs/size_mlp.yaml
"""

# intent: train size MLP with ordinal cross-entropy and importance weighting
# status: done
# next: run after body embedding model is trained
# confidence: high

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path("services/recommendation")))

SIZE_LABELS = ["XXS", "XS", "S", "M", "L", "XL", "XXL"]
SIZE_MAP = {
    0: "XXS", 1: "XXS", 2: "XS", 3: "XS", 4: "S",
    5: "S", 6: "S", 7: "M", 8: "M",
    9: "M", 10: "L", 11: "L", 12: "L",
    13: "L", 14: "XL", 15: "XL", 16: "XL",
    17: "XXL", 18: "XXL", 19: "XXL", 20: "XXL",
}


def ordinal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lambda_ord: float = 0.1,
) -> torch.Tensor:
    """Ordinal cross-entropy loss.

    L = -sum(y_k * log(y_hat_k)) + lambda_ord * sum((k - k*)^2 * y_hat_k)
    """
    ce = F.cross_entropy(logits, targets, reduction="none")
    probs = F.softmax(logits, dim=-1)
    k = torch.arange(logits.size(1), device=logits.device, dtype=torch.float32)
    k_star = targets.float().unsqueeze(1)
    ordinal = ((k.unsqueeze(0) - k_star) ** 2 * probs).sum(dim=1)
    return (ce + lambda_ord * ordinal).mean()


def train(config_path: str):
    """Run MLP training."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # Load data
    df = pd.read_parquet(config["data"]["dataset"])
    df = df[df["height_cm"].notna() & df["weight_kg"].notna()].copy()
    df = df[df["fit"] == "fit"].copy()
    df["size_label"] = df["size"].map(SIZE_MAP)
    df = df[df["size_label"].notna()].copy()

    label_to_idx = {s: i for i, s in enumerate(SIZE_LABELS)}
    df["size_idx"] = df["size_label"].map(label_to_idx)

    # Build features
    df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)
    df["waist_hip_ratio"] = df.get("waist_hip_ratio", pd.Series(0.8, index=df.index))
    df["shoulder_hip_ratio"] = df.get("shoulder_hip_ratio", pd.Series(1.1, index=df.index))

    feature_cols = ["height_cm", "weight_kg", "bmi", "age", "waist_hip_ratio", "shoulder_hip_ratio"]
    X_body = df[feature_cols].fillna(0).values.astype(np.float32)

    # Pad with zero body embedding (128 dims) until embedding model is trained
    X_embed = np.zeros((len(df), 128), dtype=np.float32)
    X = np.hstack([X_body, X_embed])
    y = df["size_idx"].values.astype(np.int64)

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_split"], stratify=y, random_state=42
    )

    # Importance weighting (Vietnamese population)
    sample_weights = np.ones(len(X_train), dtype=np.float32)
    iw = config["training"].get("importance_weighting", {})
    if iw.get("enabled", False):
        heights = X_train[:, 0]
        weights = X_train[:, 1]
        bmis = X_train[:, 2]

        w_h = np.clip(
            stats.norm.pdf(heights, iw["vn_height_mean"], iw["vn_height_std"]) /
            (stats.norm.pdf(heights, heights.mean(), heights.std()) + 1e-10),
            iw["clip_min"], iw["clip_max"]
        )
        w_w = np.clip(
            stats.norm.pdf(weights, iw["vn_weight_mean"], iw["vn_weight_std"]) /
            (stats.norm.pdf(weights, weights.mean(), weights.std()) + 1e-10),
            iw["clip_min"], iw["clip_max"]
        )
        sample_weights = (w_h * w_w).astype(np.float32)
        sample_weights /= sample_weights.mean()

        eff_n = (sample_weights.sum() ** 2) / (sample_weights ** 2).sum()
        logger.info("Importance weighting: ESS = %.0f / %d", eff_n, len(X_train))

    # DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(sample_weights),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"])

    # Model
    from src.size_mlp import SizeMLP
    model_cfg = config["model"]
    model = SizeMLP(
        input_dim=model_cfg.get("input_dim", 134),
        hidden_dims=model_cfg.get("hidden_dims", [256, 256, 128]),
        output_dim=model_cfg.get("output_dim", 7),
        dropout=model_cfg.get("dropout", [0.3, 0.3, 0.2]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    lambda_ord = config["training"]["loss"]["lambda_ordinal"]

    # Training loop
    best_acc = 0.0
    checkpoint_dir = Path(config["checkpoints"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)

            logits = model(batch_x)
            loss = ordinal_cross_entropy(logits, batch_y, lambda_ord)
            # Apply importance weights
            weighted_loss = (loss * batch_w.mean())

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item()

        scheduler.step()

        # Evaluate
        if epoch % 10 == 0:
            model.eval()
            correct = 0
            within1 = 0
            total = 0

            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    logits = model(batch_x)
                    preds = logits.argmax(dim=1)
                    correct += (preds == batch_y).sum().item()
                    within1 += (torch.abs(preds - batch_y) <= 1).sum().item()
                    total += len(batch_y)

            acc = correct / total
            w1 = within1 / total
            avg_loss = total_loss / len(train_loader)

            logger.info(
                "Epoch %d/%d — Loss: %.4f, Acc: %.3f, W1: %.3f",
                epoch, config["training"]["epochs"], avg_loss, acc, w1
            )

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), checkpoint_dir / "best.pt")

    # Save final
    torch.save(model.state_dict(), checkpoint_dir / "final.pt")
    logger.info("Training complete. Best accuracy: %.3f", best_acc)


def main():
    parser = argparse.ArgumentParser(description="Train Size MLP")
    parser.add_argument("--config", type=str, default="training/configs/size_mlp.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train(args.config)


if __name__ == "__main__":
    main()
