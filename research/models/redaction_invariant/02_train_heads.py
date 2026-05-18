"""Stage 2 — train attribute-classification heads on cached CLIP features.

Trains four head variants and writes their state-dicts + per-variant test metrics.
Each variant has separate heads for {neckline (8-way), silhouette (6-way),
sleeve (4-way), pattern (5-way)}.

Variants (with the λ in CE + λ·consistency_loss):
    redacted_only       — train on emb_red only, no consistency loss
    unredacted_only     — train on emb_unred only, no consistency loss
    paired_no_consist   — train on both inputs averaged, consistency λ=0
    paired_consist_lo   — train on both, λ=0.10
    paired_consist_md   — train on both, λ=0.50
    paired_consist_hi   — train on both, λ=1.00

Why these variants:
- redacted_only is the deployment baseline (we only have redacted images at inference)
- unredacted_only is the upper bound (full information)
- paired_with_consistency is the proposed method — uses redacted at inference but
  trains with a consistency loss that pulls red/unred embeddings together

A win for the paper: paired_with_consistency closes the gap toward unredacted_only
while using ONLY redacted inputs at test time.

Runtime: ~5 min total on A100 (data is in memory).
"""

# intent: train the 6 variants and dump test metrics for stage 3 evaluation
# status: ready to run after 01_extract_features.py completes
# next: run stage 3 to compare variants and produce the paper table
# confidence: high

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
FEAT = Path(__file__).parent / "features.npz"
PARQUET = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
OUT_DIR = Path(__file__).parent / "checkpoints"
OUT_DIR.mkdir(exist_ok=True, parents=True)
METRICS_OUT = Path(__file__).parent / "stage2_metrics.json"

# Attribute label spaces (match the parquet's categorical labels)
ATTR_SPACES = {
    "neckline":   ["v_neck","deep_v_neck","scoop","crew","boat","high_neck","sweetheart","off_shoulder"],
    "silhouette": ["bodycon","fitted","semi_fitted","a_line","loose","oversized"],
    "sleeve":     ["sleeveless","short","three_quarter","long"],
    "pattern":    ["solid","vertical_stripe","horizontal_stripe","small_print","large_print"],
}

VARIANTS = [
    ("redacted_only",      "single", 0.0),
    ("unredacted_only",    "single", 0.0),
    ("paired_no_consist",  "paired", 0.0),
    ("paired_consist_lo",  "paired", 0.10),
    ("paired_consist_md",  "paired", 0.50),
    ("paired_consist_hi",  "paired", 1.00),
]


class MultiHead(nn.Module):
    """One head per attribute, sharing a backbone embedding input."""
    def __init__(self, dim=512, hidden=256, label_spaces=ATTR_SPACES):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.2),
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden, len(vals))
            for name, vals in label_spaces.items()
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {name: head(h) for name, head in self.heads.items()}

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


def encode_label(name: str, value: str) -> int | None:
    space = ATTR_SPACES[name]
    return space.index(value) if value in space else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not FEAT.exists():
        sys.exit(f"!! {FEAT} not found — run 01_extract_features.py first")

    print(f"  loading features from {FEAT}…")
    data = np.load(FEAT, allow_pickle=True)
    gids = data["gids"]
    emb_red = data["emb_red"]
    emb_unred = data["emb_unred"]
    print(f"  features: gids={gids.shape}, emb shapes={emb_red.shape}")

    print(f"  loading labels from {PARQUET}…")
    df = pd.read_parquet(PARQUET, columns=[
        "garment_id", "neckline", "silhouette", "sleeve", "pattern",
    ])
    df = df.set_index("garment_id")

    # Align features to labels and encode labels
    rows = []
    keep_idx = []
    for i, gid in enumerate(gids):
        if gid not in df.index:
            continue
        r = df.loc[gid]
        encoded = {name: encode_label(name, r[name]) for name in ATTR_SPACES}
        if any(v is None for v in encoded.values()):
            continue
        rows.append(encoded)
        keep_idx.append(i)

    keep_idx = np.array(keep_idx)
    emb_red = emb_red[keep_idx]
    emb_unred = emb_unred[keep_idx]
    label_df = pd.DataFrame(rows)
    print(f"  aligned: {len(label_df):,} pairs with all 4 valid attribute labels")

    # 80/20 train/test split with fixed seed
    idx = np.arange(len(label_df))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=args.seed)
    print(f"  train n={len(train_idx):,}  test n={len(test_idx):,}")

    # ── Train each variant ──
    label_tensor = {name: torch.from_numpy(label_df[name].values.astype(np.int64))
                    for name in ATTR_SPACES}
    e_red_t = torch.from_numpy(emb_red).float()
    e_unr_t = torch.from_numpy(emb_unred).float()

    torch.manual_seed(args.seed)
    metrics_by_variant: dict[str, dict] = {}

    for variant_name, mode, lam in VARIANTS:
        print(f"\n  ── variant: {variant_name} (mode={mode}, λ={lam}) ──")
        model = MultiHead(dim=emb_red.shape[1]).to(args.device)
        opt = torch.optim.AdamW(model.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay)
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        train_e_r = e_red_t[train_idx].to(args.device)
        train_e_u = e_unr_t[train_idx].to(args.device)
        train_lbl = {n: label_tensor[n][train_idx].to(args.device)
                     for n in ATTR_SPACES}

        test_e_r = e_red_t[test_idx].to(args.device)
        test_e_u = e_unr_t[test_idx].to(args.device)
        test_lbl = {n: label_tensor[n][test_idx].numpy() for n in ATTR_SPACES}

        n_train = len(train_idx)
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(n_train, device=args.device)
            ep_losses = []
            for s in range(0, n_train, args.batch_size):
                b = perm[s:s + args.batch_size]
                if mode == "single" and variant_name == "redacted_only":
                    x = train_e_r[b]
                    logits = model(x)
                    loss = sum(ce(logits[n], train_lbl[n][b]) for n in ATTR_SPACES)
                elif mode == "single" and variant_name == "unredacted_only":
                    x = train_e_u[b]
                    logits = model(x)
                    loss = sum(ce(logits[n], train_lbl[n][b]) for n in ATTR_SPACES)
                else:  # paired
                    logits_r = model(train_e_r[b])
                    logits_u = model(train_e_u[b])
                    loss_ce = sum(
                        ce(logits_r[n], train_lbl[n][b])
                        + ce(logits_u[n], train_lbl[n][b])
                        for n in ATTR_SPACES
                    )
                    if lam > 0:
                        feat_r = model.features(train_e_r[b])
                        feat_u = model.features(train_e_u[b])
                        loss_consist = mse(feat_r, feat_u)
                        loss = loss_ce + lam * loss_consist
                    else:
                        loss = loss_ce
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_losses.append(loss.item())
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f"   epoch {ep+1:2d}/{args.epochs}  mean loss = {np.mean(ep_losses):.4f}")

        # ── Evaluate (always on REDACTED test embeddings — that's the deployment scenario) ──
        model.eval()
        with torch.inference_mode():
            logits_test = model(test_e_r)
        per_attr_f1 = {}
        for n in ATTR_SPACES:
            preds = logits_test[n].argmax(dim=-1).cpu().numpy()
            per_attr_f1[n] = float(f1_score(test_lbl[n], preds, average="macro"))
        avg = float(np.mean(list(per_attr_f1.values())))

        # Also evaluate on UNREDACTED test embeddings (the "upper-bound oracle" scenario)
        with torch.inference_mode():
            logits_test_un = model(test_e_u)
        per_attr_f1_un = {}
        for n in ATTR_SPACES:
            preds = logits_test_un[n].argmax(dim=-1).cpu().numpy()
            per_attr_f1_un[n] = float(f1_score(test_lbl[n], preds, average="macro"))
        avg_un = float(np.mean(list(per_attr_f1_un.values())))

        metrics_by_variant[variant_name] = {
            "lambda": lam,
            "mode": mode,
            "per_attribute_f1_redacted_eval": per_attr_f1,
            "avg_f1_redacted_eval":           avg,
            "per_attribute_f1_unredacted_eval": per_attr_f1_un,
            "avg_f1_unredacted_eval":           avg_un,
        }
        print(f"   redacted eval  avg F1 = {avg:.4f}")
        print(f"   unredacted eval avg F1 = {avg_un:.4f}")

        ckpt = OUT_DIR / f"{variant_name}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"   wrote {ckpt}")

    METRICS_OUT.write_text(json.dumps(metrics_by_variant, indent=2, default=float))
    print(f"\n  wrote {METRICS_OUT}")


if __name__ == "__main__":
    main()
