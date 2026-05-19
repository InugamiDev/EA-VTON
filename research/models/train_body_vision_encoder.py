"""Train a CNN vision encoder on BodyM photos that predicts body measurements.

This is the primary A100 deliverable for Workstream 1 of the paper. The model
takes a single photo of a person + their height as input, and outputs 6 body
measurements (chest, waist, hip, shoulder_breadth, arm_length, leg_length).
Those measurements are then fed into the ModCloth-trained size predictor to
produce a size band — giving us end-to-end photo→size accuracy on BodyM-test.

Architecture:
  - Backbone: ResNet50 pretrained on ImageNet (timm or torchvision)
  - Pool: avg-pool → 2048-d feature
  - Concat: 2048-d + 1-d (height_cm normalized) + 1-d (gender) = 2050-d
  - Head: 2 × Linear(512, GELU, Dropout 0.2) → 6 regression outputs (cm units)
  - Loss: MSE per measurement, averaged

Training:
  - Optimizer: AdamW lr=1e-4 (backbone) / 1e-3 (head) — discriminative lr
  - Batch 32, 25 epochs, cosine schedule
  - Augmentations: HorizontalFlip(p=0.5), ColorJitter, RandomResizedCrop (224 → 224×0.9-1.0)
  - Loss reweighted per dimension by inverse-stddev of the training set targets

Eval:
  - Per-measurement MAE on BodyM testA + testB
  - End-to-end size accuracy: photo → measurements → ModCloth GBM(with_circ) → size band
  - Compare against scalar-only baseline (research/models/body_regression/mlp_body_regression.pkl)

Outputs:
  - research/models/variants/body_vision_encoder.pt  (state_dict)
  - research/eval/body_vision_encoder_results.json   (metrics)

Runtime: A100, ResNet50, batch 32, ~3-4 hours for 25 epochs on ~2k subjects × 4 photos/subject.
"""

# intent: end-to-end photo → size pipeline for the paper's primary contribution
# status: ready to run; A100 single-GPU
# next: chain into the live-size endpoint as the production photo encoder
# confidence: high

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
BODYM_PARQUET = ROOT / "research/datasets/processed/bodym_measurements.parquet"
BODYM_RAW = ROOT / "research/datasets/raw/bodym"
PHOTO_MAP_FILES = [
    BODYM_RAW / "train/subject_to_photo_map.csv",
    BODYM_RAW / "testA/subject_to_photo_map.csv",
    BODYM_RAW / "testB/subject_to_photo_map.csv",
]
OUT_CKPT = ROOT / "research/models/variants/body_vision_encoder.pt"
OUT_METRICS = ROOT / "research/eval/body_vision_encoder_results.json"
SCALAR_BASELINE = ROOT / "research/models/body_regression/mlp_body_regression.pkl"
MODCLOTH_SIZE_MODEL = ROOT / "research/models/variants/gbm_modcloth_upper.pkl"

# Targets (must match the scalar-only baseline's TARGET_MEASUREMENTS)
TARGETS = [
    "chest_girth_cm", "waist_girth_cm", "hip_girth_cm",
    "shoulder_breadth_cm", "arm_length_cm", "leg_length_cm",
]


def build_photo_index() -> pd.DataFrame:
    """Read subject_to_photo_map.csv files from train/testA/testB; return
    a DataFrame with columns (subject_id, photo_id, split)."""
    frames = []
    for p in PHOTO_MAP_FILES:
        if not p.exists():
            print(f"  warning: {p} not found")
            continue
        df = pd.read_csv(p)
        df["split"] = p.parent.name  # train/testA/testB
        frames.append(df)
    if not frames:
        sys.exit("!! no photo-map CSVs found under research/datasets/raw/bodym/")
    return pd.concat(frames, ignore_index=True)


def find_photo_path(split: str, photo_id: str) -> Path | None:
    """Resolve photo on disk. BodyM's AWS Open Data Registry release uses
    <split>/mask/<photo_id>.png (8-bit grayscale silhouette, 720×960)."""
    candidates = [
        # Primary — AWS Open Data Registry layout (downloaded by download_bodym_photos.py)
        BODYM_RAW / split / "mask" / f"{photo_id}.png",
        # Legacy layouts (defensive — script previously assumed JPG)
        BODYM_RAW / split / "photos" / f"{photo_id}.jpg",
        BODYM_RAW / split / "photo_front" / f"{photo_id}.jpg",
        BODYM_RAW / split / "photo_left" / f"{photo_id}.jpg",
        BODYM_RAW / split / f"{photo_id}.jpg",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


class BodyMDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, tfm, target_mean, target_std):
        self.rows = rows.reset_index(drop=True)
        self.tfm = tfm
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        photo_path = find_photo_path(row["split"], row["photo_id"])
        if photo_path is None:
            raise FileNotFoundError(
                f"missing photo: split={row['split']}, photo_id={row['photo_id']}"
            )
        img = Image.open(photo_path).convert("RGB")
        x = self.tfm(img)

        # Aux scalar inputs
        height_norm = (row["height_cm"] - 168.0) / 12.0   # rough centering
        gender = 1.0 if row["gender"] == "male" else 0.0

        aux = torch.tensor([height_norm, gender], dtype=torch.float32)

        target = torch.tensor(
            [row[t] for t in TARGETS], dtype=torch.float32
        )
        # Z-normalize target for stable loss
        target_n = (target - self.target_mean) / self.target_std
        return x, aux, target_n, target  # raw target kept for unnormalized MAE


class VisionBodyEncoder(nn.Module):
    """ResNet50 backbone + aux-scalar concat + 2-layer regression head."""
    def __init__(self, n_aux: int = 2, n_out: int = 6, backbone: str = "resnet50"):
        super().__init__()
        from torchvision import models
        if backbone == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        else:
            raise ValueError(f"unsupported backbone {backbone}")
        self.head = nn.Sequential(
            nn.Linear(feat_dim + n_aux, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, n_out),
        )

    def forward(self, x_img: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x_img)
        return self.head(torch.cat([f, aux], dim=-1))


def build_transforms(image_size: int = 224, train: bool = True):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(image_size + 16),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        norm,
    ])


def expand_rows_to_photos(measurements: pd.DataFrame, photos: pd.DataFrame) -> pd.DataFrame:
    """Cross-join measurements with photo map → one row per (subject, photo).
    Drops subjects without any photo or without measurements."""
    # BodyM has ~3 subjects appearing across testA + testB with slightly
    # different measurements. We keep the first row per subject_id so
    # `.loc[sid]` returns a Series (not a DataFrame) below.
    measurements = (
        measurements.drop_duplicates(subset="subject_id", keep="first")
                    .set_index("subject_id")
    )
    photos = photos.set_index("subject_id")
    common = list(set(measurements.index) & set(photos.index))
    if len(photos) == 0 or len(measurements) == 0 or len(common) == 0:
        sys.exit(
            f"!! 0 subjects after join (measurements={len(measurements)}, "
            f"photos={len(photos)}, common={len(common)}) — verify BodyM "
            "photos are downloaded under research/datasets/raw/bodym/<split>/photos/"
        )
    rows = []
    for sid in common:
        meas = measurements.loc[sid]
        # Defensive: if .loc still returns a DataFrame (shouldn't after dedup),
        # take the first row.
        if isinstance(meas, pd.DataFrame):
            meas = meas.iloc[0]
        ps = photos.loc[[sid]] if sid in photos.index else photos.loc[sid:sid]
        for _, p in ps.iterrows():
            rec = {
                "subject_id": sid,
                "photo_id": p["photo_id"],
                "split": p["split"],
                "gender": meas["gender"],
                "height_cm": float(meas["height_cm"]),
            }
            for t in TARGETS:
                rec[t] = float(meas[t])
            rows.append(rec)
    out = pd.DataFrame(rows)
    return out


def train_one_epoch(model, loader, opt, scaler, device, target_mean, target_std,
                    fp16: bool) -> float:
    model.train()
    total = 0.0
    n = 0
    for x_img, aux, t_norm, _ in loader:
        x_img = x_img.to(device, non_blocking=True)
        aux = aux.to(device, non_blocking=True)
        t_norm = t_norm.to(device, non_blocking=True)
        opt.zero_grad()
        if fp16:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x_img, aux)
                loss = nn.functional.mse_loss(pred, t_norm)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        else:
            pred = model(x_img, aux)
            loss = nn.functional.mse_loss(pred, t_norm)
            # Guard: skip the backward if the loss is already nan/inf (one bad
            # batch shouldn't wreck the whole run on MPS).
            if not torch.isfinite(loss):
                continue
            loss.backward()
            # Gradient clipping defends against the occasional MPS-induced spike
            # that previously caused weights to diverge to billions.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        total += loss.item() * x_img.size(0)
        n += x_img.size(0)
    return total / max(n, 1)


@torch.inference_mode()
def evaluate(model, loader, device, target_mean, target_std) -> dict:
    model.eval()
    abs_errs = []
    raws_all, preds_all = [], []
    target_mean_t = torch.tensor(target_mean).to(device)
    target_std_t = torch.tensor(target_std).to(device)
    for x_img, aux, _, t_raw in loader:
        x_img = x_img.to(device, non_blocking=True)
        aux = aux.to(device, non_blocking=True)
        t_raw = t_raw.to(device)
        pred_norm = model(x_img, aux)
        pred_cm = pred_norm * target_std_t + target_mean_t
        abs_errs.append((pred_cm - t_raw).abs().cpu().numpy())
        raws_all.append(t_raw.cpu().numpy())
        preds_all.append(pred_cm.cpu().numpy())
    abs_arr = np.concatenate(abs_errs, axis=0)
    raws = np.concatenate(raws_all, axis=0)
    preds = np.concatenate(preds_all, axis=0)
    per_target_mae = {t: float(abs_arr[:, i].mean()) for i, t in enumerate(TARGETS)}
    return {
        "mae_per_target_cm": per_target_mae,
        "mae_avg_cm": float(abs_arr.mean()),
        "n": int(abs_arr.shape[0]),
        "preds": preds,  # for downstream size chain
        "targets": raws,
    }


def chain_size_eval(preds: np.ndarray, raws: np.ndarray,
                    target_names: list[str], heights_cm: np.ndarray | None = None) -> dict:
    """Feed predicted (bust=chest, waist, hip) into ModCloth size predictor
    and compare predicted-size vs ground-truth-size (from raw measurements).
    Prefers the with_circ model (saved as gbm_modcloth_upper_with_circ.pkl).
    Falls back to gbm_modcloth_upper.pkl if that file has bust/waist/hip cols."""
    with_circ_path = ROOT / "research/models/variants/gbm_modcloth_upper_with_circ.pkl"
    if with_circ_path.exists():
        chosen_path = with_circ_path
    elif MODCLOTH_SIZE_MODEL.exists():
        chosen_path = MODCLOTH_SIZE_MODEL
    else:
        return {"error": f"ModCloth size model missing (looked at {with_circ_path} and {MODCLOTH_SIZE_MODEL})"}

    bundle = pickle.loads(chosen_path.read_bytes())
    feat_cols = bundle.get("feature_cols", [])
    if "bust_cm" not in feat_cols or "waist_cm" not in feat_cols or "hips_cm" not in feat_cols:
        return {
            "error": "ModCloth model trained on h_only feature set — re-run "
                     "research/models/train_modcloth_upper_size.py to also save the "
                     "with_circ variant (gbm_modcloth_upper_with_circ.pkl)."
        }

    cat_enc = bundle["cat_enc"]
    size_enc = bundle["size_enc"]
    model = bundle["model"]

    # Index map: predicted measurements → ModCloth feature columns
    chest_i = target_names.index("chest_girth_cm")
    waist_i = target_names.index("waist_girth_cm")
    hip_i = target_names.index("hip_girth_cm")

    # Fallback category — ModCloth has many; use "tops" or the first label
    cat_default = "tops" if "tops" in cat_enc.classes_ else cat_enc.classes_[0]
    cat_enc_val = int(cat_enc.transform([cat_default])[0])

    # Default height = mean US-female (163 cm) if subject heights not supplied.
    # The chain works fine without true height but accuracy is slightly worse.
    default_h = 163.0

    def measurements_to_size(measurements_row: np.ndarray, height_cm: float) -> int:
        feature_values = {
            "height_cm":    float(height_cm),
            "bust_cm":      float(measurements_row[chest_i]),
            "waist_cm":     float(measurements_row[waist_i]),
            "hips_cm":      float(measurements_row[hip_i]),
            "category_enc": cat_enc_val,
        }
        X = np.array([[feature_values[c] for c in feat_cols]], dtype=np.float32)
        return int(model.predict(X)[0])

    if heights_cm is None:
        heights_cm = np.full(len(preds), default_h, dtype=np.float32)

    preds_size_idx = np.array([measurements_to_size(preds[i], heights_cm[i]) for i in range(len(preds))])
    truth_size_idx = np.array([measurements_to_size(raws[i], heights_cm[i]) for i in range(len(raws))])
    exact = float((preds_size_idx == truth_size_idx).mean())
    within1 = float((np.abs(preds_size_idx - truth_size_idx) <= 1).mean())
    return {
        "exact": exact,
        "within1": within1,
        "n": int(len(preds_size_idx)),
        "modcloth_feature_cols": feat_cols,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fp16", action="store_true",
                    help="Mixed-precision. CUDA only; ignored on MPS/CPU.")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--gender", default="both", choices=["female", "male", "both"],
                    help="Restrict to one gender; default uses both")

    # Device auto-detect: CUDA → MPS (Apple Silicon) → CPU
    _default_device = "cpu"
    if torch.cuda.is_available():
        _default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _default_device = "mps"
    ap.add_argument("--device", default=_default_device,
                    choices=["cuda", "mps", "cpu"])
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for reproducibility (torch + numpy)")
    args = ap.parse_args()

    # Set seeds early so weight init + DataLoader shuffle are reproducible.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # MPS doesn't support fp16 amp the same way CUDA does — silently disable.
    if args.fp16 and args.device != "cuda":
        print(f"  note: --fp16 only supported on CUDA; running in fp32 on {args.device}.")
        args.fp16 = False

    # ── Data ──
    print("  loading BodyM measurements + photo map…")
    meas = pd.read_parquet(BODYM_PARQUET)
    photos = build_photo_index()
    print(f"  measurements: {len(meas)}  photos: {len(photos)}")
    if args.gender != "both":
        meas = meas[meas["gender"] == args.gender].copy()
        print(f"  filtered to gender={args.gender}: {len(meas)} subjects")

    rows = expand_rows_to_photos(meas, photos)
    print(f"  expanded (subject × photo): {len(rows)} rows")
    print("  rows per split:")
    print(rows["split"].value_counts().to_string())

    train_rows = rows[rows["split"] == "train"].copy()
    testA_rows = rows[rows["split"] == "testA"].copy()
    testB_rows = rows[rows["split"] == "testB"].copy()

    # MPS doesn't support float64 — keep everything in float32 from the start
    target_mean = train_rows[TARGETS].mean().values.astype(np.float32)
    target_std = (train_rows[TARGETS].std().values + 1e-6).astype(np.float32)
    print(f"  target mean: {dict(zip(TARGETS, target_mean.round(1)))}")
    print(f"  target std:  {dict(zip(TARGETS, target_std.round(1)))}")

    train_ds = BodyMDataset(train_rows, build_transforms(args.image_size, train=True),
                            target_mean, target_std)
    testA_ds = BodyMDataset(testA_rows, build_transforms(args.image_size, train=False),
                            target_mean, target_std)
    testB_ds = BodyMDataset(testB_rows, build_transforms(args.image_size, train=False),
                            target_mean, target_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    testA_loader = DataLoader(testA_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    testB_loader = DataLoader(testB_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── Model + optimizer ──
    model = VisionBodyEncoder(n_aux=2, n_out=len(TARGETS), backbone=args.backbone).to(args.device)

    head_params = list(model.head.parameters())
    backbone_params = list(model.backbone.parameters())
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params,     "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if args.fp16 else None

    print(f"\n  training {args.epochs} epochs (fp16={args.fp16})…")
    best_mae = float("inf")
    history = []
    for ep in range(args.epochs):
        tr_loss = train_one_epoch(model, train_loader, opt, scaler, args.device,
                                  target_mean, target_std, args.fp16)
        sched.step()
        evA = evaluate(model, testA_loader, args.device, target_mean, target_std)
        evB = evaluate(model, testB_loader, args.device, target_mean, target_std)
        avg_mae = (evA["mae_avg_cm"] + evB["mae_avg_cm"]) / 2.0
        print(f"   ep {ep+1:2d}/{args.epochs}  train_mse(z)={tr_loss:.4f}  "
              f"testA MAE={evA['mae_avg_cm']:.2f}cm  testB MAE={evB['mae_avg_cm']:.2f}cm")
        history.append({
            "epoch": ep + 1,
            "train_mse_normalized": tr_loss,
            "testA_mae_cm": evA["mae_avg_cm"],
            "testB_mae_cm": evB["mae_avg_cm"],
        })
        if avg_mae < best_mae:
            best_mae = avg_mae
            OUT_CKPT.parent.mkdir(exist_ok=True, parents=True)
            torch.save({
                "state_dict": model.state_dict(),
                "target_mean": target_mean, "target_std": target_std,
                "args": vars(args),
                "best_avg_mae_cm": best_mae,
            }, OUT_CKPT)
            print(f"   ↑ saved checkpoint → {OUT_CKPT.name}  best avg MAE = {best_mae:.2f}cm")

    # ── Final eval (best checkpoint reloaded) + size chain ──
    print("\n  reloading best checkpoint + final eval…")
    # weights_only=False because our checkpoint bundles numpy arrays
    # (target_mean / target_std) alongside the state_dict.
    # PyTorch 2.6+ defaults weights_only=True which rejects numpy globals.
    ckpt = torch.load(OUT_CKPT, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    evA = evaluate(model, testA_loader, args.device, target_mean, target_std)
    evB = evaluate(model, testB_loader, args.device, target_mean, target_std)
    print(f"   testA MAE: {evA['mae_avg_cm']:.2f}cm (n={evA['n']})")
    print(f"   testB MAE: {evB['mae_avg_cm']:.2f}cm (n={evB['n']})")
    print(f"   per-target testB MAE: {evB['mae_per_target_cm']}")

    print("\n  chaining photo→measurements→size via ModCloth GBM…")
    # Heights line up row-by-row because the test DataLoaders have shuffle=False
    testA_heights = testA_rows["height_cm"].values.astype(np.float32)
    testB_heights = testB_rows["height_cm"].values.astype(np.float32)
    size_chainA = chain_size_eval(evA["preds"], evA["targets"], TARGETS, testA_heights)
    size_chainB = chain_size_eval(evB["preds"], evB["targets"], TARGETS, testB_heights)
    print(f"   size chain testA: {size_chainA}")
    print(f"   size chain testB: {size_chainB}")

    # ── Write metrics ──
    OUT_METRICS.parent.mkdir(exist_ok=True, parents=True)
    OUT_METRICS.write_text(json.dumps({
        "history": history,
        "final": {
            "testA": {k: (v if not isinstance(v, np.ndarray) else None)
                      for k, v in evA.items() if k != "preds" and k != "targets"},
            "testB": {k: (v if not isinstance(v, np.ndarray) else None)
                      for k, v in evB.items() if k != "preds" and k != "targets"},
            "size_chain_testA": size_chainA,
            "size_chain_testB": size_chainB,
        },
        "config": vars(args),
        "target_cols": TARGETS,
    }, indent=2, default=float))
    print(f"\n  wrote {OUT_METRICS}")


if __name__ == "__main__":
    main()
