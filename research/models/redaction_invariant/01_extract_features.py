"""Stage 1 — extract CLIP image features for ALL paired (redacted, unredacted) images.

Output: research/models/redaction_invariant/features.npz with arrays:
    gids        — (N,) string garment IDs
    emb_red     — (N, 512) L2-normalized CLIP embeddings of redacted images
    emb_unred   — (N, 512) L2-normalized CLIP embeddings of original images

Where N ≈ 177,303 (items with both versions on disk).

Runtime: ~25-35 min on A100 with batch_size=256, fp16. Single pass; cache the result.

Usage on the A100:
    python research/models/redaction_invariant/01_extract_features.py \\
        --batch-size 256 --num-workers 8 --device cuda --fp16
"""

# intent: pre-extract CLIP features once so all downstream training is fast
# status: ready to run
# next: run on A100; output drops to features.npz
# confidence: high

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[3]
PARQUET = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
OUT = Path(__file__).parent / "features.npz"


class PairedImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocess):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        unred = ROOT / row["image_local_path"]
        red = ROOT / row["image_local_path"].replace("/images/", "/images_redacted/", 1)
        try:
            img_u = Image.open(unred).convert("RGB")
            img_r = Image.open(red).convert("RGB")
            return {
                "gid": row["garment_id"],
                "img_u": self.preprocess(img_u),
                "img_r": self.preprocess(img_r),
                "ok": True,
            }
        except Exception as e:  # noqa: BLE001
            return {"gid": row["garment_id"], "ok": False, "err": str(e)}


def collate(batch):
    ok = [b for b in batch if b.get("ok")]
    if not ok:
        return None
    return {
        "gid": [b["gid"] for b in ok],
        "img_u": torch.stack([b["img_u"] for b in ok]),
        "img_r": torch.stack([b["img_r"] for b in ok]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true",
                    help="Use fp16 for the forward pass; A100 supports bf16/fp16")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N rows (smoke test)")
    args = ap.parse_args()

    try:
        import open_clip
    except ImportError:
        sys.exit("!! pip install open-clip-torch")

    df = pd.read_parquet(PARQUET, columns=["garment_id", "image_local_path", "redaction_status"])
    df = df[df["redaction_status"] == "face_detected_redacted"].copy()
    print(f"  rows with redacted pair: {len(df):,}")
    if args.limit:
        df = df.head(args.limit)
        print(f"  limited to {len(df):,} rows (smoke test)")

    print(f"  loading CLIP ViT-B-32 to {args.device}…")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(args.device).eval()
    if args.fp16 and args.device == "cuda":
        model = model.half()

    ds = PairedImageDataset(df, preprocess)
    loader = DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=True, drop_last=False,
    )

    gids: list[str] = []
    embs_u: list[np.ndarray] = []
    embs_r: list[np.ndarray] = []

    print(f"  forward-pass over {len(ds)} pairs…")
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            if batch is None:
                continue
            img_u = batch["img_u"].to(args.device, non_blocking=True)
            img_r = batch["img_r"].to(args.device, non_blocking=True)
            if args.fp16 and args.device == "cuda":
                img_u = img_u.half()
                img_r = img_r.half()
            eu = model.encode_image(img_u)
            er = model.encode_image(img_r)
            eu = eu / (eu.norm(dim=-1, keepdim=True) + 1e-8)
            er = er / (er.norm(dim=-1, keepdim=True) + 1e-8)
            gids.extend(batch["gid"])
            embs_u.append(eu.float().cpu().numpy())
            embs_r.append(er.float().cpu().numpy())
            if step % 50 == 0:
                done = len(gids)
                pct = 100 * done / len(ds)
                print(f"   step {step:5d}  done {done}/{len(ds)} ({pct:.1f}%)")

    gids_arr = np.array(gids)
    emb_unred = np.concatenate(embs_u, axis=0)
    emb_red = np.concatenate(embs_r, axis=0)
    print(f"\n  final shapes: gids={gids_arr.shape}, emb_unred={emb_unred.shape}, emb_red={emb_red.shape}")
    np.savez_compressed(OUT, gids=gids_arr, emb_unred=emb_unred, emb_red=emb_red)
    print(f"  wrote {OUT} ({OUT.stat().st_size/1024/1024:.0f} MB)")


if __name__ == "__main__":
    main()
