"""Auto-label DeepFashion2 upper-body items with our style taxonomy (batched).

Pipeline per batch of N items:
  1. Parallel image load + bbox crop + CLIP preprocess (torch DataLoader workers)
  2. Stack into a [N, 3, 224, 224] tensor → single CLIP forward pass on MPS/CUDA/CPU
  3. Per-item LAB k-means on cropped image (CPU)
  4. Write checkpoint parquet every CHECKPOINT_EVERY items so runs are resumable.

Output schema (one row per garment item):
  split, image_id, image_path, item_idx, category_id, category_name, bbox,
  occlusion, viewpoint,
  neckline, neckline_conf, silhouette, silhouette_conf, pattern, pattern_conf,
  sleeve, primary_color_lab, color_temperature, color_value, clip_embedding

clip_embedding is a 512-dim float vector (ViT-B-32 image embedding) — enables
catalog visual similarity / two-tower retrieval downstream without re-running.
"""

# intent: batched zero-shot labeling of 140k DeepFashion2 upper-body items
# status: done
# next: build DuckDB view over output parquet + wire into recommender
# confidence: high

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "research/datasets/raw/deepfashion2"
OUT = ROOT / "research/datasets/processed/deepfashion2"

PROMPT_BANK: dict[str, list[tuple[str, str]]] = {
    "neckline": [
        ("v_neck",       "a top with a v-neck collar"),
        ("deep_v_neck",  "a top with a deep plunging v-neckline"),
        ("scoop",        "a top with a scoop neckline"),
        ("crew",         "a top with a round crew neckline"),
        ("boat",         "a top with a wide boat neckline"),
        ("high_neck",    "a top with a high neck or turtleneck"),
        ("sweetheart",   "a top with a sweetheart neckline"),
        ("off_shoulder", "an off-the-shoulder top"),
    ],
    "silhouette": [
        ("bodycon",     "a tight bodycon top hugging the body"),
        ("fitted",      "a fitted top following the body shape"),
        ("semi_fitted", "a semi-fitted top with light shaping"),
        ("a_line",      "an a-line top flaring outward"),
        ("loose",       "a loose flowing top"),
        ("oversized",   "an oversized baggy top"),
    ],
    "pattern": [
        ("solid",              "a solid plain colored top"),
        ("vertical_stripe",    "a top with vertical stripes"),
        ("horizontal_stripe",  "a top with horizontal stripes"),
        ("small_print",        "a top with a small all-over print or pattern"),
        ("large_print",        "a top with a large bold print"),
    ],
}

SLEEVE_BY_CATEGORY = {1: "short", 2: "long", 3: "short", 4: "long", 5: "sleeveless", 6: "sleeveless"}


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1] → CIE LAB. Vectorized over N×3 arrays."""
    rgb_lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = rgb_lin @ M.T
    xn, yn, zn = 0.95047, 1.0, 1.08883
    def f(t):
        return np.where(t > 0.008856, t ** (1 / 3), 7.787 * t + 16 / 116)
    fx, fy, fz = f(xyz[:, 0] / xn), f(xyz[:, 1] / yn), f(xyz[:, 2] / zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


class UpperBodyDataset:
    """Lazy dataset that crops bbox and applies CLIP preprocess (workers safe)."""

    def __init__(self, df, preprocess, margin: float = 0.08):
        self.df = df
        self.preprocess = preprocess
        self.margin = margin

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = RAW / row.image_path
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return idx, None, None

        w, h = img.size
        x1, y1, x2, y2 = row.bbox
        bw, bh = x2 - x1, y2 - y1
        if bw <= 4 or bh <= 4:
            return idx, None, None
        mx, my = int(bw * self.margin), int(bh * self.margin)
        x1 = max(0, int(x1) - mx)
        y1 = max(0, int(y1) - my)
        x2 = min(w, int(x2) + mx)
        y2 = min(h, int(y2) + my)
        crop = img.crop((x1, y1, x2, y2))

        clip_tensor = self.preprocess(crop)
        small = np.asarray(crop.resize((96, 96)))  # for LAB analysis
        return idx, clip_tensor, small


def collate(batch):
    import torch
    valid = [(idx, t, c) for idx, t, c in batch if t is not None]
    if not valid:
        return [], None, []
    indices = [v[0] for v in valid]
    tensors = torch.stack([v[1] for v in valid])
    crops = [v[2] for v in valid]
    return indices, tensors, crops


def load_clip(device: str):
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(device)

    text_embeds: dict[str, tuple[list[str], "torch.Tensor"]] = {}
    with torch.inference_mode():
        for dim, pairs in PROMPT_BANK.items():
            labels = [p[0] for p in pairs]
            prompts = [p[1] for p in pairs]
            tokens = tokenizer(prompts).to(device)
            embeds = model.encode_text(tokens)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            text_embeds[dim] = (labels, embeds)
    return model, preprocess, text_embeds


def _color_from_crop(crop_arr: np.ndarray, kmeans_cls) -> tuple[list[float], str, str]:
    """Compute dominant color from a small RGB crop array (96×96×3 uint8)."""
    try:
        lab_pixels = _rgb_to_lab(crop_arr.reshape(-1, 3).astype(np.float64) / 255.0)
        km = kmeans_cls(n_clusters=3, n_init=3, random_state=42)
        km.fit(lab_pixels)
        centers = km.cluster_centers_
        counts = np.bincount(km.labels_, minlength=3)
        order = np.argsort(-counts)
        best = order[0]
        for i in order:
            L_, a_, b_ = centers[i]
            if not (L_ > 92 and float(np.hypot(a_, b_)) < 6):
                best = i
                break
        Lc, ac, bc = centers[best]
        chroma = float(np.hypot(ac, bc))
        temperature = "neutral"
        if chroma >= 8:
            hue = float(np.degrees(np.arctan2(bc, ac)))
            temperature = "warm" if -45 < hue < 100 else "cool"
        if Lc < 35:
            value = "dark"
        elif Lc < 70:
            value = "medium"
        else:
            value = "light"
        return [float(Lc), float(ac), float(bc)], temperature, value
    except Exception:
        return [50.0, 0.0, 0.0], "neutral", "medium"


CHECKPOINT_EVERY = 10_000


def main() -> None:
    import torch
    from torch.utils.data import DataLoader
    from sklearn.cluster import KMeans

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--output", default=str(OUT / "items_upper_labeled.parquet"))
    ap.add_argument("--checkpoint-dir", default=str(OUT / "checkpoints"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("  MPS unavailable, falling back to CPU")
        args.device = "cpu"

    src = OUT / "items_upper.parquet"
    if not src.exists():
        print(f"!! {src} missing", file=sys.stderr)
        return

    table = pq.read_table(src)
    df = table.to_pandas()
    if args.limit > 0:
        df = df.iloc[: args.limit].copy()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        already: set[tuple[str, int]] = set()
        for p in sorted(ckpt_dir.glob("part-*.parquet")):
            t = pq.read_table(p, columns=["image_id", "item_idx"])
            for iid, idx in zip(t.column("image_id").to_pylist(), t.column("item_idx").to_pylist()):
                already.add((iid, idx))
        if already:
            print(f"  resume: skipping {len(already)} already-labeled items")
            mask = ~df.apply(lambda r: (r.image_id, int(r.item_idx)) in already, axis=1)
            df = df[mask].reset_index(drop=True)

    n = len(df)
    if n == 0:
        print("  nothing left to label")
        return

    print(f"  labeling {n} items  device={args.device}  batch={args.batch_size}  workers={args.num_workers}")
    print(f"  loading CLIP ViT-B-32...")
    model, preprocess, text_embeds = load_clip(args.device)
    print(f"  loaded.")

    dataset = UpperBodyDataset(df, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        shuffle=False,
        persistent_workers=args.num_workers > 0,
    )

    out_rows: list[dict] = []
    skipped = 0
    t_start = time.time()
    items_done = 0
    part_idx = len(list(ckpt_dir.glob("part-*.parquet")))
    last_log_items = 0

    for batch_no, (indices, tensors, crops) in enumerate(loader):
        if not indices:
            skipped += args.batch_size
            continue

        tensors = tensors.to(args.device)
        with torch.inference_mode():
            img_embed = model.encode_image(tensors)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

            attr_results = {}
            for dim, (labels, text_embed) in text_embeds.items():
                scores = img_embed @ text_embed.T
                probs = scores.softmax(dim=-1).cpu().numpy()
                idxs = probs.argmax(axis=-1)
                attr_results[dim] = (labels, idxs, probs)

            embed_cpu = img_embed.cpu().numpy().astype(np.float32)

        for k, df_idx in enumerate(indices):
            row = df.iloc[df_idx]
            color_lab, temperature, value = _color_from_crop(crops[k], KMeans)

            out_rows.append({
                "split": row.split,
                "image_id": row.image_id,
                "image_path": row.image_path,
                "item_idx": int(row.item_idx),
                "category_id": int(row.category_id),
                "category_name": row.category_name,
                "bbox": list(row.bbox),
                "occlusion": int(row.occlusion) if row.occlusion is not None else 0,
                "viewpoint": int(row.viewpoint) if row.viewpoint is not None else 0,
                "neckline": attr_results["neckline"][0][int(attr_results["neckline"][1][k])],
                "neckline_conf": float(attr_results["neckline"][2][k, attr_results["neckline"][1][k]]),
                "silhouette": attr_results["silhouette"][0][int(attr_results["silhouette"][1][k])],
                "silhouette_conf": float(attr_results["silhouette"][2][k, attr_results["silhouette"][1][k]]),
                "pattern": attr_results["pattern"][0][int(attr_results["pattern"][1][k])],
                "pattern_conf": float(attr_results["pattern"][2][k, attr_results["pattern"][1][k]]),
                "sleeve": SLEEVE_BY_CATEGORY.get(int(row.category_id), "unknown"),
                "primary_color_lab": color_lab,
                "color_temperature": temperature,
                "color_value": value,
                "clip_embedding": embed_cpu[k].tolist(),
            })
            items_done += 1

        if items_done - last_log_items >= 500:
            elapsed = time.time() - t_start
            rate = items_done / elapsed if elapsed > 0 else 0
            eta_s = (n - items_done) / rate if rate > 0 else 0
            print(f"    {items_done}/{n} | {rate:.1f} items/s | ETA {eta_s/60:.1f} min | skipped {skipped}")
            last_log_items = items_done

        if len(out_rows) >= CHECKPOINT_EVERY:
            ckpt_path = ckpt_dir / f"part-{part_idx:04d}.parquet"
            pq.write_table(pa.Table.from_pylist(out_rows), ckpt_path)
            print(f"    checkpoint → {ckpt_path.name} ({len(out_rows)} rows)")
            out_rows = []
            part_idx += 1

    if out_rows:
        ckpt_path = ckpt_dir / f"part-{part_idx:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(out_rows), ckpt_path)
        print(f"  final checkpoint → {ckpt_path.name}")

    all_parts = sorted(ckpt_dir.glob("part-*.parquet"))
    print(f"\n  merging {len(all_parts)} checkpoints → {args.output}")
    tables = [pq.read_table(p) for p in all_parts]
    merged = pa.concat_tables(tables, promote_options="default")
    pq.write_table(merged, args.output)
    print(f"  wrote {args.output} ({len(merged)} rows)")

    elapsed = time.time() - t_start
    print(f"\n  total: {elapsed/60:.1f} min, {items_done/elapsed:.1f} items/s, {skipped} skipped")


if __name__ == "__main__":
    main()
