"""Auto-label DeepFashion-with-masks (HuggingFace parquet shards).

Same Fit-Flatter-Match attribute taxonomy as the DeepFashion2 labeler, but
adapted for the HF parquet schema (images embedded as bytes, plus per-item
binary masks for clean garment crops).

Filters: gender=WOMEN, cloth_type ∈ {upper-body categories}.
Uses the binary mask to compute a tight garment bbox before CLIP encoding.

Output: research/datasets/processed/deepfashion_with_masks/items_upper_labeled.parquet
"""

# intent: label DF-with-masks upper-body items, expand combined catalog past 170k
# status: done
# next: union with DF2 catalog via build_combined_style_catalog.py
# confidence: high

from __future__ import annotations

import argparse
import glob
import io
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "research/datasets/raw/female_clothing_style_v1/deepfashion_with_masks/data"
OUT_DIR = ROOT / "research/datasets/processed/deepfashion_with_masks"

# DF-with-masks cloth_type → our DF2-compatible category_id + name.
# Categories 1-6 in our DF2 taxonomy: 1 short_sleeve_top, 2 long_sleeve_top,
# 3 short_sleeve_outwear, 4 long_sleeve_outwear, 5 vest, 6 sling.
CLOTH_TYPE_MAP: dict[str, tuple[int, str]] = {
    "Tees_Tanks":          (1, "short_sleeve_top"),     # may be sleeveless — refined via CLIP
    "Graphic_Tees":        (1, "short_sleeve_top"),
    "Shirts_Polos":        (1, "short_sleeve_top"),
    "Blouses_Shirts":      (2, "long_sleeve_top"),      # most are long
    "Sweaters":            (2, "long_sleeve_top"),
    "Sweatshirts_Hoodies": (2, "long_sleeve_top"),
    "Cardigans":           (4, "long_sleeve_outwear"),
    "Jackets_Coats":       (4, "long_sleeve_outwear"),
    "Jackets_Vests":       (5, "vest"),
    "Suiting":             (4, "long_sleeve_outwear"),
}

UPPER_TYPES = set(CLOTH_TYPE_MAP.keys())

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
    "sleeve": [
        ("sleeveless",     "a sleeveless top with bare arms"),
        ("short",          "a short-sleeve top"),
        ("three_quarter",  "a three-quarter sleeve top"),
        ("long",           "a long-sleeve top"),
    ],
}


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb_lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = rgb_lin @ M.T
    xn, yn, zn = 0.95047, 1.0, 1.08883
    f = lambda t: np.where(t > 0.008856, t ** (1 / 3), 7.787 * t + 16 / 116)
    fx, fy, fz = f(xyz[:, 0] / xn), f(xyz[:, 1] / yn), f(xyz[:, 2] / zn)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def _color_from_masked_pixels(rgb_arr: np.ndarray, mask_arr: np.ndarray) -> tuple[list[float], str, str]:
    """K-means dominant color, sampling only mask=True pixels (true garment color)."""
    from sklearn.cluster import KMeans

    if mask_arr is not None:
        m = mask_arr > 127  # binarize
        pixels = rgb_arr[m]
    else:
        pixels = rgb_arr.reshape(-1, 3)

    if len(pixels) < 50:
        return [50.0, 0.0, 0.0], "neutral", "medium"

    if len(pixels) > 3000:
        idx = np.random.choice(len(pixels), 3000, replace=False)
        pixels = pixels[idx]

    try:
        lab = _rgb_to_lab(pixels.astype(np.float64) / 255.0)
        km = KMeans(n_clusters=3, n_init=3, random_state=42)
        km.fit(lab)
        centers = km.cluster_centers_
        counts = np.bincount(km.labels_, minlength=3)
        order = np.argsort(-counts)
        best = order[0]
        # With mask we trust the largest cluster — no background to skip
        Lc, ac, bc = centers[best]
    except Exception:
        return [50.0, 0.0, 0.0], "neutral", "medium"

    chroma = float(np.hypot(ac, bc))
    if chroma < 8:
        temperature = "neutral"
    else:
        hue = float(np.degrees(np.arctan2(bc, ac)))
        temperature = "warm" if -45 < hue < 100 else "cool"
    if Lc < 35:
        value = "dark"
    elif Lc < 70:
        value = "medium"
    else:
        value = "light"
    return [float(Lc), float(ac), float(bc)], temperature, value


def _bbox_from_mask(mask: Image.Image) -> tuple[int, int, int, int] | None:
    """Return tight bbox around non-zero mask pixels."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    rows = np.any(arr > 127, axis=1)
    cols = np.any(arr > 127, axis=0)
    if not rows.any() or not cols.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(x1), int(y1), int(x2), int(y2)


class DFMaskDataset:
    """Yields (image_path_idx, clip_tensor, masked_rgb) per item.
    Holds the loaded shard tables in memory (small enough — ~350MB each)."""

    def __init__(self, shard_paths: list[str], preprocess, margin: float = 0.05):
        self.shards = []
        for p in shard_paths:
            t = pq.read_table(p)
            self.shards.append(t)
        self.preprocess = preprocess
        self.margin = margin
        self._index = []  # list of (shard_i, row_i, pid, cloth_type, caption, pose)
        for si, t in enumerate(self.shards):
            genders = t.column("gender").to_pylist()
            cloths  = t.column("cloth_type").to_pylist()
            pids    = t.column("pid").to_pylist()
            poses   = t.column("pose").to_pylist()
            caps    = t.column("caption").to_pylist()
            for ri in range(len(t)):
                if genders[ri] != "WOMEN" or cloths[ri] not in UPPER_TYPES:
                    continue
                self._index.append((si, ri, pids[ri], cloths[ri], caps[ri], poses[ri]))

    def __len__(self) -> int:
        return len(self._index)

    def get(self, idx: int):
        si, ri, pid, cloth_type, caption, pose = self._index[idx]
        t = self.shards[si]
        img_bytes = t.column("images")[ri].as_py()["bytes"]
        mask_bytes = t.column("mask")[ri].as_py()["bytes"]

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        except Exception:
            return None

        bbox = _bbox_from_mask(mask)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            return None

        # Loosen bbox slightly
        mx, my = int(bw * self.margin), int(bh * self.margin)
        w, h = img.size
        x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx); y2 = min(h, y2 + my)

        crop = img.crop((x1, y1, x2, y2))
        mask_crop = mask.crop((x1, y1, x2, y2))
        clip_tensor = self.preprocess(crop)

        return idx, clip_tensor, np.asarray(crop), np.asarray(mask_crop), pid, cloth_type, caption, pose


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


CHECKPOINT_EVERY = 5_000


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output", default=str(OUT_DIR / "items_upper_labeled.parquet"))
    ap.add_argument("--checkpoint-dir", default=str(OUT_DIR / "checkpoints"))
    args = ap.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"

    shard_paths = sorted(glob.glob(str(RAW_DIR / "*.parquet")))
    if not shard_paths:
        print(f"!! no shards in {RAW_DIR}", file=sys.stderr); return

    print(f"  loading CLIP ViT-B-32 on {args.device}...")
    model, preprocess, text_embeds = load_clip(args.device)
    print(f"  loaded.")

    ds = DFMaskDataset(shard_paths, preprocess)
    n = len(ds)
    if args.limit > 0: n = min(n, args.limit)
    print(f"  filtered to {n} upper-body WOMEN items")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict] = []
    t_start = time.time()
    items_done = 0
    last_log = 0
    skipped = 0
    part_idx = 0
    batch_size = args.batch_size

    # Process in batches
    i = 0
    while i < n:
        batch_items = []
        while len(batch_items) < batch_size and i < n:
            entry = ds.get(i)
            i += 1
            if entry is not None:
                batch_items.append(entry)
            else:
                skipped += 1

        if not batch_items:
            break

        tensors = torch.stack([e[1] for e in batch_items]).to(args.device)
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

        for k, (idx_in_ds, _t, rgb_arr, mask_arr, pid, cloth_type, caption, pose) in enumerate(batch_items):
            color_lab, temp, value = _color_from_masked_pixels(rgb_arr, mask_arr)
            cat_id, cat_name = CLOTH_TYPE_MAP.get(cloth_type, (1, "short_sleeve_top"))
            out_rows.append({
                "split": "train",
                "image_id": pid,                         # using DF-mm pid (not 6-digit DF2 id)
                "image_path": "",                        # not stored on disk per row
                "item_idx": idx_in_ds,
                "category_id": cat_id,
                "category_name": cat_name,
                "bbox": [0.0, 0.0, float(rgb_arr.shape[1]), float(rgb_arr.shape[0])],
                "occlusion": 0,
                "viewpoint": 1 if "front" in pose else (2 if "back" in pose else 3),
                "cloth_type_orig": cloth_type,
                "pose": pose,
                "caption": caption,
                "neckline": attr_results["neckline"][0][int(attr_results["neckline"][1][k])],
                "neckline_conf": float(attr_results["neckline"][2][k, attr_results["neckline"][1][k]]),
                "silhouette": attr_results["silhouette"][0][int(attr_results["silhouette"][1][k])],
                "silhouette_conf": float(attr_results["silhouette"][2][k, attr_results["silhouette"][1][k]]),
                "pattern": attr_results["pattern"][0][int(attr_results["pattern"][1][k])],
                "pattern_conf": float(attr_results["pattern"][2][k, attr_results["pattern"][1][k]]),
                "sleeve": attr_results["sleeve"][0][int(attr_results["sleeve"][1][k])],
                "sleeve_conf": float(attr_results["sleeve"][2][k, attr_results["sleeve"][1][k]]),
                "primary_color_lab": color_lab,
                "color_temperature": temp,
                "color_value": value,
                "clip_embedding": embed_cpu[k].tolist(),
                "source": "deepfashion_with_masks",
            })
            items_done += 1

        if items_done - last_log >= 500:
            elapsed = time.time() - t_start
            rate = items_done / elapsed
            eta = (n - items_done) / rate if rate > 0 else 0
            print(f"    {items_done}/{n} | {rate:.1f} items/s | ETA {eta/60:.1f} min | skipped {skipped}", flush=True)
            last_log = items_done

        if len(out_rows) >= CHECKPOINT_EVERY:
            ckpt = ckpt_dir / f"part-{part_idx:04d}.parquet"
            pq.write_table(pa.Table.from_pylist(out_rows), ckpt)
            print(f"    checkpoint → {ckpt.name} ({len(out_rows)} rows)", flush=True)
            out_rows = []
            part_idx += 1

    if out_rows:
        ckpt = ckpt_dir / f"part-{part_idx:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(out_rows), ckpt)
        print(f"  final checkpoint → {ckpt.name}", flush=True)

    parts = sorted(ckpt_dir.glob("part-*.parquet"))
    tables = [pq.read_table(p) for p in parts]
    merged = pa.concat_tables(tables, promote_options="default") if tables else pa.Table.from_pylist([])
    pq.write_table(merged, args.output)
    print(f"\n  wrote {args.output} ({len(merged)} rows)")
    print(f"  total: {(time.time()-t_start)/60:.1f} min, {items_done/(time.time()-t_start):.1f} items/s, {skipped} skipped")


if __name__ == "__main__":
    main()
