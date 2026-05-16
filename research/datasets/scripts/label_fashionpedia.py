"""Auto-label Fashionpedia upper-body items via the same CLIP-zero-shot stack.

Fashionpedia categories 0-9 are full garments; 13+ are clothing parts. We keep
upper-body categories only: shirt_blouse(0), top_tshirt_sweatshirt(1), sweater(2),
cardigan(3), jacket(4), vest(5), coat(9).

Per image: each object's bbox becomes one labeled garment row.

Output: research/datasets/processed/fashionpedia/items_upper_labeled.parquet
"""

# intent: label Fashionpedia upper-body objects with same taxonomy as DF2/DFM
# status: done
# next: union with DF2/DFM in combined catalog
# confidence: high

from __future__ import annotations

import argparse
import glob
import io
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "research/datasets/raw/fashionpedia/data"
OUT_DIR = ROOT / "research/datasets/processed/fashionpedia"

# Fashionpedia → our DF2-compatible category. Mappings are deliberate:
#   our 1 short_sleeve_top, 2 long_sleeve_top, 3 short_sleeve_outwear,
#   4 long_sleeve_outwear, 5 vest, 6 sling
FP_CATEGORY_MAP: dict[int, tuple[int, str]] = {
    0: (2, "long_sleeve_top"),       # shirt_blouse — defaults long; refined via CLIP sleeve
    1: (1, "short_sleeve_top"),      # top_tshirt_sweatshirt
    2: (2, "long_sleeve_top"),       # sweater
    3: (4, "long_sleeve_outwear"),   # cardigan
    4: (4, "long_sleeve_outwear"),   # jacket
    5: (5, "vest"),                  # vest
    9: (4, "long_sleeve_outwear"),   # coat
}
UPPER_CATS = set(FP_CATEGORY_MAP.keys())

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


def _color_from_crop(rgb_arr: np.ndarray) -> tuple[list[float], str, str]:
    from sklearn.cluster import KMeans
    pixels = rgb_arr.reshape(-1, 3)
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
        for i in order:
            L_, a_, b_ = centers[i]
            if not (L_ > 92 and float(np.hypot(a_, b_)) < 6):
                best = i; break
        Lc, ac, bc = centers[best]
    except Exception:
        return [50.0, 0.0, 0.0], "neutral", "medium"
    chroma = float(np.hypot(ac, bc))
    temperature = "neutral"
    if chroma >= 8:
        hue = float(np.degrees(np.arctan2(bc, ac)))
        temperature = "warm" if -45 < hue < 100 else "cool"
    value = "dark" if Lc < 35 else ("medium" if Lc < 70 else "light")
    return [float(Lc), float(ac), float(bc)], temperature, value


def load_clip(device: str):
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(device)
    text_embeds = {}
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(glob.glob(str(RAW_DIR / "*.parquet")))
    print(f"  shards: {len(shard_paths)}")
    print(f"  loading CLIP on {args.device}...")
    model, preprocess, text_embeds = load_clip(args.device)
    print("  loaded.")

    # First pass: enumerate upper-body items across all shards
    items: list[tuple[int, int, int, int, list[float]]] = []
    # (shard_idx, row_idx, obj_idx_in_row, category_id, bbox)
    for si, p in enumerate(shard_paths):
        t = pq.read_table(p, columns=["objects"])
        objs = t.column("objects").to_pylist()
        for ri, o in enumerate(objs):
            cats = o.get("category", [])
            bboxes = o.get("bbox", [])
            for k, c in enumerate(cats):
                if c in UPPER_CATS:
                    items.append((si, ri, k, c, bboxes[k]))
        print(f"  shard {si}: {len(t)} rows; cumulative upper-body items {len(items):,}")

    if args.limit > 0:
        items = items[: args.limit]
    n = len(items)
    print(f"  total upper-body items to label: {n:,}")

    # Cache shards in memory by shard index (need access to image bytes)
    shard_cache: dict[int, "pa.Table"] = {}

    def get_image_bytes(si: int, ri: int) -> bytes | None:
        if si not in shard_cache:
            shard_cache[si] = pq.read_table(shard_paths[si], columns=["image"])
        return shard_cache[si].column("image")[ri].as_py()["bytes"]

    out_rows: list[dict] = []
    t_start = time.time()
    items_done = 0
    last_log = 0
    skipped = 0
    part_idx = 0
    batch_size = args.batch_size

    i = 0
    while i < n:
        # Build batch
        batch_meta: list[tuple] = []
        batch_tensors: list = []
        batch_arrs: list = []

        while len(batch_tensors) < batch_size and i < n:
            si, ri, oi, cat, bbox = items[i]
            i += 1
            img_bytes = get_image_bytes(si, ri)
            if not img_bytes:
                skipped += 1; continue
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                skipped += 1; continue
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            if bw < 10 or bh < 10:
                skipped += 1; continue
            mx, my = bw * 0.05, bh * 0.05
            w, h = img.size
            crop = img.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)),
                             min(w, int(x2 + mx)), min(h, int(y2 + my))))
            batch_tensors.append(preprocess(crop))
            batch_arrs.append(np.asarray(crop))
            batch_meta.append((si, ri, oi, cat, bbox, crop.size))

        if not batch_tensors:
            continue

        tensors = torch.stack(batch_tensors).to(args.device)
        with torch.inference_mode():
            img_embed = model.encode_image(tensors)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
            attr = {}
            for dim, (labels, txt) in text_embeds.items():
                scores = img_embed @ txt.T
                probs = scores.softmax(dim=-1).cpu().numpy()
                idxs = probs.argmax(axis=-1)
                attr[dim] = (labels, idxs, probs)
            embed_cpu = img_embed.cpu().numpy().astype(np.float32)

        for k, ((si, ri, oi, cat, bbox, sz), rgb_arr) in enumerate(zip(batch_meta, batch_arrs)):
            cat_id, cat_name = FP_CATEGORY_MAP[cat]
            color_lab, temp, value = _color_from_crop(rgb_arr)
            out_rows.append({
                "split": "train",
                "image_id": f"fp_{si}_{ri}",
                "image_path": "",                     # embedded in parquet, not on disk
                "item_idx": oi,
                "category_id": cat_id,
                "category_name": cat_name,
                "fp_category_id": cat,                # keep raw FP category too
                "bbox": list(bbox),
                "occlusion": 0,
                "viewpoint": 1,
                "neckline": attr["neckline"][0][int(attr["neckline"][1][k])],
                "neckline_conf": float(attr["neckline"][2][k, attr["neckline"][1][k]]),
                "silhouette": attr["silhouette"][0][int(attr["silhouette"][1][k])],
                "silhouette_conf": float(attr["silhouette"][2][k, attr["silhouette"][1][k]]),
                "pattern": attr["pattern"][0][int(attr["pattern"][1][k])],
                "pattern_conf": float(attr["pattern"][2][k, attr["pattern"][1][k]]),
                "sleeve": attr["sleeve"][0][int(attr["sleeve"][1][k])],
                "sleeve_conf": float(attr["sleeve"][2][k, attr["sleeve"][1][k]]),
                "primary_color_lab": color_lab,
                "color_temperature": temp,
                "color_value": value,
                "clip_embedding": embed_cpu[k].tolist(),
                "source": "fashionpedia",
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
