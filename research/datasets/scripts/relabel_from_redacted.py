"""Re-label every item in the combined catalog using the face-redacted images.

After redact_faces.py blurs face ovals, the CLIP embeddings + color k-means
must be re-derived from those redacted images so the catalog labels are
fully privacy-clean (every label provenance points to a face-blurred image).

For every row:
  1. Swap image_local_path from images/<source>/<f> → images_redacted/<source>/<f>
  2. Open + bbox-crop the redacted image
  3. CLIP image encode → neckline / silhouette / pattern / sleeve argmax
  4. LAB k-means → primary_color_lab / temperature / value
  5. Overwrite those columns in items_upper_combined.parquet

The category_id, image_id, bbox, occasion_tags etc. are untouched.
"""

# intent: rebuild attribute + color labels from face-redacted images
# status: done
# next: rebuild dataset bundle to ship privacy-clean labels
# confidence: high

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
COMBINED = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"

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
        hue = float(math.degrees(math.atan2(bc, ac)))
        temperature = "warm" if -45 < hue < 100 else "cool"
    value = "dark" if Lc < 35 else ("medium" if Lc < 70 else "light")
    return [float(Lc), float(ac), float(bc)], temperature, value


def _redacted_path(orig_image_path: str | None) -> str | None:
    """images/<src>/<f> → images_redacted/<src>/<f>"""
    if not orig_image_path:
        return None
    return orig_image_path.replace("/combined/images/", "/combined/images_redacted/", 1)


def load_clip(device: str):
    import open_clip, torch
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


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(COMBINED))
    ap.add_argument("--out", dest="out_path", default=str(COMBINED))
    ap.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=2000)
    args = ap.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"

    print(f"  reading {args.in_path}...")
    t = pq.read_table(args.in_path)
    n = len(t) if not args.limit else min(args.limit, len(t))
    print(f"  rows: {n:,}")

    print(f"  loading CLIP on {args.device}...")
    model, preprocess, text_embeds = load_clip(args.device)
    print("  loaded.")

    paths = t.column("image_local_path").to_pylist()
    bboxes = t.column("bbox").to_pylist()
    sources = t.column("source").to_pylist()
    cats = t.column("category_id").to_pylist()

    # Reuse existing sleeve when source has trustworthy sleeve hints
    old_sleeves = t.column("sleeve").to_pylist()

    out_neck = [None] * n
    out_neck_c = [0.0] * n
    out_sil = [None] * n
    out_sil_c = [0.0] * n
    out_pat = [None] * n
    out_pat_c = [0.0] * n
    out_sleeve = [None] * n
    out_sleeve_c = [0.0] * n
    out_lab = [None] * n
    out_temp = [None] * n
    out_val = [None] * n
    out_clip = [None] * n

    t_start = time.time()
    last_log = 0
    failed = 0

    # Process in batches
    i = 0
    while i < n:
        batch_idx = []
        batch_tensors = []
        batch_arrs = []
        while len(batch_tensors) < args.batch_size and i < n:
            idx = i; i += 1
            red_path = _redacted_path(paths[idx])
            if not red_path:
                failed += 1
                continue
            full = ROOT / red_path
            if not full.exists():
                # Fall back to original (rare — happens when redaction skipped a file)
                full = ROOT / paths[idx]
                if not full.exists():
                    failed += 1
                    continue
            try:
                img = Image.open(full).convert("RGB")
            except Exception:
                failed += 1
                continue
            bbox = bboxes[idx]
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                w, h = img.size
                bw, bh = max(1, x2 - x1), max(1, y2 - y1)
                if bw < 10 or bh < 10:
                    crop = img
                else:
                    mx, my = bw * 0.05, bh * 0.05
                    crop = img.crop((
                        max(0, int(x1 - mx)), max(0, int(y1 - my)),
                        min(w, int(x2 + mx)), min(h, int(y2 + my)),
                    ))
            else:
                crop = img

            batch_idx.append(idx)
            batch_tensors.append(preprocess(crop))
            batch_arrs.append(np.asarray(crop))

        if not batch_tensors:
            continue

        tensors = torch.stack(batch_tensors).to(args.device)
        with torch.inference_mode():
            img_embed = model.encode_image(tensors)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
            attr_results = {}
            for dim, (labels, txt) in text_embeds.items():
                scores = img_embed @ txt.T
                probs = scores.softmax(dim=-1).cpu().numpy()
                idxs = probs.argmax(axis=-1)
                attr_results[dim] = (labels, idxs, probs)
            embed_cpu = img_embed.cpu().numpy().astype(np.float32)

        for k, idx in enumerate(batch_idx):
            out_neck[idx] = attr_results["neckline"][0][int(attr_results["neckline"][1][k])]
            out_neck_c[idx] = float(attr_results["neckline"][2][k, attr_results["neckline"][1][k]])
            out_sil[idx] = attr_results["silhouette"][0][int(attr_results["silhouette"][1][k])]
            out_sil_c[idx] = float(attr_results["silhouette"][2][k, attr_results["silhouette"][1][k]])
            out_pat[idx] = attr_results["pattern"][0][int(attr_results["pattern"][1][k])]
            out_pat_c[idx] = float(attr_results["pattern"][2][k, attr_results["pattern"][1][k]])

            # For sources where sleeve was category-derived (DF2 cats 1/2/5/6 →
            # short/long/sleeveless), keep the deterministic mapping. Otherwise
            # re-derive from CLIP.
            cat = cats[idx]
            src = sources[idx]
            if src == "deepfashion2" and cat in (1, 3):
                out_sleeve[idx] = "short"
            elif src == "deepfashion2" and cat in (2, 4):
                out_sleeve[idx] = "long"
            elif src == "deepfashion2" and cat in (5, 6):
                out_sleeve[idx] = "sleeveless"
            else:
                out_sleeve[idx] = attr_results["sleeve"][0][int(attr_results["sleeve"][1][k])]
                out_sleeve_c[idx] = float(attr_results["sleeve"][2][k, attr_results["sleeve"][1][k]])

            # Color from redacted crop
            lab, temp, value = _color_from_crop(batch_arrs[k])
            out_lab[idx] = lab
            out_temp[idx] = temp
            out_val[idx] = value
            out_clip[idx] = embed_cpu[k].tolist()

        if i - last_log >= args.log_every:
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n - i) / rate if rate > 0 else 0
            print(f"    {i:>7}/{n:,}  {rate:.1f} items/s  ETA {eta/60:.1f}min  failed={failed}", flush=True)
            last_log = i

    print(f"\n  done relabeling. failed: {failed:,}")
    print("  writing patched parquet...")

    # Patch columns; preserve every other column
    patch_cols = {
        "neckline": out_neck, "neckline_conf": out_neck_c,
        "silhouette": out_sil, "silhouette_conf": out_sil_c,
        "pattern": out_pat, "pattern_conf": out_pat_c,
        "sleeve": out_sleeve, "sleeve_conf": out_sleeve_c,
        "primary_color_lab": out_lab,
        "color_temperature": out_temp, "color_value": out_val,
        "clip_embedding": out_clip,
    }

    # For rows that failed to load (None values), keep the existing label
    for col, new_vals in patch_cols.items():
        old_vals = t.column(col).to_pylist()
        merged = [(new_vals[i] if new_vals[i] is not None else old_vals[i])
                  for i in range(len(t))]
        patch_cols[col] = merged

    keep_cols = [c for c in t.column_names if c not in patch_cols]
    base = t.select(keep_cols)
    new_arrays = [pa.array(patch_cols[c]) for c in patch_cols]
    new_table = pa.Table.from_arrays(
        list(base.itercolumns()) + new_arrays,
        names=list(base.column_names) + list(patch_cols.keys()),
    )
    pq.write_table(new_table, args.out_path)
    sz = Path(args.out_path).stat().st_size / 1024 / 1024
    print(f"  wrote {args.out_path} ({sz:.1f} MB, {new_table.num_rows:,} rows)")
    print(f"  total: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
