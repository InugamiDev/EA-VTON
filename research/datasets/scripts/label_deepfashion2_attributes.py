"""Auto-label DeepFashion2 upper-body items with our style taxonomy.

For each upper-body item:
  1. Crop the garment using its bounding box (with small margin)
  2. CLIP zero-shot classify per attribute dimension:
       - neckline ∈ {v_neck, deep_v_neck, scoop, crew, boat, high_neck, sweetheart, off_shoulder}
       - silhouette ∈ {bodycon, fitted, semi_fitted, a_line, loose, oversized}
       - pattern ∈ {solid, vertical_stripe, horizontal_stripe, small_print, large_print}
  3. Derive sleeve_length from DeepFashion2 category_id (1=short, 2=long, 5=sleeveless, ...)
  4. Extract dominant color via LAB k-means (k=3, ignore the largest cluster if it's near-white background)
  5. Map LAB color → temperature ∈ {warm, cool, neutral} and value ∈ {light, medium, dark}

Outputs:
  research/datasets/processed/deepfashion2/items_upper_labeled.parquet
    one row per item with the manifest columns + 6 new attribute columns + dominant LAB.

Run on a sample first (--limit 500) to validate before full 140k run.
"""

# intent: auto-label DeepFashion2 garments with our 7-dim style taxonomy
# status: in_progress
# next: validate on 500-item sample, then run full 140k batch overnight
# confidence: medium  (CLIP zero-shot accuracy varies per attribute)

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "research/datasets/raw/deepfashion2"
OUT = ROOT / "research/datasets/processed/deepfashion2"

# ── Attribute taxonomy + CLIP text prompts ──

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

# Derive sleeve from DeepFashion2 category_id (no CLIP needed)
SLEEVE_BY_CATEGORY = {
    1: "short",   # short sleeve top
    2: "long",    # long sleeve top
    3: "short",   # short sleeve outwear
    4: "long",    # long sleeve outwear
    5: "sleeveless",  # vest
    6: "sleeveless",  # sling
}


def load_clip(device: str):
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(device)

    # Pre-encode all text prompts (one-time cost)
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


def crop_garment(image_path: Path, bbox: list[float], margin: float = 0.08) -> Image.Image | None:
    """Crop an image to the bounding box with a small margin, returning None on error."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 4 or bh <= 4:
        return None
    mx, my = int(bw * margin), int(bh * margin)
    x1 = max(0, int(x1) - mx)
    y1 = max(0, int(y1) - my)
    x2 = min(w, int(x2) + mx)
    y2 = min(h, int(y2) + my)
    return img.crop((x1, y1, x2, y2))


def classify_attributes(
    crop: Image.Image,
    model,
    preprocess,
    text_embeds: dict,
    device: str,
) -> dict[str, tuple[str, float]]:
    """Return {dim: (label, confidence)} for each attribute dimension."""
    import torch

    img_tensor = preprocess(crop).unsqueeze(0).to(device)
    with torch.inference_mode():
        img_embed = model.encode_image(img_tensor)
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

        results = {}
        for dim, (labels, text_embed) in text_embeds.items():
            scores = (img_embed @ text_embed.T).squeeze(0)
            probs = scores.softmax(dim=-1).cpu().numpy()
            idx = int(probs.argmax())
            results[dim] = (labels[idx], float(probs[idx]))
    return results


def dominant_color_lab(
    crop: Image.Image, k: int = 3, sample_size: int = 4000
) -> tuple[list[float], str, str]:
    """K-means cluster pixels in CIE LAB space, return (lab, temp, value) of the largest non-background cluster."""
    from sklearn.cluster import KMeans
    from PIL import ImageCms

    crop_small = crop.resize((128, 128))
    arr = np.asarray(crop_small).reshape(-1, 3).astype(np.float64) / 255.0

    # RGB -> LAB via simple matrix conversion (approx; good enough for clustering)
    # Using sRGB → XYZ → LAB
    def rgb_to_lab(rgb):
        # Linearize sRGB
        rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        # to XYZ (D65)
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ])
        xyz = rgb @ M.T
        # to LAB
        xn, yn, zn = 0.95047, 1.0, 1.08883
        f = lambda t: np.where(t > 0.008856, t ** (1 / 3), 7.787 * t + 16 / 116)
        fx, fy, fz = f(xyz[:, 0] / xn), f(xyz[:, 1] / yn), f(xyz[:, 2] / zn)
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        return np.stack([L, a, b], axis=-1)

    lab = rgb_to_lab(arr)
    if len(lab) > sample_size:
        idx = np.random.choice(len(lab), sample_size, replace=False)
        lab_sample = lab[idx]
    else:
        lab_sample = lab

    km = KMeans(n_clusters=k, n_init=3, random_state=42)
    km.fit(lab_sample)
    centers = km.cluster_centers_
    counts = np.bincount(km.labels_, minlength=k)

    # Find largest non-background cluster (L > 92 with chroma < 6 = white-ish background)
    order = np.argsort(-counts)
    best = order[0]
    for i in order:
        L, a, b = centers[i]
        chroma = np.hypot(a, b)
        if not (L > 92 and chroma < 6):
            best = i
            break

    L, a, b = centers[best]
    chroma = np.hypot(a, b)

    # Hue → temperature
    if chroma < 8:
        temperature = "neutral"
    else:
        hue = np.degrees(np.arctan2(b, a))
        # Warm: hue in roughly (-30, 100) -- reds, oranges, yellows
        # Cool: hue in (100, 260) -- greens, blues, purples
        temperature = "warm" if -45 < hue < 100 else "cool"

    # Lightness → value
    if L < 35:
        value = "dark"
    elif L < 70:
        value = "medium"
    else:
        value = "light"

    return [float(L), float(a), float(b)], temperature, value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all upper-body items")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--batch-log", type=int, default=200)
    ap.add_argument("--output", default=str(OUT / "items_upper_labeled.parquet"))
    args = ap.parse_args()

    src = OUT / "items_upper.parquet"
    if not src.exists():
        print(f"!! {src} not found — run build_deepfashion2_manifest.py first", file=sys.stderr)
        return

    table = pq.read_table(src)
    df = table.to_pandas()
    if args.limit > 0:
        df = df.iloc[: args.limit].copy()

    n = len(df)
    print(f"  labeling {n} upper-body items on {args.device}")
    print(f"  loading CLIP ViT-B-32 (laion2b)...")
    model, preprocess, text_embeds = load_clip(args.device)
    print(f"  loaded.")

    out_rows: list[dict] = []
    skipped = 0
    for i, row in enumerate(df.itertuples(index=False)):
        if i % args.batch_log == 0:
            print(f"    {i}/{n}  (skipped {skipped})")

        img_path = RAW / row.image_path
        crop = crop_garment(img_path, row.bbox)
        if crop is None:
            skipped += 1
            continue

        try:
            attrs = classify_attributes(crop, model, preprocess, text_embeds, args.device)
            lab, temp, value = dominant_color_lab(crop)
        except Exception as e:
            skipped += 1
            continue

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
            "neckline": attrs["neckline"][0],
            "neckline_conf": attrs["neckline"][1],
            "silhouette": attrs["silhouette"][0],
            "silhouette_conf": attrs["silhouette"][1],
            "pattern": attrs["pattern"][0],
            "pattern_conf": attrs["pattern"][1],
            "sleeve": SLEEVE_BY_CATEGORY.get(int(row.category_id), "unknown"),
            "primary_color_lab": lab,
            "color_temperature": temp,
            "color_value": value,
        })

    print(f"\n  labeled {len(out_rows)} items, skipped {skipped}")

    out_table = pa.Table.from_pylist(out_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, out_path)
    print(f"  wrote {out_path}")

    # Quick distribution summary
    print("\n  Attribute distribution:")
    from collections import Counter
    for dim in ["neckline", "silhouette", "pattern", "sleeve", "color_temperature", "color_value"]:
        c = Counter(r[dim] for r in out_rows)
        top = ", ".join(f"{k}={v}" for k, v in c.most_common(5))
        print(f"    {dim:18s} {top}")


if __name__ == "__main__":
    main()
