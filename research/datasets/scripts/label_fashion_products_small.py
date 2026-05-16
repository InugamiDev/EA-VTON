"""Auto-label fashion_products_small (Myntra) upper-body items.

Differences from DF2/DFM/Fashionpedia:
  - Already has human-labeled gender, masterCategory, subCategory, articleType, baseColour
  - Whole-image crop (no bbox — these are catalog product shots on white bg)
  - baseColour string maps directly to LAB via a fixed name→LAB table
  - articleType maps directly to category_id (no CLIP needed for category)
  - CLIP only runs for neckline / silhouette / pattern / sleeve (the truly missing dims)

Filters: gender ∈ {Women, Unisex} AND subCategory='Topwear'
"""

# intent: label fashion_products_small upper-body items using human attributes + CLIP supplementation
# status: done
# next: union into combined catalog
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
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "research/datasets/raw/fashion_products_small/data"
OUT_DIR = ROOT / "research/datasets/processed/fashion_products_small"

# articleType → (category_id, category_name, sleeve hint)
ARTICLE_TYPE_MAP: dict[str, tuple[int, str, str | None]] = {
    "Tshirts":          (1, "short_sleeve_top",    "short"),
    "Tops":             (1, "short_sleeve_top",    None),
    "Tunics":           (2, "long_sleeve_top",     None),
    "Shirts":           (2, "long_sleeve_top",     "long"),
    "Kurtas":           (2, "long_sleeve_top",     "long"),
    "Kurta Sets":       (2, "long_sleeve_top",     "long"),
    "Sweatshirts":      (2, "long_sleeve_top",     "long"),
    "Sweaters":         (2, "long_sleeve_top",     "long"),
    "Jackets":          (4, "long_sleeve_outwear", "long"),
    "Blazers":          (4, "long_sleeve_outwear", "long"),
    "Rain Jacket":      (4, "long_sleeve_outwear", "long"),
    "Nehru Jackets":    (4, "long_sleeve_outwear", "long"),
    "Innerwear Vests":  (5, "vest",                "sleeveless"),
    "Lounge Tshirts":   (1, "short_sleeve_top",    "short"),
}
UPPER_ARTICLE_TYPES = set(ARTICLE_TYPE_MAP.keys())

# baseColour name → CIE LAB (approximate, hand-curated)
COLOR_NAME_LAB: dict[str, list[float]] = {
    "Black":         [10.0, 0.0, 0.0],
    "White":         [98.0, 0.0, 0.0],
    "Grey":          [60.0, 0.0, 0.0],
    "Charcoal":      [25.0, 0.0, 0.0],
    "Silver":        [78.0, 0.0, 0.0],
    "Blue":          [44.0, 8.0, -55.0],
    "Navy Blue":     [16.0, 4.0, -22.0],
    "Sea Green":     [56.0, -30.0, -2.0],
    "Teal":          [52.0, -25.0, -10.0],
    "Turquoise Blue":[68.0, -28.0, -10.0],
    "Green":         [50.0, -32.0, 28.0],
    "Olive":         [44.0, -8.0, 38.0],
    "Yellow":        [86.0, -2.0, 80.0],
    "Mustard":       [74.0, 10.0, 70.0],
    "Gold":          [70.0, 12.0, 50.0],
    "Orange":        [62.0, 50.0, 70.0],
    "Red":           [42.0, 60.0, 40.0],
    "Maroon":        [22.0, 38.0, 18.0],
    "Pink":          [78.0, 22.0, 4.0],
    "Magenta":       [42.0, 60.0, -20.0],
    "Purple":        [32.0, 38.0, -50.0],
    "Lavender":      [78.0, 14.0, -18.0],
    "Brown":         [38.0, 14.0, 26.0],
    "Beige":         [78.0, 4.0, 16.0],
    "Tan":           [66.0, 8.0, 24.0],
    "Khaki":         [60.0, 0.0, 32.0],
    "Mauve":         [58.0, 14.0, -2.0],
    "Coffee Brown":  [32.0, 12.0, 20.0],
    "Burgundy":      [22.0, 34.0, 12.0],
    "Rust":          [38.0, 36.0, 38.0],
    "Cream":         [92.0, 2.0, 14.0],
    "Off White":     [94.0, 1.0, 6.0],
    "Peach":         [82.0, 18.0, 24.0],
    "Coral":         [70.0, 38.0, 30.0],
    "Bronze":        [50.0, 18.0, 32.0],
    "Copper":        [56.0, 28.0, 36.0],
    "Multi":         [60.0, 0.0, 0.0],
}

# usage → occasion tags
USAGE_MAP: dict[str, list[str]] = {
    "Casual":   ["casual", "date"],
    "Formal":   ["work", "formal"],
    "Sports":   ["casual"],
    "Ethnic":   ["formal", "party"],
    "Party":    ["party", "date"],
    "Travel":   ["casual"],
    "Smart Casual": ["work", "casual"],
}

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


def _color_temperature_value(lab: list[float]) -> tuple[str, str]:
    L, a, b = lab
    chroma = float(np.hypot(a, b))
    temperature = "neutral"
    if chroma >= 8:
        hue = float(np.degrees(np.arctan2(b, a)))
        temperature = "warm" if -45 < hue < 100 else "cool"
    value = "dark" if L < 35 else ("medium" if L < 70 else "light")
    return temperature, value


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


CHECKPOINT_EVERY = 2_000


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

    # Read + filter all rows up-front (small dataset)
    rows: list[dict] = []
    for p in sorted(glob.glob(str(RAW_DIR / "*.parquet"))):
        t = pq.read_table(p)
        f = pc.and_(
            pc.is_in(t.column("gender"), value_set=pa.array(["Women", "Unisex"])),
            pc.equal(t.column("subCategory"), "Topwear"),
        )
        ft = t.filter(f).to_pylist()
        for r in ft:
            if r["articleType"] in UPPER_ARTICLE_TYPES:
                rows.append(r)
    if args.limit > 0:
        rows = rows[: args.limit]
    n = len(rows)
    print(f"  filtered to {n} upper-body items (Women+Unisex Topwear with mapped articleType)")

    print(f"  loading CLIP on {args.device}...")
    model, preprocess, text_embeds = load_clip(args.device)
    print("  loaded.")

    out_rows: list[dict] = []
    t_start = time.time()
    items_done = 0
    last_log = 0
    skipped = 0
    part_idx = 0

    i = 0
    while i < n:
        batch_meta: list[dict] = []
        batch_tensors: list = []
        while len(batch_tensors) < args.batch_size and i < n:
            r = rows[i]; i += 1
            try:
                img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            except Exception:
                skipped += 1; continue
            batch_tensors.append(preprocess(img))
            batch_meta.append(r)

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

        for k, r in enumerate(batch_meta):
            article = r["articleType"]
            cat_id, cat_name, sleeve_hint = ARTICLE_TYPE_MAP.get(article, (1, "short_sleeve_top", None))
            colour = r.get("baseColour") or "Multi"
            lab = COLOR_NAME_LAB.get(colour, [60.0, 0.0, 0.0])
            temp, value = _color_temperature_value(lab)
            usage = r.get("usage") or "Casual"
            occasion = USAGE_MAP.get(usage, ["casual"])

            sleeve = sleeve_hint or "short"
            out_rows.append({
                "split": "train",
                "image_id": f"fps_{r['id']}",
                "image_path": "",
                "item_idx": 0,
                "category_id": cat_id,
                "category_name": cat_name,
                "article_type": article,
                "bbox": [0.0, 0.0, 60.0, 80.0],   # nominal; image already cropped
                "occlusion": 0,
                "viewpoint": 1,
                "title": r.get("productDisplayName"),
                "base_colour": colour,
                "usage": usage,
                "neckline": attr["neckline"][0][int(attr["neckline"][1][k])],
                "neckline_conf": float(attr["neckline"][2][k, attr["neckline"][1][k]]),
                "silhouette": attr["silhouette"][0][int(attr["silhouette"][1][k])],
                "silhouette_conf": float(attr["silhouette"][2][k, attr["silhouette"][1][k]]),
                "pattern": attr["pattern"][0][int(attr["pattern"][1][k])],
                "pattern_conf": float(attr["pattern"][2][k, attr["pattern"][1][k]]),
                "sleeve": sleeve,
                "primary_color_lab": lab,
                "color_temperature": temp,
                "color_value": value,
                "occasion_tags": occasion,
                "clip_embedding": embed_cpu[k].tolist(),
                "source": "fashion_products_small",
            })
            items_done += 1

        if items_done - last_log >= 250:
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
