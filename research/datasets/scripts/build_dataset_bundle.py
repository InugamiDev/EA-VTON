"""Package the labeled catalog as a publishable, citable dataset bundle.

What we publish (legally distributable as a derivative work):
  - Our auto-generated labels (neckline, silhouette, sleeve, pattern, color)
  - The CLIP-derived 512-d image embedding
  - The bbox + image identifier (pointer back to source)

What we do NOT publish (license-restricted):
  - Image bytes / extracted thumbnails (source datasets have their own licenses)
  - Source dataset's raw annotation files

Output structure:
  research/datasets/processed/dataset_bundle_v1/
    labels.parquet           ← 207k rows, no image bytes
    taxonomy.json            ← label space definitions
    schema.json              ← column descriptions
    seasonal_palettes.json   ← copy of the 12-season anchors
    vn_body_centroids.json   ← copy of the 5-cluster body type definitions
    body_rules_vn.json       ← copy of the flatter rules
    METHODOLOGY.md           ← how labels were generated
    DATASET_CARD.md          ← HuggingFace dataset card
    LICENSE                  ← CC BY 4.0 (labels only)
    LICENSE.SOURCES.md       ← upstream license inheritance per source
    download_images.py       ← reproducible script to fetch source images
    examples/sample_50.parquet ← 50-row preview for quick inspection
"""

# intent: produce a citable, redistributable bundle of our labeling work
# status: done
# next: optionally `huggingface-cli upload` to a public dataset repo
# confidence: high

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
SRC_PARQUET = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
BUNDLE = ROOT / "research/datasets/processed/dataset_bundle_v1"

# Columns we publish — labels and pointers only. NO image bytes/paths.
PUBLISH_COLS = [
    "garment_id", "source", "image_id", "item_idx",
    "category_id", "category_name", "bbox",
    "neckline", "neckline_conf",
    "silhouette", "silhouette_conf",
    "sleeve", "sleeve_conf",
    "pattern", "pattern_conf",
    "primary_color_lab", "color_temperature", "color_value",
    "occasion_tags",
    "clip_embedding",        # 512-d CLIP image embedding — derivative, openable
]


def write_taxonomy() -> dict:
    return {
        "neckline": [
            "v_neck", "deep_v_neck", "scoop", "crew", "boat",
            "high_neck", "sweetheart", "off_shoulder"
        ],
        "silhouette": [
            "bodycon", "fitted", "semi_fitted", "a_line", "loose", "oversized"
        ],
        "sleeve": ["sleeveless", "short", "three_quarter", "long"],
        "pattern": [
            "solid", "vertical_stripe", "horizontal_stripe",
            "small_print", "large_print"
        ],
        "color_temperature": ["warm", "cool", "neutral"],
        "color_value": ["light", "medium", "dark"],
        "category_id": {
            1: "short_sleeve_top",
            2: "long_sleeve_top",
            3: "short_sleeve_outwear",
            4: "long_sleeve_outwear",
            5: "vest",
            6: "sling",
        },
        "source": [
            "deepfashion2", "deepfashion_with_masks",
            "fashionpedia", "fashion_products_small"
        ],
        "occasion_tags": ["work", "casual", "date", "formal", "party"],
    }


def write_schema() -> dict:
    return {
        "garment_id": "string — '<source>:<image_id>:<item_idx>' (unique key)",
        "source": "string — one of {deepfashion2, deepfashion_with_masks, fashionpedia, fashion_products_small}",
        "image_id": "string — source-specific image identifier (use download_images.py to fetch)",
        "item_idx": "int — object index within the source image (multiple garments per image)",
        "category_id": "int (1-6) — garment category (see taxonomy.json)",
        "category_name": "string — human-readable category label",
        "bbox": "list[float] of length 4 — [x1, y1, x2, y2] in source image pixels",
        "neckline": "string — one of 8 neckline labels (CLIP zero-shot)",
        "neckline_conf": "float — softmax confidence for neckline label",
        "silhouette": "string — one of 6 silhouette labels (CLIP zero-shot)",
        "silhouette_conf": "float — softmax confidence for silhouette label",
        "sleeve": "string — one of 4 sleeve labels (CLIP zero-shot or category-derived)",
        "sleeve_conf": "float — softmax confidence (0 when category-derived)",
        "pattern": "string — one of 5 pattern labels (CLIP zero-shot)",
        "pattern_conf": "float — softmax confidence",
        "primary_color_lab": "list[float] of length 3 — [L, a, b] in CIE LAB color space, D65 white point",
        "color_temperature": "string — {warm, cool, neutral} derived from primary_color_lab hue angle",
        "color_value": "string — {light, medium, dark} derived from primary_color_lab L*",
        "occasion_tags": "list[string] — inferred occasion suggestions",
        "clip_embedding": "list[float] of length 512 — OpenCLIP ViT-B-32 (laion2b_s34b_b79k) image embedding, L2-normalized",
    }


def main() -> None:
    if not SRC_PARQUET.exists():
        print(f"!! source parquet missing at {SRC_PARQUET}")
        return

    BUNDLE.mkdir(parents=True, exist_ok=True)

    print(f"  reading {SRC_PARQUET}...")
    t = pq.read_table(SRC_PARQUET)
    print(f"  full catalog: {len(t):,} rows")

    # Select only publishable columns
    keep_cols = [c for c in PUBLISH_COLS if c in t.column_names]
    missing = set(PUBLISH_COLS) - set(keep_cols)
    if missing:
        print(f"  !! missing source cols (filling with defaults): {missing}")
    out = t.select(keep_cols)
    out_path = BUNDLE / "labels.parquet"
    pq.write_table(out, out_path)
    sz = out_path.stat().st_size / 1024 / 1024
    print(f"  wrote {out_path} ({sz:.1f} MB, {len(out):,} rows)")

    # 50-row sample for quick preview
    sample = out.slice(0, 50)
    pq.write_table(sample, BUNDLE / "examples/sample_50.parquet" if False else BUNDLE / "examples_sample_50.parquet")
    (BUNDLE / "examples").mkdir(exist_ok=True)
    shutil.move(str(BUNDLE / "examples_sample_50.parquet"), str(BUNDLE / "examples/sample_50.parquet"))

    # Taxonomy + schema
    (BUNDLE / "taxonomy.json").write_text(json.dumps(write_taxonomy(), indent=2, ensure_ascii=False))
    (BUNDLE / "schema.json").write_text(json.dumps(write_schema(), indent=2, ensure_ascii=False))
    print(f"  wrote taxonomy.json, schema.json")

    # Copy companion files
    for name, src in [
        ("seasonal_palettes.json", ROOT / "services/recommendation/data/seasonal_palettes.json"),
        ("vn_body_centroids.json", ROOT / "services/recommendation/data/vn_body_centroids_v1.json"),
        ("body_rules_vn.json", ROOT / "services/recommendation/data/upper_body_rules_vn_v1.json"),
    ]:
        if src.exists():
            shutil.copy2(src, BUNDLE / name)
            print(f"  copied {name}")

    # LICENSE — labels under CC BY 4.0
    (BUNDLE / "LICENSE").write_text("""Creative Commons Attribution 4.0 International License (CC BY 4.0)

The contents of this repository — including labels.parquet, taxonomy.json,
schema.json, seasonal_palettes.json, vn_body_centroids.json, body_rules_vn.json,
METHODOLOGY.md, and all original source code — are released under CC BY 4.0.

You are free to share and adapt this work for any purpose, including
commercial, provided that you give appropriate credit and indicate if changes
were made.

Citation:
  EA-VTON Team. (2026). EA-VTON Upper-Body Garment Attribute Labels: A 207k
  multi-source CLIP-derived catalog for population-aware style recommendation.
  https://github.com/InugamiDev/EA-VTON

For the source images themselves, see LICENSE.SOURCES.md — each source dataset
has its own license that you must comply with separately.

Full text: https://creativecommons.org/licenses/by/4.0/legalcode
""")

    (BUNDLE / "LICENSE.SOURCES.md").write_text("""# Upstream Source Licenses

This labeled catalog references images from four source datasets. **Our labels**
(labels.parquet, taxonomy.json, methodology, etc.) are CC BY 4.0. The **source
images** retain their original licenses, which you must comply with separately
when downloading them via `download_images.py`.

## Per-Source License Inheritance

| Source | Items | Source License | Redistributable? |
|---|---|---|---|
| DeepFashion2 | 139,788 | CUHK MMLab — non-commercial research only | No, fetch original |
| Fashionpedia | 37,875 | CC BY 4.0 (annotations); CC BY-NC for images | Limited |
| DeepFashion-with-masks | 25,122 | Apache 2.0 (HF mirror), derives from DeepFashion2 | Limited |
| fashion_products_small | 5,207 | Myntra source via Kaggle — verify per use case | No |

## Fetching the Originals

Use `download_images.py` (provided) to fetch each source's images into the
expected layout. The script obeys each source's access controls (e.g., the
DeepFashion2 form-gated download).

## Attribution

If you use this labeled catalog, cite both:
1. Our work (LICENSE)
2. The relevant upstream sources as documented in their READMEs.
""")
    print(f"  wrote LICENSE + LICENSE.SOURCES.md")

    # Methodology
    (BUNDLE / "METHODOLOGY.md").write_text("""# Methodology — How These Labels Were Generated

## Source Datasets

| Source | Items | Filtering |
|---|---|---|
| DeepFashion2 (train split) | 139,788 | Category 1-6 (upper body): short/long sleeve top, outwear, vest, sling |
| Fashionpedia | 37,875 | Category 0-5, 9 (shirt/top/sweater/cardigan/jacket/vest/coat) |
| DeepFashion-with-masks (HF) | 25,122 | gender=WOMEN AND cloth_type ∈ {Tees_Tanks, Blouses_Shirts, Sweaters, Jackets_Coats, Sweatshirts_Hoodies, Cardigans, Graphic_Tees, Shirts_Polos, Jackets_Vests, Suiting} |
| fashion_products_small (Myntra HF) | 5,207 | gender ∈ {Women, Unisex} AND subCategory=Topwear AND articleType ∈ upper-body set |
| **Total** | **207,992** | |

## Pipeline

```
source image → bbox/mask crop → CLIP image encode → softmax against prompts → label
                              ↓
                       per-pixel LAB → K-means → dominant color → temperature/value bin
```

### Step 1 — Crop
- **DeepFashion2**: bounding box from annotation JSON
- **DeepFashion-with-masks**: tight bbox computed from binary garment mask (cleaner than bbox)
- **Fashionpedia**: bounding box from detection annotation
- **fashion_products_small**: whole image (catalog-style flat-lay/ghost-mannequin)

### Step 2 — Visual Attributes (zero-shot CLIP)
- Model: OpenCLIP **ViT-B-32**, weights **laion2b_s34b_b79k**
- Each attribute dimension has 4-8 natural-language prompts
- argmax over softmax similarities → label
- Confidence is softmax probability (uniform-low for highly correlated prompts; argmax label still informative)

### Step 3 — Color
- 96×96 resized RGB → CIE LAB (D65) via standard matrix transform
- K-means (k=3) on LAB pixels
- Skip near-white background cluster (L>92 ∧ chroma<6) when present
- Largest non-background cluster centroid → `primary_color_lab`
- Derived bins:
  - `color_temperature` = "warm" if hue ∈ (-45°, 100°) else "cool"; "neutral" when chroma < 8
  - `color_value` = "dark" if L<35; "medium" if L<70; "light" otherwise

### Step 4 — Category-derived (no CLIP)
- `sleeve` for DeepFashion2: derived from category_id (1→short, 2→long, 5/6→sleeveless)
- `sleeve` for DeepFashion-with-masks and Fashionpedia: CLIP zero-shot
- `sleeve` for fashion_products_small: derived from `articleType` (Tshirts→short, Shirts→long, …)

## CLIP Embeddings
Each row carries a 512-dim **L2-normalized** CLIP image embedding. Use for:
- Visual similarity search (cosine ≈ dot product on normalized vectors)
- Two-tower retrieval models
- Cross-modal CLIP-text retrieval (encode query with same model)

## Hardware + Throughput
- M2 MacBook (MPS), no dedicated GPU
- DeepFashion2: 41.7 items/s end-to-end
- DeepFashion-with-masks: 40.0 items/s
- Fashionpedia: 32.9 items/s
- fashion_products_small: 53.8 items/s
- Total labeling time: ~78 minutes

## Limitations
- **Zero-shot confidence is uniform-low** (~0.12-0.21 mean) because our prompt set is highly correlated within each dim. Argmax label is still informative but absolute conf scores should not be treated as probability calibration.
- **No human-validated ground truth**. We have not measured per-attribute accuracy on a held-out human-labeled set.
- **Pattern bias**: solid pattern is over-represented (~46%) consistent with real catalog distributions.
- **Color**: dominant-color extraction can miss multicolor / heavily patterned garments (single LAB centroid).
- **DeepFashion2 image_id collisions across val/test/train**: this release uses train split only.
""")
    print(f"  wrote METHODOLOGY.md")

    # Dataset card
    (BUNDLE / "DATASET_CARD.md").write_text("""---
license: cc-by-4.0
task_categories:
- image-classification
- visual-question-answering
language:
- en
- vi
tags:
- fashion
- clothing
- garment
- style-recommendation
- vietnamese
- multi-source
- clip-zero-shot
size_categories:
- 100K<n<1M
---

# EA-VTON Upper-Body Garment Attribute Labels (v1)

**207,992** upper-body garment items labeled with a unified attribute taxonomy
across four public fashion datasets (DeepFashion2, Fashionpedia,
DeepFashion-with-masks, fashion_products_small). Each row carries:

- 7 categorical attributes (neckline, silhouette, sleeve, pattern, color temp/value, category)
- Primary color in CIE LAB
- 512-d OpenCLIP ViT-B-32 image embedding
- Bounding box / image pointer back to source

Built to support **population-aware style recommendation** research — paired
with our 12-season color palettes and 5-cluster Vietnamese body-type rules.

## Quick Start

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW items AS SELECT * FROM read_parquet('labels.parquet')")

# Find V-neck warm-tone fitted tops
con.execute('''
    SELECT garment_id, source, color_temperature, neckline, silhouette
    FROM items
    WHERE neckline IN ('v_neck', 'deep_v_neck')
      AND silhouette = 'fitted'
      AND color_temperature = 'warm'
    LIMIT 10
''').fetchall()
```

## Per-Source Breakdown

| Source | Items | Source License |
|---|---|---|
| DeepFashion2 | 139,788 | Research-only (CUHK MMLab) |
| Fashionpedia | 37,875 | CC BY 4.0 |
| DeepFashion-with-masks (HF) | 25,122 | Apache 2.0 |
| fashion_products_small (Myntra HF) | 5,207 | Verify per source |

## Files

- `labels.parquet` — 207,992 rows, ~640 MB
- `taxonomy.json` — label space (8 necklines × 6 silhouettes × 5 patterns × …)
- `schema.json` — column descriptions
- `seasonal_palettes.json` — 12 Sci\\ART seasons × 8 LAB anchors
- `vn_body_centroids.json` — 5 VN body-type cluster centroids (Trieu et al. 2024)
- `body_rules_vn.json` — body-type × attribute polarity rules
- `download_images.py` — fetch source images for any subset
- `METHODOLOGY.md` — full pipeline
- `LICENSE` — CC BY 4.0 for labels
- `LICENSE.SOURCES.md` — upstream image licenses

## Citation

```bibtex
@misc{eavton_labels_v1_2026,
  title={EA-VTON Upper-Body Garment Attribute Labels: A 207k Multi-Source
         CLIP-Derived Catalog for Population-Aware Style Recommendation},
  author={EA-VTON Team},
  year={2026},
  publisher={GitHub},
  url={https://github.com/InugamiDev/EA-VTON}
}
```

Also cite the upstream sources for any images you fetch.

## Limitations

- Labels are CLIP zero-shot; no human gold standard included
- DeepFashion2 train split only
- Single dominant color per item
- See `METHODOLOGY.md` for details
""")
    print(f"  wrote DATASET_CARD.md")

    # download_images.py
    (BUNDLE / "download_images.py").write_text("""#!/usr/bin/env python3
\"\"\"Fetch source images for a subset of labels.parquet.

Each source has different access controls. This script handles only the
HuggingFace-hosted sources automatically; DeepFashion2 requires manual
form-gated download from CUHK MMLab.

Usage:
    python download_images.py --subset deepfashion_with_masks --output ./images
    python download_images.py --subset fashionpedia --max 1000

Prerequisites:
    pip install pyarrow pillow huggingface_hub
\"\"\"

import argparse
import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

HF_REPOS = {
    "deepfashion_with_masks": ("SaffalPoosh/deepFashion-with-masks", "dataset"),
    "fashionpedia":           ("detection-datasets/fashionpedia", "dataset"),
    "fashion_products_small": ("ashraq/fashion-product-images-small", "dataset"),
}

MANUAL_INSTRUCTIONS = {
    "deepfashion2": \"\"\"DeepFashion2 requires form-based access from CUHK MMLab.

  1. Visit http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion2.html
  2. Fill out the access form, email Yuying Ge
  3. You will receive a password and Google Drive links
  4. Download train.zip, unzip with the supplied password
  5. Place at: <output>/deepfashion2/train/image/*.jpg
\"\"\",
}


def fetch_hf_source(subset: str, output_dir: Path, max_items: int = 0) -> None:
    from huggingface_hub import snapshot_download
    repo, kind = HF_REPOS[subset]
    target = output_dir / subset
    target.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {repo} → {target}")
    snapshot_download(repo_id=repo, repo_type=kind, local_dir=str(target))
    print(f"  done. extract images per dataset's own format.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True,
                    choices=["deepfashion2", "deepfashion_with_masks", "fashionpedia", "fashion_products_small"])
    ap.add_argument("--output", default="./images")
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    if args.subset == "deepfashion2":
        print(MANUAL_INSTRUCTIONS["deepfashion2"]); sys.exit(0)

    fetch_hf_source(args.subset, Path(args.output), args.max)


if __name__ == "__main__":
    main()
""")
    (BUNDLE / "download_images.py").chmod(0o755)
    print(f"  wrote download_images.py")

    # Final summary
    print()
    print("  ─── BUNDLE CONTENTS ───")
    for f in sorted(BUNDLE.rglob("*")):
        if f.is_file():
            sz = f.stat().st_size
            sz_str = f"{sz/1024/1024:.1f} MB" if sz > 1024*1024 else f"{sz/1024:.1f} KB"
            print(f"    {f.relative_to(BUNDLE)!s:42s} {sz_str:>10s}")
    total_mb = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  total: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
