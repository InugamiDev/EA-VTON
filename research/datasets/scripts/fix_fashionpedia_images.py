"""Patch Fashionpedia image paths in the combined parquet.

The Fashionpedia labeler stored image_id = "fp_<shard>_<row>" but the combined
builder looked up by raw parquet's int image_id and missed them all. This
script:
  1. Loads the labeled Fashionpedia parquet
  2. Parses image_id → (shard_idx, row_idx)
  3. Extracts image bytes from the raw parquet at that row
  4. Saves to disk under images/fashionpedia/<sha>.jpg
  5. Builds a (garment_id → image_local_path) mapping
  6. Rewrites the combined parquet with patched paths for Fashionpedia rows
"""

# intent: backfill missing Fashionpedia image paths in combined catalog
# status: done
# next: verify rec service surfaces Fashionpedia items in top-K
# confidence: high

from __future__ import annotations

import glob
import hashlib
import io
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
COMBINED = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
RAW_DIR = ROOT / "research/datasets/raw/fashionpedia/data"
IMG_DIR = ROOT / "research/datasets/processed/combined/images/fashionpedia"


def main() -> None:
    if not COMBINED.exists():
        print(f"!! combined parquet missing at {COMBINED}", file=sys.stderr)
        return

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load all raw shards into memory (the image column only)
    raw_shards = sorted(glob.glob(str(RAW_DIR / "*.parquet")))
    raw_image_cols = []
    for p in raw_shards:
        t = pq.read_table(p, columns=["image"])
        raw_image_cols.append(t.column("image"))
    print(f"  loaded {len(raw_shards)} raw shards")

    # Load combined parquet
    print(f"  loading combined parquet...")
    combined = pq.read_table(COMBINED)
    sources = combined.column("source").to_pylist()
    image_ids = combined.column("image_id").to_pylist()
    image_paths = combined.column("image_local_path").to_pylist()
    print(f"  {len(combined):,} rows total")

    fp_indices = [i for i, s in enumerate(sources) if s == "fashionpedia"]
    print(f"  fashionpedia rows: {len(fp_indices):,}")

    # Backfill image paths for fashionpedia rows
    fixed = 0
    skipped = 0
    for idx in fp_indices:
        iid = image_ids[idx]
        # parse "fp_<si>_<ri>"
        parts = iid.split("_")
        if len(parts) != 3 or parts[0] != "fp":
            skipped += 1
            continue
        try:
            si = int(parts[1])
            ri = int(parts[2])
        except ValueError:
            skipped += 1
            continue

        if si >= len(raw_image_cols):
            skipped += 1; continue

        try:
            img_bytes = raw_image_cols[si][ri].as_py()["bytes"]
        except Exception:
            skipped += 1; continue

        h = hashlib.sha1(img_bytes).hexdigest()[:20]
        target = IMG_DIR / f"{h}.jpg"
        if not target.exists():
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img.save(target, format="JPEG", quality=85)
            except Exception:
                skipped += 1; continue

        rel_path = str(target.relative_to(ROOT))
        image_paths[idx] = rel_path
        fixed += 1
        if fixed % 2000 == 0:
            print(f"    fixed {fixed:,}/{len(fp_indices):,}", flush=True)

    print(f"\n  fixed {fixed:,} Fashionpedia paths, skipped {skipped:,}")

    # Rewrite combined parquet with patched paths
    print("  rewriting combined parquet...")
    new_table = combined.set_column(
        combined.column_names.index("image_local_path"),
        "image_local_path",
        pa.array(image_paths, type=pa.string()),
    )
    pq.write_table(new_table, COMBINED)
    print(f"  wrote {COMBINED} ({new_table.num_rows:,} rows)")


if __name__ == "__main__":
    main()
