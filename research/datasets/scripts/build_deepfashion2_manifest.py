"""Build a flat parquet manifest from DeepFashion2 annotations.

Source layout (after our import):
  research/datasets/raw/deepfashion2/
    train/annos/000001.json ... 191961.json
    train/image/000001.jpg
    test/json/{keypoints_test_information.json, retrieval_*.json}
    test/image/000001.jpg
    val/json/{keypoints_val_information.json, retrieval_*.json}
    val/image/  (after unzip)

Output:
  research/datasets/processed/deepfashion2/items_all.parquet      (every item)
  research/datasets/processed/deepfashion2/items_upper.parquet    (categories 1-6)

DeepFashion2 categories:
   1 short sleeve top      ← upper
   2 long sleeve top       ← upper
   3 short sleeve outwear  ← upper
   4 long sleeve outwear   ← upper
   5 vest                  ← upper
   6 sling                 ← upper
   7 shorts
   8 trousers
   9 skirt
  10 short sleeve dress
  11 long sleeve dress
  12 vest dress
  13 sling dress

Per-item annotation schema (DeepFashion2 README):
  category_id, category_name, bounding_box [x1,y1,x2,y2],
  segmentation (list of polygons), landmarks [25 triplets (x,y,vis)],
  scale {1,2,3}, viewpoint {1=front,2=side/back,3=zoom}, zoom_in,
  occlusion {1=slight, 2=medium, 3=heavy}, style (sub-style id)

Train annotations are one JSON per image with items keyed item1..itemN.
Test/val use the keypoints_*.json bundles instead.
"""

# intent: build manifest of DeepFashion2 items for style recommendation catalog
# status: done
# next: filter to upper-body, then crop garment images for CLIP embedding
# confidence: high

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "research/datasets/raw/deepfashion2"
OUT = ROOT / "research/datasets/processed/deepfashion2"

UPPER_CATEGORIES = {1, 2, 3, 4, 5, 6}

CATEGORY_NAMES = {
    1: "short_sleeve_top",
    2: "long_sleeve_top",
    3: "short_sleeve_outwear",
    4: "long_sleeve_outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeve_dress",
    11: "long_sleeve_dress",
    12: "vest_dress",
    13: "sling_dress",
}


def iter_train_items() -> Iterable[dict]:
    """Yield one record per garment item from train split."""
    annos_dir = RAW / "train/annos"
    if not annos_dir.exists():
        print(f"!! train annos not found at {annos_dir}", file=sys.stderr)
        return

    files = sorted(annos_dir.glob("*.json"))
    total = len(files)
    print(f"  train: {total} annotation files")

    for i, anno_path in enumerate(files):
        if i % 5000 == 0:
            print(f"    train progress: {i}/{total}")
        with open(anno_path, "r") as f:
            anno = json.load(f)

        image_id = anno_path.stem  # "000001"
        image_path = f"train/image/{image_id}.jpg"
        source = anno.get("source", "unknown")
        pair_id = anno.get("pair_id")

        for key, item in anno.items():
            if not key.startswith("item"):
                continue
            yield {
                "split": "train",
                "image_id": image_id,
                "image_path": image_path,
                "item_idx": int(key.replace("item", "")),
                "source": source,
                "pair_id": pair_id,
                "category_id": item["category_id"],
                "category_name": CATEGORY_NAMES.get(item["category_id"], item.get("category_name", "unknown")),
                "bbox": item["bounding_box"],
                "scale": item.get("scale"),
                "viewpoint": item.get("viewpoint"),
                "zoom_in": item.get("zoom_in"),
                "occlusion": item.get("occlusion"),
                "style": item.get("style"),
            }


def iter_test_val_items(split: str) -> Iterable[dict]:
    """Yield items from test/val split (single-file keypoints_*_information.json)."""
    info_name = f"keypoints_{'test' if split == 'test' else 'val'}_information.json"
    info_path = RAW / split / "json" / info_name
    if not info_path.exists():
        print(f"!! {split} info not found at {info_path}", file=sys.stderr)
        return

    print(f"  {split}: reading {info_path.name}")
    with open(info_path, "r") as f:
        info = json.load(f)

    images = info.get("images", info)
    if isinstance(images, dict):
        images = list(images.values())

    annotations = info.get("annotations", [])
    if not annotations:
        print(f"!! no annotations in {split} info", file=sys.stderr)
        return

    # Build image_id → file_name map from images list
    img_index = {}
    if isinstance(images, list):
        for img in images:
            img_index[img.get("id", img.get("image_id"))] = img.get("file_name", f"{img.get('id')}.jpg")

    print(f"    {split}: {len(annotations)} annotation entries")
    for ann in annotations:
        image_id_int = ann.get("image_id")
        image_id = str(image_id_int).zfill(6) if image_id_int else ann.get("file_name", "").replace(".jpg", "")
        file_name = img_index.get(image_id_int, f"{image_id}.jpg")

        yield {
            "split": split,
            "image_id": image_id,
            "image_path": f"{split}/image/{file_name}",
            "item_idx": ann.get("id", 0),  # unique per-annotation
            "source": "shop_or_user",
            "pair_id": ann.get("pair_id"),
            "category_id": ann.get("category_id"),
            "category_name": CATEGORY_NAMES.get(ann.get("category_id"), "unknown"),
            "bbox": ann.get("bbox"),
            "scale": ann.get("scale"),
            "viewpoint": ann.get("viewpoint"),
            "zoom_in": ann.get("zoom_in"),
            "occlusion": ann.get("occlusion"),
            "style": ann.get("style"),
        }


def build_manifest(splits: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for split in splits:
        print(f"── building {split} ──")
        if split == "train":
            all_records.extend(iter_train_items())
        else:
            all_records.extend(iter_test_val_items(split))

    print(f"\nTotal items collected: {len(all_records)}")

    # Cast and write all
    schema = pa.schema([
        ("split", pa.string()),
        ("image_id", pa.string()),
        ("image_path", pa.string()),
        ("item_idx", pa.int32()),
        ("source", pa.string()),
        ("pair_id", pa.int32()),
        ("category_id", pa.int32()),
        ("category_name", pa.string()),
        ("bbox", pa.list_(pa.float32())),
        ("scale", pa.int8()),
        ("viewpoint", pa.int8()),
        ("zoom_in", pa.int8()),
        ("occlusion", pa.int8()),
        ("style", pa.int8()),
    ])

    table = pa.Table.from_pylist(all_records, schema=schema)
    full_path = OUT / "items_all.parquet"
    pq.write_table(table, full_path)
    print(f"  wrote {full_path} ({len(table)} rows)")

    # Filter to upper-body and write a separate parquet for fast loading
    upper_mask = pa.compute.is_in(table.column("category_id"), value_set=pa.array(list(UPPER_CATEGORIES), pa.int32()))
    upper_table = table.filter(upper_mask)
    upper_path = OUT / "items_upper.parquet"
    pq.write_table(upper_table, upper_path)
    print(f"  wrote {upper_path} ({len(upper_table)} upper-body rows)")

    # Per-category counts
    print("\nPer-category counts:")
    cats = table.column("category_id").to_pylist()
    from collections import Counter
    for cid, n in sorted(Counter(cats).items()):
        marker = "← upper" if cid in UPPER_CATEGORIES else ""
        print(f"  {cid:2d} {CATEGORY_NAMES.get(cid, '?'):24s} {n:>7,} {marker}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test", "val"])
    args = ap.parse_args()
    build_manifest(args.splits)


if __name__ == "__main__":
    main()
