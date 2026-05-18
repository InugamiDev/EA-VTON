"""Recover items_upper_combined.parquet after it was accidentally truncated.

Source of truth: research/datasets/processed/dataset_bundle_v1/labels.parquet
    — has 208,069 rows with all labels + CLIP embeddings, but lacks image_local_path
    and redaction_status.

Recovery steps:
  1. Load bundle (208,069 rows)
  2. Reconstruct image_local_path per row from (source, image_id, item_idx)
     using the same convention the combined builder used.
  3. Add `split`, `title`, `occlusion`, `viewpoint`, `sleeve_conf` from per-source
     labeled parquets where available (some downstream code uses them).
  4. Run the redaction-status classifier inline to add `redaction_status`.
  5. Write to research/datasets/processed/combined/items_upper_combined.parquet.

This script is idempotent — running it multiple times produces the same output.
"""

# intent: restore the source-of-truth combined parquet that was truncated
# status: done
# next: never repeat the bug; the filter_person_visible script no longer overwrites this file
# confidence: high

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "research/datasets/processed/dataset_bundle_v1/labels.parquet"
OUT    = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
IMAGES = ROOT / "research/datasets/processed/combined/images"
IMAGES_REDACTED = ROOT / "research/datasets/processed/combined/images_redacted"

# Per-source labeled parquets — used to recover supplemental columns
SOURCES = {
    "deepfashion2":            ROOT / "research/datasets/processed/deepfashion2/items_upper_labeled.parquet",
    "deepfashion_with_masks":  ROOT / "research/datasets/processed/deepfashion_with_masks/items_upper_labeled.parquet",
    "fashionpedia":            ROOT / "research/datasets/processed/fashionpedia/items_upper_labeled.parquet",
    "fashion_products_small":  ROOT / "research/datasets/processed/fashion_products_small/items_upper_labeled.parquet",
}


def reconstruct_image_local_path(row: pd.Series) -> str:
    """Mirror the combined builder's path convention."""
    src = row["source"]
    img = row["image_id"]
    idx = row["item_idx"]
    if src in ("deepfashion2", "deepfashion_with_masks"):
        return f"research/datasets/processed/combined/images/{src}/{img}_{idx}.jpg"
    if src == "fashionpedia":
        return f"research/datasets/processed/combined/images/fashionpedia/{img}_{idx}.jpg"
    if src == "fashion_products_small":
        return f"research/datasets/processed/combined/images/fashion_products_small/{img}_{idx}.jpg"
    if src == "vn_dottie":
        return f"research/datasets/processed/combined/images/vn_dottie/{img}.jpg"
    if src == "vn_ivy_moda":
        return f"research/datasets/processed/combined/images/vn_ivy_moda/{img}.jpg"
    return ""


def classify_redaction(image_local_path: str, redacted_set: set, unredacted_set: set) -> str:
    if not image_local_path:
        return "not_processed"
    redacted_path = image_local_path.replace("/images/", "/images_redacted/", 1)
    if redacted_path in redacted_set:
        return "face_detected_redacted"
    if image_local_path in unredacted_set:
        return "no_face_detected"
    return "not_processed"


def main() -> None:
    if not BUNDLE.exists():
        sys.exit(f"!! {BUNDLE} not found — cannot recover")

    print(f"  loading bundle backup: {BUNDLE.name} ({BUNDLE.stat().st_size/1024/1024:.0f} MB)")
    df = pd.read_parquet(BUNDLE)
    print(f"  bundle rows: {len(df):,}, cols: {len(df.columns)}")

    # Reconstruct image_local_path
    print("  reconstructing image_local_path…")
    df["image_local_path"] = df.apply(reconstruct_image_local_path, axis=1)
    n_missing_path = (df["image_local_path"] == "").sum()
    print(f"  rows with no image_local_path: {n_missing_path}")

    # Pull supplemental columns from per-source parquets
    print("  joining supplemental columns from per-source parquets…")
    for src_name, src_path in SOURCES.items():
        if not src_path.exists():
            print(f"   - {src_name}: source parquet missing, skipping")
            continue
        sdf = pd.read_parquet(src_path)
        # Build a key matching the bundle's garment_id format
        if src_name in ("deepfashion2", "deepfashion_with_masks"):
            # garment_id = "{source}:{image_id}:{item_idx}"
            sdf["garment_id"] = src_name + ":" + sdf["image_id"].astype(str) + ":" + sdf["item_idx"].astype(str)
        elif src_name == "fashionpedia":
            sdf["garment_id"] = "fashionpedia:" + sdf["image_id"].astype(str) + ":" + sdf["item_idx"].astype(str)
        elif src_name == "fashion_products_small":
            sdf["garment_id"] = "fashion_products_small:" + sdf["image_id"].astype(str) + ":" + sdf["item_idx"].astype(str)
        # Keep only columns we need to backfill, drop columns already in bundle
        keep_cols = ["garment_id"]
        for c in ["split", "occlusion", "viewpoint", "title", "sleeve_conf"]:
            if c in sdf.columns and c not in df.columns:
                keep_cols.append(c)
        if len(keep_cols) > 1:
            sub = sdf[keep_cols].drop_duplicates(subset="garment_id")
            before = set(df.columns)
            df = df.merge(sub, on="garment_id", how="left")
            print(f"   - {src_name}: added cols {set(df.columns) - before}")

    # Add redaction_status by checking file presence
    print("  classifying redaction_status (by file existence)…")
    def repo_rel_set(dir_: Path) -> set[str]:
        if not dir_.exists():
            return set()
        return {str(p.relative_to(ROOT)) for p in dir_.rglob("*.jpg")}
    redacted_set = repo_rel_set(IMAGES_REDACTED)
    unredacted_set = repo_rel_set(IMAGES)
    print(f"   redacted files on disk: {len(redacted_set):,}")
    print(f"   unredacted files on disk: {len(unredacted_set):,}")

    df["redaction_status"] = df["image_local_path"].apply(
        lambda p: classify_redaction(p, redacted_set, unredacted_set)
    )
    print()
    print("  redaction_status distribution:")
    for s, n in df["redaction_status"].value_counts().items():
        print(f"    {s:<28s} {n:>7,}  ({n/len(df)*100:.1f}%)")

    # Write
    OUT.parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(OUT, index=False)
    print(f"\n  wrote {OUT} ({OUT.stat().st_size/1024/1024:.0f} MB, {len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
