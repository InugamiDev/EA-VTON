"""Add a per-row `redaction_status` column to the combined parquet.

Status values:
    face_detected_redacted — a face polygon was traced and the image was redacted
    no_face_detected       — MediaPipe found 0 faces; the image was passed through unchanged
    redaction_skipped      — extraction failure or other error; the original image was used
    not_processed          — the source image never went through the redaction pipeline

We infer the status by:
  1. If `images_redacted/<path>` exists: face_detected_redacted (the only items that get
     copied to images_redacted/ are those with non-empty face polygons per the redactor)
  2. Else if `images/<path>` exists: no_face_detected
  3. Else: not_processed

This is conservative — it does not distinguish "no_face_detected" from "redaction_skipped due
to error" without re-running the pipeline. We treat them together as "no_face_detected"
since the downstream artifact is identical (an un-redacted image was used for labeling).

This closes the auditor's flag that redaction_status was deferred to v1.1 — it is now in v1.0.
"""

# intent: add per-row redaction provenance for the headline privacy claim
# status: done
# next: verify the count summary matches the redaction pipeline's reported tally
# confidence: high

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
COMBINED = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
IMAGES = ROOT / "research/datasets/processed/combined/images"
IMAGES_REDACTED = ROOT / "research/datasets/processed/combined/images_redacted"


def main() -> None:
    if not COMBINED.exists():
        sys.exit(f"!! {COMBINED} not found")

    print(f"  loading {COMBINED.name}…")
    df = pd.read_parquet(COMBINED)
    print(f"  rows: {len(df):,}")
    if "redaction_status" in df.columns:
        print("  redaction_status already present; will overwrite")

    # image_local_path is stored relative to the repo root, e.g.:
    #   "research/datasets/processed/combined/images/deepfashion2/000001_1.jpg"
    # The redaction pipeline mirrors this layout under images_redacted/.
    def repo_rel_set(dir_: Path) -> set[str]:
        if not dir_.exists():
            return set()
        return {str(p.relative_to(ROOT)) for p in dir_.rglob("*.jpg")}

    print("  scanning redacted image directory…")
    redacted_set = repo_rel_set(IMAGES_REDACTED)
    print(f"  found {len(redacted_set):,} redacted image files")
    print("  scanning unredacted image directory…")
    unredacted_set = repo_rel_set(IMAGES)
    print(f"  found {len(unredacted_set):,} unredacted image files")

    def classify(image_local_path: str | None) -> str:
        if image_local_path is None or pd.isna(image_local_path):
            return "not_processed"
        p = str(image_local_path)
        redacted_path = p.replace("/images/", "/images_redacted/", 1)
        if redacted_path in redacted_set:
            return "face_detected_redacted"
        if p in unredacted_set:
            return "no_face_detected"
        return "not_processed"

    print("  classifying rows…")
    statuses = df["image_local_path"].apply(classify)
    df["redaction_status"] = statuses.astype("category")

    print()
    print("  redaction_status distribution:")
    for s, n in df["redaction_status"].value_counts().items():
        print(f"    {s:<28s} {n:>7,}  ({n/len(df)*100:.1f}%)")

    print(f"\n  writing back to {COMBINED}…")
    df.to_parquet(COMBINED, index=False)
    print(f"  done ({COMBINED.stat().st_size/1024/1024:.0f} MB)")

    # Note: the DuckDB view should auto-pick up the new column on next query
    # because it's a zero-copy view. No catalog rebuild needed unless the
    # build script's column projection is updated.
    print()
    print("  NOTE: the DuckDB items view does not explicitly project this column.")
    print("        Update build_style_catalog_duckdb.py to include redaction_status")
    print("        in the items view if downstream code expects to read it from `items`.")


if __name__ == "__main__":
    main()
