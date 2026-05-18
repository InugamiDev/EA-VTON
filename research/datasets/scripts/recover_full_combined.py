"""Full one-shot recovery of items_upper_combined.parquet.

Two-stage:
  Stage A — rerun build_combined_catalog.py to recover ALL image_local_paths
            (re-extracts embedded images from the HF parquets; idempotent if
            the .jpg files already exist on disk because the hash is content-derived).
  Stage B — merge the rich label columns (CLIP embedding + suits_g1..5 +
            best_season + style_personality + season_distance_de2000 + etc.)
            from the bundle backup at dataset_bundle_v1/labels.parquet onto
            the freshly-rebuilt parquet, keyed by garment_id.
  Stage C — re-add redaction_status column by checking file presence under
            combined/images_redacted/.

The result is items_upper_combined.parquet with:
    - 208,069 rows
    - correct image_local_path for ALL sources (not just DF2)
    - CLIP embeddings + person-categorization labels intact (from bundle)
    - redaction_status correctly populated by on-disk file check
"""

# intent: full restore after the smoke-test bug truncated the source parquet
# status: ready to run; expected ~15-30 min wall-clock (image re-extraction is the slow part)
# next: rerun build_style_catalog_duckdb.py to refresh the DuckDB view
# confidence: high

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "research/datasets/scripts"
BUILD = SCRIPTS / "build_combined_catalog.py"
BUNDLE = ROOT / "research/datasets/processed/dataset_bundle_v1/labels.parquet"
COMBINED = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
BACKUP = COMBINED.with_suffix(".pre_recovery_bak.parquet")
IMAGES = ROOT / "research/datasets/processed/combined/images"
IMAGES_REDACTED = ROOT / "research/datasets/processed/combined/images_redacted"


def stage_a_rebuild() -> None:
    print("══ Stage A — rerun build_combined_catalog.py to recover image_local_paths ══")
    if COMBINED.exists():
        shutil.copy2(COMBINED, BACKUP)
        print(f"  backed up current parquet → {BACKUP.name}")
    # Rerun the builder. This re-extracts embedded images (idempotent) and
    # writes a fresh basic items_upper_combined.parquet with correct paths.
    res = subprocess.run(
        [sys.executable, str(BUILD)],
        cwd=str(ROOT),
        capture_output=False,
    )
    if res.returncode != 0:
        sys.exit(f"!! build_combined_catalog.py exited with code {res.returncode}")
    print(f"  Stage A done; {COMBINED.name} now has {pd.read_parquet(COMBINED, columns=['garment_id']).shape[0]:,} rows")


def stage_b_merge_bundle() -> None:
    print()
    print("══ Stage B — merge rich label columns from bundle backup ══")
    if not BUNDLE.exists():
        sys.exit(f"!! bundle backup missing at {BUNDLE}; cannot merge rich labels")

    basic = pd.read_parquet(COMBINED)
    bundle = pd.read_parquet(BUNDLE)
    print(f"  basic (freshly rebuilt): {len(basic):,} rows, {len(basic.columns)} cols")
    print(f"  bundle: {len(bundle):,} rows, {len(bundle.columns)} cols")

    # Columns that the bundle has but basic build does NOT
    bundle_only_cols = ["clip_embedding",
                        "suits_g1", "suits_g2", "suits_g3", "suits_g4", "suits_g5",
                        "best_suits_cluster", "best_suits_score",
                        "best_season", "season_family", "season_distance_de2000",
                        "style_personality"]
    cols_to_merge = ["garment_id"] + [c for c in bundle_only_cols if c in bundle.columns]
    missing = [c for c in bundle_only_cols if c not in bundle.columns]
    if missing:
        print(f"  WARN: bundle missing expected cols {missing}; will leave them out")
    merged = basic.merge(bundle[cols_to_merge], on="garment_id", how="left")
    new_cols = set(merged.columns) - set(basic.columns)
    n_with_clip = int(merged["clip_embedding"].notna().sum()) if "clip_embedding" in merged.columns else 0
    print(f"  merged in {len(new_cols)} columns from bundle")
    print(f"  rows with non-null clip_embedding: {n_with_clip:,} / {len(merged):,}")

    merged.to_parquet(COMBINED, index=False)
    print(f"  rewrote {COMBINED} ({COMBINED.stat().st_size/1024/1024:.0f} MB, {len(merged.columns)} cols)")


def stage_c_redaction_status() -> None:
    print()
    print("══ Stage C — re-add redaction_status by file existence ══")
    df = pd.read_parquet(COMBINED)

    def repo_rel_set(dir_: Path) -> set[str]:
        if not dir_.exists():
            return set()
        return {str(p.relative_to(ROOT)) for p in dir_.rglob("*.jpg")}

    print("  scanning redacted image directory…")
    redacted_set = repo_rel_set(IMAGES_REDACTED)
    print(f"   found {len(redacted_set):,} redacted JPEGs")
    print("  scanning unredacted image directory…")
    unredacted_set = repo_rel_set(IMAGES)
    print(f"   found {len(unredacted_set):,} unredacted JPEGs")

    def classify(p):
        if p is None or pd.isna(p):
            return "not_processed"
        red = str(p).replace("/images/", "/images_redacted/", 1)
        if red in redacted_set:
            return "face_detected_redacted"
        if str(p) in unredacted_set:
            return "no_face_detected"
        return "not_processed"

    df["redaction_status"] = df["image_local_path"].apply(classify)
    print()
    print("  redaction_status distribution:")
    for s, n in df["redaction_status"].value_counts().items():
        print(f"    {s:<28s} {n:>7,}  ({n/len(df)*100:.1f}%)")

    df.to_parquet(COMBINED, index=False)
    print(f"\n  finalized {COMBINED} ({COMBINED.stat().st_size/1024/1024:.0f} MB)")


def main() -> None:
    stage_a_rebuild()
    stage_b_merge_bundle()
    stage_c_redaction_status()
    print()
    print("══ Recovery complete ══")
    print(f"  → next: rerun build_style_catalog_duckdb.py to refresh DuckDB view")


if __name__ == "__main__":
    main()
