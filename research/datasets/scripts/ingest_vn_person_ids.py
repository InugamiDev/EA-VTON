"""Merge person_id + face_clip_embedding from vn_person_clusters.jsonl into
items_upper_combined.parquet.

After vn_person_pipeline.py produces the JSONL with per-item person_id,
this script:
  1. Reads the catalog parquet.
  2. Adds two columns: `person_id` (int) and `face_clip_embedding`
     (list[float] of length 512, or null).
  3. For rows in vn_yody or vn_coupletx, fills from the JSONL.
  4. For all other rows, leaves these columns null/-1 (those items have
     already been face-redacted; we cannot extract face embeddings retroactively).

Output: items_upper_combined.parquet is updated in place.

Why this matters: at recommendation time, if the user uploads a selfie, we
can compute their face CLIP embedding, do nearest-neighbor against the
known person_ids in the catalog, and "you look like this model → here are
items she wears" becomes a real recommendation signal.
"""

# intent: add person_id + face_clip_embedding columns to the catalog
# status: ready (depends on vn_person_pipeline.py output)
# next: surface in /recommend-style/upper as an optional similarity boost
# confidence: high

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
COMBINED = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
PERSON_JSONL = ROOT / "research/datasets/raw/vn_brands/vn_person_clusters.jsonl"


def main() -> None:
    if not PERSON_JSONL.exists():
        sys.exit(f"!! {PERSON_JSONL} missing — run vn_person_pipeline.py first")
    if not COMBINED.exists():
        sys.exit(f"!! {COMBINED} missing")

    print(f"  loading {PERSON_JSONL.name}…")
    person_rows = []
    with open(PERSON_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            person_rows.append(json.loads(line))
    pdf = pd.DataFrame(person_rows)
    print(f"  person rows: {len(pdf)}")
    print(f"  person_id distribution:")
    print(pdf["person_id"].value_counts().head(15))

    print(f"\n  loading catalog…")
    df = pd.read_parquet(COMBINED)
    print(f"  catalog rows: {len(df):,}")

    # Initialize new columns with neutral defaults
    if "person_id" not in df.columns:
        df["person_id"] = -1
    if "face_clip_embedding" not in df.columns:
        df["face_clip_embedding"] = None

    # Build a garment_id → (person_id, face_clip_embedding) map
    person_map = {
        r["garment_id"]: (int(r["person_id"]), r["face_clip_embedding"])
        for r in person_rows
    }
    print(f"\n  mapping {len(person_map)} VN garment_ids back to the catalog…")

    n_matched = 0
    for gid, (pid, emb) in person_map.items():
        mask = df["garment_id"] == gid
        if mask.any():
            df.loc[mask, "person_id"] = pid
            # face_clip_embedding has to be assigned per-row for list type
            idx = df.index[mask][0]
            df.at[idx, "face_clip_embedding"] = emb
            n_matched += 1
    print(f"  matched + wrote {n_matched} rows")
    n_unmatched = len(person_map) - n_matched
    if n_unmatched:
        print(f"  WARNING: {n_unmatched} JSONL rows did NOT match any catalog garment_id "
              "(probably means the catalog parquet hasn't absorbed these yet — run "
              "recover_full_combined.py first)")

    df.to_parquet(COMBINED, index=False)
    print(f"\n  rewrote {COMBINED.name} (now has person_id + face_clip_embedding cols)")
    print(f"  rows with person_id != -1: {(df['person_id'] != -1).sum():,}")
    print(f"  rows with face_clip_embedding non-null: {df['face_clip_embedding'].notna().sum():,}")


if __name__ == "__main__":
    main()
