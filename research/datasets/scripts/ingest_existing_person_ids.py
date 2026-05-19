"""Merge person_id from existing_person_clusters.jsonl into the catalog parquet.

Same shape as ingest_vn_person_ids.py but for the 208k pre-existing catalog.
NO face_clip_embedding column populated for these rows — only the integer
person_id. The embeddings were discarded by cluster_existing_faces.py per
the privacy-first design.

After this runs, you can query the catalog like:
    SELECT garment_id, source, person_id, cluster_size
    FROM items
    WHERE person_id >= 0
    AND cluster_size >= 5
    ORDER BY cluster_size DESC

to find catalog models who appear in ≥5 items.
"""

# intent: anonymised person_id merge — no biometric data persists
# status: ready (depends on cluster_existing_faces.py output)
# next: rebuild DuckDB view to surface person_id in /recommend-style/upper
# confidence: high

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
COMBINED = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
PERSON_JSONL = ROOT / "research/datasets/processed/combined/existing_person_clusters.jsonl"


def main() -> None:
    if not PERSON_JSONL.exists():
        sys.exit(f"!! {PERSON_JSONL} missing — run cluster_existing_faces.py first")
    if not COMBINED.exists():
        sys.exit(f"!! {COMBINED} missing")

    print(f"  loading {PERSON_JSONL.name}…")
    person_rows = []
    with open(PERSON_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            person_rows.append(json.loads(line))
    print(f"  person rows: {len(person_rows):,}")

    sizes = pd.Series([r["cluster_size"] for r in person_rows]).value_counts().sort_index()
    print(f"  cluster-size distribution (top 10):")
    print(sizes.head(10).to_string())

    print(f"\n  loading catalog…")
    df = pd.read_parquet(COMBINED)
    print(f"  catalog rows: {len(df):,}")

    if "person_id" not in df.columns:
        df["person_id"] = -1
    if "cluster_size" not in df.columns:
        df["cluster_size"] = 1

    # Build map; namespace existing person_ids so they don't collide with VN ones
    # (the VN ones use 0..89; we offset existing ones by +1000 to keep them disjoint)
    OFFSET = 1000
    person_map = {
        r["garment_id"]: (int(r["person_id"]) + OFFSET, int(r["cluster_size"]))
        for r in person_rows
    }
    print(f"\n  merging {len(person_map):,} person_ids (offset {OFFSET})…")

    # Only overwrite where current person_id is -1 (don't clobber VN ones)
    n_matched = 0
    for gid, (pid, csz) in person_map.items():
        mask = df["garment_id"] == gid
        if not mask.any():
            continue
        idx = df.index[mask][0]
        # only assign if currently unassigned
        if df.at[idx, "person_id"] == -1:
            df.at[idx, "person_id"] = pid
            df.at[idx, "cluster_size"] = csz
            n_matched += 1
    print(f"  matched + wrote {n_matched} rows")

    n_with_pid = (df["person_id"] != -1).sum()
    print(f"\n  total rows with person_id != -1 now: {n_with_pid:,}")

    df.to_parquet(COMBINED, index=False)
    print(f"  rewrote {COMBINED.name} ({COMBINED.stat().st_size/1024/1024:.0f} MB)")


if __name__ == "__main__":
    main()
