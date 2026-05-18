"""Filter the combined catalog to items where a PERSON is visibly wearing the garment.

Problem: the 208k catalog mixes (a) on-model shots and (b) flatlay/product-only shots
without a person. The hand-crafted body-rule scoring (`Flatter` in FFM) and the cold-start
calibration both assume the silver labels reflect "on a person" suitability. Flatlay-only
images bias those labels.

Solution: run MediaPipe PoseLandmarker on every image (or its redacted version). Keep only
items where the pose detector returns BOTH shoulder landmarks AND at least one hip
landmark with visibility ≥ 0.5. That guarantees an upper-body posed frame.

Output:
    research/datasets/processed/combined/items_upper_combined_personvisible.parquet
    research/datasets/processed/combined/person_visibility_report.json

Runtime: ~1.5-2 hours on CPU (multiprocessing 8 workers); 208,069 images, ~5 imgs/sec/worker.
On A100 this is still CPU-bound — MediaPipe doesn't accelerate on GPU.

Usage:
    .venv/bin/python3 research/datasets/scripts/filter_person_visible.py \\
        --workers 8 --batch-log 500 --use-redacted
"""

# intent: produce a clean person-visible subset; remove flatlay/product-only shots
# status: ready to run
# next: rebuild the DuckDB view over the filtered parquet for downstream consumers
# confidence: high

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PARQUET = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
OUT_PARQUET = ROOT / "research/datasets/processed/combined/items_upper_combined_personvisible.parquet"
REPORT = ROOT / "research/datasets/processed/combined/person_visibility_report.json"
POSE_MODEL = ROOT / "services/feature-service/models/pose_landmarker_heavy.task"

# MediaPipe pose landmark indices (33-point model)
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
VISIBILITY_THRESHOLD = 0.5


def init_worker(use_redacted: bool):
    """Each worker initializes its own PoseLandmarker (not picklable)."""
    global _lm, _use_redacted
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=str(POSE_MODEL))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    _lm = vision.PoseLandmarker.create_from_options(options)
    _use_redacted = use_redacted


def classify_one(row_args: tuple) -> dict:
    """Run pose detection on one image; return a dict with the keep decision."""
    gid, image_local_path = row_args
    try:
        from PIL import Image
        import mediapipe as mp
        import numpy as _np

        if _use_redacted:
            path = ROOT / image_local_path.replace("/images/", "/images_redacted/", 1)
            if not path.exists():
                # fall back to unredacted if no redacted version
                path = ROOT / image_local_path
        else:
            path = ROOT / image_local_path

        if not path.exists():
            return {"garment_id": gid, "keep": False, "reason": "file_missing"}

        img = _np.asarray(Image.open(path).convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        result = _lm.detect(mp_image)

        if not result.pose_landmarks:
            return {"garment_id": gid, "keep": False, "reason": "no_pose"}

        # Take the first detected pose
        landmarks = result.pose_landmarks[0]
        ls = landmarks[LM_LEFT_SHOULDER].visibility
        rs = landmarks[LM_RIGHT_SHOULDER].visibility
        lh = landmarks[LM_LEFT_HIP].visibility
        rh = landmarks[LM_RIGHT_HIP].visibility

        both_shoulders = ls >= VISIBILITY_THRESHOLD and rs >= VISIBILITY_THRESHOLD
        any_hip = lh >= VISIBILITY_THRESHOLD or rh >= VISIBILITY_THRESHOLD

        if both_shoulders and any_hip:
            return {
                "garment_id": gid, "keep": True, "reason": "ok",
                "vis_lsho": float(ls), "vis_rsho": float(rs),
                "vis_lhip": float(lh), "vis_rhip": float(rh),
            }
        elif both_shoulders:
            return {"garment_id": gid, "keep": False, "reason": "shoulders_only_no_hip",
                    "vis_lsho": float(ls), "vis_rsho": float(rs),
                    "vis_lhip": float(lh), "vis_rhip": float(rh)}
        else:
            return {"garment_id": gid, "keep": False, "reason": "shoulders_not_visible",
                    "vis_lsho": float(ls), "vis_rsho": float(rs),
                    "vis_lhip": float(lh), "vis_rhip": float(rh)}
    except Exception as e:  # noqa: BLE001
        return {"garment_id": gid, "keep": False, "reason": f"error:{type(e).__name__}"}


def iter_chunks(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-log", type=int, default=500,
                    help="print progress every N rows")
    ap.add_argument("--use-redacted", action="store_true",
                    help="run pose detection on redacted images (faces already removed)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N rows (for smoke testing)")
    args = ap.parse_args()

    if not POSE_MODEL.exists():
        sys.exit(f"!! pose model not found at {POSE_MODEL}")

    df = pd.read_parquet(PARQUET)
    print(f"  catalog rows: {len(df):,}")
    if args.limit:
        df = df.head(args.limit)
        print(f"  limited to {len(df):,} rows")

    row_args = list(zip(df["garment_id"].tolist(), df["image_local_path"].tolist()))

    print(f"  spawning {args.workers} workers; use_redacted={args.use_redacted}")
    results: list[dict] = []
    with Pool(processes=args.workers,
              initializer=init_worker,
              initargs=(args.use_redacted,)) as pool:
        for i, r in enumerate(pool.imap_unordered(classify_one, row_args, chunksize=32), 1):
            results.append(r)
            if i % args.batch_log == 0:
                kept = sum(1 for x in results if x["keep"])
                print(f"   {i}/{len(row_args)}  kept={kept}  drop_rate={1 - kept/i:.3f}")

    # Build keep_map and reason histogram
    keep_map = {r["garment_id"]: r["keep"] for r in results}
    reason_counts: dict[str, int] = {}
    for r in results:
        reason_counts[r["reason"]] = reason_counts.get(r["reason"], 0) + 1

    df["person_visible"] = df["garment_id"].map(keep_map).fillna(False)
    n_kept = int(df["person_visible"].sum())
    n_dropped = int((~df["person_visible"]).sum())
    print()
    print(f"  kept (person visible): {n_kept:,}  ({n_kept/len(df)*100:.1f}%)")
    print(f"  dropped: {n_dropped:,}  ({n_dropped/len(df)*100:.1f}%)")
    print()
    print("  drop reasons:")
    for reason, n in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:<28s} {n:>7,}  ({n/len(df)*100:.1f}%)")

    # Save filtered parquet (person-visible only)
    filtered = df[df["person_visible"]].drop(columns=["person_visible"]).copy()
    filtered.to_parquet(OUT_PARQUET, index=False)
    print(f"\n  wrote person-visible subset: {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1024/1024:.0f} MB, {len(filtered):,} rows)")

    # Write the per-row visibility flag to a SIDECAR file, NOT back to the source parquet.
    # Writing to the source caused a data-loss bug when this script was run with --limit
    # (the limited dataframe overwrote the full source). Use the sidecar + a join on garment_id
    # if downstream code needs to access the flag in the context of the full catalog.
    SIDECAR = PARQUET.with_name("person_visibility_flags.parquet")
    df[["garment_id", "person_visible"]].to_parquet(SIDECAR, index=False)
    print(f"  wrote per-row flag sidecar: {SIDECAR}")
    print(f"  (NOTE: source parquet {PARQUET.name} is NOT modified — join the sidecar on garment_id if needed)")

    REPORT.write_text(json.dumps({
        "total_rows": int(len(df)),
        "kept": n_kept,
        "dropped": n_dropped,
        "drop_rate": float(n_dropped / len(df)),
        "reasons": reason_counts,
        "filter_criteria": {
            "both_shoulders_visibility_gte": VISIBILITY_THRESHOLD,
            "at_least_one_hip_visibility_gte": VISIBILITY_THRESHOLD,
            "pose_detection_confidence_min": 0.4,
        },
        "use_redacted_for_detection": bool(args.use_redacted),
    }, indent=2, default=float))
    print(f"  wrote report: {REPORT}")


if __name__ == "__main__":
    main()
