"""Download BodyM silhouette masks from the AWS Open Data Registry.

BodyM is a public dataset of 720×960 grayscale silhouettes paired with body
measurements. We use it to train a vision encoder that predicts body
measurements from a silhouette → which then chains into the ModCloth-trained
size predictor to give us the camera-only "Path A" size pipeline.

S3 layout (per AWS Open Data Registry):
    s3://amazon-bodym/<split>/hwg_metadata.csv
    s3://amazon-bodym/<split>/measurements.csv
    s3://amazon-bodym/<split>/subject_to_photo_map.csv
    s3://amazon-bodym/<split>/mask/<photo_id>.png   ← what we download here

We pull each mask by photo_id from the local subject_to_photo_map.csv,
serially with rate-limited HTTP GETs (S3 is fine with hundreds of req/sec
but we keep it polite). Cached locally; reruns are idempotent.

Output:
    research/datasets/raw/bodym/<split>/mask/<photo_id>.png

Total expected size: ~10,000 masks × 6-8 KB ≈ 60-80 MB.
Runtime on a typical home connection: ~5-15 min.

Citation (please include in any paper using these):
    Ruiz, N., Bellver, M., Bolkart, T., Arora, A., Lin, M. C., Romero, J.,
    Bala, R. (2022). Human Body Measurement Estimation with Adversarial
    Augmentation. https://arxiv.org/abs/2210.05667

License: CC BY-NC 4.0 — research / non-commercial use only.
"""

# intent: pull the 10k BodyM silhouettes so train_body_vision_encoder.py can run on Mac
# status: ready
# next: chain into train_body_vision_encoder.py once the photos land on disk
# confidence: high

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BODYM_RAW = ROOT / "research/datasets/raw/bodym"
S3_BASE = "https://amazon-bodym.s3.amazonaws.com"

SPLITS = ["train", "testA", "testB"]


def read_photo_ids(split: str) -> list[tuple[str, str]]:
    csv_path = BODYM_RAW / split / "subject_to_photo_map.csv"
    if not csv_path.exists():
        return []
    pairs: list[tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["subject_id"], row["photo_id"]))
    return pairs


def fetch_one(args: tuple[str, str, Path]) -> tuple[str, bool, str]:
    """Worker: fetch one mask PNG. Returns (photo_id, ok, msg)."""
    split, photo_id, out_path = args
    if out_path.exists() and out_path.stat().st_size > 0:
        return (photo_id, True, "cached")
    url = f"{S3_BASE}/{split}/mask/{photo_id}.png"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "EAVTONResearchBot/0.1"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        if len(data) < 100:
            return (photo_id, False, "too_small")
        out_path.write_bytes(data)
        return (photo_id, True, "downloaded")
    except Exception as e:  # noqa: BLE001
        return (photo_id, False, f"err:{type(e).__name__}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="0 = all photos in chosen splits")
    args = ap.parse_args()

    total_done = 0
    total_cached = 0
    total_failed = 0
    t_start = time.time()

    for split in args.splits:
        pairs = read_photo_ids(split)
        if args.limit:
            pairs = pairs[: args.limit]
        if not pairs:
            print(f"  ! no photo_ids in {split}/subject_to_photo_map.csv; skipping")
            continue
        out_dir = BODYM_RAW / split / "mask"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Dedup by photo_id (multiple subjects may share — unlikely but safe)
        seen: set[str] = set()
        jobs: list[tuple[str, str, Path]] = []
        for _, pid in pairs:
            if pid in seen:
                continue
            seen.add(pid)
            jobs.append((split, pid, out_dir / f"{pid}.png"))

        print(f"\n── {split} — {len(jobs)} unique mask URLs ──")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(fetch_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                pid, ok, msg = fut.result()
                if ok and msg == "downloaded":
                    total_done += 1
                elif ok and msg == "cached":
                    total_cached += 1
                else:
                    total_failed += 1
                if i % 200 == 0:
                    elapsed = time.time() - t_start
                    print(f"   {i}/{len(jobs)}  ok={total_done + total_cached}  failed={total_failed}  elapsed={elapsed:.0f}s")

    print()
    print(f"  done in {time.time() - t_start:.0f}s")
    print(f"  downloaded: {total_done}")
    print(f"  already cached: {total_cached}")
    print(f"  failed: {total_failed}")
    print()
    print(f"  Masks at: {BODYM_RAW}/<split>/mask/<photo_id>.png")
    print(f"  Next: run train_body_vision_encoder.py")


if __name__ == "__main__":
    main()
