"""Detect + redact faces in every image of the combined catalog.

Uses MediaPipe Face Detection (model_selection=1 — full-range, ~5ms per image
on CPU). For each detected face: applies Gaussian blur over the bbox region
with a fixed-feature margin around it. Outputs to a parallel directory so
originals remain intact:

    combined/images/<source>/<name>.jpg
       │
       ▼
    combined/images_redacted/<source>/<name>.jpg   (blurred faces)

Multiprocessing pool for throughput. Resumable: skips files where the redacted
output already exists.

Options:
    --mode  blur | box  | pixelate
    --margin  fraction of bbox to expand redaction (default 0.25)
    --workers  parallel processes (default 6)
"""

# intent: privacy-safe redact faces across all catalog images for distribution
# status: done
# next: optionally rebuild the bundle with redacted images bundled in
# confidence: high

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[3]
IMG_ROOT = ROOT / "research/datasets/processed/combined/images"
OUT_ROOT = ROOT / "research/datasets/processed/combined/images_redacted"

# Sources to redact (skip any subdir not listed)
SOURCES = ["deepfashion2", "deepfashion_with_masks", "fashionpedia",
           "fashion_products_small", "vn_brands"]


def _detect_face_bboxes(rgb_array: np.ndarray, detector) -> list[tuple[int, int, int, int]]:
    """Return list of (x1, y1, x2, y2) face bboxes in absolute pixel coords.

    Uses MediaPipe Tasks FaceDetector (BlazeFace short-range).
    """
    import mediapipe as mp_lib
    h, w = rgb_array.shape[:2]
    mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb_array)
    res = detector.detect(mp_image)
    if not res.detections:
        return []
    bboxes = []
    for det in res.detections:
        bb = det.bounding_box
        x1 = int(bb.origin_x)
        y1 = int(bb.origin_y)
        x2 = int(bb.origin_x + bb.width)
        y2 = int(bb.origin_y + bb.height)
        bboxes.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
    return bboxes


def _redact_image(img: Image.Image, bboxes: list, mode: str, margin: float) -> Image.Image:
    """Apply redaction to face bboxes in the image. Returns modified copy."""
    if not bboxes:
        return img
    out = img.copy()
    w, h = out.size

    for x1, y1, x2, y2 in bboxes:
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * margin), int(bh * margin)
        x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx); y2 = min(h, y2 + my)
        if x2 <= x1 or y2 <= y1:
            continue

        region = out.crop((x1, y1, x2, y2))
        if mode == "blur":
            radius = max(8, min(bw, bh) // 4)
            region = region.filter(ImageFilter.GaussianBlur(radius=radius))
        elif mode == "pixelate":
            small = region.resize((max(1, (x2 - x1) // 16), max(1, (y2 - y1) // 16)),
                                  resample=Image.Resampling.BILINEAR)
            region = small.resize(region.size, resample=Image.Resampling.NEAREST)
        else:  # "box"
            draw = ImageDraw.Draw(region)
            draw.rectangle([0, 0, region.size[0] - 1, region.size[1] - 1], fill=(0, 0, 0))
        out.paste(region, (x1, y1))

    return out


def _process_one(args: tuple) -> tuple[str, int]:
    """Worker: redact one image. Returns (source, n_faces_detected)."""
    in_path, out_path, mode, margin = args
    if out_path.exists():
        return ("skipped", 0)
    try:
        img = Image.open(in_path).convert("RGB")
    except Exception:
        return ("failed", 0)

    # Lazy import inside worker so each process gets its own detector
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    global _DETECTOR
    try:
        _DETECTOR
    except NameError:
        model_path = str(ROOT / "research/models/blaze_face_short_range.tflite")
        opts = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=0.4,
        )
        _DETECTOR = mp_vision.FaceDetector.create_from_options(opts)

    arr = np.asarray(img)
    bboxes = _detect_face_bboxes(arr, _DETECTOR)
    if bboxes:
        img = _redact_image(img, bboxes, mode, margin)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=85)
    return ("ok", len(bboxes))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="blur", choices=["blur", "pixelate", "box"])
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    args = ap.parse_args()

    # Collect all input/output paths
    jobs: list[tuple] = []
    for src in args.sources:
        src_dir = IMG_ROOT / src
        if not src_dir.exists():
            print(f"  ! {src_dir} missing, skipping")
            continue
        # Handle nested brand dirs under vn_brands (raw/vn_brands/images/<brand>/...)
        if src == "vn_brands":
            # Pull from raw scrape location since combined doesn't have these
            raw_vn = ROOT / "research/datasets/raw/vn_brands/images"
            if raw_vn.exists():
                for brand_dir in raw_vn.iterdir():
                    if not brand_dir.is_dir(): continue
                    for jpg in brand_dir.glob("*.jpg"):
                        out = OUT_ROOT / "vn_brands" / brand_dir.name / jpg.name
                        jobs.append((jpg, out, args.mode, args.margin))
        else:
            for jpg in src_dir.glob("*.jpg"):
                out = OUT_ROOT / src / jpg.name
                jobs.append((jpg, out, args.mode, args.margin))

    if args.limit > 0:
        jobs = jobs[: args.limit]

    print(f"  total images to process: {len(jobs):,}")
    print(f"  mode={args.mode}, margin={args.margin}, workers={args.workers}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    done = 0
    skipped = 0
    failed = 0
    faces_total = 0
    redacted_imgs = 0

    with mp.Pool(processes=args.workers) as pool:
        for status, n_faces in pool.imap_unordered(_process_one, jobs, chunksize=20):
            if status == "ok":
                done += 1
                faces_total += n_faces
                if n_faces > 0:
                    redacted_imgs += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1

            if (done + skipped + failed) % 1000 == 0:
                elapsed = time.time() - t_start
                rate = (done + skipped + failed) / elapsed if elapsed > 0 else 0
                pct = 100 * (done + skipped + failed) / len(jobs)
                eta = (len(jobs) - done - skipped - failed) / rate if rate > 0 else 0
                print(f"    {done+skipped+failed:>7}/{len(jobs):,}  ({pct:.1f}%)  "
                      f"{rate:.1f} img/s  faces={faces_total:,}  redacted={redacted_imgs:,}  "
                      f"ETA {eta/60:.1f}min", flush=True)

    total = time.time() - t_start
    print()
    print(f"  ── done in {total/60:.1f} min ──")
    print(f"  processed: {done:,}  skipped (cached): {skipped:,}  failed: {failed:,}")
    print(f"  faces detected: {faces_total:,}  images with redaction: {redacted_imgs:,}")
    print(f"  output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
