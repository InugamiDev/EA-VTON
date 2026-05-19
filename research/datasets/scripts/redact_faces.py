"""Face-shape redaction using MediaPipe FaceLandmarker.

Unlike a bbox blur (which leaves an ugly rectangular block), this:
  1. Detects 478 face mesh landmarks per face via MediaPipe Tasks FaceLandmarker
  2. Builds a closed polygon from the FACE_OVAL landmark indices (36 points
     tracing the actual face perimeter)
  3. Creates a soft-edged binary mask from the polygon
  4. Gaussian-blurs the image and composites: blurred WHERE face oval, original
     everywhere else (hair, neck, garment all untouched)

Outputs to combined/images_redacted/ in parallel to combined/images/, so all
downstream label scripts can be re-pointed at the redacted tree.
"""

# intent: privacy-safe face redaction using face-shaped mask, not bbox rectangle
# status: done
# next: re-run all labelers on redacted images for fully privacy-clean labels
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
MODEL_PATH = ROOT / "research/models/face_landmarker.task"

SOURCES = ["deepfashion2", "deepfashion_with_masks", "fashionpedia",
           "fashion_products_small", "vn_brands"]

# 36 landmark indices that form the closed FACE_OVAL polygon in MediaPipe's
# 478-point face mesh, traced in order around the face perimeter.
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


def _face_polygon(landmarks, img_w: int, img_h: int) -> list[tuple[int, int]]:
    """Convert normalized landmarks → pixel polygon along FACE_OVAL."""
    return [
        (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
        for i in FACE_OVAL_INDICES
    ]


def _redact_with_mesh(img: Image.Image, faces_landmarks: list,
                      blur_radius: int = 18, dilate_iters: int = 2) -> Image.Image:
    """Blur ONLY the face-oval region for every detected face."""
    if not faces_landmarks:
        return img
    w, h = img.size

    # Build cumulative mask from all face polygons
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for face in faces_landmarks:
        poly = _face_polygon(face, w, h)
        if len(poly) >= 3:
            draw.polygon(poly, fill=255)

    # Slightly dilate mask so edges of the face aren't visible at oval boundary
    for _ in range(dilate_iters):
        mask = mask.filter(ImageFilter.MaxFilter(5))
    # Feather the mask edge so blur transition isn't a hard line
    mask = mask.filter(ImageFilter.GaussianBlur(radius=4))

    # Gaussian blur applied only where mask > 0 via composite
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.composite(blurred, img, mask)


def _process_one(args: tuple) -> tuple[str, int]:
    """Worker: redact one image. Returns (status, n_faces_detected)."""
    in_path, out_path, blur_radius = args
    if out_path.exists():
        return ("skipped", 0)
    try:
        img = Image.open(in_path).convert("RGB")
    except Exception:
        return ("failed", 0)

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import mediapipe as mp_lib

    global _LANDMARKER
    try:
        _LANDMARKER
    except NameError:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=0.4,
        )
        _LANDMARKER = mp_vision.FaceLandmarker.create_from_options(opts)

    arr = np.asarray(img)
    mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=arr)
    result = _LANDMARKER.detect(mp_image)
    faces = result.face_landmarks  # list[list[NormalizedLandmark]]

    if faces:
        img = _redact_with_mesh(img, faces, blur_radius=blur_radius)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=88)
    return ("ok", len(faces))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blur-radius", type=int, default=18,
                    help="Gaussian blur radius applied inside face oval")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    ap.add_argument("--overwrite", action="store_true",
                    help="Reprocess images even if redacted output exists")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        print(f"!! model missing at {MODEL_PATH}")
        print("   Download with:")
        print("   curl -L -o {} https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task".format(MODEL_PATH))
        return

    # Collect all jobs
    jobs: list[tuple] = []
    for src in args.sources:
        # vn_brands is special-cased: reads from raw/vn_brands/images/<brand>/
        # instead of combined/images/<src>/. Don't gate it on combined/images existing.
        if src == "vn_brands":
            raw_vn = ROOT / "research/datasets/raw/vn_brands/images"
            if not raw_vn.exists():
                print(f"  ! {raw_vn} missing, skipping vn_brands")
                continue
            for brand_dir in raw_vn.iterdir():
                if not brand_dir.is_dir():
                    continue
                # VN scrapers preserve source extension (.jpg, .png, .webp).
                # Output is always .jpg for consistency with the rest of the
                # downstream pipeline.
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                    for src_img in brand_dir.glob(ext):
                        out = OUT_ROOT / "vn_brands" / brand_dir.name / (src_img.stem + ".jpg")
                        if args.overwrite or not out.exists():
                            jobs.append((src_img, out, args.blur_radius))
            continue

        # Other sources use combined/images/<src>/
        src_dir = IMG_ROOT / src
        if not src_dir.exists():
            print(f"  ! {src_dir} missing, skipping")
            continue
        for jpg in src_dir.glob("*.jpg"):
            out = OUT_ROOT / src / jpg.name
            if args.overwrite or not out.exists():
                jobs.append((jpg, out, args.blur_radius))

    if args.limit > 0:
        jobs = jobs[: args.limit]

    print(f"  total images to process: {len(jobs):,}")
    print(f"  blur_radius={args.blur_radius}  workers={args.workers}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    done = skipped = failed = 0
    faces_total = redacted_imgs = 0

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
                pct = 100 * (done + skipped + failed) / len(jobs) if jobs else 0
                eta = (len(jobs) - done - skipped - failed) / rate if rate > 0 else 0
                print(f"    {done+skipped+failed:>7}/{len(jobs):,}  ({pct:.1f}%)  "
                      f"{rate:.1f} img/s  faces={faces_total:,}  redacted={redacted_imgs:,}  "
                      f"ETA {eta/60:.1f}min", flush=True)

    total = time.time() - t_start
    print()
    print(f"  ── done in {total/60:.1f} min ──")
    print(f"  processed: {done:,}  skipped: {skipped:,}  failed: {failed:,}")
    print(f"  faces detected: {faces_total:,}  images with redaction: {redacted_imgs:,}")
    print(f"  output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
