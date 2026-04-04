#!/usr/bin/env python3
"""Compare FASHN VTON output quality across different step counts.

Runs inference at 10, 20, 30 steps and saves:
- Raw output for each
- Enhanced output (face restore + skin correct + upscale + sharpen) for each
- Quality metrics comparison

Usage:
    python test_step_comparison.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path(__file__).parent / "test_results" / "step_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PERSON = Path(__file__).parent / "vendor" / "IDM-VTON" / "gradio_demo" / "example" / "human" / "00034_00.jpg"
GARMENT = Path(__file__).parent / "vendor" / "IDM-VTON" / "gradio_demo" / "example" / "cloth" / "04469_00.jpg"
WEIGHTS_DIR = Path(__file__).parent / "weights" / "fashn-vton"

STEP_COUNTS = [10, 20, 30]


def run_comparison():
    import torch
    import numpy as np
    from PIL import Image
    from fashn_vton import TryOnPipeline
    from skimage.metrics import structural_similarity as ssim

    from services.pipeline.postprocessing.face_restore import restore_face
    from services.pipeline.postprocessing.skin_correction import match_skin_tone
    from services.pipeline.postprocessing.detail_enhance import (
        upscale_to_original,
        frequency_sharpen,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Person: {PERSON}")
    print(f"Garment: {GARMENT}")
    print(f"Steps to compare: {STEP_COUNTS}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Load model once
    print("Loading model...")
    t0 = time.time()
    pipe = TryOnPipeline(weights_dir=str(WEIGHTS_DIR), device=device)
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    person_img = Image.open(PERSON).convert("RGB")
    garment_img = Image.open(GARMENT).convert("RGB")
    original_img = person_img.copy()

    results = []

    for steps in STEP_COUNTS:
        print(f"{'='*60}")
        print(f"  Running {steps} steps...")
        print(f"{'='*60}")

        # Inference
        t0 = time.time()
        result = pipe(
            person_image=person_img,
            garment_image=garment_img,
            category="tops",
            garment_photo_type="flat-lay",
            num_samples=1,
            num_timesteps=steps,
            guidance_scale=1.5,
            seed=42,
        )
        inference_time = time.time() - t0
        raw_img = result.images[0]

        # Save raw
        raw_path = OUTPUT_DIR / f"raw_{steps}steps.png"
        raw_img.save(str(raw_path), "PNG")
        print(f"  Raw: {raw_img.size} in {inference_time:.1f}s → {raw_path.name}")

        # Postprocess: face + skin + upscale + sharpen
        t0 = time.time()
        img, face_d = restore_face(raw_img, original_img)
        img, skin_d = match_skin_tone(img, original_img)
        img, up_d = upscale_to_original(img, original_img)
        img = frequency_sharpen(img, strength=1.5)
        postprocess_time = time.time() - t0

        # Save enhanced
        enhanced_path = OUTPUT_DIR / f"enhanced_{steps}steps.png"
        img.save(str(enhanced_path), "PNG")
        print(f"  Enhanced: {img.size} in {postprocess_time:.1f}s → {enhanced_path.name}")

        # Quality metrics
        # 1. Face SSIM (compare face region with original)
        face_ssim = face_d.get("face_ssim", None)

        # 2. Overall SSIM vs original (non-garment regions should be preserved)
        raw_resized = raw_img.resize(original_img.size, Image.LANCZOS)
        raw_arr = np.array(raw_resized)
        orig_arr = np.array(original_img)

        # Crop top 30% for face comparison
        h = orig_arr.shape[0]
        face_h = int(h * 0.3)
        face_region_ssim = ssim(
            orig_arr[:face_h], raw_arr[:face_h],
            channel_axis=2, data_range=255
        )

        # 3. Sharpness (Laplacian variance — higher = sharper)
        import cv2
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 4. Edge density in garment region
        diff = np.mean(np.abs(raw_arr.astype(float) - orig_arr.astype(float)), axis=2)
        garment_mask = diff > 30
        garment_region = np.array(raw_resized)[garment_mask]
        garment_std = float(np.std(garment_region)) if len(garment_region) > 0 else 0

        metrics = {
            "steps": steps,
            "inference_time": round(inference_time, 1),
            "postprocess_time": round(postprocess_time, 1),
            "face_ssim": round(face_ssim, 4) if face_ssim else None,
            "face_region_ssim": round(float(face_region_ssim), 4),
            "sharpness": round(sharpness, 1),
            "garment_detail": round(garment_std, 1),
            "skin_delta": skin_d.get("skin_delta_lab"),
        }
        results.append(metrics)
        print(f"  Metrics: {metrics}")
        print()

    # Summary table
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Steps':>6} | {'Time':>7} | {'Face SSIM':>10} | {'Sharpness':>10} | {'Garment Detail':>14} | {'Skin Delta':>10}")
    print(f"{'-'*6}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}-+-{'-'*10}")
    for m in results:
        print(
            f"{m['steps']:>6} | {m['inference_time']:>6.1f}s | "
            f"{m['face_region_ssim']:>10.4f} | {m['sharpness']:>10.1f} | "
            f"{m['garment_detail']:>14.1f} | {m['skin_delta'] or 'n/a':>10}"
        )

    # Write metrics to file
    metrics_path = OUTPUT_DIR / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("FASHN VTON v1.5 — Step Comparison\n")
        f.write(f"Device: {device}\n")
        f.write(f"Person: {PERSON.name}\n")
        f.write(f"Garment: {GARMENT.name}\n")
        f.write(f"Seed: 42, Category: tops, Guidance: 1.5\n\n")
        f.write(f"{'Steps':>6} | {'Inference':>10} | {'Face SSIM':>10} | {'Sharpness':>10} | {'Garment Detail':>14} | {'Skin Delta':>10}\n")
        f.write(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}-+-{'-'*10}\n")
        for m in results:
            f.write(
                f"{m['steps']:>6} | {m['inference_time']:>9.1f}s | "
                f"{m['face_region_ssim']:>10.4f} | {m['sharpness']:>10.1f} | "
                f"{m['garment_detail']:>14.1f} | {m['skin_delta'] or 'n/a':>10}\n"
            )
        f.write(f"\nHigher sharpness = more detail. Higher garment_detail = more texture variation.\n")
        f.write(f"Face SSIM closer to 1.0 = better face preservation.\n")
        f.write(f"Lower skin delta = less color correction needed = better raw output.\n")

    print(f"\nMetrics saved: {metrics_path}")
    print(f"All results in: {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    run_comparison()
