#!/usr/bin/env python3
"""Standalone FASHN VTON v1.5 model test.

Run this directly — no server needed. Tests each component independently.

Usage:
    python test_model.py                    # Run all tests
    python test_model.py --quick            # Skip slow inference test
    python test_model.py --steps 5          # Fewer steps (faster, lower quality)
    python test_model.py --person <path>    # Custom person image
    python test_model.py --garment <path>   # Custom garment image
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Test images from IDM-VTON examples
TEST_DIR = Path(__file__).parent / "vendor" / "IDM-VTON" / "gradio_demo" / "example"
DEFAULT_PERSON = TEST_DIR / "human" / "00034_00.jpg"
DEFAULT_GARMENT = TEST_DIR / "cloth" / "04469_00.jpg"
WEIGHTS_DIR = Path(__file__).parent / "weights" / "fashn-vton"
OUTPUT_DIR = Path("/tmp/fitview_test_results")


def _header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _ok(msg: str, elapsed: float | None = None):
    t = f" ({elapsed:.1f}s)" if elapsed is not None else ""
    print(f"  [OK] {msg}{t}")


def _fail(msg: str, err: Exception | None = None):
    print(f"  [FAIL] {msg}")
    if err:
        print(f"         {err}")


def _skip(msg: str):
    print(f"  [SKIP] {msg}")


# ── Test 1: Environment ──────────────────────────────────────────────
def test_environment():
    _header("1. Environment")

    # Python version
    v = sys.version_info
    _ok(f"Python {v.major}.{v.minor}.{v.micro}")

    # torch
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _ok(f"PyTorch {torch.__version__} (device: {device})")
        if torch.cuda.is_available():
            _ok(f"CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            _ok(f"VRAM: {vram:.1f} GB")
    except ImportError:
        _fail("PyTorch not installed")
        return False

    # fashn_vton
    try:
        from fashn_vton import TryOnPipeline
        _ok("fashn-vton package installed")
    except ImportError:
        _fail("fashn-vton not installed — run: git clone https://github.com/fashn-AI/fashn-vton-1.5.git && pip install -e ./fashn-vton-1.5")
        return False

    # onnxruntime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        _ok(f"onnxruntime {ort.__version__} (providers: {', '.join(providers)})")
    except ImportError:
        _fail("onnxruntime not installed — run: pip install onnxruntime-gpu (CUDA) or onnxruntime (CPU)")
        return False

    # PIL
    try:
        from PIL import Image
        _ok("Pillow installed")
    except ImportError:
        _fail("Pillow not installed")
        return False

    return True


# ── Test 2: Weights ──────────────────────────────────────────────────
def test_weights():
    _header("2. Model Weights")

    model_path = WEIGHTS_DIR / "model.safetensors"
    yolox_path = WEIGHTS_DIR / "dwpose" / "yolox_l.onnx"
    dwpose_path = WEIGHTS_DIR / "dwpose" / "dw-ll_ucoco_384.onnx"

    all_ok = True
    for name, path in [
        ("model.safetensors", model_path),
        ("dwpose/yolox_l.onnx", yolox_path),
        ("dwpose/dw-ll_ucoco_384.onnx", dwpose_path),
    ]:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            _ok(f"{name} ({size_mb:.0f} MB)")
        else:
            _fail(f"{name} missing at {path}")
            all_ok = False

    if not all_ok:
        print("\n  To download weights, run:")
        print("    python -c \"from services.pipeline.backends.huggingface_vton import _download_weights; _download_weights()\"")

    return all_ok


# ── Test 3: Model Loading ────────────────────────────────────────────
def test_model_loading():
    _header("3. Model Loading")

    import torch
    from fashn_vton import TryOnPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ok(f"Target device: {device}")

    t0 = time.time()
    try:
        pipe = TryOnPipeline(weights_dir=str(WEIGHTS_DIR), device=device)
        elapsed = time.time() - t0
        _ok(f"TryOnPipeline loaded", elapsed)

        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
            _ok(f"VRAM usage: {vram_mb:.0f} MB")

        return pipe
    except Exception as e:
        _fail(f"Model loading failed", e)
        return None


# ── Test 4: Test Images ──────────────────────────────────────────────
def test_images(person_path: Path, garment_path: Path):
    _header("4. Test Images")

    from PIL import Image

    all_ok = True
    for name, path in [("Person", person_path), ("Garment", garment_path)]:
        if path.exists():
            img = Image.open(path)
            _ok(f"{name}: {path.name} ({img.size[0]}x{img.size[1]}, {img.mode})")
        else:
            _fail(f"{name} image not found: {path}")
            all_ok = False

    return all_ok


# ── Test 5: Inference ─────────────────────────────────────────────────
def test_inference(pipe, person_path: Path, garment_path: Path, steps: int, category: str):
    _header(f"5. Inference ({steps} steps, category={category})")

    import torch
    from PIL import Image

    person_img = Image.open(person_path).convert("RGB")
    garment_img = Image.open(garment_path).convert("RGB")

    _ok(f"Person: {person_img.size}, Garment: {garment_img.size}")

    t0 = time.time()
    try:
        result = pipe(
            person_image=person_img,
            garment_image=garment_img,
            category=category,
            garment_photo_type="flat-lay",
            num_samples=1,
            num_timesteps=steps,
            guidance_scale=1.5,
            seed=42,
        )
        elapsed = time.time() - t0

        output_img = result.images[0]
        _ok(f"Inference complete: {output_img.size}", elapsed)

        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            _ok(f"Peak VRAM: {vram_mb:.0f} MB")

        # Save result
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"test_result_{steps}steps.png"
        output_img.save(str(out_path), "PNG")
        _ok(f"Saved: {out_path}")

        return output_img

    except Exception as e:
        _fail(f"Inference failed", e)
        import traceback
        traceback.print_exc()
        return None


# ── Test 6: Postprocessing ────────────────────────────────────────────
def test_postprocessing(result_img, person_path: Path):
    _header("6. Postprocessing (original)")

    from PIL import Image

    original_img = Image.open(person_path).convert("RGB")
    current_img = result_img

    # Face restoration
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from services.pipeline.postprocessing.face_restore import restore_face

        t0 = time.time()
        current_img, details = restore_face(current_img, original_img)
        elapsed = time.time() - t0
        _ok(f"Face restoration: restored={details.get('restored')}, ssim={details.get('face_ssim', 'n/a')}", elapsed)
    except Exception as e:
        _fail(f"Face restoration", e)

    # Skin tone correction
    try:
        from services.pipeline.postprocessing.skin_correction import match_skin_tone

        t0 = time.time()
        current_img, details = match_skin_tone(current_img, original_img)
        elapsed = time.time() - t0
        _ok(f"Skin correction: corrected={details.get('corrected')}, delta={details.get('skin_delta_lab', 'n/a')}", elapsed)
    except Exception as e:
        _fail(f"Skin correction", e)

    # Save before enhancement
    out_path = OUTPUT_DIR / "test_result_postprocessed.png"
    current_img.save(str(out_path), "PNG")
    _ok(f"Saved (before enhancement): {out_path}")

    return current_img


# ── Test 6b: Detail Enhancement (realism) ────────────────────────────
def test_detail_enhancement(result_img, person_path: Path):
    _header("6b. Detail Enhancement (realism)")

    from PIL import Image
    from services.pipeline.postprocessing.detail_enhance import (
        upscale_to_original,
        soften_garment_boundaries,
        harmonize_lighting,
        frequency_sharpen,
    )

    original_img = Image.open(person_path).convert("RGB")
    current_img = result_img
    _ok(f"Input: {current_img.size}, Original: {original_img.size}")

    # Upscale to original resolution
    try:
        t0 = time.time()
        current_img, details = upscale_to_original(current_img, original_img)
        elapsed = time.time() - t0
        _ok(f"Upscale: {details}", elapsed)
    except Exception as e:
        _fail(f"Upscale", e)

    # Soften garment boundaries
    try:
        t0 = time.time()
        current_img, details = soften_garment_boundaries(current_img, original_img)
        elapsed = time.time() - t0
        _ok(f"Boundary soften: {details}", elapsed)
    except Exception as e:
        _fail(f"Boundary soften", e)

    # Harmonize lighting
    try:
        t0 = time.time()
        current_img, details = harmonize_lighting(current_img, original_img)
        elapsed = time.time() - t0
        _ok(f"Lighting harmonize: {details}", elapsed)
    except Exception as e:
        _fail(f"Lighting harmonize", e)

    # Frequency sharpening
    try:
        t0 = time.time()
        current_img = frequency_sharpen(current_img, strength=1.5)
        elapsed = time.time() - t0
        _ok(f"Frequency sharpen: strength=1.5", elapsed)
    except Exception as e:
        _fail(f"Frequency sharpen", e)

    # Save enhanced result
    out_path = OUTPUT_DIR / "test_result_enhanced.png"
    current_img.save(str(out_path), "PNG")
    _ok(f"Saved (enhanced): {out_path}")

    return current_img


# ── Test 7: Category Mapping ─────────────────────────────────────────
def test_categories():
    _header("7. Category Mapping")

    sys.path.insert(0, str(Path(__file__).parent))
    from services.pipeline.backends.huggingface_vton import _CATEGORY_MAP

    categories = {"tops": 0, "bottoms": 0, "one-pieces": 0}
    for k, v in _CATEGORY_MAP.items():
        categories[v] += 1

    for cat, count in categories.items():
        _ok(f"{cat}: {count} mappings")

    _ok(f"Total: {len(_CATEGORY_MAP)} garment types mapped")
    return True


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test FASHN VTON v1.5 model")
    parser.add_argument("--quick", action="store_true", help="Skip inference test")
    parser.add_argument("--steps", type=int, default=20, help="Diffusion steps (default: 20)")
    parser.add_argument("--person", type=str, default=str(DEFAULT_PERSON), help="Person image path")
    parser.add_argument("--garment", type=str, default=str(DEFAULT_GARMENT), help="Garment image path")
    parser.add_argument("--category", type=str, default="tops", choices=["tops", "bottoms", "one-pieces"])
    args = parser.parse_args()

    person_path = Path(args.person)
    garment_path = Path(args.garment)

    print("\nFASHN VTON v1.5 — Standalone Model Test")
    print(f"Weights: {WEIGHTS_DIR}")
    print(f"Output:  {OUTPUT_DIR}")

    results = {}

    # Test 1: Environment
    results["env"] = test_environment()
    if not results["env"]:
        print("\n[ABORT] Fix environment issues first.")
        sys.exit(1)

    # Test 2: Weights
    results["weights"] = test_weights()
    if not results["weights"]:
        print("\n[ABORT] Download weights first.")
        sys.exit(1)

    # Test 3: Model loading
    pipe = test_model_loading()
    results["loading"] = pipe is not None
    if pipe is None:
        print("\n[ABORT] Model loading failed.")
        sys.exit(1)

    # Test 4: Images
    results["images"] = test_images(person_path, garment_path)

    # Test 5: Inference
    if args.quick:
        _header("5. Inference")
        _skip("--quick flag set")
        results["inference"] = None
        result_img = None
    elif not results["images"]:
        _header("5. Inference")
        _skip("No test images")
        results["inference"] = None
        result_img = None
    else:
        result_img = test_inference(pipe, person_path, garment_path, args.steps, args.category)
        results["inference"] = result_img is not None

    # Test 6: Postprocessing
    if result_img is not None:
        postprocessed_img = test_postprocessing(result_img, person_path)
    else:
        _header("6. Postprocessing")
        _skip("No inference result")
        postprocessed_img = None

    # Test 6b: Detail Enhancement
    if postprocessed_img is not None:
        test_detail_enhancement(postprocessed_img, person_path)
    else:
        _header("6b. Detail Enhancement")
        _skip("No postprocessed result")

    # Test 7: Categories
    results["categories"] = test_categories()

    # Summary
    _header("Summary")
    total = 0
    passed = 0
    for name, result in results.items():
        if result is None:
            print(f"  {name}: skipped")
        elif result:
            print(f"  {name}: PASS")
            passed += 1
            total += 1
        else:
            print(f"  {name}: FAIL")
            total += 1

    print(f"\n  {passed}/{total} tests passed")
    if results.get("inference"):
        print(f"  Results saved to: {OUTPUT_DIR}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
