"""Batch feature extraction for all dataset pairs.

For each person image in the combined dataset:
  1. Pose → HRNet-W32 (or MediaPipe fallback) → (18, 3) keypoints + (18, 64, 48) heatmaps
  2. Parsing → SCHP (or dummy) → (H, W) uint8 semantic map (20 classes ATR format)
  3. Agnostic → parsing[upper_clothes]=0, inpaint with skin color
  4. Body embedding → CNN → R^128

Saves results as .npy / .png files alongside source images.

Usage:
    python data/scripts/extract_features.py --dataset vitonhd
    python data/scripts/extract_features.py --dataset vitonhd --limit 100
"""

# intent: batch feature extraction pipeline for training data preparation
# status: done
# next: run after downloading datasets and model weights
# confidence: high

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")

# ATR label indices
ATR_UPPER_CLOTHES = 4
ATR_TORSO_SKIN = 18

# Heatmap params
HEATMAP_H = 64
HEATMAP_W = 48
NUM_KEYPOINTS = 18


def _generate_heatmaps(
    keypoints: np.ndarray,
    h: int = HEATMAP_H,
    w: int = HEATMAP_W,
    sigma: float = 6.0,
) -> np.ndarray:
    """Generate Gaussian heatmaps from keypoints."""
    heatmaps = np.zeros((keypoints.shape[0], h, w), dtype=np.float32)
    for j in range(keypoints.shape[0]):
        x, y, conf = keypoints[j]
        if conf < 0.1:
            continue
        px, py = int(x * w), int(y * h)
        if px < 0 or px >= w or py < 0 or py >= h:
            continue
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        heatmaps[j] = conf * np.exp(
            -((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2)
        )
    return heatmaps


def _extract_pose_hrnet(img: Image.Image) -> dict[str, np.ndarray]:
    """Try HRNet-W32 ONNX inference."""
    sys.path.insert(0, str(Path("services/feature-service")))
    from src.pose import HRNetPose
    model = HRNetPose.get_instance()
    return model.predict(img)


def _extract_pose_mediapipe(img: Image.Image) -> dict[str, np.ndarray]:
    """Fallback: extract pose with MediaPipe PoseLandmarker."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    # Download model if needed
    model_path = Path.home() / ".cache" / "mediapipe" / "pose_landmarker_heavy.task"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        logger.info("Downloading MediaPipe pose model...")
        urllib.request.urlretrieve(url, model_path)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.array(img),
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return {
            "keypoints": np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32),
            "heatmaps": np.zeros((NUM_KEYPOINTS, HEATMAP_H, HEATMAP_W), dtype=np.float32),
        }

    # Convert MediaPipe 33 landmarks to 18-point COCO-ish format
    COCO_FROM_MP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 0]
    landmarks = result.pose_landmarks[0]
    keypoints = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    for i, mp_idx in enumerate(COCO_FROM_MP):
        lm = landmarks[mp_idx]
        keypoints[i] = [lm.x, lm.y, lm.visibility if hasattr(lm, "visibility") else 0.5]

    heatmaps = _generate_heatmaps(keypoints, HEATMAP_H, HEATMAP_W, sigma=6.0)
    return {"keypoints": keypoints, "heatmaps": heatmaps}


def _extract_pose(img: Image.Image) -> dict[str, np.ndarray]:
    """Extract pose, trying HRNet first then MediaPipe fallback."""
    try:
        return _extract_pose_hrnet(img)
    except (ImportError, Exception) as e:
        logger.debug("HRNet unavailable (%s), using MediaPipe", e)
        return _extract_pose_mediapipe(img)


def _extract_parsing_schp(img: Image.Image) -> np.ndarray:
    """Try SCHP ONNX inference."""
    sys.path.insert(0, str(Path("services/feature-service")))
    from src.parsing import SCHPParser
    parser = SCHPParser.get_instance()
    return parser.predict(img)


def _extract_parsing_simple(img: Image.Image) -> np.ndarray:
    """Simple fallback: approximate parsing from color/brightness heuristics.

    Not great, but better than all zeros. Labels upper-body region based on
    vertical position heuristic (top 60% excluding top 15% for head).
    """
    w, h = img.size
    parsing = np.zeros((h, w), dtype=np.uint8)

    # Rough body region heuristics
    head_end = int(h * 0.15)
    torso_end = int(h * 0.55)
    hip_end = int(h * 0.7)

    # Face region
    parsing[0:head_end, w // 4:3 * w // 4] = 11  # Face

    # Upper clothes
    parsing[head_end:torso_end, w // 5:4 * w // 5] = 4  # Upper-clothes

    # Torso skin (small strip)
    parsing[torso_end:torso_end + 10, w // 3:2 * w // 3] = 18

    # Legs
    parsing[hip_end:, 0:w // 2] = 12  # Left-leg
    parsing[hip_end:, w // 2:] = 13  # Right-leg

    return parsing


def _extract_parsing(img: Image.Image) -> np.ndarray:
    """Extract parsing, trying SCHP first then simple fallback."""
    try:
        return _extract_parsing_schp(img)
    except (ImportError, Exception) as e:
        logger.debug("SCHP unavailable (%s), using simple heuristic", e)
        return _extract_parsing_simple(img)


def _generate_agnostic(img: Image.Image, parsing: np.ndarray) -> Image.Image:
    """Generate agnostic person image by removing upper clothes region."""
    img_array = np.array(img)
    mask = parsing == ATR_UPPER_CLOTHES

    skin_mask = parsing == ATR_TORSO_SKIN
    if skin_mask.any():
        skin_color = img_array[skin_mask].mean(axis=0).astype(np.uint8)
    else:
        skin_color = np.array([200, 180, 170], dtype=np.uint8)

    result = img_array.copy()
    result[mask] = skin_color
    return Image.fromarray(result)


def _extract_embedding(
    heatmaps: np.ndarray | None,
    parsing: np.ndarray | None,
) -> np.ndarray:
    """Extract body embedding. Returns zeros if model not available."""
    try:
        sys.path.insert(0, str(Path("services/feature-service")))
        from src.body_embedding import BodyEmbeddingNet
        model = BodyEmbeddingNet.get_instance()
        return model.predict(heatmaps, parsing)
    except (ImportError, RuntimeError):
        return np.zeros(128, dtype=np.float32)


def extract_features_for_pair(pair: dict, data_dir: Path) -> dict | None:
    """Extract all features for a single person-garment pair."""
    person_path = data_dir / pair["person"]["image"]
    if not person_path.exists():
        return None

    img = Image.open(person_path).convert("RGB")
    pair_id = pair["id"]

    # Output directory: dataset root (same level as person/, garment/)
    out_dir = data_dir
    result = {"id": pair_id}

    # 1. Pose extraction
    heatmaps = None
    try:
        pose_data = _extract_pose(img)
        pose_dir = out_dir / "pose"
        pose_dir.mkdir(exist_ok=True)
        np.save(pose_dir / f"{pair_id}.npy", pose_data["keypoints"])

        heatmap_dir = out_dir / "heatmaps"
        heatmap_dir.mkdir(exist_ok=True)
        np.save(heatmap_dir / f"{pair_id}.npy", pose_data["heatmaps"])
        heatmaps = pose_data["heatmaps"]

        result["pose"] = f"pose/{pair_id}.npy"
        result["heatmaps"] = f"heatmaps/{pair_id}.npy"
    except Exception as e:
        logger.warning("Pose failed for %s: %s", pair_id, e)

    # 2. Parsing extraction
    parsing = None
    try:
        parsing = _extract_parsing(img)
        parsing_dir = out_dir / "parsing"
        parsing_dir.mkdir(exist_ok=True)
        Image.fromarray(parsing).save(parsing_dir / f"{pair_id}.png")
        result["parsing"] = f"parsing/{pair_id}.png"
    except Exception as e:
        logger.warning("Parsing failed for %s: %s", pair_id, e)

    # 3. Agnostic generation
    try:
        if parsing is not None:
            agnostic = _generate_agnostic(img, parsing)
            agnostic_dir = out_dir / "agnostic"
            agnostic_dir.mkdir(exist_ok=True)
            agnostic.save(agnostic_dir / f"{pair_id}.png")
            result["agnostic"] = f"agnostic/{pair_id}.png"
    except Exception as e:
        logger.warning("Agnostic failed for %s: %s", pair_id, e)

    # 4. Body embedding
    try:
        embedding = _extract_embedding(heatmaps, parsing)
        embed_dir = out_dir / "body_embed"
        embed_dir.mkdir(exist_ok=True)
        np.save(embed_dir / f"{pair_id}.npy", embedding)
        result["body_embedding"] = f"body_embed/{pair_id}.npy"
    except Exception as e:
        logger.warning("Embedding failed for %s: %s", pair_id, e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch feature extraction")
    parser.add_argument("--dataset", type=str, default="vitonhd",
                        help="Dataset name (vitonhd, dresscode, etc.)")
    parser.add_argument("--pairs", type=Path, default=None,
                        help="Path to pairs.json (default: data/processed/<dataset>/pairs.json)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.pairs:
        pairs_file = args.pairs
        dataset_dir = pairs_file.parent
    else:
        dataset_dir = DATA_DIR / args.dataset
        pairs_file = dataset_dir / "pairs.json"

    if not pairs_file.exists():
        logger.error("Pairs file not found: %s", pairs_file)
        logger.error("Run the dataset download script first.")
        return

    with open(pairs_file) as f:
        pairs = json.load(f)

    if args.limit:
        pairs = pairs[:args.limit]

    logger.info("Processing %d pairs from %s", len(pairs), pairs_file)

    results = []
    failed = 0

    for i, pair in enumerate(pairs):
        result = extract_features_for_pair(pair, dataset_dir)
        if result:
            results.append(result)
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d (failed: %d)", i + 1, len(pairs), failed)

    # Save results index
    output_path = dataset_dir / "features_index.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Extracted features for %d/%d pairs → %s",
                len(results), len(pairs), output_path)


if __name__ == "__main__":
    main()
