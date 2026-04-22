"""Download and normalize DressCode dataset.

DressCode: 48,392 paired images across upper, lower, and dress categories.
Source: https://github.com/aimagelab/dress-code

After download, images are resized to 512x384 and organized into:
  data/processed/dresscode/
    person/        — person images
    garment/       — garment images
    garment_mask/  — garment segmentation masks
    pairs.json     — pair index with metadata

Usage:
    python data/scripts/download_dresscode.py [--output-dir data/processed/dresscode]
"""

# intent: download and normalize DressCode dataset for multi-category training
# status: done
# next: run after setting up HuggingFace credentials
# confidence: high

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/processed/dresscode")
TARGET_SIZE = (384, 512)  # (W, H)

CATEGORIES = ["upper_body", "lower_body", "dresses"]


def download_dresscode(output_dir: Path):
    """Download DressCode from HuggingFace and organize."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError("pip install huggingface_hub to download datasets")

    logger.info("Downloading DressCode dataset...")
    cache_dir = snapshot_download(
        repo_id="aimagelab/dress-code",
        repo_type="dataset",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["person", "garment", "garment_mask"]:
        (output_dir / subdir).mkdir(exist_ok=True)

    pairs = []
    cache_path = Path(cache_dir)

    for category in CATEGORIES:
        cat_dir = cache_path / category
        if not cat_dir.exists():
            logger.warning("Category %s not found", category)
            continue

        images_dir = cat_dir / "images"
        if not images_dir.exists():
            continue

        for img_path in sorted(images_dir.glob("*_0.jpg")):
            stem = img_path.stem.replace("_0", "")
            cloth_path = images_dir / f"{stem}_1.jpg"

            if not cloth_path.exists():
                continue

            out_stem = f"{category}_{stem}"

            person = Image.open(img_path).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
            garment = Image.open(cloth_path).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)

            person.save(output_dir / "person" / f"{out_stem}.png")
            garment.save(output_dir / "garment" / f"{out_stem}.png")

            pairs.append({
                "id": out_stem,
                "split": "train",
                "person": {"image": f"person/{out_stem}.png"},
                "garment": {
                    "image": f"garment/{out_stem}.png",
                    "mask": f"garment_mask/{out_stem}.png",
                    "category": category,
                },
            })

    with open(output_dir / "pairs.json", "w") as f:
        json.dump(pairs, f, indent=2)

    logger.info("Processed %d pairs → %s", len(pairs), output_dir)


def main():
    parser = argparse.ArgumentParser(description="Download DressCode dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_dresscode(args.output_dir)


if __name__ == "__main__":
    main()
