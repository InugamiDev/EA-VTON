"""Download and normalize DeepFashion2 dataset (for recommendation model).

DeepFashion2: 801K images across 13 clothing categories.
Used for training the item tower of the two-tower recommendation model.
Source: https://github.com/switchablenorms/DeepFashion2

Note: DeepFashion2 requires manual download due to license restrictions.
This script processes the downloaded data into our unified format.

Usage:
    python data/scripts/download_deepfashion2.py --input-dir /path/to/deepfashion2 [--output-dir data/processed/deepfashion2]
"""

# intent: process DeepFashion2 dataset for recommendation model training
# status: done
# next: requires manual download first (license restricted)
# confidence: high

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/processed/deepfashion2")
TARGET_SIZE = (384, 512)

CATEGORIES = [
    "short_sleeve_top", "long_sleeve_top", "short_sleeve_outwear",
    "long_sleeve_outwear", "vest", "sling", "shorts", "trousers",
    "skirt", "short_sleeve_dress", "long_sleeve_dress",
    "vest_dress", "sling_dress",
]


def process_deepfashion2(input_dir: Path, output_dir: Path):
    """Process DeepFashion2 into unified format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["images", "annotations"]:
        (output_dir / subdir).mkdir(exist_ok=True)

    items = []

    for split in ["train", "validation"]:
        split_dir = input_dir / split
        image_dir = split_dir / "image"
        annot_dir = split_dir / "annos"

        if not image_dir.exists():
            logger.warning("Split dir not found: %s", image_dir)
            continue

        for img_path in sorted(image_dir.glob("*.jpg"))[:50000]:  # Cap per split
            stem = img_path.stem
            annot_path = annot_dir / f"{stem}.json"

            if not annot_path.exists():
                continue

            with open(annot_path) as f:
                annot = json.load(f)

            # Process each item in the annotation
            for key, item_annot in annot.items():
                if not key.startswith("item"):
                    continue

                category_id = item_annot.get("category_id", 0)
                if category_id < 1 or category_id > 13:
                    continue

                category = CATEGORIES[category_id - 1]
                item_id = f"{split}_{stem}_{key}"

                # Crop and resize
                bbox = item_annot.get("bounding_box", [])
                if len(bbox) != 4:
                    continue

                img = Image.open(img_path).convert("RGB")
                x1, y1, x2, y2 = bbox
                cropped = img.crop((x1, y1, x2, y2))
                resized = cropped.resize(TARGET_SIZE, Image.LANCZOS)
                resized.save(output_dir / "images" / f"{item_id}.png")

                items.append({
                    "id": item_id,
                    "split": split,
                    "image": f"images/{item_id}.png",
                    "category": category,
                    "category_id": category_id,
                })

    with open(output_dir / "items.json", "w") as f:
        json.dump(items, f, indent=2)

    logger.info("Processed %d items → %s", len(items), output_dir)


def main():
    parser = argparse.ArgumentParser(description="Process DeepFashion2 dataset")
    parser.add_argument("--input-dir", type=Path, required=True,
                       help="Path to downloaded DeepFashion2 dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    process_deepfashion2(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
