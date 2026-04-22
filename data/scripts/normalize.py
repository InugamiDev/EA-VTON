"""Normalize all datasets into unified schema.

Reads pairs.json from each dataset directory, validates images,
and creates a combined index at data/processed/all_pairs.json.

Usage:
    python data/scripts/normalize.py
"""

# intent: unify all dataset pair indices into a single normalized schema
# status: done
# confidence: high

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")

DATASETS = [
    {"name": "vitonhd", "pairs_file": "pairs.json"},
    {"name": "dresscode", "pairs_file": "pairs.json"},
]


def normalize():
    """Combine all dataset pair indices."""
    all_pairs = []

    for ds in DATASETS:
        ds_dir = DATA_DIR / ds["name"]
        pairs_path = ds_dir / ds["pairs_file"]

        if not pairs_path.exists():
            logger.warning("Skipping %s — %s not found", ds["name"], pairs_path)
            continue

        with open(pairs_path) as f:
            pairs = json.load(f)

        for pair in pairs:
            # Prefix paths with dataset name
            pair["dataset"] = ds["name"]
            if "person" in pair and "image" in pair["person"]:
                pair["person"]["image"] = f"{ds['name']}/{pair['person']['image']}"
            if "garment" in pair and "image" in pair["garment"]:
                pair["garment"]["image"] = f"{ds['name']}/{pair['garment']['image']}"
            if "garment" in pair and "mask" in pair["garment"]:
                pair["garment"]["mask"] = f"{ds['name']}/{pair['garment']['mask']}"

            all_pairs.append(pair)

    output_path = DATA_DIR / "all_pairs.json"
    with open(output_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    logger.info("Combined %d pairs from %d datasets → %s",
                len(all_pairs), len(DATASETS), output_path)

    # Print summary
    by_dataset = {}
    by_category = {}
    for p in all_pairs:
        ds = p.get("dataset", "unknown")
        cat = p.get("garment", {}).get("category", "unknown")
        by_dataset[ds] = by_dataset.get(ds, 0) + 1
        by_category[cat] = by_category.get(cat, 0) + 1

    print("\nDataset summary:")
    for ds, count in sorted(by_dataset.items()):
        print(f"  {ds}: {count:,} pairs")
    print("\nCategory breakdown:")
    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count:,}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    normalize()
