#!/usr/bin/env python3
"""Fetch source images for a subset of labels.parquet.

Each source has different access controls. This script handles only the
HuggingFace-hosted sources automatically; DeepFashion2 requires manual
form-gated download from CUHK MMLab.

Usage:
    python download_images.py --subset deepfashion_with_masks --output ./images
    python download_images.py --subset fashionpedia --max 1000

Prerequisites:
    pip install pyarrow pillow huggingface_hub
"""

import argparse
import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

HF_REPOS = {
    "deepfashion_with_masks": ("SaffalPoosh/deepFashion-with-masks", "dataset"),
    "fashionpedia":           ("detection-datasets/fashionpedia", "dataset"),
    "fashion_products_small": ("ashraq/fashion-product-images-small", "dataset"),
}

MANUAL_INSTRUCTIONS = {
    "deepfashion2": """DeepFashion2 requires form-based access from CUHK MMLab.

  1. Visit http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion2.html
  2. Fill out the access form, email Yuying Ge
  3. You will receive a password and Google Drive links
  4. Download train.zip, unzip with the supplied password
  5. Place at: <output>/deepfashion2/train/image/*.jpg
""",
}


def fetch_hf_source(subset: str, output_dir: Path, max_items: int = 0) -> None:
    from huggingface_hub import snapshot_download
    repo, kind = HF_REPOS[subset]
    target = output_dir / subset
    target.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {repo} → {target}")
    snapshot_download(repo_id=repo, repo_type=kind, local_dir=str(target))
    print(f"  done. extract images per dataset's own format.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True,
                    choices=["deepfashion2", "deepfashion_with_masks", "fashionpedia", "fashion_products_small"])
    ap.add_argument("--output", default="./images")
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    if args.subset == "deepfashion2":
        print(MANUAL_INSTRUCTIONS["deepfashion2"]); sys.exit(0)

    fetch_hf_source(args.subset, Path(args.output), args.max)


if __name__ == "__main__":
    main()
