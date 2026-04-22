"""Download and normalize VITON-HD dataset.

VITON-HD: ~11,647 paired person-garment images.
Source: forgeml/viton_hd on HuggingFace (parquet format with embedded images).

After download, images are extracted and organized into:
  data/processed/vitonhd/
    person/        — person images
    garment/       — garment images (flat-lay cloth)
    garment_mask/  — garment segmentation masks
    agnostic/      — agnostic person images (garment region removed)
    pairs.json     — pair index with metadata

Usage:
    python data/scripts/download_vitonhd.py [--output-dir data/processed/vitonhd]
"""

# intent: download and normalize VITON-HD dataset for training
# status: done
# next: run feature extraction after download
# confidence: high

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/processed/vitonhd")
TARGET_SIZE = (384, 512)  # (W, H)


def download_vitonhd(output_dir: Path):
    """Download VITON-HD from HuggingFace parquet format and organize."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("pip install huggingface_hub to download datasets")

    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pip install pandas pyarrow to process parquet files")

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["person", "garment", "garment_mask", "agnostic"]:
        (output_dir / subdir).mkdir(exist_ok=True)

    pairs = []
    repo_id = "forgeml/viton_hd"
    n_shards = 8

    for shard_idx in range(n_shards):
        shard_file = f"data/train-{shard_idx:05d}-of-{n_shards:05d}-*.parquet"

        # List actual shard filenames
        from huggingface_hub import HfApi
        api = HfApi()
        files = list(api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo="data"))
        shard_files = [f.path for f in files if f.path.startswith(f"data/train-{shard_idx:05d}")]

        if not shard_files:
            logger.warning("Shard %d not found", shard_idx)
            continue

        shard_path = shard_files[0]
        logger.info("Downloading shard %d/%d: %s", shard_idx + 1, n_shards, shard_path)

        local_path = hf_hub_download(repo_id, shard_path, repo_type="dataset")
        df = pd.read_parquet(local_path)

        for row_idx, row in df.iterrows():
            stem = f"viton_{shard_idx:02d}_{row_idx:05d}"

            try:
                # Extract person image
                person_bytes = row["image"]["bytes"]
                person = Image.open(io.BytesIO(person_bytes)).convert("RGB")
                person = person.resize(TARGET_SIZE, Image.LANCZOS)
                person.save(output_dir / "person" / f"{stem}.png")

                # Extract garment image
                cloth_bytes = row["cloth"]["bytes"]
                garment = Image.open(io.BytesIO(cloth_bytes)).convert("RGB")
                garment = garment.resize(TARGET_SIZE, Image.LANCZOS)
                garment.save(output_dir / "garment" / f"{stem}.png")

                # Extract garment mask
                mask_bytes = row["cloth_mask"]["bytes"]
                mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
                mask = mask.resize(TARGET_SIZE, Image.NEAREST)
                mask.save(output_dir / "garment_mask" / f"{stem}.png")

                # Extract agnostic if available
                if row.get("agnostic") is not None and row["agnostic"].get("bytes"):
                    agn_bytes = row["agnostic"]["bytes"]
                    agnostic = Image.open(io.BytesIO(agn_bytes)).convert("RGB")
                    agnostic = agnostic.resize(TARGET_SIZE, Image.LANCZOS)
                    agnostic.save(output_dir / "agnostic" / f"{stem}.png")

                pairs.append({
                    "id": stem,
                    "split": "train",
                    "person": {"image": f"person/{stem}.png"},
                    "garment": {
                        "image": f"garment/{stem}.png",
                        "mask": f"garment_mask/{stem}.png",
                        "category": "upper_body",
                    },
                })
            except Exception as e:
                logger.warning("Failed to process row %s: %s", stem, e)
                continue

        logger.info("Shard %d: processed %d rows, total pairs: %d",
                     shard_idx + 1, len(df), len(pairs))

    # Write pairs index
    with open(output_dir / "pairs.json", "w") as f:
        json.dump(pairs, f, indent=2)

    logger.info("Processed %d pairs → %s", len(pairs), output_dir)


def main():
    parser = argparse.ArgumentParser(description="Download VITON-HD dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_vitonhd(args.output_dir)


if __name__ == "__main__":
    main()
