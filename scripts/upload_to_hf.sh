#!/usr/bin/env bash
# Upload the labeled catalog + redacted images to a PRIVATE Hugging Face dataset
# so the A100 collaborator can pull everything with one command.
#
# Total upload: ~14-15 GB
#   - items_upper_combined.parquet            642 MB
#   - dataset_bundle_v1/labels.parquet        640 MB  (backup of the same labels)
#   - combined/images_redacted.tar.zst       ~6-9 GB compressed (13 GB raw, zst -19 ≈ ~50% on JPEGs)
#   - combined/images.tar.zst                ~2-3 GB compressed
#   - person_visibility_flags.parquet          ~5 MB
#
# IMPORTANT — license posture:
#   Source images are bound by their upstream licenses:
#     - DeepFashion2: CUHK research-only, NOT publicly redistributable
#     - Fashionpedia: CC BY 4.0 — OK
#     - DeepFashion-with-masks: Apache 2.0 — OK
#     - Myntra/fashion_products_small: verify per use
#     - VN brand scrapes: research-fair-use only
#   Therefore this MUST be a PRIVATE dataset on HF, shared only with collaborators
#   who agree to the same upstream license terms.
#
# Prereqs (one-time on the lead's machine):
#   pip install -U "huggingface_hub[cli]" zstandard
#   .venv/bin/hf auth login            # paste a write-scoped token
#
# Note: HF rebranded the CLI from `huggingface-cli` to `hf`. Both ship with
# huggingface_hub[cli] but only `hf` is supported going forward. This script
# uses `hf` directly (preferring the venv install at .venv/bin/hf).
#
# Usage:
#   bash scripts/upload_to_hf.sh <hf-username> <dataset-name>
#   Example:
#     bash scripts/upload_to_hf.sh InugamiDev eavton-internal-v1

set -euo pipefail

HF_USER="${1:?usage: bash scripts/upload_to_hf.sh <hf-username> <dataset-name>}"
DATASET_NAME="${2:?usage: bash scripts/upload_to_hf.sh <hf-username> <dataset-name>}"
REPO_ID="${HF_USER}/${DATASET_NAME}"

ROOT="$(git rev-parse --show-toplevel)"
COMBINED_DIR="${ROOT}/research/datasets/processed/combined"
BUNDLE_DIR="${ROOT}/research/datasets/processed/dataset_bundle_v1"
STAGING="${ROOT}/.hf_staging"

echo "── Creating staging directory at ${STAGING}"
mkdir -p "${STAGING}"

echo "── Compressing redacted images (zstd -10, multithreaded)"
if [ ! -f "${STAGING}/images_redacted.tar.zst" ]; then
  tar --use-compress-program="zstd -10 -T0" \
      -cf "${STAGING}/images_redacted.tar.zst" \
      -C "${COMBINED_DIR}" images_redacted/
  echo "   wrote ${STAGING}/images_redacted.tar.zst ($(du -h "${STAGING}/images_redacted.tar.zst" | cut -f1))"
else
  echo "   already exists, skipping"
fi

echo "── Compressing unredacted images (for downstream tasks that need both)"
if [ ! -f "${STAGING}/images.tar.zst" ]; then
  tar --use-compress-program="zstd -10 -T0" \
      -cf "${STAGING}/images.tar.zst" \
      -C "${COMBINED_DIR}" images/
  echo "   wrote ${STAGING}/images.tar.zst ($(du -h "${STAGING}/images.tar.zst" | cut -f1))"
else
  echo "   already exists, skipping"
fi

echo "── Staging parquet labels"
cp "${COMBINED_DIR}/items_upper_combined.parquet" "${STAGING}/"
cp "${BUNDLE_DIR}/labels.parquet" "${STAGING}/labels_bundle.parquet"

# Optional sidecars
[ -f "${COMBINED_DIR}/person_visibility_flags.parquet" ] && \
  cp "${COMBINED_DIR}/person_visibility_flags.parquet" "${STAGING}/" || true
[ -f "${COMBINED_DIR}/person_visibility_report.json" ] && \
  cp "${COMBINED_DIR}/person_visibility_report.json" "${STAGING}/" || true

# A short manifest the collaborator reads first
cat > "${STAGING}/README.md" <<EOF
# eavton-internal — private collaborator data drop

This dataset is PRIVATE. Source images carry the licenses of their upstream
datasets (DeepFashion2, Fashionpedia, etc.) — do not redistribute.

## Contents

| File | Size (approx) | What it is |
|---|---|---|
| items_upper_combined.parquet | 642 MB | 208,069 items × 37 cols, includes CLIP embeddings + person-labels |
| labels_bundle.parquet | 640 MB | Same data, the v1 dataset-release bundle backup |
| images_redacted.tar.zst | ~6-9 GB | Face-mesh-redacted JPEGs, one per item (~177k files) |
| images.tar.zst | ~2-3 GB | Original (unredacted) JPEGs for paired-training tasks |
| person_visibility_flags.parquet | ~5 MB | Per-row flag from MediaPipe pose filter (optional) |

## Extraction on the A100

\`\`\`bash
# 0. Install + auth (one-time)
pip install -U "huggingface_hub[cli]" zstandard
hf auth login

# 1. From the cloned repo root, pull the dataset (~14-15 GB)
hf download ${REPO_ID} --repo-type dataset \\
    --local-dir .hf_collab_pull

# 2. Place the parquets
mkdir -p research/datasets/processed/combined research/datasets/processed/dataset_bundle_v1
cp .hf_collab_pull/items_upper_combined.parquet research/datasets/processed/combined/
cp .hf_collab_pull/labels_bundle.parquet research/datasets/processed/dataset_bundle_v1/labels.parquet
[ -f .hf_collab_pull/person_visibility_flags.parquet ] && \\
    cp .hf_collab_pull/person_visibility_flags.parquet research/datasets/processed/combined/

# 3. Extract images (zstd -d, multithread, ~3-5 min on A100 SSD)
tar --use-compress-program=zstd -xf .hf_collab_pull/images_redacted.tar.zst \\
    -C research/datasets/processed/combined/
tar --use-compress-program=zstd -xf .hf_collab_pull/images.tar.zst \\
    -C research/datasets/processed/combined/

# 4. Rebuild DuckDB view
python research/datasets/scripts/build_style_catalog_duckdb.py
\`\`\`

You are NOT downloading BodyM photos via this dataset — those come from Meta's
public BodyM release per COLLABORATOR_SETUP.md §4.
EOF

echo "── Uploading to https://huggingface.co/datasets/${REPO_ID}"
echo "   (HF will create the dataset if it doesn't exist; default visibility = PRIVATE)"

# Resolve hf binary: prefer venv install, fall back to PATH.
HF_BIN="${ROOT}/.venv/bin/hf"
if [ ! -x "${HF_BIN}" ]; then
    if command -v hf >/dev/null 2>&1; then
        HF_BIN="hf"
    else
        echo "!! could not find the 'hf' CLI."
        echo "   Install with: pip install -U \"huggingface_hub[cli]\""
        echo "   Then auth:    .venv/bin/hf auth login"
        exit 1
    fi
fi

"${HF_BIN}" upload "${REPO_ID}" "${STAGING}" . \
    --repo-type dataset --private \
    --commit-message "Initial private upload — labels + redacted images for collaborator A100 training"

echo ""
echo "── Done. Next step:"
echo "   1. Visit https://huggingface.co/datasets/${REPO_ID}/settings"
echo "      → set visibility to PRIVATE if not already"
echo "      → invite the collaborator (her HF username) as a member"
echo "   2. Send her the dataset URL + her HF access token instructions"
echo ""
echo "   Staging files left in ${STAGING} — delete after she confirms download:"
echo "      rm -rf ${STAGING}"
