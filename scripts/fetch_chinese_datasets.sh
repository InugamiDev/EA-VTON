#!/usr/bin/env bash
# Fetch Chinese-affiliated fashion datasets to a local cache. NON-distributable —
# the collaborator runs this directly on her A100 after she has signed each
# upstream license. We do not include these in the HF private dataset because
# the licenses (CUHK research-only for DeepFashion-derived data) forbid both
# public AND private redistribution by us.
#
# Sources covered:
#   DeepFashion-MultiModal (CUHK Liu Lab + NTU + SenseTime) — ~5.4 GB images,
#     44,096 items, manual parsing+keypoint annotations, textual descriptions.
#     License: same as DeepFashion (CUHK research-only).
#     Citation: Jiang et al., SIGGRAPH 2022, "Text2Human: Text-Driven Controllable
#     Human Image Generation".
#     Repo: https://github.com/yumingj/DeepFashion-MultiModal
#
#   SHHQ-1.0 (SenseHuman / StyleGAN-Human) — 40,000 full-body images.
#     License: research-only.
#     Citation: Fu et al., ECCV 2022.
#     Repo: https://github.com/stylegan-human/StyleGAN-Human
#
# Prereqs:
#   pip install gdown                # Google Drive downloader
#   For SHHQ-1.0: read the StyleGAN-Human repo README and accept the EULA before
#     running this script.
#
# Usage:
#   bash scripts/fetch_chinese_datasets.sh [--no-shhq]
#   The --no-shhq flag skips StyleGAN-Human (you may not want it if license
#   acceptance hasn't happened yet).

set -euo pipefail

INCLUDE_SHHQ=1
if [ "${1:-}" = "--no-shhq" ]; then
    INCLUDE_SHHQ=0
fi

ROOT="$(git rev-parse --show-toplevel)"
RAW="${ROOT}/research/datasets/raw"
DF_MM="${RAW}/deepfashion_multimodal"
SHHQ="${RAW}/shhq_1_0"

# ──────────────────────────────────────────────────────────────────────────
# 1. DeepFashion-MultiModal
# ──────────────────────────────────────────────────────────────────────────
echo "── DeepFashion-MultiModal ──"
mkdir -p "${DF_MM}"

if ! command -v gdown >/dev/null 2>&1; then
    echo "!! gdown not installed. Run: pip install gdown"
    exit 1
fi

# Google Drive file IDs (from https://github.com/yumingj/DeepFashion-MultiModal#download-links)
declare -A DF_MM_FILES
DF_MM_FILES[image.zip]="1U2PljA7NE57jcSSzPs21ZurdIPXdYZtN"        # 5.4 GB — 44,096 JPGs (750×1101)
DF_MM_FILES[parsing.zip]="1r-5t-VgDaAQidZLVgWtguaG7DvMoyUv9"      # 90 MB  — 12,701 PNG masks
DF_MM_FILES[labels.zip]="11WoM5ZFwWpVjrIvZajW0g8EmQCNKMAWH"       # 575 KB — shape/fabric/color labels
DF_MM_FILES[textual_descriptions.zip]="1d1TRm8UMcQhZCb6HpPo8l3OPEin4Ztk2"  # 11 MB

for filename in "${!DF_MM_FILES[@]}"; do
    fileid="${DF_MM_FILES[$filename]}"
    target="${DF_MM}/${filename}"
    if [ -f "${target}" ]; then
        echo "   already downloaded: ${filename}"
        continue
    fi
    echo "   downloading ${filename} (id=${fileid})..."
    gdown "https://drive.google.com/uc?id=${fileid}" -O "${target}" --no-check-certificate
done

echo "   unzipping..."
cd "${DF_MM}"
for z in *.zip; do
    [ -f "${z}" ] || continue
    if [ ! -d "${z%.zip}" ]; then
        unzip -q "${z}"
    fi
done
cd "${ROOT}"

# ──────────────────────────────────────────────────────────────────────────
# 2. SHHQ-1.0 (optional)
# ──────────────────────────────────────────────────────────────────────────
if [ "${INCLUDE_SHHQ}" -eq 1 ]; then
    echo ""
    echo "── SHHQ-1.0 ──"
    mkdir -p "${SHHQ}"
    echo "   The StyleGAN-Human team requires you to sign their EULA before download."
    echo "   1. Visit:  https://github.com/stylegan-human/StyleGAN-Human"
    echo "   2. Read the README → 'Note: pls fill out this form to download SHHQ-1.0'"
    echo "   3. Fill in the Google Form they link"
    echo "   4. They reply by email with a personalized download link"
    echo "   5. Place the downloaded SHHQ-1.0.tar.gz under ${SHHQ}/ and extract"
    echo ""
    echo "   This step is NOT automated because the license requires per-user gating."
else
    echo ""
    echo "── SHHQ-1.0 skipped (--no-shhq flag) ──"
fi

# ──────────────────────────────────────────────────────────────────────────
# 3. Next steps for the collaborator
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "── Next ──"
echo "   1. Run the build_combined_catalog.py to ingest these new sources."
echo "      python research/datasets/scripts/build_combined_catalog.py \\"
echo "          --sources deepfashion_multimodal"
echo "   2. Run the relabel pipeline on the new items only (~1 hour A100):"
echo "      python research/datasets/scripts/relabel_from_redacted.py \\"
echo "          --filter source=deepfashion_multimodal"
echo "   3. Rebuild the DuckDB view: build_style_catalog_duckdb.py"
echo ""
echo "   Reminder: DeepFashion-MultiModal images cannot be redistributed (CUHK"
echo "   research-only license). Train + evaluate locally; do NOT push images to HF."
