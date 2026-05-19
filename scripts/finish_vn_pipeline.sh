#!/usr/bin/env bash
# Chain script: Steps 2 → 3 → 4 of the VN new-data pipeline.
#
# Assumes vn_person_pipeline.py (Step 1+2) has already produced
# research/datasets/raw/vn_brands/vn_person_clusters.jsonl.
#
# Runs:
#   - ingest_vn_person_ids.py    — copies person_id + face_clip_embedding into the parquet
#   - redact_faces.py             — face-mesh-oval redaction on raw/vn_brands/images/{yody,coupletx}/
#   - relabel_from_redacted.py    — CLIP zero-shot attributes on the redacted images
#   - derive_person_labels.py     — body-cluster suitability + season + style_personality
#   - build_style_catalog_duckdb.py — refresh the DuckDB view
#
# Total runtime: ~30-60 min on CPU for 615 items.
#
# Usage:
#   bash scripts/finish_vn_pipeline.sh

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
PY="${ROOT}/.venv/bin/python3"
cd "${ROOT}"

echo "══ Step A — ingest person_ids into catalog ══"
"${PY}" research/datasets/scripts/ingest_vn_person_ids.py

echo ""
echo "══ Step B — face-mesh oval redaction on the 615 new VN images ══"
"${PY}" research/datasets/scripts/redact_faces.py \
    --sources vn_brands --workers 6

echo ""
echo "══ Step C — CLIP zero-shot relabel (only the new rows with placeholder labels) ══"
echo "   Note: relabel_from_redacted.py runs over the WHOLE catalog by default."
echo "   For our 615-item incremental update, only rows where image_local_path now"
echo "   points to an images_redacted/vn_brands/... path get re-labeled meaningfully;"
echo "   the existing 208k rows are re-labeled to the same values (idempotent)."
echo "   To skip the redundant re-label of pre-existing rows, set the --limit flag:"
echo "      python research/datasets/scripts/relabel_from_redacted.py --limit 615 --device cpu"
echo "   (Filter by source column not currently supported in that script — extend if needed.)"
"${PY}" research/datasets/scripts/relabel_from_redacted.py --device cpu

echo ""
echo "══ Step D — derive person-categorization labels (body cluster, season, style) ══"
"${PY}" research/datasets/scripts/derive_person_labels.py

echo ""
echo "══ Step E — rebuild DuckDB view ══"
"${PY}" research/datasets/scripts/build_style_catalog_duckdb.py

echo ""
echo "══ Done ══"
echo "  items_upper_combined.parquet is now fully labeled."
echo "  Next: hf upload to refresh the private dataset on Hugging Face."
