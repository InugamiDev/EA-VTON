# Female Style Rec Annotation

This directory contains auto-label outputs and human review batches for the style recommendation dataset.

## Auto Labels

Auto-labels are generated from redacted manifests and existing source-derived labels. They do not infer identity or sensitive traits.

Current totals across train/val/test:

- Records: 8,194
- Auto-accepted: 7,076
- Needs review: 1,118
- Confidence threshold: 0.72

Use these files for weak-supervision style-rec training:

- `auto_labels/accepted_labels.jsonl`
- `auto_labels_val/accepted_labels.jsonl`
- `auto_labels_test/accepted_labels.jsonl`

Review these queues before treating labels as high quality:

- `auto_labels/review_queue.jsonl`
- `auto_labels_val/review_queue.jsonl`
- `auto_labels_test/review_queue.jsonl`

## Human Review

`batch_0001/` is the first 100-record manual review batch. Open the local labeling UI through the dataset server:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

From `research/datasets/`, open:

```text
http://127.0.0.1:8765/annotation/female_style_rec_v1/batch_0001/index.html
```

Export reviewed labels from the browser as JSONL or CSV when done.

## Refresh Commands

```bash
python3 research/datasets/scripts/auto_label_female_style_rec.py
python3 research/datasets/scripts/auto_label_female_style_rec.py \
  --source research/datasets/processed/female_style_rec_v1/images_val.jsonl \
  --output-dir research/datasets/annotation/female_style_rec_v1/auto_labels_val
python3 research/datasets/scripts/auto_label_female_style_rec.py \
  --source research/datasets/processed/female_style_rec_v1/images_test.jsonl \
  --output-dir research/datasets/annotation/female_style_rec_v1/auto_labels_test
python3 research/datasets/scripts/start_female_style_labeling.py --batch-size 100
```
