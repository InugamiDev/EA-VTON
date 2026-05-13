# Auto Labels

Auto-labeled style recommendation records generated from redacted manifests and existing source-derived labels.

Files:

- `auto_labels.jsonl`: all auto-label records with confidence and status.
- `accepted_labels.jsonl`: labels accepted at the configured confidence threshold.
- `review_queue.jsonl`: low-confidence records for human verification.
- `auto_labels.csv` and `review_queue.csv`: spreadsheet-friendly versions.
- `stats.json`: counts and review-reason distribution.

These labels do not infer identity or sensitive traits. They should be treated as weak labels until human QA confirms enough samples per class.
