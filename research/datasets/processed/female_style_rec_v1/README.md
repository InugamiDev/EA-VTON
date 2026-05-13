# Female Style Recommendation v1

Processed dataset package built from redacted clothing style manifests.

Files:

- `images_train.jsonl`, `images_val.jsonl`, `images_test.jsonl`: image-level records with redacted paths and labels.
- `image_labels.csv`: flattened primary outfit labels for classifier baselines.
- `garment_labels.csv`: one row per labeled garment.
- `person_profiles.jsonl`: repeated-image style aggregates by person.
- `style_interactions.jsonl`: positive person-to-style-item observations for retrieval/recommender training.
- `class_maps.json`: stable integer ids for categorical labels.
- `stats.json`: source, counts, and label distribution summary.

Use this package for style-rec pilots only. External dataset rows use `dataset_label` gender provenance and must not be treated as first-party self-identification.
