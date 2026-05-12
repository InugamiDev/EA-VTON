# Selected Sources

Last selected: 2026-05-08.

## Use Now

| Source | Role | Local data | Decision |
|---|---|---:|---|
| `deepfashion_with_masks_all_capped20` | External redacted pretraining | 34,142 records, 6,694 people | Use for garment/style classifier pretraining and smoke tests. |
| `deepfashion_with_masks_balanced_10_20` | Same-person pilot | 8,194 records, 589 people | Use for person-level split and repeated-image aggregation pilots. |
| `participant_intake_v1` | Final target collection | 100,000-row roster | Use to collect the real consented 10,000 x 10 corpus. |
| `amazon_review_2023_fashion` | Recommendation sidecar | Pending download | Use only product/review metadata, ratings, and item graph data. |
| `local_catalog_pilot` | Pipeline smoke test | Local garment catalog | Use only for schema and pipeline checks. |

## Do Not Use For People Images

Do not crawl arbitrary people images from social media, search results, videos, dating apps, school/work pages, forums, or private communities.

Do not use Amazon review user-posted images in the people-image manifest. Amazon is selected only for product/review metadata unless a separate rights/privacy review explicitly approves image use.

## Deferred

The following are useful but not selected until access and terms are cleared:

- `deepfashion2`
- `fashionpedia`
- `modanet`
- `viton_hd`
- `dresscode`
- `deepfashion_multimodal_parts2whole`

## Blocked

The following are not selected for this dataset:

- `luperson`
- `luperson_nl`
- `last_reid`
- `entire_id`

They are ReID or public-camera/movie-derived sources and do not pass this dataset's consent, gender-label, privacy, license, and fashion-label gates.

## Build Order

1. Train initial garment/style classifiers from `manifest_deepfashion_with_masks_all_capped20.jsonl`.
2. Test person-level repeated-image aggregation on `manifest_deepfashion_with_masks_balanced_10_20.jsonl`.
3. Add Amazon Reviews 2023 product/review metadata as a sidecar after download succeeds.
4. Fill the final training corpus through participant intake, redaction, labeling, and strict manifest validation.
