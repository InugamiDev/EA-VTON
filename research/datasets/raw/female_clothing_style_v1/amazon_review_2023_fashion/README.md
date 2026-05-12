# Amazon Reviews 2023 fashion categories

This source is registered for recommendation-side product/review metadata only. It is not a people-image collection source.

Homepage: https://amazon-reviews-2023.github.io/

License status: `research_dataset_terms_required_before_training`

Relevant fashion categories:

- `Amazon_Fashion`: 2.0M users, 825.9K items, 2.5M ratings.
- `Clothing_Shoes_and_Jewelry`: 22.6M users, 7.2M items, 66.0M ratings.

Direct files tracked in `sources.json`:

- `amazon_fashion_reviews`
- `amazon_fashion_metadata`
- `clothing_shoes_jewelry_reviews`
- `clothing_shoes_jewelry_metadata`
- `clothing_shoes_jewelry_ratings_0core`

Use notes:

- Strip reviewer identifiers from exported features unless a privacy review approves a pseudonymous interaction graph.
- Use product metadata and product images for item-side embeddings after rights review.
- Do not add user-posted review images to `female_clothing_style_v1` people-image manifests without separate rights/privacy approval.
- Prefer this source over Amazon Reviews 2018 for new recommendation experiments; keep 2018 only for reproducibility.
