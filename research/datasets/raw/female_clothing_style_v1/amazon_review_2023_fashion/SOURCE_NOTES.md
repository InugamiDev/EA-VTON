# Amazon Reviews 2023 fashion categories

This source has small direct files plus larger form-gated category files. It is registered for product/review metadata only; user-posted images must not enter the people-image training manifest without separate rights review.

Homepage: https://amazon-reviews-2023.github.io/

License status: `research_dataset_terms_required_before_training`

Direct files:
- `amazon_fashion_metadata`: https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Amazon_Fashion.jsonl.gz
- `amazon_fashion_reviews`: https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Amazon_Fashion.jsonl.gz
- `clothing_shoes_jewelry_metadata`: https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz
- `clothing_shoes_jewelry_ratings_0core`: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/Clothing_Shoes_and_Jewelry.csv.gz
- `clothing_shoes_jewelry_reviews`: https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz

Notes: Prefer this over the 2018 version for new recommendation experiments. Use product metadata, product images, ratings, reviews, and item graphs as a sidecar source only. Strip reviewer identifiers from exported training features and exclude user-posted review images from the people-image manifest unless a separate rights/privacy review approves them.
