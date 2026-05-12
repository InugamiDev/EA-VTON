# Amazon Review Data 2018 fashion categories

This source has small direct files plus larger form-gated category files. It is registered for product/review metadata only; user-posted images must not enter the people-image training manifest without separate rights review.

Homepage: https://nijianmo.github.io/amazon/index.html

License status: `research_dataset_terms_required_before_training`

Direct files:
- `amazon_fashion_5core`: https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION_5.json.gz
- `amazon_fashion_ratings`: https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION.csv
- `clothing_shoes_jewelry_5core`: https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Clothing_Shoes_and_Jewelry_5.json.gz
- `clothing_shoes_jewelry_ratings`: https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Clothing_Shoes_and_Jewelry.csv

Notes: Useful for product metadata, style fields, ratings, reviews, and co-purchase graphs. Reviews may contain reviewer IDs/names and user-posted images; strip identifying fields and do not add user images to the redacted people manifest without a separate rights/privacy review. Full per-category review and metadata files are form-gated.
