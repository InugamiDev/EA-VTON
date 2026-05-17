# Upstream Source Licenses

This labeled catalog references images from four source datasets. **Our labels**
(labels.parquet, taxonomy.json, methodology, etc.) are CC BY 4.0. The **source
images** retain their original licenses, which you must comply with separately
when downloading them via `download_images.py`.

## Per-Source License Inheritance

| Source | Items | Source License | Redistributable? |
|---|---|---|---|
| DeepFashion2 | 139,788 | CUHK MMLab — non-commercial research only | No, fetch original |
| Fashionpedia | 37,875 | CC BY 4.0 (annotations); CC BY-NC for images | Limited |
| DeepFashion-with-masks | 25,122 | Apache 2.0 (HF mirror), derives from DeepFashion2 | Limited |
| fashion_products_small | 5,207 | Myntra source via Kaggle — verify per use case | No |

## Fetching the Originals

Use `download_images.py` (provided) to fetch each source's images into the
expected layout. The script obeys each source's access controls (e.g., the
DeepFashion2 form-gated download).

## Attribution

If you use this labeled catalog, cite both:
1. Our work (LICENSE)
2. The relevant upstream sources as documented in their READMEs.
