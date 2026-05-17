---
license: cc-by-4.0
task_categories:
- image-classification
- visual-question-answering
language:
- en
- vi
tags:
- fashion
- clothing
- garment
- style-recommendation
- vietnamese
- multi-source
- clip-zero-shot
size_categories:
- 100K<n<1M
---

# EA-VTON Upper-Body Garment Attribute Labels (v1)

**207,992** upper-body garment items labeled with a unified attribute taxonomy
across four public fashion datasets (DeepFashion2, Fashionpedia,
DeepFashion-with-masks, fashion_products_small). Each row carries:

- 7 categorical attributes (neckline, silhouette, sleeve, pattern, color temp/value, category)
- Primary color in CIE LAB
- 512-d OpenCLIP ViT-B-32 image embedding
- Bounding box / image pointer back to source

Built to support **population-aware style recommendation** research — paired
with our 12-season color palettes and 5-cluster Vietnamese body-type rules.

## Quick Start

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW items AS SELECT * FROM read_parquet('labels.parquet')")

# Find V-neck warm-tone fitted tops
con.execute('''
    SELECT garment_id, source, color_temperature, neckline, silhouette
    FROM items
    WHERE neckline IN ('v_neck', 'deep_v_neck')
      AND silhouette = 'fitted'
      AND color_temperature = 'warm'
    LIMIT 10
''').fetchall()
```

## Per-Source Breakdown

| Source | Items | Source License |
|---|---|---|
| DeepFashion2 | 139,788 | Research-only (CUHK MMLab) |
| Fashionpedia | 37,875 | CC BY 4.0 |
| DeepFashion-with-masks (HF) | 25,122 | Apache 2.0 |
| fashion_products_small (Myntra HF) | 5,207 | Verify per source |

## Files

- `labels.parquet` — 207,992 rows, ~640 MB
- `taxonomy.json` — label space (8 necklines × 6 silhouettes × 5 patterns × …)
- `schema.json` — column descriptions
- `seasonal_palettes.json` — 12 Sci\ART seasons × 8 LAB anchors
- `vn_body_centroids.json` — 5 VN body-type cluster centroids (Trieu et al. 2024)
- `body_rules_vn.json` — body-type × attribute polarity rules
- `download_images.py` — fetch source images for any subset
- `METHODOLOGY.md` — full pipeline
- `LICENSE` — CC BY 4.0 for labels
- `LICENSE.SOURCES.md` — upstream image licenses

## Citation

```bibtex
@misc{eavton_labels_v1_2026,
  title={EA-VTON Upper-Body Garment Attribute Labels: A 207k Multi-Source
         CLIP-Derived Catalog for Population-Aware Style Recommendation},
  author={EA-VTON Team},
  year={2026},
  publisher={GitHub},
  url={https://github.com/InugamiDev/EA-VTON}
}
```

Also cite the upstream sources for any images you fetch.

## Limitations

- Labels are CLIP zero-shot; no human gold standard included
- DeepFashion2 train split only
- Single dominant color per item
- See `METHODOLOGY.md` for details
