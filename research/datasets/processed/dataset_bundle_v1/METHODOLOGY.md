# Methodology — How These Labels Were Generated

## Source Datasets

| Source | Items | Filtering |
|---|---|---|
| DeepFashion2 (train split) | 139,788 | Category 1-6 (upper body): short/long sleeve top, outwear, vest, sling |
| Fashionpedia | 37,875 | Category 0-5, 9 (shirt/top/sweater/cardigan/jacket/vest/coat) |
| DeepFashion-with-masks (HF) | 25,122 | gender=WOMEN AND cloth_type ∈ {Tees_Tanks, Blouses_Shirts, Sweaters, Jackets_Coats, Sweatshirts_Hoodies, Cardigans, Graphic_Tees, Shirts_Polos, Jackets_Vests, Suiting} |
| fashion_products_small (Myntra HF) | 5,207 | gender ∈ {Women, Unisex} AND subCategory=Topwear AND articleType ∈ upper-body set |
| **Total** | **207,992** | |

## Pipeline

```
source image → bbox/mask crop → CLIP image encode → softmax against prompts → label
                              ↓
                       per-pixel LAB → K-means → dominant color → temperature/value bin
```

### Step 1 — Crop
- **DeepFashion2**: bounding box from annotation JSON
- **DeepFashion-with-masks**: tight bbox computed from binary garment mask (cleaner than bbox)
- **Fashionpedia**: bounding box from detection annotation
- **fashion_products_small**: whole image (catalog-style flat-lay/ghost-mannequin)

### Step 2 — Visual Attributes (zero-shot CLIP)
- Model: OpenCLIP **ViT-B-32**, weights **laion2b_s34b_b79k**
- Each attribute dimension has 4-8 natural-language prompts
- argmax over softmax similarities → label
- Confidence is softmax probability (uniform-low for highly correlated prompts; argmax label still informative)

### Step 3 — Color
- 96×96 resized RGB → CIE LAB (D65) via standard matrix transform
- K-means (k=3) on LAB pixels
- Skip near-white background cluster (L>92 ∧ chroma<6) when present
- Largest non-background cluster centroid → `primary_color_lab`
- Derived bins:
  - `color_temperature` = "warm" if hue ∈ (-45°, 100°) else "cool"; "neutral" when chroma < 8
  - `color_value` = "dark" if L<35; "medium" if L<70; "light" otherwise

### Step 4 — Category-derived (no CLIP)
- `sleeve` for DeepFashion2: derived from category_id (1→short, 2→long, 5/6→sleeveless)
- `sleeve` for DeepFashion-with-masks and Fashionpedia: CLIP zero-shot
- `sleeve` for fashion_products_small: derived from `articleType` (Tshirts→short, Shirts→long, …)

## CLIP Embeddings
Each row carries a 512-dim **L2-normalized** CLIP image embedding. Use for:
- Visual similarity search (cosine ≈ dot product on normalized vectors)
- Two-tower retrieval models
- Cross-modal CLIP-text retrieval (encode query with same model)

## Hardware + Throughput
- M2 MacBook (MPS), no dedicated GPU
- DeepFashion2: 41.7 items/s end-to-end
- DeepFashion-with-masks: 40.0 items/s
- Fashionpedia: 32.9 items/s
- fashion_products_small: 53.8 items/s
- Total labeling time: ~78 minutes

## Limitations
- **Zero-shot confidence is uniform-low** (~0.12-0.21 mean) because our prompt set is highly correlated within each dim. Argmax label is still informative but absolute conf scores should not be treated as probability calibration.
- **No human-validated ground truth**. We have not measured per-attribute accuracy on a held-out human-labeled set.
- **Pattern bias**: solid pattern is over-represented (~46%) consistent with real catalog distributions.
- **Color**: dominant-color extraction can miss multicolor / heavily patterned garments (single LAB centroid).
- **DeepFashion2 image_id collisions across val/test/train**: this release uses train split only.
