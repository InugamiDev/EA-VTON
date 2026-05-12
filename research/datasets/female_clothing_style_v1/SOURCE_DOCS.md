# Source Docs Packet

Last checked: 2026-05-08.

This packet records the source documentation needed for the `female_clothing_style_v1` collection target:

- Target: 10,000 consenting adult participants x 10 images each = 100,000 redacted person images.
- Use case: clothing style recommendation, outfit classification, garment attribute prediction, and fit/flattery ranking.
- Hard privacy rule: do not crawl arbitrary people from social media, dating apps, schools, workplaces, forums, image search results, video screenshots, or private communities.
- Allowed collection paths: direct participant upload with explicit consent, public/research datasets whose terms allow the intended ML use, and product-only metadata/images after rights review.

## Target Decision

The 10,000 x 10 target should be treated as a first-party participant collection. Public fashion datasets are useful for garment parsing, taxonomy pretraining, try-on research, and product/review graph bootstrapping, but they do not honestly satisfy the same-person target unless they pass all of these gates:

- Person-level grouping exists and can be audited.
- Each accepted person has at least 10 usable images.
- The female label provenance is known; first-party data must be self-identified, while external rows stay `gender_label.source=dataset_label`.
- The license permits the actual training purpose.
- Faces are redacted and metadata is stripped before labeling/training.
- Splits are person-level, with no same-person leakage across train/val/test/holdout.

Current local approved/downloaded strict subset is far below target: `manifest_deepfashion_with_masks_balanced_10_20.jsonl` has 589 people and 8,194 records. The remaining target gap is 9,411 people and 91,806 images.

## Primary Collection Path

| Source | Role | Why |
|---|---|---|
| Direct participant intake | Primary target source | Only path that reliably provides consent, self-identified female labels, and exactly 10 images per same participant. |
| DeepFashion-derived local imports | External pretraining/pilot source | Useful for garment/style parsing experiments, but labels are dataset labels and do not count as self-identified participant data. |
| Product/review datasets | Sidecar recommendation source | Useful for item metadata, size/color preferences, ratings, reviews, and co-purchase graphs. User-posted review images must not enter the people-image manifest without a separate rights/privacy review. |

## Source Matrix

| Source | Current documented scale | Useful for | Target decision | Docs |
|---|---:|---|---|---|
| Direct participant intake | 100,000 planned rows | Final training corpus, preference/fit data, self-identified labels | Required for the 10,000 x 10 target | `README.md`, `schema.json`, `collection_policy.md` |
| DeepFashion-MultiModal | 44,096 high-resolution human images, 12,701 full-body images | Human parsing, keypoints, DensePose, clothing shape/fabric/color labels, captions | Research/pretraining only; non-commercial terms and no first-party consent | https://github.com/yumingj/DeepFashion-MultiModal |
| DeepFashion MultiModal Parts2Whole | About 41,500 reference-target pairs | Same-person pose pairs, multimodal human-generation research | Metadata audit found 498 WOMEN-labeled loose pids with 10+ images and 0 strict same-outfit groups; not enough for target | https://huggingface.co/datasets/huanngzh/DeepFashion-MultiModal-Parts2Whole |
| DeepFashion2 | 491K images, 801K clothing items, 13 categories, 873K commercial-consumer pairs | Garment detection, segmentation, landmarks, consumer-to-shop retrieval | Good garment source; not a 10-images-per-person preference corpus | https://github.com/switchablenorms/DeepFashion2 |
| DeepFashion with masks | Local audit: 35,646 WOMEN-labeled rows, 6,694 pids, 589 pids with 10+ images | Local segmentation/classification pilot | Imported subset is useful for pilots; not enough for target and gender source is dataset label | https://huggingface.co/datasets/SaffalPoosh/deepFashion-with-masks |
| Fashionpedia | 48,825 clothing images, 27 main categories, 19 apparel parts, 294 attributes | Attribute taxonomy, instance segmentation, garment attribute classifiers | Use annotations/ontology after terms review; images follow source platform terms | https://fashionpedia.github.io/home/ |
| ModaNet | Polygon annotations over 13 fashion categories | Segmentation benchmark and category mapping | Research/benchmark source only; archived repo and image-source terms need review | https://github.com/eBay/modanet |
| VITON-HD | 11,647 train and 2,032 test frontal-view woman/top-clothing pairs | Upper-body virtual try-on and paired garment/person tests | Non-commercial try-on research source; not a style preference corpus | https://github.com/shadow2496/VITON-HD |
| Dress Code | 53,792 garments and 107,584 images across upper body, lower body, dresses | Multi-category virtual try-on, keypoints, skeletons, human parsing, DensePose | Access-gated; not released to private companies; not a target collection source | https://github.com/aimagelab/dress-code |
| Amazon Reviews 2023 | 571.54M reviews total; Amazon Fashion has 2.5M ratings; Clothing Shoes and Jewelry has 66.0M ratings | Recommendation interactions, product metadata, product images, item graph | Product/review sidecar only. Strip reviewer IDs/names; exclude user-posted images from people manifest | https://amazon-reviews-2023.github.io/ |
| Amazon Reviews 2018 | 233.1M reviews total; Amazon Fashion has 883,636 reviews; Clothing Shoes and Jewelry has 32,292,099 reviews | Reproducible older recommendation benchmark | Prefer 2023 for new rec work unless reproducing older baselines | https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html |

## Fetch And Crawl Rules

Use fixed source registries and source-specific importers. Do not write a general web crawler for person photos.

1. For participant data, generate a roster with:
   `python3 research/datasets/scripts/create_female_clothing_intake_roster.py --people 10000 --images-per-person 10`
2. Collect images only through the guarded intake flow or another consented upload channel.
3. Store raw identifiable media only in restricted storage.
4. Redact faces before annotation and model training.
5. Import external datasets only after the `sources.json` entry has a license/status that allows the intended use.
6. Keep user-review images out of the people-image manifest unless a separate rights/privacy review explicitly approves them.
7. Validate every training manifest before training:
   `python3 research/datasets/scripts/validate_female_clothing_style_manifest.py research/datasets/female_clothing_style_v1/manifest_train.jsonl --strict-counts`

## Labeling Plan

Use the existing `schema.json` and `label_taxonomy.json` as the canonical label contract.

- Image-level labels: occasion, formality, aesthetic tags, color mood, and layering.
- Garment labels: family, category, visibility, layer role, bbox when available, fit, silhouette, color, pattern, material, fabric weight, structure, and category-specific attributes.
- Privacy labels: rights basis/status/scope, face redaction status/method, and quality gates.
- Human QA: sample every external import by source, sample every auto-label model by category/aesthetic, and reject records with visible faces after redaction.

## Training Split Rules

- Split by `person_id`, never by image.
- Keep first-party participants and external dataset records traceable by source in metadata outside the training manifest.
- Do not mix first-party self-identified labels with dataset-inferred labels for demographic evaluation.
- Hold out a participant-level set before auto-label model tuning to avoid leakage through repeated outfits or repeated identities.

## Practical Dataset Build

Phase 1 creates usable pretraining data without pretending the target is complete:

- Use local DeepFashion-derived redacted records for garment/style classifier pilots.
- Use Fashionpedia/ModaNet/DeepFashion2 annotations for garment taxonomy and segmentation pretraining after terms review.
- Use Amazon Reviews 2023 metadata and reviews for item-side embeddings and collaborative filtering features, excluding reviewer identifiers from model features unless a privacy review approves a pseudonymous interaction graph.

Phase 2 reaches the actual target:

- Recruit 10,000 consenting adult participants.
- Collect 10 images per participant according to the slot guide in `README.md`.
- Redact, label, validate, and split at person level.
- Train the style recommender on redacted images plus non-identifying labels and item metadata.
