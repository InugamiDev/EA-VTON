# Female Clothing Style Dataset v1

## Scope

This dataset is for a women-focused clothing style recommendation system. It is not limited to tops: each image can contain a full outfit with multiple labeled garments.

- Target cohort: consenting adult participants who self-identify as female. Licensed external datasets may use `gender_label.source=dataset_label`; do not treat those rows as self-identification or person-preference ground truth.
- Target size: 10,000 participants x 10 images each = 100,000 redacted images.
- Clothing scope: tops, bottoms, dresses/jumpsuits, outerwear, footwear, and visible accessories.
- Training scope: style recommendation, outfit classification, garment attribute prediction, and fit/flattery ranking.
- Privacy rule: training and annotation manifests must point to face-redacted images. Raw identifiable images stay in restricted storage and are not used by model training.

See `SOURCE_DOCS.md` for the fetched source documentation matrix, crawl/fetch rules, target gap, and recommended acquisition plan. See `SELECTED_SOURCES.md` and `selected_sources.json` for the current selected build inputs.

## Collection Rules

Use only consented or properly licensed data. Do not scrape private social media, dating apps, school/work pages, or any source where neither explicit subject consent nor dataset-level ML training rights are available.

Each participant should contribute 10 images covering style variation:

| Slot | Requirement |
|------|-------------|
| 01 | casual everyday outfit |
| 02 | work or smart-casual outfit |
| 03 | dress or one-piece outfit if the participant wears one |
| 04 | layered outfit with outerwear or cardigan |
| 05 | fitted or semi-fitted outfit |
| 06 | relaxed or oversized outfit |
| 07 | dark/neutral color outfit |
| 08 | light/colorful/patterned outfit |
| 09 | side or three-quarter view |
| 10 | participant-selected favorite outfit |

If a participant does not wear a specific category, keep the slot but label the visible garments accurately. Do not force dresses, heels, skirts, or other gender stereotypes.

## Required Processing Order

1. Ingest raw images into restricted storage.
2. Create a raw intake manifest with local `source_image_uri` and labeled `face_boxes`.
3. Run face redaction with `research/datasets/scripts/redact_faces_from_manifest.py`.
4. Label only redacted images.
5. Validate the training manifest with `research/datasets/scripts/validate_female_clothing_style_manifest.py`.
6. Train models only from redacted image paths and non-identifying labels.

## Local Intake Flow

The app now exposes a guarded local intake route for first-party participant batches:

- Frontend: `http://localhost:3000/intake`
- API status: `GET /api/intake/status`
- API submission: `POST /api/intake/participant`

The submission endpoint accepts `10-20` JPEG/PNG/WebP images for one consenting participant, requires explicit training and internal-research consent, and writes raw records to restricted local storage under `research/datasets/restricted/female_clothing_style_v1/intake`. That directory is gitignored and is not served from `/uploads`.

Submitted records are raw intake records only. They are not training records until face boxes are labeled, `redact_faces_from_manifest.py` writes redacted images, and the resulting manifest passes validation.

## Label Structure

Each image has image-level outfit labels and a `garments` array. Label every visible primary garment, not only the most prominent one.

Per-image labels:

- `outfit.occasion`: `casual`, `work`, `date`, `formal`, `party`, `athleisure`, `lounge`, `outdoor`, `travel`, `unknown`
- `outfit.formality`: `very_casual`, `casual`, `smart_casual`, `business`, `formal`, `unknown`
- `outfit.aesthetic_tags`: multi-label, see `label_taxonomy.json`
- `outfit.color_mood`: `monochrome`, `neutral`, `warm`, `cool`, `colorful`, `high_contrast`, `unknown`
- `outfit.layering`: `single_layer`, `light_layering`, `heavy_layering`, `unknown`

Per-garment labels:

- `family`: `top`, `bottom`, `dress_jumpsuit`, `outerwear`, `footwear`, `accessory`
- `category`: family-compatible class from `label_taxonomy.json`
- `visibility`: `primary`, `secondary`, `partial`
- `layer_role`: `base`, `mid`, `outer`, `full_body`, `footwear`, `accessory`
- `attributes.fit`: `bodycon`, `fitted`, `semi_fitted`, `straight`, `relaxed`, `oversized`, `unknown`
- `attributes.silhouette`: category-specific shape label such as `a_line`, `wide_leg`, `boxy`, `wrap`, `unknown`
- `attributes.color_family`, `pattern`, `fabric_weight`, `structure`, `material_tags`
- Category-specific optional attributes: neckline, sleeve length, hem length, rise, leg shape, dress length, closure.

Avoid labels that identify the person. Body-shape labels are optional and should be derived from measurements/pose only when consented and needed for fit research.

## Manifest Format

Training manifests are JSONL: one JSON object per redacted image. See `schema.json`.

Minimum example:

```json
{"person_id":"p_000001","image_id":"p_000001_01","split":"train","gender_label":{"value":"female","source":"self_identified"},"age_band":"25_34","redacted_image_uri":"redacted/p_000001_01.jpg","face_redaction":{"status":"redacted","method":"black_box"},"rights":{"basis":"first_party_consent","status":"granted","scope":"style_rec_training"},"garments":[{"garment_id":"g01","family":"top","category":"shirt","visibility":"primary","layer_role":"base","attributes":{"fit":"semi_fitted","silhouette":"straight","color_family":"white","pattern":"solid","fabric_weight":"medium","structure":"structured","material_tags":["cotton"],"neckline":"collared","sleeve_length":"long","hem_length":"regular","closure":"button"}},{"garment_id":"g02","family":"bottom","category":"trousers","visibility":"primary","layer_role":"base","attributes":{"fit":"straight","silhouette":"straight","color_family":"black","pattern":"solid","fabric_weight":"medium","structure":"structured","material_tags":["woven"],"rise":"mid","leg_shape":"straight"}}],"outfit":{"occasion":"work","formality":"business","aesthetic_tags":["minimal","classic"],"color_mood":"neutral","layering":"single_layer"},"quality":{"usable":true,"face_visible_after_redaction":false,"garments_visible":true}}
```

## Validation

```bash
python3 research/datasets/scripts/validate_female_clothing_style_manifest.py research/datasets/female_clothing_style_v1/manifest_train.jsonl --strict-counts
```

Strict counts require 10,000 people and exactly 10 images per person. During pilot collection, omit `--strict-counts`.

## Privacy Gates

- Reject records with `quality.face_visible_after_redaction=true`.
- Reject records missing `face_redaction` or `redacted_image_uri`.
- Reject training manifests containing `source_image_uri`, `raw_image_uri`, names, usernames, profile URLs, or contact info.
- Store consent IDs, source account IDs, or dataset-native person IDs separately from the training manifest if they can identify a person.
- Keep raw images encrypted/restricted and delete them if retention is not explicitly covered by consent or dataset license terms.
