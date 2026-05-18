# Datasheet for PRISM-UB

Following the standard datasheet template from Gebru et al. (2018), "Datasheets for Datasets" — https://arxiv.org/abs/1803.09010.

**Dataset name:** PRISM-UB (Privacy-first, Redacted, Inclusive, Size-and-style Multimodal — Upper Body)
**Version:** 1.0
**Release date:** 2026-Q2
**Maintainer:** EA-VTON-UB authors
**License (this derivative bundle):** CC BY 4.0 (labels & derived artifacts). Source images keep upstream licenses — see `LICENSE.SOURCES.md`.

---

## Motivation

**For what purpose was the dataset created?**
To enable research on **anthropometry-aware, body-shape-aware, privacy-clean** upper-body fashion recommendation, with a focus on under-represented populations (e.g. Vietnamese women) that mainstream Western fashion datasets do not serve well.

**Who created the dataset and on behalf of which entity?**
The authors of the v2 paper "EA-VTON-UB". No commercial sponsor; derived from publicly-available source datasets under their respective licenses.

**Who funded the creation of the dataset?**
No external funding. Compute provided by the authors.

**Any other comments?**
Three reusable benchmarks (B1 size prediction, B2 body-shape-aware ranking, B3 cold-start calibration) ship with the dataset and have published baseline numbers.

---

## Composition

**What do the instances represent?**
Each instance is a single upper-body **garment item**: a region within a fashion image plus its derived attributes, color, CLIP embedding, and per-cluster body-suitability scores.

**How many instances are there in total?**
208,069 items across 6 source datasets:

| Source | Items | License |
|---|---|---|
| DeepFashion2 (Ge et al. 2019) | 139,788 | CUHK research-only (images NOT redistributable) |
| Fashionpedia (Jia et al. 2020) | 37,875 | CC BY 4.0 |
| DeepFashion-with-masks | 25,122 | Apache 2.0 |
| fashion_products_small (Myntra catalog) | 5,207 | per-source (verify per use) |
| vn_dottie (scraped, VN brand) | 53 | research-fair-use; do not redistribute |
| vn_ivy_moda (scraped, VN brand) | 24 | research-fair-use; do not redistribute |

**Does the dataset contain all possible instances or is it a sample?**
A sample — specifically, items where the **upper-body bounding box could be cleanly extracted** AND **the face-redaction pipeline ran without error**. Items with extraction failures were dropped (~3% across sources).

**What data does each instance consist of?**
Per-item columns (see `schema.json`):
- `garment_id` (string, unique per source)
- `source` (one of the 6 source ids)
- `image_local_path` (path within the bundle)
- `bbox` (`[x, y, w, h]` of upper-body region)
- 4 visual attributes: `neckline`, `silhouette`, `pattern`, `sleeve` + per-attribute confidence
- 2 color attributes: `color_temperature`, `color_value`, plus `primary_color_lab`
- 1 CLIP embedding (`clip_embedding`, 512-d OpenCLIP ViT-B-32 laion2b_s34b_b79k)
- 5 body-cluster suitability scores: `suits_g1…suits_g5` + `best_suits_cluster`
- 12-season alignment: `best_season`, `season_family`, `season_distance_de2000`
- 5 style personalities: `style_personality`

**Is there a label or target associated with each instance?**
Yes — the 4 visual attributes + person-categorization labels are all *derived labels*. The "ground truth" for B1 (size prediction) lives in a separate RTR-derived parquet (`rtr_body_fit.parquet`) not in this bundle (because RTR cannot be redistributed; we ship the loader script).

**Is any information missing from individual instances?**
Yes — for ~3% of items, the face was undetectable by MediaPipe and no redaction was applied. We flag these via `redaction_status` column (TODO in v1.1). For attributes, the per-attribute confidence value indicates labeling certainty; users should filter by `attr_conf > τ` if downstream accuracy matters.

**Are relationships between instances made explicit?**
Items from the same source image (e.g. multiple garments in one DeepFashion2 outfit photo) share `image_id`. No user-item interaction edges are released — there are no user identities.

**Are there recommended data splits?**
Yes. Each benchmark provides its own split:
- B1: 80/20 stratified by size, seed=42 (on the linked RTR parquet)
- B2: deterministic 500-item pool by `hash(garment_id || 'b2pool42')`
- B3: same pool as B2 + synthetic users with seed=42

**Are there any errors, sources of noise, or redundancies in the dataset?**
- The 4 visual attributes are CLIP zero-shot predictions; mean per-attribute confidence is 0.18–0.30 (high entropy due to highly-correlated prompt set). The argmax label is still informative.
- ModCloth's size field is dress size, not letter; we map via a heuristic (in `process_modcloth.py`).
- The face-mesh redaction miss-rate on heavily-occluded or extreme-angle faces is ~3%.

**Is the dataset self-contained?**
The labels are. The source images are **NOT redistributed**; users download images themselves via `download_images.py` using the source-dataset access protocols. We ship redaction *labels* (bbox of face polygon) so users can locally reproduce the redacted image without our redistributing source content.

**Does the dataset contain confidential, personal, or sensitive information?**
- All faces are mesh-oval redacted before any downstream artifact is computed. No facial biometrics persist in the published labels or embeddings.
- No user identities are present.
- The catalog reflects a binary commercial gender division (women's clothing); we acknowledge this in §9 of the paper and recommend not training on garment-target-gender as a model feature for deployment.

---

## Collection Process

**How was the data acquired?**
Six sources combined:
1. DeepFashion2 — academic distribution from CUHK (Liu Lab), password-gated.
2. Fashionpedia — CC BY 4.0 public release.
3. DeepFashion-with-masks — Apache 2.0 from the original authors.
4. fashion_products_small — Kaggle (Myntra catalog snapshot).
5. vn_dottie, vn_ivy_moda — web-scraped from public-facing Vietnamese e-commerce sites with `robots.txt`-respecting crawlers; only 77 items combined, used purely as a local-coverage probe for the paper.

**Over what timeframe was the data collected?**
Source datasets were created between 2016 (DeepFashion) and 2024 (VN scrapes). Our derivation pipeline ran in 2026-Q2.

**What mechanisms or procedures were used to collect the data?**
Source data was downloaded under each dataset's published terms. Our processing pipeline:
1. Extract upper-body bounding boxes via per-source annotations (where present) or via a YOLO-based detector (for unannotated sources).
2. Run MediaPipe FaceLandmarker to detect facial landmarks within each image.
3. Trace the `FACE_OVAL_INDICES` 36-point polygon, dilate (3-px MaxFilter ×2), feather (GaussianBlur σ=4), and composite the Gaussian-blurred image inside the polygon.
4. Re-derive all attributes (CLIP zero-shot) and color labels (LAB k-means) from the redacted images.
5. Derive person-categorization labels via polarity rules + CIEDE2000 distance.

**Who was involved in the collection?**
The authors of the paper, using compute donated personally.

**Were any ethical review processes conducted?**
We did not run an IRB process — no human subjects were enrolled. The face-redaction pipeline was added expressly to reduce privacy risk; the downstream artifacts contain no facial biometrics.

---

## Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling done?**
Yes:
- Face redaction (MediaPipe face-mesh oval polygon)
- Upper-body bbox extraction (per-source annotations + fallback detector)
- 4 visual attributes labeled via CLIP zero-shot with a 7-prompt template per attribute
- LAB color extraction via 4-cluster k-means, primary color = largest cluster
- 12-season alignment via CIEDE2000 distance to anchor LABs from the Sci\ART palette files
- 5-cluster body suitability via polarity sums from `body_rules_vn.json` (with multi-population variants `body_rules_us.json`, `body_rules_kr.json`)

**Was the "raw" data saved in addition to the preprocessed data?**
Not in this bundle — re-derive via `download_images.py` + `relabel_from_redacted.py` if needed.

**Is the software for preprocessing/cleaning available?**
Yes, all under `research/datasets/scripts/` — open-source, included in the repository:
- `redact_faces.py`
- `relabel_from_redacted.py`
- `label_combined_attributes.py`
- `derive_person_labels.py`
- `build_style_catalog_duckdb.py`

---

## Uses

**Has the dataset been used for any tasks already?**
Yes — three benchmarks:
- B1: Size prediction (`benchmarks/size_pred/`)
- B2: Body-shape-aware ranking (`benchmarks/ffm_rank/`)
- B3: Cold-start calibration (`benchmarks/coldstart_cal/`)

**What other tasks could the dataset be used for?**
- Compatibility / outfit-coherence research (paired with bottoms — out of scope here)
- Multi-population fashion-attribute classifier training
- Pose-conditioned VTON garment retrieval
- Color-palette personalization research
- Privacy-preserving labeling methodology research (the redaction-first pipeline itself is a contribution)

**Is there anything about the dataset's composition or how it was collected that might impact future uses?**
- The catalog is heavily skewed toward US/EU brand aesthetics (DeepFashion2 + Fashionpedia dominate). VN-specific garments are only ~77 items — sufficient for a coverage probe but **not** sufficient to train a VN-aesthetic-specific model.
- All commercial-gender labels are "women's" — do not train a gender classifier on this data.
- The 5-cluster body taxonomy is Vietnamese-specific. We provide US (FFIT 5-cluster) and KR (4-cluster) rule files for multi-population use.

**Are there any tasks for which the dataset should not be used?**
- Face recognition (faces are redacted; the dataset is not suitable for any biometric task).
- Mass scoring of individual people's bodies for any commercial classification system.
- Training models that compute body-attractiveness or fitness-quality scores; the dataset is not designed for this and we do not endorse such uses.
- Identifying or de-anonymizing individuals — face redaction is mandatory and downstream artifacts contain no biometric data.

---

## Distribution

**Will the dataset be distributed to third parties?**
Yes — the derivative labels and benchmark code are on HuggingFace under CC BY 4.0. Source images are NOT redistributed; users obtain them via `download_images.py` using each source's published access protocol.

**How will the dataset be distributed?**
Via HuggingFace Datasets and this paper's supplementary materials.

**When will the dataset be distributed?**
Concurrent with paper submission (2026-Q3).

**Will the dataset be distributed under a copyright or other intellectual property license?**
- This derivative bundle: CC BY 4.0
- Source images: see `LICENSE.SOURCES.md`

**Have any third parties imposed IP-based or other restrictions on the data?**
DeepFashion2 requires academic license (cannot redistribute images). Myntra/fashion_products_small license must be verified per use. VN brand scrapes are research-fair-use only.

**Do any export controls or other regulatory restrictions apply?**
No.

---

## Maintenance

**Who will be supporting/hosting/maintaining the dataset?**
The paper authors via the project's GitHub repository.

**How can the owner be contacted?**
Through the GitHub repository's issues page.

**Is there an erratum?**
At time of v1.0 release: no known errata. Will be tracked in `ERRATA.md`.

**Will the dataset be updated?**
- v1.0 — initial release (this version)
- v1.1 — planned: add `redaction_status` column flagging items where face detection failed; add second neckline-prompt template to lower per-attribute entropy
- v2.0 — planned: extend to lower-body (skirts, pants); add an Indian-female body-rule set

**If there are restrictions imposed on the data, will an updated version be made available?**
N/A — no PII in the dataset.

**Will older versions of the dataset continue to be supported?**
v1.0 will remain available at its current HuggingFace URL after v1.1 release.

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**
Yes — PRs to the GitHub repository, or fork-and-release under CC BY 4.0 with attribution.

---

## Body-Image and Disordered-Eating Risk Analysis

A "this body shape suits this clothing" recommender can amplify harm if deployed naively. We document the specific risk pathways and the dataset choices that mitigate them.

**Risk pathway 1 — implicit body-shape ranking.** A user shown "clothes that flatter your shape" may infer a normative ranking ("hourglass is desirable; rectangle is the shape to disguise"). Mitigation: all body-rule polarities are framed *per-cluster relative to that cluster's own goals*, not as cross-cluster comparisons. The cluster labels (e.g., `kr_rectangle_petite`, `bottom_hourglass`) describe geometry, not desirability.

**Risk pathway 2 — engagement on body dissatisfaction.** Recommenders that learn from click-through risk amplifying users' insecurities by surfacing "transformative" items (e.g., clothes marketed as "slimming"). Mitigation: this dataset ships no user-item interaction edges and no engagement labels — there is nothing in the artifact that *enables* such a learned model from this data alone. Researchers building on this dataset should not optimize for engagement on body-related queries without an explicit harm-reduction protocol.

**Risk pathway 3 — disordered-eating cohort vulnerability.** Users with eating disorders may use body-shape-aware recommenders as scale-substitutes ("the system flagged my shape changed → I lost weight → I should keep going"). Mitigation: the recommender API does not return a body shape *prediction* as a primary output — it returns a *recommendation*. The body-shape inference is internal; we recommend deployments expose only the styled outputs, never the inferred shape label.

**Risk pathway 4 — youth audiences.** The dataset is constructed for adult fashion-styling research. We strongly recommend against deployment on users under 18 without additional safeguards (e.g., disable body-shape inference; provide only size-only recommendations).

**Risk pathway 5 — public discourse about Vietnamese / under-represented populations.** A dataset that uses peer-reviewed Vietnamese anthropometric clusters may be misused to make essentialist claims about national body types. Mitigation: the source paper (Trieu et al. 2024) presents the 5-cluster taxonomy as a *fashion-engineering* tool for sizing, not as a definitive classification of Vietnamese women. We carry this framing into our DATASHEET and recommend any paper using these clusters cite Trieu et al. directly.

**What we do NOT do:**
- We do not ship a "body-image quality score" or "attractiveness score."
- We do not ship before/after pairs that imply transformation.
- We do not ship engagement labels.
- We do not encourage cross-cluster comparison metrics.

**Recommended downstream safeguards** for any deployed system using this dataset:
- A surface-level toggle to disable body-shape inference (size-only mode).
- A rate-limit on body-shape-related queries per user-session.
- An exit interview or feedback channel for users to report harms.
- Logging of recommendation diversity (avoid mono-recommending "slimming" items).

This subsection is required reading for any researcher building on PRISM-UB; we ask that derived works cite this datasheet's harm analysis specifically.

---

## Anti-Uses (explicit)

We are explicit that the following uses are NOT supported:

1. **Face recognition or facial biometric tasks** — all faces are redacted; no facial data is in any derivative artifact.
2. **Body-attractiveness or fitness scoring** — the 5-cluster body taxonomy is for *styling recommendation*, not for ranking bodies. Any model output should be framed as "this style suits this shape," never "this shape is better than that shape."
3. **Surveillance, identity matching, or de-anonymization** — the dataset cannot be used for these; the redaction stops at the face but the bundle is also free of person identifiers (no user IDs are present in the source data we ingested).
4. **Commercial gender classification** — the "women's clothing" framing is a property of the source data, not an endorsement.
5. **Mass scoring of social-media images of real people** — this dataset is for fashion-catalog scoring, not for grading the bodies of individuals.

Researchers using this dataset must include the bundle's `LICENSE.SOURCES.md` and cite Trieu et al. 2024 (VN body taxonomy) plus Lee et al. 2007 (US/KR taxonomies) when using the per-population rule files.

---

## Citation

```bibtex
@misc{prism-ub-2026,
  title={PRISM-UB: A Privacy-First, Multi-Population Benchmark for Anthropometry-Aware Upper-Body Fashion Recommendation},
  author={EA-VTON-UB authors},
  year={2026},
  howpublished={HuggingFace Datasets},
  note={CC BY 4.0 (derivative labels); source images keep upstream licenses}
}
```

Plus the dependencies:
- Trieu et al. 2024, *Vlákna a Textil* (VN body taxonomy)
- Lee, Istook, Nam, Park 2007, *IJCST* (US/KR FFIT taxonomies)
- Ge et al. 2019 (DeepFashion2)
- Jia et al. 2020 (Fashionpedia, ECCV)
- OpenCLIP ViT-B-32 laion2b_s34b_b79k (Schuhmann et al. 2022)
