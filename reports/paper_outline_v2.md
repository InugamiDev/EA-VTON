# Paper Outline v2 — NeurIPS D&B Pivot

**Status:** v2 — pivot from RecSys industry track to **NeurIPS 2027 Datasets & Benchmarks** (CORE A*)
**Date:** 2026-05-18 | **Supersedes:** `paper_outline_v1.md`

---

## Why v2 supersedes v1

Two findings forced the pivot:

1. **The copula+PSIS+α=0.75 reweighting gives +0.6 pp exact over uniform on the female-upper subset (ablation table below).** Reviewers at any A*-tier *methods* venue will reject this as marginal. The headline of v1 was the wrong headline.
2. **A* main-track probability for v1 was 3–5%, even with deferred work** (per the venue agent's analysis). The realistic A* path is *not* a methods paper.

What survives the pivot — and becomes the centerpiece — is **the dataset**:
- 208,069 upper-body items across 6 sources, unified to a single schema
- Face-redaction (MediaPipe FaceLandmarker oval, not bbox) applied **prior to label derivation** — no other major fashion benchmark does this
- Person-categorization labels: 5-cluster VN body suitability, 12 Sci\ART seasons, 5 style personalities, CLIP-ViT-B-32 embeddings
- Multi-population body-rule extension (added in v2; see §6)
- CC BY 4.0 labels + per-source license inheritance documented

The literature scan (12 targeted searches, 5 paper-level fetches) found **zero** major fashion benchmarks that face-redact prior to label derivation. That is the uncontested gap.

---

## 1. Working title

> **EA-VTON-UB: A Privacy-First, Multi-Population Benchmark for Anthropometry-Aware Upper-Body Fashion Recommendation**

Working acronym: **PRISM-UB** (Privacy-first, Redacted, Inclusive, Size-and-style Multimodal — Upper Body). Title and acronym are placeholder; the artifact decisions are what matter.

---

## 2. Target Venue and Why

**Primary**: NeurIPS 2027 Datasets and Benchmarks Track (CORE A*).
**Expected deadline**: ~mid-May 2027 (NeurIPS 2026 D&B abstract was 2026-05-04 — already passed).
**Track acceptance rate**: ~28% recent ([NeurIPS D&B 2024](https://neurips.cc/Conferences/2024/CallForDatasetsBenchmarks)).
**Why this venue, not a methods venue:**
- D&B reviewers evaluate the *dataset artifact* and its *benchmarks*, not algorithmic novelty.
- Our novelty (privacy-first redaction methodology + multi-population body-rule extension) is artifact novelty, which is exactly what they reward.
- We sidestep the "incremental methods" trap that would tank a SIGIR/KDD main-track submission.

**Parallel target**: RecSys 2026 industry track (CORE A) for the recsys angle — keep it as a faster publishable record.

**Cushion target**: ICLR 2027 papers-with-dataset-contribution + SIGIR 2027 Resource Paper track (CORE A*). Submit to whichever closes first.

---

## 3. The Three Contribution Claims (sharpened)

1. **A 208,069-item upper-body fashion catalog with redaction-first methodology.** All published attributes, embeddings, and labels are derived from face-mesh oval-redacted images. The release stance — *no downstream artifact is derived from a non-redacted image* — is documented and reproducible via `download_images.py` + `relabel_from_redacted.py`. No prior fashion benchmark (DeepFashion, DeepFashion2, Fashionpedia, FashionGen, Polyvore) makes this guarantee.

2. **Person-categorization label layer with multi-population body rules.** We provide (a) 5-cluster Vietnamese body suitability scores from peer-reviewed anthropometric clusters (Trieu et al. 2024, *Vlákna a Textil*), (b) 12 Sci\ART seasonal palette scores per item (CIEDE2000 to anchor LABs), (c) 5 style personality labels (CLIP zero-shot), (d) 512-d OpenCLIP ViT-B-32 embeddings, (e) **new in v2: at least one additional population's body-rule set (e.g., US or KR)** sourced from a peer-reviewed anthropometric study, demonstrating the schema is not VN-specific.

3. **Three reusable benchmarks on the catalog**, each with public splits, baseline numbers, and an automatic evaluator:
   - **B1 — Size prediction** with a copula-tempered reweighted baseline (the size GBM; reported honestly with +0.6 pp gain over uniform).
   - **B2 — Body-shape-aware item ranking** with NDCG@K and item-level explanation DAGs.
   - **B3 — Cold-start preference calibration** with Kendall's τ and leave-one-out recall@K.

   These are designed so a non-fashion researcher can use the dataset for new methods.

---

## 4. The Ablation (now honestly reported)

Full 6-row ablation on the female-upper-body subset of RTR (n_train=8,750, n_test=2,188):

| Variant | Tempering | Exact | Within-1 | MAE | VN-Exact | VN-W1 | k̂ |
|---|---|---:|---:|---:|---:|---:|---:|
| uniform | — | 0.516 | 0.698 | 1.004 | 0.497 | 0.734 | — |
| independence | α=1.0 | 0.517 | 0.696 | 1.014 | 0.521 | 0.722 | — |
| copula | α=1.0 | **0.523** | 0.705 | 0.998 | **0.533** | 0.734 | — |
| copula+PSIS | α=1.0 | 0.516 | 0.700 | 1.014 | 0.497 | 0.722 | −0.09 |
| **copula+PSIS α=0.75** (default) | α=0.75 | 0.522 | **0.707** | **0.990** | 0.509 | **0.740** | −0.09 |
| copula+PSIS α=0.50 | α=0.50 | 0.518 | 0.699 | 1.011 | 0.497 | 0.710 | −0.09 |

Honest interpretation:
- PSIS k̂ = −0.09 → the importance-ratio tails are not heavy; PSIS is essentially a no-op here.
- The chosen α=0.75 variant wins on Within-1, MAE, VN-Within-1, but **loses to plain copula on full Exact and VN-Exact**.
- This is reported as a **negative finding** for the reweighting: "for the specific RTR → VN-anthropometric transfer, principled covariate-shift correction yields marginal gains; the within-1 improvement matters in practice but the gain is small." That framing turns a weak result into useful evidence for the community.

The ablation gives B1 a *protocol* and an honest baseline — it does not have to give us a state-of-the-art number.

---

## 5. Multi-Population Body Rules (new in v2 — gap-closer)

To address the "VN-only is too narrow" reviewer objection, v2 extends the schema with at least one additional peer-reviewed population. Candidates:

| Population | Source | Cluster method | Status |
|---|---|---|---|
| Vietnamese female | Trieu et al. 2024, *Vlákna a Textil* | 5-cluster k-means on 480 3D scans | ✅ already in v1 |
| US female | Lee et al. 2007 / CAESAR (US sub-sample) | 4-cluster shape factor analysis | ⏳ to add |
| Korean female | Size-Korea (Kim et al. 2010 / 2020) | 5-cluster body-type proportions | ⏳ candidate |
| Indian female | Pandey et al. 2018 (sample n≈300) | 4-cluster anthropometric typology | ⏳ candidate |

Adding 1 extra population is enough to defuse the "narrow scope" objection. Adding 2 is enough to claim "multi-population schema."

---

## 6. The Three Benchmarks (specifications go in §7 of the paper)

### B1 — Size Prediction (`size_pred_v1`)
- **Task**: predict size ∈ {XXS, XS, S, M, L, XL, XXL} from (height_cm, weight_kg, BMI, age, body_type_cluster, garment_category).
- **Train / val / test**: 80/10/10 stratified by size, fixed seed 42.
- **Metrics**: Exact, Within-1, MAE, per-cluster recall.
- **Baselines**: rule-based, uniform GBM, copula+PSIS α=0.75 GBM (this paper's default).
- **Evaluator**: `benchmarks/size_pred/evaluate.py`.

### B2 — Body-shape-aware Item Ranking (`fit_flatter_match_v1`)
- **Task**: given (user body cluster, predicted size, occasion), rank a 500-item candidate pool.
- **Ground truth**: silver labels from per-cluster body rules + user-pick simulations.
- **Metrics**: NDCG@10, MRR, coverage of seasonal palettes in top-10.
- **Baselines**: random, color-only (CIEDE2000), shape-only (Flatter), and the FFM combination.
- **Evaluator**: `benchmarks/ffm_rank/evaluate.py`.

### B3 — Cold-Start Calibration (`coldstart_calibration_v1`)
- **Task**: given 3-5 liked items (out of 18 stratified seeds), re-rank a 500-item pool to maximize recall of held-out liked items.
- **Metrics**: leave-one-out Recall@10, Kendall's τ vs. base ranking.
- **Baselines**: random seed selection + random centroid, DPP seed selection + centroid (per Nguyen UAI 2024), our SQL-stratified seed + centroid.
- **Evaluator**: `benchmarks/coldstart_cal/evaluate.py`.

Pre-empts the Nguyen UAI 2024 reviewer objection by including DPP as a baseline.

---

## 7. Section Map (~12 pages D&B template)

| § | Title | Length | Headlines |
|---|---|---|---|
| 1 | Motivation | 1.0 p | Underserved populations + privacy-first datasets are gaps; we close both. |
| 2 | Related Datasets | 1.0 p | DeepFashion(2), Fashionpedia, FashionGen, Polyvore; explicit redaction-policy comparison table. |
| 3 | Dataset Composition | 2.0 p | Sources, sizes, licenses, per-source statistics, schema diagram. |
| 4 | Redaction Methodology | 1.5 p | MediaPipe FaceLandmarker FACE_OVAL polygon + dilate + feather; miss-rate analysis; downstream label re-derivation. |
| 5 | Person-Categorization Labels | 1.5 p | VN clusters (Trieu); 12-season CIEDE2000; CLIP zero-shot; multi-population extension. |
| 6 | Benchmark Suite | 2.0 p | B1/B2/B3 task definitions, splits, metrics, baseline numbers. |
| 7 | Datasheet (Gebru et al. template) | 1.0 p | Motivation, composition, collection, preprocessing, uses, distribution, maintenance. |
| 8 | Limitations & Ethics | 0.5 p | Female-only scope; MediaPipe miss-rate ~3%; size-shaming risk; biases. |
| 9 | Conclusion | 0.5 p | Reusable artifact + invitation to extend. |
| App | Reproducibility | 1.0 p | Hashes, requirements, runtime, hardware. |

---

## 8. Reviewer Red-Flag Pre-empts (from gap analysis)

1. **"CMCAN already does body-shape-aware fashion."** → §2 explicitly contrasts: CMCAN is population-agnostic SMPL-based; we are population-specific with peer-reviewed anthropometric basis. CMCAN does not address size transfer at all.

2. **"Nguyen et al. UAI 2024 already does cold-start with diverse seeds."** → B3 includes Nguyen-style DPP as a baseline; we claim *operational simplicity* (one SQL query) at minimal accuracy cost, not method novelty.

3. **"Your VN-subset is RTR users with h<160, w<55 — not real VN women."** → §8 calls this out as a sanity proxy; B1 reports it as such, not as a primary result.

4. **"You're just running an off-the-shelf face redactor."** → §4 reframes: the contribution is the *redaction-first release stance* + the demonstrated downstream-task feasibility, not the redactor itself.

5. **"Why CC BY 4.0 on labels when source images can't be redistributed?"** → §3 documents the per-source license matrix; `download_images.py` is the way reproducible reuse works.

6. **"FFM is hand-engineered, no learning."** → §6 (B2) frames FFM as a *baseline* on a benchmark, not the central claim. The dataset enables a learned alternative; we don't claim to be that alternative.

---

## 9. Probability Estimates (updated honestly)

| Path | Probability of A* accept | Notes |
|---|---|---|
| v1 outline → SIGIR/KDD/WWW main track | 3–5% | "no methodological novelty" |
| v2 outline → NeurIPS 2027 D&B | **~25–35%** if multi-population + 3 benchmarks + datasheet land; up to ~45% if a non-VN user study lands too |
| v2 outline → ICLR 2027 dataset-track | 15–25% | Slightly stricter than D&B on ML novelty |
| v2 outline → SIGIR 2027 Resource Paper | 20–30% | Good fit; tighter scope |
| v1 fallback → RecSys 2026 industry (A) | 35–50% | Submit in parallel for the publishable record |

Best honest read: **~30–40% A*** if v2 lands the multi-population extension, the datasheet, all three benchmark numbers, and explicitly pre-empts CMCAN + Nguyen. The user's 70% expectation is reachable only with one additional asset: a small-N user study (N≥30 VN women) showing the cold-start calibration improves perceived fit. We should plan that study.

---

## 10. Concrete Next Actions

1. **Add 1 non-VN body-rule set** (US or KR) — done in 1–2 days. Sources via WebSearch + peer-reviewed PDFs.
2. **Write the 3 benchmark protocol files** in `benchmarks/{size_pred,ffm_rank,coldstart_cal}/` — done in 2–3 days.
3. **Run B3 calibration evaluation now** (Kendall's τ + leave-one-out recall, with DPP baseline). 1 day.
4. **Update dataset card** to full Gebru et al. template with biases + anti-uses. 1 day.
5. **Draft §1 + §2** (motivation + related datasets table with redaction-policy column). 4 hours.
6. **Plan and schedule the user study** (N=30 VN women, 5 min each = ~3 hours of subject time). 1 week to recruit + run.
7. **Submit to RecSys 2026 industry track** with v1 scope in parallel — locks in the publishable record. Target deadline window: check.

If all six land cleanly, A* probability rises to ~40–50%. Reaching the user's 70% target requires the user study to also return a statistically-significant improvement.

---

*This file supersedes v1; the v1 file is kept for reference and to compare what changed.*
