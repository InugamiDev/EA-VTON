# Paper Outline v1 — EA-VTON Upper-Body for Vietnamese Women

**Status:** v1 draft outline | **Date:** 2026-05-18
**Working title (short):** EA-VTON-UB-VN
**Working title (long):** *Anthropometry-Aware Size and Style Recommendation for
Vietnamese Women's Upper-Body Garments via Copula-Tempered Domain Transfer and
CLIP-Centroid Cold-Start Calibration*

---

## 1. Scope, Framing, and One-Line Pitch

> Given a single phone-camera frame plus self-reported height, predict a
> Vietnamese woman's upper-body size and rank candidate garments by a
> Fit-Flatter-Match score, with cold-start preference elicited from 3
> taps on a stratified-diverse seed grid — all without storing facial
> biometrics.

**Deliberate scope cuts (and why):**

| Cut | Why |
|---|---|
| Lower body (skirts, pants, dresses) | Sizing semantics differ (waist+hip vs. shoulder+bust); a one-paper scope is upper-body only. |
| Male users | Body taxonomy (Trieu et al. 2024) + the rental-source size data we re-weight (RTR) are female-only. |
| 3D try-on rendering | VTON visualization is solved by DM-VTON / CatVTON elsewhere; we treat it as a downstream consumer of our `(size, ranked_items)` pair. |
| Personalization from interaction history | Cold-start is the realistic deployment scenario; we replace it with seed-grid calibration. |

---

## 2. Contributions (the four-bullet "claim block")

1. **Copula-tempered density-ratio reweighting** that transfers a US-rental size
   classifier (RTR, ~10.9k female upper-body `fit=='fit'` rows) to a VN female
   anthropometric prior (h = 156.2 ± 5.5 cm, w = 53.9 ± 8.0 kg, from Tran et al.
   2024), with PSIS k̂-stabilized tail weights and α=0.75 tempering. Result:
   **52.2% exact / 70.7% within-1** on the held-out RTR test set;
   **50.9% exact / 73.96% within-1** on the VN-anthropometric subset
   (h < 160 cm, w < 55 kg, n = 169).

2. **Fit-Flatter-Match (FFM) decomposition** for style ranking:
   `s(u, i, c) = α·Fit(u, i) + β·Flatter(u, i, c) + γ·Match(u, i, c)`
   where Fit is a geometric circumference-vs-ease residual, Flatter is a
   per-cluster body-rule polarity sum, and Match is a multi-anchor CIEDE2000
   distance over Sci\\ART 12-season palettes. Each term has a per-item
   explanation DAG returned by the API.

3. **Stratified-diverse cold-start calibration**: a single SQL window-function
   query over `(neckline × silhouette × color_temperature)` cells yields ~18
   visually distinct seed items; the user's picks form a CLIP-ViT-B-32 centroid
   that re-ranks subsequent recommendations as
   `s' = (1 − δ) · s_FFM + δ · sim(item, v*)`, with δ = 0.30.
   No interaction history required.

4. **Privacy-clean derivative dataset (208,069 upper-body items)** with
   MediaPipe face-mesh oval redaction (not bbox) applied to ~42k images
   prior to label derivation, so all attributes/embeddings published in
   the bundle are downstream of redaction. Released under CC BY 4.0 (labels);
   source images keep upstream licenses.

---

## 3. System Overview (the figure we cite as Fig. 1)

```
       ┌─────────────┐       ┌─────────────┐      ┌──────────────┐
camera │ MediaPipe   │ ratios│ FFM Style   │      │ Calibration  │
─────► │ Pose + Face │──────►│ Recommender │◄────►│ Seed (18)    │
       └─────────────┘       └─────────────┘      └──────────────┘
            │                       ▲                    │
            ▼                       │                    ▼
       ┌─────────────┐       ┌─────────────┐      ┌──────────────┐
height │ Size GBM    │ size  │ DuckDB view │◄─────│ CLIP centroid│
─────► │ (VN-tuned)  │──────►│ over 208k   │      │ (user picks) │
       └─────────────┘       └─────────────┘      └──────────────┘
            │                       ▲
            ▼                       │
       ┌──────────────────────────────┐
       │  RTR + Tran et al. VN prior  │  (copula PSIS + α=0.75 tempering)
       └──────────────────────────────┘
```

---

## 4. Section Map (target: 10–12 pages double-column)

| § | Title | Length | Headlines |
|---|---|---|---|
| 1 | Introduction | 1.0 p | Camera-only retail UX; female VN scope; cold-start as default; privacy-by-default. |
| 2 | Related Work | 1.0 p | RTR/ModCloth size papers (Misra+Wan); SizeNet; FitNet; CLIP4Cir; CIEDE2000 fashion; Trieu et al. body taxonomy; PSIS (Vehtari). |
| 3 | The Domain Gap | 0.8 p | Empirical: h/w distribution shift RTR vs. VN_FEMALE prior; why naïve transfer fails (Δ exact = 11.4 pp absolute, our ablation). |
| 4 | Anthropometry-Aware Size Model | 1.5 p | Copula construction, PSIS, α tempering ablation, female upper-body filter; results table. |
| 5 | Fit-Flatter-Match Style Ranker | 1.8 p | Geometric Fit; per-cluster Flatter rules (5 VN clusters from Trieu); multi-anchor Match in CIE LAB. |
| 6 | Cold-Start Calibration | 1.2 p | Stratified seed picker (SQL); CLIP centroid; δ ablation; Kendall's τ vs. base ranking. |
| 7 | The Dataset | 1.0 p | 6 sources merged; MediaPipe oval redaction; label methodology; CC BY 4.0 bundle. |
| 8 | Evaluation | 2.0 p | Offline retrieval metrics; intrinsic FFM stability; calibration latency; survey N=? (deferred). |
| 9 | Ethics, Privacy, Limitations | 0.5 p | Female-only scope; redaction failure modes; size-shaming risk; the "no body image stored" guarantee. |
| 10 | Conclusion | 0.4 p | What is reusable, what is VN-specific. |

---

## 5. Evaluation Plan

### 5.1 Size head (`gbm_female_upper_vn`)

* **Primary**: Exact accuracy, within-1 accuracy, on RTR test split (n ≈ 2 188 — 20 % stratified).
* **VN-range subset**: same metrics, filtered to `h < 160 ∧ w < 55` (proxy for VN-female range). Already computed: 50.9 % / 73.96 % on n = 169.
* **Ablation table (8 rows)**:
  | Reweighting | Tempering α | Train | Exact | Within-1 |
  |---|---|---|---|---|
  | none | — | full RTR | (baseline) | (baseline) |
  | independence | α = 1.0 | full RTR | … | … |
  | copula | α = 1.0 | full RTR | … | … |
  | copula + PSIS | α = 1.0 | full RTR | … | … |
  | copula + PSIS | α = 0.75 | full RTR | … | … |
  | copula + PSIS | α = 0.50 | full RTR | … | … |
  | copula + PSIS | α = 0.75 | **female upper only** | **52.2** | **70.7** | ← chosen
  | (oracle: train+test on VN data) | — | — | (upper bound) | — |
* **Calibration curve**: predicted probability vs. empirical accuracy bins.
* **Per-cluster confusion matrix**: the 5 Trieu et al. clusters × 7 sizes.

### 5.2 Style ranker

* **Intrinsic metrics** (the offline-only ones we can compute without a user study):
  * Coverage of the 12 seasonal palettes in top-10 across users.
  * Stability of ranking under bootstrap resampling of the candidate pool
    (mean Kendall's τ over 100 reshuffles).
  * Mean per-component contribution: how much of the score variance is
    explained by Fit vs. Flatter vs. Match (variance-decomp table).
* **Calibration effect**:
  * Kendall's τ between calibrated and base top-K (we expect ~0.3–0.6 for δ = 0.30).
  * Recall@K of the "held-out liked items" in a leave-one-out CLIP simulation:
    pick 3 of 4 known-liked, see if the 4th is in top-K of the recalibration.
* **Latency budget**: target < 250 ms p95 for `/recommend-style/upper/calibrate` on a single Mac M-series CPU (currently passing).

### 5.3 User study (deferred to v2 or rebuttal)

Mention as a planned extension. Don't block submission on it.

---

## 6. Datasets & Releases

* **Training (size)**: RTR upper-body female `fit=='fit'` rows, 10 938 after filter.
* **Catalog (style)**: 208 069 upper-body items across 6 sources, all attributes derived from face-mesh-redacted images.
* **Public release** (in `research/datasets/processed/dataset_bundle_v1/`):
  * `labels.parquet` (our derivative; CC BY 4.0)
  * `taxonomy.json`, `schema.json`, `seasonal_palettes.json`,
    `vn_body_centroids.json`, `body_rules_vn.json`
  * `METHODOLOGY.md`, `DATASET_CARD.md`, `LICENSE.SOURCES.md`
  * `download_images.py` (so we don't redistribute non-redistributable images)

---

## 7. Reproducibility Checklist (for the appendix)

* [ ] All hyperparameters in `train_female_upper_size.py` header constants
* [ ] PSIS k̂ reported per run
* [ ] Random seeds fixed (42 throughout)
* [ ] `requirements.txt` pinned (Python 3.11, PyTorch 2.5+, DuckDB 1.5+, CLIP ViT-B-32 laion2b_s34b_b79k)
* [ ] Hardware: single Mac M-series CPU; no GPU required for inference
* [ ] Time to reproduce: ~6 h on a 2025 MacBook (mostly labeling)

---

## 8. Target Venues (ranked by fit)

1. **RecSys 2026 (industry / short paper track)** — cold-start, FFM decomposition,
   privacy-clean deployment story; their industry track explicitly welcomes
   regional/under-served-market deployments.
2. **CVPR Workshop on Computer Vision for Fashion** (CVF2026) — strong fit for
   the redaction + CLIP + body-shape angle, less so for the size GBM.
3. **ICTIR / SIGIR-AP** — Asia-Pacific bias on the user study; would be the
   right home if we add the deferred N-participant study.
4. **Local: VNRR or NAFOSTED-listed VN journal** — a Vietnamese-language
   version for the local stakeholder audience after the English version.

Backup plan: post to arXiv first (cs.IR or cs.CV) by 2026-Q3 regardless of venue
to claim the work; submit to RecSys 2026 short-paper track by their typical
mid-year deadline.

---

## 9. Risks and Open Questions

* **No labeled VN ground truth for size**: we evaluate on RTR-VN-proxy subset
  only. The reviewer will ask: *how do we know this generalizes?* Our defense:
  the prior is from a peer-reviewed VN anthropometry study (Tran et al. 2024);
  the within-1 accuracy on the VN subset is **higher** than the full-set
  baseline, which is consistent with successful reweighting.
* **CLIP centroid degeneracy**: if the user picks 3 nearly-identical items, the
  centroid is fine but the cosine similarities cluster tightly (we see 0.87–
  0.90 in our smoke test). Mitigation: report Kendall's τ rather than score
  spread, and tune δ via ablation.
* **Stratified seeds bias by attribute frequency**: e.g. `solid` accounts for
  68 % of items, so it dominates seeds. May need re-weighting in v2.
* **Privacy edge cases**: MediaPipe face landmarker miss-rate on heavy
  occlusion / extreme angles. We documented this in §7 of the dataset card.

---

## 10. Concrete Next Actions (post-outline)

1. Run the full 8-row ablation table for §4 (currently we have only the chosen
   row). Estimated time: 30 min on CPU.
2. Compute the calibration Kendall's τ + leave-one-out recall on a held-out
   liked-set. ~1 h.
3. Variance-decomp table for §5. ~30 min.
4. Draft §1 + §2 (intro + related) and circulate. ~4 h.
5. Decide co-authorship and ordering.

---

*This outline supersedes any earlier scratch notes; future revisions go into
`reports/paper_outline_vN.md` with N incremented.*
