# EA-VTON Research Plan

## Paper Title (Working)

**EA-VTON: Ethnicity-Aware Virtual Try-On with Anthropometric Fit Prediction**

*Subtitle: Bridging the Gap Between Visual Try-On and Accurate Size Recommendation for Underrepresented Populations*

---

## 1. Research Problem

Current virtual try-on (VTON) systems produce visually plausible images but **ignore whether the garment actually fits the person**. A user sees "this shirt looks nice on me" but has no idea if they should order size S or M. This disconnect causes:

- **High return rates** (30-40% in online fashion, primarily due to wrong size)
- **Population bias** — models are trained on Western body types; predictions fail for Asian populations where body proportions differ significantly
- **Decoupled systems** — size recommendation and virtual try-on exist as separate, unconnected tools

**Research gap**: No existing work integrates population-specific anthropometric modeling with virtual try-on to produce both *visually accurate* and *size-aware* results.

---

## 2. What We Use vs. What We Build

| Component | Source | Our Contribution |
|-----------|--------|-----------------|
| Image VTON backbone | FASHN v1.5 (Apache 2.0, cited) | None — used as baseline |
| Body measurement estimation | Linear regression on population data | **Novel Vietnamese-calibrated formulas** |
| Population adaptation | Importance-weighted sampling | **Novel density ratio reweighting RTR→Vietnamese** |
| Size-fit prediction | EA-VTON size engine | **Novel ethnicity-aware sizing** |
| Fit-aware VTON rendering | NEW — does not exist | **Novel contribution** |
| Evaluation framework | NEW metrics | **Novel fit-accuracy metrics** |

### Analogy
Like how Stable Diffusion papers cite U-Net and CLIP but contribute the diffusion framework — we cite FASHN VTON but contribute the **anthropometric-aware pipeline** around it.

---

## 3. Novel Contributions (5)

### C1: Vietnamese Anthropometric Model
- Regression formulas calibrated to Vietnamese body clusters (Tran et al. 2024)
- 5 body type clusters with prevalence data
- Demonstrates US model overestimates Vietnamese chest by ~5.8cm

**Status**: DONE (docs/02, ea_size_engine.py)

### C2: Importance-Weighted Population Adaptation
- Reweight RTR (US, 192K rows) to simulate Vietnamese population without collecting new data
- Density ratio estimation via Gaussian PDF ratio
- ESS analysis showing 12.6% effective sample retention

**Status**: DONE (docs/03, ea-vton/models/train_size_rec.py)

### C3: Fit-Aware Virtual Try-On Rendering
- Given: person photo + garment + predicted fit (tight/normal/loose)
- Output: VTON image that **visually reflects the predicted fit**
- Method: post-process FASHN output using garment mask deformation
  - Tight fit → compress garment mask inward, add fabric tension cues
  - Loose fit → expand garment mask outward, add drape cues
  - Normal fit → minimal adjustment

**Status**: NOT BUILT — this is the key missing piece

### C4: Photo-Based Body Estimation (Optional Enhancement)
- Estimate height/weight from a single photo using body landmarks
- MediaPipe pose → segment lengths → height regression
- Eliminates manual measurement input

**Status**: NOT BUILT — nice-to-have, not required for paper

### C5: Fit-Accuracy Evaluation Metrics
- **Size Accuracy (SA)**: does the predicted size match ground truth?
- **Fit Visual Consistency (FVC)**: does the rendered garment visually match the predicted fit level?
- **Population Fairness (PF)**: accuracy gap between US and Vietnamese test subjects
- Standard VTON metrics: FID, SSIM, LPIPS (for visual quality baseline)

**Status**: NOT BUILT — needed for evaluation section

---

## 4. System Architecture (Paper Version)

```
Input: Person Photo (I_p) + Garment Image (I_g) + Body Info (h, w)
                    │                │                    │
                    ▼                ▼                    ▼
           ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐
           │  Person       │  │  Garment    │  │  Anthropometric  │
           │  Preprocessor │  │  Processor  │  │  Module (C1+C2)  │
           └──────┬───────┘  └──────┬──────┘  └────────┬─────────┘
                  │                 │                   │
                  │                 │           ┌───────▼────────┐
                  │                 │           │ Size Predictor │
                  │                 │           │ (S, M, L...)   │
                  │                 │           │ + Fit Level    │
                  │                 │           │ (tight/normal/ │
                  │                 │           │  loose)        │
                  │                 │           └───────┬────────┘
                  ▼                 ▼                   │
           ┌─────────────────────────────┐             │
           │     VTON Backbone           │             │
           │     (FASHN v1.5)            │             │
           │     I_p + I_g → I_vton      │             │
           └──────────┬──────────────────┘             │
                      │                                │
                      ▼                                ▼
           ┌─────────────────────────────────────────────┐
           │          Fit-Aware Renderer (C3)             │
           │                                             │
           │  1. Detect garment mask in I_vton            │
           │  2. Get fit_level from Size Predictor        │
           │  3. Deform garment region:                   │
           │     - tight  → inward compression + tension  │
           │     - loose  → outward expansion + drape     │
           │     - normal → pass-through                  │
           │  4. Composite back onto person               │
           └──────────────┬──────────────────────────────┘
                          │
                          ▼
                  Output: I_result
                  + Size Recommendation
                  + Confidence Score
                  + Fit Visualization
```

---

## 5. Implementation Plan

### Phase 1: Foundation (DONE)
- [x] Anthropometric formulas (US + Vietnamese)
- [x] Importance-weighted sampling pipeline
- [x] Train baseline + EA-VTON size models
- [x] FASHN VTON inference pipeline
- [x] Basic frontend + API
- [x] Documentation (docs 01-09)

### Phase 2: Fit-Aware Rendering (TODO — core paper contribution)

#### 2a. Fit Level Prediction
- Input: estimated body measurements + garment size chart
- Output: fit_level ∈ {very_tight, tight, normal, loose, very_loose}
- Logic:
  ```
  chest_diff = estimated_chest - garment_chest_midpoint
  if chest_diff > +4cm  → very_tight (garment too small)
  if chest_diff > +2cm  → tight
  if |chest_diff| ≤ 2cm → normal
  if chest_diff < -2cm  → loose
  if chest_diff < -4cm  → very_loose
  ```
- File: `backend/services/fit_predictor.py`

#### 2b. Garment Mask Deformation
- Extract garment mask from VTON output (FashnHumanParser)
- Apply fit-dependent deformation:
  - **Tight**: erode mask by N pixels, apply slight inward warp, add subtle fabric stretch lines
  - **Loose**: dilate mask by N pixels, apply slight outward warp, soften edges for drape effect
  - **Normal**: no deformation
- Deformation magnitude proportional to |chest_diff|
- File: `backend/services/pipeline/postprocessing/fit_renderer.py`

#### 2c. Integration
- Wire fit_predictor into pipeline (after Stage 3 inference, before Stage 5 composite)
- Pass fit_level to garment_composite
- Return fit info in API response
- File: modify `backend/services/pipeline/__init__.py`

### Phase 3: Evaluation Framework (TODO)

#### 3a. Build Evaluation Dataset
- Need: person photos with **known body measurements** wearing **known garment sizes**
- Options:
  - BodyM dataset (Amazon) — has measurements but no VTON ground truth
  - Manually annotate a small set (50-100 person-garment pairs with fit labels)
  - Synthetic: generate pairs using FASHN at different "person sizes" and compare
- File: `ea-vton/evaluation/build_eval_set.py`

#### 3b. Implement Metrics
- **Size Accuracy (SA)**: `correct_predictions / total_predictions`
  - Measure separately for US and Vietnamese populations
- **Within-1 Accuracy (W1A)**: prediction within ±1 size of ground truth
- **Fit Visual Consistency (FVC)**:
  - Render same person with tight/normal/loose prediction
  - Compute garment area ratio (tight should have smaller garment area)
  - Automated: `garment_mask_area(tight) < garment_mask_area(normal) < garment_mask_area(loose)`
- **Population Fairness (PF)**: `|SA_us - SA_vn|` (lower = more fair)
- **Standard VTON metrics**: FID, SSIM, LPIPS (for visual quality — should not degrade vs. baseline FASHN)
- File: `ea-vton/evaluation/metrics.py`

#### 3c. Run Experiments
- Experiment 1: Size accuracy — EA-VTON vs. US-only baseline on Vietnamese subjects
- Experiment 2: Visual quality — EA-VTON fit rendering vs. vanilla FASHN (FID, SSIM)
- Experiment 3: Fit consistency — does tight/loose rendering produce visually distinguishable results?
- Experiment 4: Ablation — remove importance weighting, remove fit rendering, measure degradation
- File: `ea-vton/evaluation/run_experiments.py`

### Phase 4: Paper Writing (TODO)

#### Paper Outline
1. **Abstract** — Problem (VTON ignores fit), Method (EA-VTON), Results (X% size accuracy improvement, fit-aware rendering)
2. **Introduction** — Return rate problem, population bias, our contributions
3. **Related Work**
   - Image VTON: VITON, CP-VTON, HR-VITON, IDM-VTON, FASHN
   - Size recommendation: traditional (size charts), ML-based, body estimation
   - Gap: no work combines both
4. **Method**
   - 4.1 Anthropometric Module (C1)
   - 4.2 Population Adaptation via Importance Weighting (C2)
   - 4.3 Fit-Aware Rendering (C3)
   - 4.4 Pipeline Integration
5. **Experiments**
   - 5.1 Datasets and setup
   - 5.2 Size prediction accuracy (Table 1)
   - 5.3 Visual quality (Table 2)
   - 5.4 Fit consistency (Table 3)
   - 5.5 Ablation study (Table 4)
   - 5.6 Population fairness analysis
6. **Discussion** — Limitations (photo-based estimation accuracy, garment deformation realism), future work (video, distillation)
7. **Conclusion**

#### Target Venues
- **Tier 1**: CVPR, ECCV, ICCV (competitive, need strong visual results)
- **Tier 2**: WACV, ACCV, ACM MM (more achievable)
- **Applied**: Fashion & AI workshops at CVPR/ECCV, RecSys
- **Journal**: IEEE TPAMI, Pattern Recognition, TMM

---

## 6. What Needs GPU Compute

| Task | GPU Needed | Time Estimate |
|------|-----------|---------------|
| FASHN inference for eval set | 1x A100 | ~1 hour (1000 pairs) |
| FID computation | 1x any GPU | ~30 min |
| Fit rendering experiments | CPU OK | ~2 hours |
| (Optional) Fine-tune FASHN | 4x A100 | Days-weeks |
| (Optional) Distillation | 4-8x A100 | Weeks |

Core paper does NOT require fine-tuning or distillation. Those are future work.

---

## 7. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fit deformation looks unrealistic | High — kills C3 | Start with subtle adjustments; ablate to find sweet spot |
| Vietnamese eval data insufficient | Medium | Use synthetic approach (importance-weighted RTR) |
| Reviewers say "just engineering" | High | Emphasize population fairness angle and novel metrics |
| FID degrades with fit rendering | Medium | Measure delta; if >5%, reduce deformation strength |
| FASHN quality insufficient | Low | FASHN SoTA on VITON-HD benchmark; quality is adequate |

---

## 8. Timeline

| Week | Milestone |
|------|-----------|
| 1 | Fit predictor (2a) + fit renderer prototype (2b) |
| 2 | Integration into pipeline (2c) + simple frontend for fit visualization |
| 3 | Evaluation dataset construction (3a) + metrics implementation (3b) |
| 4 | Run experiments (3c) + generate tables/figures |
| 5-6 | Paper writing (draft) |
| 7 | Internal review + revisions |
| 8 | Submission |

---

## 9. File Map (What Exists vs. What To Build)

```
research-try-out/
├── docs/
│   ├── 01-system-architecture.md        ✅ exists
│   ├── 02-body-measurement-estimation.md ✅ exists
│   ├── 03-importance-weighted-sampling.md ✅ exists
│   ├── 04-size-recommendation.md        ✅ exists
│   ├── 05-virtual-tryon-pipeline.md     ✅ exists
│   ├── 06-quality-scoring.md            ✅ exists
│   ├── 07-datasets.md                   ✅ exists
│   ├── 08-benchmarks.md                 ✅ exists
│   ├── 09-video-tryon-pipeline.md       ✅ exists (Plan A/B)
│   └── 10-research-plan.md             ✅ THIS FILE
│
├── backend/
│   ├── services/
│   │   ├── ea_size_engine.py            ✅ exists
│   │   ├── fit_predictor.py             🔨 TODO (Phase 2a)
│   │   ├── pipeline/
│   │   │   ├── __init__.py              ✅ exists (needs Phase 2c mod)
│   │   │   └── postprocessing/
│   │   │       ├── garment_composite.py ✅ exists
│   │   │       └── fit_renderer.py      🔨 TODO (Phase 2b)
│   │   └── distillation/               ⏳ future work
│   └── routers/
│       ├── tryon.py                     ✅ exists
│       ├── tryon_simple.py              ✅ exists (just added)
│       └── size_recommendation.py       ✅ exists
│
├── ea-vton/
│   ├── models/
│   │   ├── train_size_rec.py            ✅ exists
│   │   ├── baseline_size_model.pkl      ✅ exists
│   │   └── eavton_size_model.pkl        ✅ exists
│   ├── evaluation/
│   │   ├── build_eval_set.py            🔨 TODO (Phase 3a)
│   │   ├── metrics.py                   🔨 TODO (Phase 3b)
│   │   └── run_experiments.py           🔨 TODO (Phase 3c)
│   └── paper/
│       └── ea-vton-paper.tex            🔨 TODO (Phase 4)
│
└── frontend/
    └── src/app/
        ├── simple/page.tsx              ✅ exists (just added)
        └── try-on/page.tsx              ✅ exists
```

---

## 10. One-Line Summary

**EA-VTON takes a standard VTON backbone (FASHN) and makes it population-aware and fit-aware — so users don't just see "how it looks" but "how it fits."**
