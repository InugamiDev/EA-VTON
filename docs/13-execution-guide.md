# EA-VTON Execution Guide: What To Do, How, and Why

> This document separates RESEARCH (size recommendation) from ENGINEERING (VTON demo).
> Updated 2026-04-20 after exhaustive literature verification.

---

## PART A: SIZE RECOMMENDATION (The Research)

### The Problem Statement

> Given: A user's body info (height, weight, age, body type) + garment category
> Predict: The correct clothing size (XXS through XXL)
> Constraint: Training data is US population (RTR, 192K). Target users are Vietnamese.
> No Vietnamese fit-feedback data exists.

### Why This Is Novel

No paper has combined all three:
1. **Copula-based density ratio** for cross-population transfer (not independent Gaussians)
2. **PSIS** for importance weight stabilization (not hard clipping)
3. **CORN ordinal loss** for size prediction (not standard cross-entropy)

Closest work:
- Kumar & Parkinson 2018 did anthropometric reweighting but for ergonomic design, not ML
- Sheikh et al. 2019 did ML on RTR but assumed single population
- Neither combined density ratio estimation with ordinal regression for sizing

---

### DATA REQUIREMENTS

| Dataset | What It Provides | Download | Size | Status |
|---------|-----------------|----------|------|--------|
| **RTR** | 192K US rental reviews: height, weight, body_type, fit feedback (small/fit/large), garment category, size | [McAuley Lab](https://cseweb.ucsd.edu/~jmcauley/datasets.html) or [Kaggle](https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation) | ~120MB raw | HAVE IT |
| **ModCloth** | 82K reviews, same format as RTR. Second source dataset for generalization validation | Same links as RTR (bundled together) | ~40MB | NEED TO DOWNLOAD |
| **Tran et al. 2024** | 5 Vietnamese female body clusters (height, weight, chest, waist, hip means + std per cluster) | Hardcoded from [paper PDF](http://vat.ft.tul.cz/2024/1/VaT_2024_1_1.pdf) | N/A | HAVE IT |
| **ANSUR II** (optional) | 6K US military, 93 measurements. Validates body estimation formulas | [Penn State](https://www.openlab.psu.edu/ansur2/) | Free | OPTIONAL |

**What we DON'T need for the rec paper:**
- VITON-HD (that's for VTON demo only)
- BodyM (that's for image-based body estimation, not tabular rec)
- Any image data

---

### APPROACH: 4 Steps

#### Step 1: Data Preparation
**What**: Load RTR, filter to fit="fit" rows (~20K), extract features
**Features**: height_cm, weight_kg, age, body_type (encoded), category (encoded)
**Target**: size label (XXS-XXL, 7 ordinal classes)
**Split**: 80/20 stratified by size, seed=42

**Also**: Download and process ModCloth with same pipeline as second test dataset.

**Train?** No. Just data processing.
**Time**: 30 minutes.

#### Step 2: Importance Weighting (Copula + PSIS)
**What**: Compute density ratio w(x) = P(x|Vietnamese) / P(x|US) for each training sample

**Current method (broken)**:
```
w = P(h|VN) / P(h|US) × P(w|VN) / P(w|US) × P(bmi|VN) / P(bmi|US)
```
Independent Gaussians. Ignores h-w correlation (r=0.72). BMI redundant. Result: +0.2pp only.

**Proposed method**:
1. Drop BMI (derived from h and w — redundant feature)
2. Fit marginals: height ~ Normal, weight ~ LogNormal (skewed right)
3. Compute rank correlations from RTR data
4. Build Gaussian copula for joint P(height, weight | US) and P(height, weight | VN)
5. Density ratio = P_VN(h,w) / P_US(h,w) per sample
6. Apply PSIS (Vehtari et al. 2024): fit GPD to tail weights, smooth extremes
7. Normalize to mean=1

**Why Copula over alternatives?**
- KLIEP (Sugiyama 2007): good but sensitive to kernel bandwidth with only 2 features
- uLSIF: can produce negative weights
- KMM: O(n^2), too slow for 192K samples
- Copula: explicitly separates marginals from correlation structure, interpretable, matches the anthropometric domain where we know the marginal shapes

**Why PSIS over hard clipping?**
- Current [0.01, 100] clipping is arbitrary
- PSIS has theoretical guarantees (GPD tail fitting)
- Diagnostic: shape parameter k tells you if the shift is too extreme (k > 0.7 = warning)

**Train?** No. Statistical computation only.
**Time**: 1-2 hours to implement.
**Key output**: importance weights array + ESS diagnostic + k-hat diagnostic

#### Step 3: Size Prediction Model (CORN)
**What**: Train ordinal classification model: body features + importance weights → size

**Baseline models (must implement ALL for comparison)**:
1. GBM + no weighting (ERM baseline)
2. GBM + independent Gaussian weights (current weak method)
3. GBM + copula+PSIS weights
4. MLP + standard CE + no weighting
5. MLP + standard CE + copula+PSIS weights
6. **MLP + CORN loss + copula+PSIS weights** (proposed method)

**Why CORN over standard CE?**
- CE treats all misclassifications equally: predicting XXL when truth is S gets same penalty as predicting M
- CORAL (Cao 2020) fixes this with binary subtasks but has weight-sharing constraint
- CORN (Shi & Raschka 2023) uses conditional probability chain rule WITHOUT weight-sharing — strictly better than CORAL empirically
- Size is inherently ordinal: XXS < XS < S < M < L < XL < XXL

**Why GBM as baseline instead of just MLP?**
- Ferrari Dacrema (RecSys 2019) showed many neural approaches don't beat simple baselines
- GBM is known best-in-class for tabular data with <100 features
- If MLP+CORN+copula can't beat GBM+copula, the ordinal loss isn't contributing

**Architecture (MLP)**:
```
Input (6 features) → Linear(128) + LayerNorm + GELU + Dropout(0.3)
                    → Linear(128) + LayerNorm + GELU + Dropout(0.3)
                    → CORN head (K-1 binary outputs for K=7 classes)
```
Small network — 6 input features don't need depth.

**Train?** YES.
- GBM: < 1 minute per variant
- MLP: ~5 minutes per variant on CPU
- Total: ~30 minutes for all 6 model variants

#### Step 4: Evaluation
**What**: Compare all 6 models across multiple population subsets

**Metrics** (what the community expects):
| Metric | What It Measures | Formula |
|--------|-----------------|---------|
| Top-1 Accuracy | Exact size match | mean(pred == true) |
| Within-1 Accuracy | Off by ≤1 size | mean(|ord(pred) - ord(true)| ≤ 1) |
| MAE (ordinal) | Average size distance | mean(|ord(pred) - ord(true)|) |
| Directional Bias | Over/under-sizing tendency | mean(ord(pred) - ord(true)) |
| ESS | Weight stability diagnostic | (Σw)² / Σ(w²) |
| PSIS k-hat | Tail diagnostic | GPD shape parameter |

**Subsets to evaluate on** (from our GO/NO-GO script):
| Subset | Definition | What It Tests |
|--------|-----------|---------------|
| Full test set | All RTR test rows | Overall accuracy (should not degrade much) |
| VN-range | h<160 AND w<55 | Target population proxy |
| VN Cluster 1 | h<155 AND w<50 | Extreme small bodies |
| VN Cluster 3 | 153<h<160 AND 48<w<56 | Dominant VN body type (35.9%) |
| US-typical | h>163 AND w>58 | Sanity check (should stay similar) |

**Success criteria**:
- Within-1 accuracy on VN-range: **≥ +2pp over unweighted baseline**
- Within-1 accuracy on full test: **≤ 1pp degradation**
- Directional bias on VN-range: **closer to 0** than baseline
- ESS with copula+PSIS: **higher** than with independent Gaussian

**Also run on ModCloth** to prove method generalizes beyond RTR.

**Train?** No. Just inference + metrics.
**Time**: 10 minutes.

---

### ABLATION STUDIES (for paper)

| Ablation | What It Proves |
|----------|---------------|
| Remove PSIS (raw copula weights) | PSIS stabilizes training, recovers ESS |
| Replace copula with independent Gaussian | Correlation modeling matters |
| Replace CORN with standard CE | Ordinal structure matters |
| Replace CORN with CORAL | CORN > CORAL (no weight sharing) |
| Height only vs weight only vs joint | Multivariate necessary |
| Vary VN target parameters ±1σ | Sensitivity to population assumptions |

---

### PAPER STRUCTURE

```
1. Introduction
   - Online fashion return rates (20-40%, 53% due to size)
   - No cross-population sizing exists
   - Contribution summary

2. Related Work
   - Size recommendation (Misra 2018, Sheikh 2019, Sembium 2017/2018)
   - Covariate shift (Shimodaira 2000, Sugiyama 2007, Fang 2020/2023)
   - Anthropometric transfer (Kumar 2018, Tran 2024, Lee 2007)
   - Ordinal regression (CORAL 2020, CORN 2023)

3. Problem Formulation
   - Causal DAG (Bareinboim & Pearl transportability)
   - Covariate shift assumption: P(Y|X) invariant across populations
   - Density ratio estimation objective

4. Method
   4.1 Copula Density Ratio Estimation
   4.2 PSIS Weight Stabilization
   4.3 CORN Ordinal Classification
   4.4 End-to-end pipeline

5. Experiments
   5.1 Datasets (RTR, ModCloth, VN priors)
   5.2 Baselines (6 model variants)
   5.3 Main results (Table: accuracy across population subsets)
   5.4 Ablation studies
   5.5 Sensitivity analysis

6. Discussion
   - Limitations (P(Y|X) assumption, no VN ground truth)
   - Fairness implications
   - Future work (VN user study, SizeKorea as second target)

7. Conclusion
```

**Target venue**: FashionXRecsys @ RecSys 2026 (workshop, 4-6 pages)
**Stretch target**: RecSys 2026 main track (full paper, 10 pages)

---

### TIMELINE

| Week | Task | Output |
|------|------|--------|
| 1 | Implement copula + PSIS weighting | `adapt_rtr_to_vn_v2.py` with copula weights |
| 1 | Download + process ModCloth | `modcloth_body_fit.parquet` |
| 2 | Implement CORN loss + MLP | `train_size_corn.py` |
| 2 | Train all 6 model variants | 6 model checkpoints |
| 2 | Run full evaluation + ablations | Results tables |
| 3 | Write paper draft | LaTeX 4-6 pages |
| 4 | Iterate on results, sensitivity analysis | Final tables + figures |

**Total research effort: ~4 weeks**

---

## PART B: VIRTUAL TRY-ON (The Demo)

### What This Is

Engineering work to make a visual demo. **No research contribution here.** Use pretrained models.

### Options

| Model | Type | Size | Speed on M2 | Quality | Effort |
|-------|------|------|-------------|---------|--------|
| **DM-VTON** | GAN (distilled) | 37MB | Real-time | OK | Low — download + wrap |
| **CatVTON** | Diffusion (SD 1.5) | ~900MB | 3-10 min/img | Good | Medium — OpenVINO setup |
| **Mobile-VTON** | Diffusion (distilled) | ~400MB | ~80s (NPU) | Good | Medium — CVPR 2026, new |

**Recommendation**: Start with DM-VTON for instant demo, add CatVTON for high-quality output.

### What To Do

1. Download pretrained weights
2. Write inference wrapper (Python, ~100 lines)
3. Connect to API gateway (`apps/api/src/routes/tryon.js`)
4. Test with sample images
5. Done

**No training. No novel architecture. No paper contribution.**

### Integration Point

The rec system predicts size → VTON visualizes that size on the user's body.
This is a UX feature, not a research contribution.

```
User uploads photo + enters height
  → Rec system: "You should wear size M"
  → VTON: shows user wearing size M garment (visual confirmation)
```

---

## PART C: ADVISOR NOTES

### What reviewers will ask

1. **"Why not just collect Vietnamese data?"**
   - Answer: Data collection requires thousands of users + garment access + fit feedback infrastructure. Our method works with zero target data. This is the entire point.

2. **"How do you know P(Y|X) is invariant?"**
   - Answer: We acknowledge this limitation explicitly. Our contribution is the covariate shift component. Cultural preference shift requires future work with target population feedback.

3. **"Your Vietnamese evaluation is synthetic — how do you validate?"**
   - Answer: We evaluate on the VN-range subset of RTR (real US users who happen to have Vietnamese-range bodies). This is a proxy, not synthetic. We also report sensitivity analysis varying VN parameters.

4. **"GBM already gets 83% within-1. Why bother?"**
   - Answer: 83% is on US bodies. On VN-range bodies (h<160, w<55) it drops to 70%. That's the gap we're closing.

5. **"Why not use image-based body estimation?"**
   - Answer: Separate contribution. This paper focuses on the population transfer methodology. Image input is future work that our framework supports.

### What makes a strong submission

1. **Clean ablation table** — every component justified by removing it and showing degradation
2. **Multiple subsets** — don't just report overall accuracy, show per-cluster results
3. **ESS diagnostics** — show that copula+PSIS actually recovers ESS vs independent Gaussian
4. **ModCloth replication** — same method, different dataset, similar improvement = strong
5. **Honest limitations** — acknowledge P(Y|X) assumption, lack of VN ground truth

### Common mistakes to avoid

1. Don't oversell — this is a methodological contribution (copula + PSIS + CORN for sizing), not a new model architecture
2. Don't compare against SOTA diffusion VTON — that's a different problem
3. Don't report only overall accuracy — the whole point is subset performance
4. Don't ignore the baseline warning (Ferrari Dacrema 2019) — always include simple GBM
5. Don't claim "Vietnamese size recommendation system" — claim "cross-population transfer framework validated on Vietnamese target"
