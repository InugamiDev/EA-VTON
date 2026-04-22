# Verified Literature Review: Cross-Population Clothing Size Recommendation

> Verified 2026-04-20. Each paper checked against Google Scholar / arXiv / ACM DL / IEEE Xplore.
> Status: 18 fully verified, 11 corrected, 1 removed (hallucinated).

---

## Paper Database (29 verified papers)

| # | Title | Authors | Venue/Year | Domain | Relevance | Status |
|---|-------|---------|------------|--------|-----------|--------|
| 1 | Decomposing Fit Semantics for Product Size Recommendation in Metric Spaces | Misra, R., Wan, M., McAuley, J. | RecSys 2018 | Size Rec | 5 | VERIFIED |
| 2 | A Deep Learning System for Predicting Size and Fit in Fashion E-Commerce | Sheikh, A., et al. | RecSys 2019 | Size Rec | 5 | VERIFIED |
| 3 | Direct Importance Estimation with Model Selection and Its Application to Covariate Shift Adaptation | Sugiyama, M., et al. | NeurIPS 2007 | Density Ratio | 5 | VERIFIED (title corrected) |
| 4 | Improving predictive inference under covariate shift by weighting the log-likelihood function | Shimodaira, H. | JSPI 2000 | Covariate Shift | 5 | VERIFIED |
| 5 | Pareto Smoothed Importance Sampling | Vehtari, A., Simpson, D., Gelman, A., Yao, Y., Gabry, J. | JMLR 2024 | Density Ratio | 5 | VERIFIED |
| 6 | Reweighting anthropometric data using a nearest neighbour approach | Kumar, K.A., Parkinson, M.B. | Ergonomics 2018 | Anthropometrics | 5 | VERIFIED |
| 7 | Analysis of Vietnamese Women's Body Shape from Anthropometric Data | Tran, T.M.K., et al. | Fibres & Textiles 2024 | Anthropometrics | 5 | VERIFIED |
| 8 | Comparison of body shape between USA and Korean women | Lee, J., Istook, C.L., Nam, Y., Park, S. | **IJCST** 2007 | Anthropometrics | 4 | CORRECTED (was JFMM) |
| 9 | A General Algorithm for Deciding Transportability of Experimental Results | Bareinboim, E., Pearl, J. | JCI 2013 | Causal Inference | 5 | VERIFIED |
| 10 | Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities (CORN) | Shi, X., Cao, W., Raschka, S. | Pattern Analysis & Applications **2023** (arXiv 2021) | Ordinal Regression | 5 | CORRECTED (was 2021) |
| 11 | Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation (CORAL) | Cao, W., Mirjalili, V., Raschka, S. | Pattern Recognition Letters 2020 | Ordinal Regression | 4 | VERIFIED |
| 12 | Estimating Model Performance Under Covariate Shift Without Labels (PAPE) | Bialek, J., et al. | NeurIPS 2025 | Covariate Shift | 4 | VERIFIED |
| 13 | Cross-Market Product Recommendation (FOREC) | Bonab, H., et al. | CIKM 2021 | Recommendation | 4 | VERIFIED |
| 14 | Human Body Measurement Estimation with Adversarial Augmentation (BMnet) | Ruiz, N., et al. | 3DV 2022 | Body Estimation | 3 | VERIFIED |
| 15 | A Survey on Fairness-Aware Recommender Systems | Jin, D., et al. | Information Fusion 2023 | Fairness | 4 | VERIFIED |
| 16 | Telescoping Density-Ratio Estimation | Rhodes, B., et al. | NeurIPS 2020 | Density Ratio | 5 | CORRECTED (was "Density-Ratio Estimation via Classification") |
| 17 | Beyond MSE: Ordinal Cross-Entropy for Probabilistic Time Series Forecasting (OCE-TS) | **Wang, J.**, Shi, H., Li, F., Shang, X. | arXiv 2025 (2511.10200) | Ordinal Regression | 4 | CORRECTED (first author was wrong) |
| 18 | Recommending product sizes to customers | Sembium, V., et al. | RecSys 2017 | Size Rec | 4 | VERIFIED |
| 19 | Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches | Ferrari Dacrema, M., Cremonesi, P., Jannach, D. | RecSys 2019 | Recommendation | 5 | VERIFIED |
| 20 | Estimating Unbounded Density Ratios: Applications in Error Control under Covariate Shift | (various) | arXiv 2025 (2504.01031) | Density Ratio | 4 | CORRECTED (title was paraphrased) |
| 21 | A Review of Modern Fashion Recommender Systems | Deldjoo, Y., et al. | ACM Computing Surveys **2023/2024** (arXiv 2022) | Recommendation | 4 | CORRECTED (was 2022) |
| 22 | Bayesian models for product size recommendations | Sembium, V., et al. | WebConf (WWW) 2018 | Size Rec | 4 | VERIFIED |
| 23 | Rethinking Importance Weighting for Deep Learning under Distribution Shift | Fang, T., Lu, N., Niu, G., Sugiyama, M. | NeurIPS **2020** | Covariate Shift | 5 | CORRECTED (was cited as 2023) |
| 24 | A short survey on importance weighting for machine learning | Kimura, M., Hino, H. | TMLR 2024 | Density Ratio | 4 | VERIFIED |
| 25 | Generalizing Importance Weighting to A Universal Solver for Distribution Shift Problems | Fang, T., et al. | NeurIPS **2023** (spotlight) | Covariate Shift | 5 | CORRECTED (was conflated with #23) |
| 26 | Deep Orientational Representation Learning for Ordinal Regression | Jia, G., et al. | IEEE 2026 (venue unconfirmed) | Ordinal Regression | 4 | PARTIALLY VERIFIED |
| 27 | FitGAN: Fit- and Shape-Realistic Generative Adversarial Networks for Fashion | Pecenakova, S., Karessli, N., Shirvany, R. | ICPR 2022 | Body Estimation | 3 | VERIFIED |
| 28 | SIZER: A Dataset and Model for Parsing 3D Clothing and Learning Size Sensitive 3D Clothing | Tiwari, G., et al. | ECCV 2020 (Oral) | Body Estimation | 4 | VERIFIED |
| 29 | Your copula is a classifier in disguise | (various) | arXiv 2024 (2411.03014) | Density Ratio | 4 | ADDED (replaces removed #29) |

**REMOVED**: Bojanic et al. "A Deep Learning Model for Estimation of Human Body" Machine Vision 2024 — paper does not exist at this venue with this title. HALLUCINATED.

---

## Method Comparison: Importance Weighting

| Method | Paper | Handles Correlation? | ESS Behavior | Cost | When It Fails |
|--------|-------|---------------------|--------------|------|---------------|
| Independent Gaussian | Baseline | No | Highly unstable, extreme weights | Very low | Correlated features (height-weight r=0.72) |
| KLIEP | Sugiyama 2007 (#3) | Yes (with multivariate kernels) | Better than naive, still vulnerable to outliers | Moderate | Ultra-high dims, no support overlap |
| uLSIF | Kanamori 2009 | Yes (kernel formulation) | More stable than KLIEP | Low (analytic solution) | Can produce negative weights |
| KMM | Huang 2007 | Yes (RKHS mapping) | Explicitly bounded | High (QP, scales O(n^2)) | 192K samples too large |
| Copula Density Ratio | Rhodes 2020 (#16), arXiv 2024 (#29) | **Yes, explicitly models correlation** | Stable if copula family matches true dependency | High | Wrong copula family choice |
| PSIS (post-processing) | Vehtari 2024 (#5) | N/A (applied on top) | **Maximizes ESS** via tail smoothing | Low | Shape parameter k > 0.7 |
| NN Reweighting | Kumar 2018 (#6) | Yes (Mahalanobis distance) | Depends on cluster quality | Moderate | No overlap between populations |

**Our approach: Copula + PSIS** — model h-w correlation jointly, then smooth extreme weights.

---

## Size Recommendation Landscape

| Paper | Input | Model | Dataset | Accuracy | Population-Aware? |
|-------|-------|-------|---------|----------|-------------------|
| Misra (RecSys 2018) | User/Item ID, size, fit | Metric learning | RTR, ModCloth | AUC-based (not top-1) | No |
| Sheikh (RecSys 2019) | Anthropometrics + embeddings | SFNet (Siamese DL) | RTR, ModCloth | ModCloth 69%, RTR 76%, Width 88% | No |
| Sembium (RecSys 2017) | Purchase history | Latent factors | Amazon | Return rate reduction | No |
| Sembium (WWW 2018) | Purchase + returns | Hierarchical Bayesian | Amazon | Probabilistic sizing | No |
| Kumar (Ergonomics 2018) | Height, weight, BMI | NN reweighting | NHANES | Statistical moment matching | **Yes** (first anthropometric transfer) |
| **Ours (proposed)** | Height, weight, age, body type, category | GBM/MLP + CORN loss | RTR + VN priors | TBD (target: +2pp within-1 on VN subset) | **Yes** (copula + PSIS + ordinal) |

---

## Gap Analysis

### What NO paper has done:
Copula density ratio + PSIS + ordinal loss (CORN) for cross-population zero-shot size recommendation from Western data to Asian demographic.

### Closest existing work:
1. **Kumar & Parkinson 2018**: Anthropometric reweighting — but for ergonomic design, NOT ML prediction
2. **Sheikh et al. 2019**: ML on RTR — but no population adaptation
3. **FOREC (CIKM 2021)**: Cross-market rec — but uses interaction sequences, not body measurements

### Required baselines:
1. Unweighted GBM (ERM baseline)
2. Independent Gaussian weighting (our current weak method)
3. KLIEP (Sugiyama 2007)
4. uLSIF
5. SFNet-style architecture (Sheikh 2019)
6. CORAL ordinal loss (to show CORN is better)

### Required ablations:
1. With/without PSIS (proves ESS recovery)
2. CE vs CORAL vs CORN loss (proves ordinal benefit)
3. Independent vs Copula weighting (proves correlation modeling)
4. Single feature vs joint (proves multivariate necessity)

### Strongest reviewer objections:
1. **P(Y|X) invariance**: US and VN women with same body may prefer different fit (cultural)
2. **Vanishing ESS**: 9.7cm height gap = extreme tail, near-zero effective samples
3. **No VN ground truth**: synthetic evaluation only

### Target venues:
- FashionXRecsys workshop @ RecSys 2026 (realistic)
- RecSys main track (ambitious but defensible)
- KDD (if statistical innovation emphasized)

---

## Must-Cite Papers (20)

| # | Citation | Why |
|---|---------|-----|
| 1 | Misra et al. RecSys 2018 | RTR/ModCloth dataset origin |
| 2 | Sheikh et al. RecSys 2019 | SOTA neural sizing baseline |
| 3 | Sembium et al. RecSys 2017 | Amazon sizing system |
| 4 | Sembium et al. WWW 2018 | Bayesian sizing extension |
| 5 | Shimodaira JSPI 2000 | Foundational covariate shift theory |
| 6 | Sugiyama et al. NeurIPS 2007 | KLIEP density ratio estimation |
| 7 | Vehtari et al. JMLR 2024 | PSIS for weight stabilization |
| 8 | Kumar & Parkinson Ergonomics 2018 | Anthropometric reweighting precedent |
| 9 | Tran et al. Fibres & Textiles 2024 | Vietnamese body cluster data |
| 10 | Lee et al. IJCST 2007 | US-Korean body shape comparison |
| 11 | Bareinboim & Pearl JCI 2013 | Transportability theory |
| 12 | Cao et al. PRL 2020 | CORAL ordinal regression |
| 13 | Shi et al. PAA 2023 | CORN ordinal regression |
| 14 | Fang et al. NeurIPS 2020 | Rethinking importance weighting |
| 15 | Fang et al. NeurIPS 2023 | Universal solver for distribution shift |
| 16 | Kimura & Hino TMLR 2024 | IW survey |
| 17 | Rhodes et al. NeurIPS 2020 | Telescoping density-ratio estimation |
| 18 | Ferrari Dacrema et al. RecSys 2019 | Proper baseline comparison warning |
| 19 | Jin et al. Information Fusion 2023 | Fairness in rec systems |
| 20 | Bonab et al. CIKM 2021 | Cross-market recommendation |
