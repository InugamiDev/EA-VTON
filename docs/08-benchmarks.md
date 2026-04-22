# Benchmarks & Evaluation Metrics

## Standard VTON Benchmarks

### VITON-HD

**Dataset**: 13,679 image pairs (person + garment), 1024×768 resolution, upper-body only
**Test set**: 2,032 pairs (paired) + unpaired evaluation
**Population**: Predominantly Western/East Asian models — no Vietnamese representation

### DressCode

**Dataset**: 53,792 image pairs, three categories (upper, lower, dresses)
**Test set**: 5,400 pairs
**Population**: Western fashion models

### Key Gap

Neither benchmark contains Vietnamese subjects or Vietnamese body proportions. EA-VTON's contribution is demonstrating that VTON models trained/evaluated on these benchmarks perform worse on underrepresented populations.

## Evaluation Metrics

### FID (Fréchet Inception Distance)

Measures distributional similarity between generated and real images:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

where $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$ are the mean and covariance of Inception-v3 features for real and generated images.

**Lower is better.** FID = 0 means identical distributions.

> **Source note.** The VITON-HD paired-setting numbers in this section are taken directly from **Table 1 of the FitDiT paper** (Jiang et al., "FitDiT: Advancing the Authentic Garment Details for High-fidelity Virtual Try-on", arXiv:2411.10499) so that all methods are compared under a single, uniform evaluation protocol. Different papers report slightly different values for the same model (e.g. the CatVTON paper reports its own FID as 5.43 paired) because of resolution, test-split, and preprocessing differences. We therefore pin the comparison table to one consistent source.

VITON-HD, **paired** setting (ground truth available):

| Model | SSIM ↑ | LPIPS ↓ | FID ↓ | KID ↓ |
|-------|--------|---------|-------|-------|
| LaDI-VTON | 0.8763 | 0.0911 | 6.6044 | 1.0672 |
| StableVTON | 0.8665 | 0.0835 | 6.8581 | 1.2553 |
| IDM-VTON | 0.8806 | 0.0789 | 6.3381 | 1.3224 |
| OOTDiffusion | 0.8513 | 0.0964 | 6.5186 | 0.8961 |
| CatVTON | 0.8694 | 0.0970 | 6.1394 | 0.9639 |
| **FitDiT** (SOTA in this table) | **0.8985** | **0.0661** | **4.7309** | **0.1895** |

VITON-HD, **unpaired** setting (FID/KID only):

| Model | FID ↓ | KID ↓ |
|-------|-------|-------|
| LaDI-VTON | 9.4095 | 1.6866 |
| StableVTON | 9.5868 | 1.4508 |
| IDM-VTON | 9.6114 | 1.6387 |
| OOTDiffusion | 9.6733 | 1.2061 |
| CatVTON | 9.1434 | 1.2666 |
| **FitDiT** | **8.2042** | **0.3421** |

**FASHN VTON v1.5**: no peer-reviewed numbers yet — the technical report is marked "coming soon" on fashn.ai/research/vton-1-5 at time of writing. We therefore do not list it in the comparison table and instead report our own FASHN evaluation as part of the EA-VTON paper's experiments section.

### SSIM (Structural Similarity Index)

Pixel-level structural comparison between generated and ground truth:

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

**Higher is better.** SSIM = 1.0 means identical.

VITON-HD paired SSIM (source: FitDiT paper Table 1):

| Model | VITON-HD SSIM↑ |
|-------|----------------|
| FitDiT | 0.8985 |
| IDM-VTON | 0.8806 |
| LaDI-VTON | 0.8763 |
| CatVTON | 0.8694 |
| StableVTON | 0.8665 |
| OOTDiffusion | 0.8513 |

### LPIPS (Learned Perceptual Image Patch Similarity)

Perceptual distance using deep features (AlexNet/VGG):

$$\text{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \|w_l \odot (\hat{x}^l_{hw} - \hat{y}^l_{hw})\|_2^2$$

where $\hat{x}^l, \hat{y}^l$ are unit-normalized activations at layer $l$ and $w_l$ are learned weights.

**Lower is better.** LPIPS = 0 means perceptually identical.

VITON-HD paired LPIPS (source: FitDiT paper Table 1):

| Model | VITON-HD LPIPS↓ |
|-------|-----------------|
| FitDiT | 0.0661 |
| IDM-VTON | 0.0789 |
| StableVTON | 0.0835 |
| LaDI-VTON | 0.0911 |
| OOTDiffusion | 0.0964 |
| CatVTON | 0.0970 |

### KID (Kernel Inception Distance)

Unbiased alternative to FID using Maximum Mean Discrepancy:

$$\text{KID} = \text{MMD}^2(\phi(X_r), \phi(X_g))$$

where $\phi$ maps images to Inception-v3 feature space and MMD uses a polynomial kernel.

**Lower is better.** More reliable than FID with small sample sizes.

## EA-VTON-Specific Metrics

Beyond standard VTON metrics, EA-VTON introduces population-aware evaluation:

### Size Recommendation Accuracy

**Top-1 Accuracy**: Exact size match rate

$$\text{Acc@1} = \frac{|\{i : \hat{y}_i = y_i\}|}{N}$$

**Within-1 Accuracy**: Predicted within one size step

$$\text{Acc@W1} = \frac{|\{i : |\text{ord}(\hat{y}_i) - \text{ord}(y_i)| \leq 1\}|}{N}$$

**Population-stratified accuracy**: Accuracy computed separately for:
- Samples in the Vietnamese-typical range ($h \in [148, 164]$, $w \in [42, 66]$)
- Samples in the US-typical range ($h \in [158, 174]$, $w \in [50, 75]$)

This reveals whether the EA-VTON model improves accuracy for the target population.

### Ethnicity Gap Score

$$\text{EGS} = \frac{\text{Acc}_{\text{EA-VTON}}^{\text{VN}} - \text{Acc}_{\text{baseline}}^{\text{VN}}}{\text{Acc}_{\text{baseline}}^{\text{VN}}}$$

Measures the relative improvement of EA-VTON over baseline for Vietnamese body types. Positive EGS = EA-VTON helps.

### Size Shift Rate

$$\text{SSR} = \frac{|\{i : s_{\text{EA-VTON},i} \neq s_{\text{baseline},i}\}|}{N}$$

Percentage of users who get a different size recommendation. High SSR = the population correction matters in practice.

### Measurement Error (MAE)

For chest estimation against known ground truth (Tran et al. cluster centroids):

$$\text{MAE}_{\text{pop}} = \frac{1}{K} \sum_{k=1}^{K} |c_k - \hat{c}_{\text{pop}}(h_k, w_k)|$$

| Model | MAE (cm) |
|-------|----------|
| Baseline (US) | ~7.9 (projected) |
| EA-VTON (VN) | ~3.9 (projected) |

(Projected from cluster-centroid verification in [02-body-measurement-estimation.md](./02-body-measurement-estimation.md). These are not measured against per-subject ground truth; they are the errors each regression makes against the 5 published Vietnamese cluster centroids, and will be replaced with per-subject MAE once we have a held-out evaluation set.)

## SOTA Comparison Table (VITON-HD paired — numbers from FitDiT paper Table 1)

| Model | Year | Venue | FID↓ | SSIM↑ | LPIPS↓ | License |
|-------|------|-------|------|-------|--------|---------|
| FitDiT | 2024 | arXiv (2411.10499) | **4.7309** | **0.8985** | **0.0661** | Research / non-commercial |
| CatVTON | 2024 | arXiv (2407.15886) | 6.1394 | 0.8694 | 0.0970 | Research |
| IDM-VTON | 2024 | ECCV | 6.3381 | 0.8806 | 0.0789 | CC BY-NC-SA 4.0 |
| OOTDiffusion | 2024 | arXiv | 6.5186 | 0.8513 | 0.0964 | CC BY-NC-SA 4.0 |
| LaDI-VTON | 2023 | ACM MM | 6.6044 | 0.8763 | 0.0911 | CC BY-NC |
| StableVTON | 2024 | CVPR | 6.8581 | 0.8665 | 0.0835 | CC BY-NC-SA 4.0 |
| FASHN v1.5 | 2025 | Blog post / HF release | *not yet published* | *not yet published* | *not yet published* | **Apache 2.0** |
| EA-VTON (ours) | 2026 | — | TBD | TBD | TBD | Apache 2.0 |

Note: CatVTON's own paper reports slightly different self-measured numbers (e.g. paired FID ≈ 5.43, SSIM ≈ 0.8704, LPIPS ≈ 0.0565) due to different evaluation protocols. We use the FitDiT-reported values here for apples-to-apples comparison.

**EA-VTON's angle**: We're not claiming SOTA on VITON-HD (unfair comparison — different objective). Our contribution is demonstrating that existing models fail for underrepresented populations and providing population-aware corrections.

## Planned Experiments

1. **Baseline replication**: Run FASHN v1.5 on VITON-HD test set, report FID/SSIM/LPIPS
2. **Population-stratified evaluation**: Create a synthetic test set with Vietnamese body proportions, compare quality scores
3. **Size recommendation A/B**: For N Vietnamese body profiles, compare baseline vs EA-VTON size accuracy against Tran et al. ground truth
4. **Ablation studies**:
   - Importance weighting vs uniform weighting
   - Vietnamese formula vs baseline formula
   - With vs without postprocessing (face restoration, skin correction)
5. **User study** (if feasible): Vietnamese users rate try-on realism and size recommendation usefulness
