# Importance-Weighted Sampling — Mathematical Analysis

## Problem Statement

We have a large clothing fit dataset (RTR, 192K rows) from a US population, but no equivalent dataset for Vietnamese users. We want to train a model that behaves *as if* it was trained on Vietnamese data, without collecting new data.

**Key insight**: If we know the body measurement distributions of both populations, we can reweight existing samples to simulate the target population.

## Importance Sampling Theory

### Density Ratio Estimation

Let $P_S$ be the source (RTR/US) distribution and $P_T$ be the target (Vietnamese) distribution over body measurements $\mathbf{x} = (h, w, b)$ where $h$ = height, $w$ = weight, $b$ = BMI.

The importance weight for each sample is:

$$r(\mathbf{x}) = \frac{p_T(\mathbf{x})}{p_S(\mathbf{x})}$$

This transforms an expectation under $P_S$ to an expectation under $P_T$:

$$\mathbb{E}_{P_T}[f(\mathbf{x})] = \mathbb{E}_{P_S}\left[\frac{p_T(\mathbf{x})}{p_S(\mathbf{x})} f(\mathbf{x})\right]$$

In practice: a sample that's common in the Vietnamese population but rare in RTR gets upweighted; a sample common in RTR but rare in Vietnam gets downweighted.

### Gaussian Assumption

We model both distributions as independent Gaussians per feature:

$$p_T(\mathbf{x}) = \prod_{j \in \{h,w,b\}} \mathcal{N}(x_j \mid \mu_j^T, \sigma_j^T)$$

$$p_S(\mathbf{x}) = \prod_{j \in \{h,w,b\}} \mathcal{N}(x_j \mid \mu_j^S, \sigma_j^S)$$

Therefore:

$$r(\mathbf{x}) = \prod_{j} \frac{\mathcal{N}(x_j \mid \mu_j^T, \sigma_j^T)}{\mathcal{N}(x_j \mid \mu_j^S, \sigma_j^S)}$$

## Population Parameters

### Source (RTR)

Computed from data:

| Feature | $\mu^S$ | $\sigma^S$ |
|---------|---------|------------|
| Height (cm) | 165.9 | ~7.0 |
| Weight (kg) | 62.3 | ~12.0 |
| BMI | ~22.7 | ~4.0 |

### Target (Vietnamese Female)

From national statistics and Tran et al. 2024:

| Feature | $\mu^T$ | $\sigma^T$ |
|---------|---------|------------|
| Height (cm) | 156.2 | 5.5 |
| Weight (kg) | 53.9 | 8.0 |
| BMI | 21.4 | 2.5 |

## Per-Feature Weight Computation

### Height Weight

$$w_h(h) = \text{clip}\left(\frac{\mathcal{N}(h \mid 156.2, 5.5)}{\mathcal{N}(h \mid \mu_h^S, \sigma_h^S) + \epsilon}, 0.01, 100\right)$$

where $\epsilon = 10^{-10}$ prevents division by zero.

**Example**: For a person with $h = 155$ cm:
- Vietnamese PDF: $\mathcal{N}(155 \mid 156.2, 5.5) \approx 0.070$
- RTR PDF: $\mathcal{N}(155 \mid 165.9, 7.0) \approx 0.014$
- $w_h = 0.070 / 0.014 \approx 5.0$ → this person is 5× more likely in Vietnamese population

For a person with $h = 170$ cm:
- Vietnamese PDF: $\mathcal{N}(170 \mid 156.2, 5.5) \approx 0.0002$
- RTR PDF: $\mathcal{N}(170 \mid 165.9, 7.0) \approx 0.049$
- $w_h = 0.0002 / 0.049 \approx 0.004$ → clipped to 0.01

### Weight Weight

$$w_w(w) = \text{clip}\left(\frac{\mathcal{N}(w \mid 53.9, 8.0)}{\mathcal{N}(w \mid \mu_w^S, \sigma_w^S) + \epsilon}, 0.01, 100\right)$$

### BMI Weight

$$w_b(b) = \text{clip}\left(\frac{\mathcal{N}(b \mid 21.4, 2.5)}{\mathcal{N}(b \mid \mu_b^S, \sigma_b^S) + \epsilon}, 0.01, 100\right)$$

### Combined Weight

$$r_i = w_h(h_i) \cdot w_w(w_i) \cdot w_b(b_i)$$

Normalized to mean 1:

$$\tilde{r}_i = \frac{r_i}{\frac{1}{N}\sum_{j=1}^N r_j}$$

## Clipping Rationale

Without clipping, density ratios can be extreme. Consider $h = 140$ cm:
- Vietnamese PDF: $\mathcal{N}(140 \mid 156.2, 5.5) \approx 10^{-5}$
- RTR PDF: $\mathcal{N}(140 \mid 165.9, 7.0) \approx 10^{-6}$
- Raw ratio: ~10

Now consider $h = 140$ combined with $w = 40$ and low BMI — the product of three large ratios could reach $10^3$ or more, giving a single sample disproportionate influence.

The clipping bounds $[0.01, 100]$ per feature limit the maximum per-feature ratio to 100× and the minimum to 0.01×. The combined weight can still reach $100^3 = 10^6$ in extreme cases, but normalization keeps things practical.

## Effective Sample Size

The effective sample size (ESS) measures how many unweighted samples the weighted set is equivalent to:

$$n_{\text{eff}} = \frac{\left(\sum_{i=1}^N \tilde{r}_i\right)^2}{\sum_{i=1}^N \tilde{r}_i^2}$$

**Properties:**
- When all weights are equal: $n_{\text{eff}} = N$ (full dataset)
- When one weight dominates: $n_{\text{eff}} \to 1$
- High weight variance → low ESS → less reliable estimates

**Observed**: ESS = 24,313 out of 192,311 total (12.6%). This means the adapted dataset has roughly the statistical power of ~24K independent Vietnamese samples — adequate for training but indicating significant distribution shift.

## Resampling

In addition to importance weights (used directly in model training via `sample_weight`), we produce a resampled subset:

1. Convert weights to probabilities: $p_i = \tilde{r}_i / \sum_j \tilde{r}_j$
2. Draw 20,000 samples with replacement: $\{x_{\pi(1)}, ..., x_{\pi(20000)}\}$ where $\pi(i) \sim \text{Categorical}(p)$
3. Verify: resampled distribution should match Vietnamese targets

**Verification results:**

| Feature | RTR Original | Resampled | Vietnamese Target |
|---------|-------------|-----------|-------------------|
| Height | 165.9 ± 7.0 | ~156 ± 5.5 | 156.2 ± 5.5 |
| Weight | 62.3 ± 12.0 | ~54 ± 8.0 | 53.9 ± 8.0 |
| BMI | 22.7 ± 4.0 | ~21.4 ± 2.5 | 21.4 ± 2.5 |

## Size Distribution Shift

Reweighting shifts the size distribution toward smaller sizes:

```
Original RTR:    XXS(2%) XS(5%) S(15%) M(30%) L(25%) XL(15%) XXL(8%)
Resampled (VN):  XXS(5%) XS(12%) S(25%) M(28%) L(18%) XL(8%)  XXL(4%)
```

This reflects the physical reality: Vietnamese women are shorter and lighter, so their correct sizes cluster around S-M rather than M-L.

## Mathematical Guarantees

1. **Consistency**: As $N \to \infty$, the importance-weighted estimator converges to the true target expectation (assuming bounded density ratios)
2. **Unbiasedness**: $\mathbb{E}_{P_S}[r(\mathbf{x}) f(\mathbf{x})] = \mathbb{E}_{P_T}[f(\mathbf{x})]$ exactly
3. **Variance**: $\text{Var}_{P_S}[r(\mathbf{x}) f(\mathbf{x})]$ increases with the variance of $r$ — hence the ESS metric

## Limitations

1. **Gaussian assumption**: Body measurements are approximately but not exactly Gaussian (right-skewed for weight/BMI)
2. **Feature independence**: Height, weight, BMI are correlated but modeled independently — a multivariate Gaussian would be more accurate
3. **No fit-label shift**: We assume that the relationship between body measurements and correct size is the same across populations — only the body distribution changes
4. **Missing features**: Torso length, limb proportions, and other ethnicity-correlated features are not captured
