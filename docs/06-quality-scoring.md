# Quality Scoring — Mathematical Analysis

## Overview

The quality scoring system evaluates virtual try-on results using 7 weighted metrics. Each metric produces a score in $[0, 1]$, combined into a final score in $[0, 100]$.

$$Q = 100 \times \sum_{i=1}^{7} w_i \cdot q_i$$

## Metric Weights

| Metric | Weight $w_i$ | Rationale |
|--------|-------------|-----------|
| Face Preservation | 0.35 | Users notice face degradation immediately |
| Structural Quality | 0.25 | Edge coherence indicates garment-body alignment |
| Artifact Detection | 0.20 | Visible artifacts break realism |
| Image Quality (brightness) | 0.10 | Part of the 0.20 image quality budget |
| Image Quality (contrast) | 0.10 | Part of the 0.20 image quality budget |
| Skin Tone Fidelity | 0.15 | Color shift is a known VTON flaw |

Note: Weights sum to 1.15, not 1.0. The final score is clamped to [0, 100].

## Metric 1: Face Preservation (SSIM)

### Setup

Extract the top 30% of the image (assumed face region). Resize both original and result to 256 × 128.

### SSIM (Structural Similarity Index)

For windows $x$ (original) and $y$ (result):

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

where:
- $\mu_x, \mu_y$: window means
- $\sigma_x^2, \sigma_y^2$: window variances
- $\sigma_{xy}$: cross-covariance
- $C_1 = (0.01 \times 255)^2 = 6.5025$
- $C_2 = (0.03 \times 255)^2 = 58.5225$

SSIM ranges in $[-1, 1]$, with 1.0 = identical.

The implementation uses `skimage.metrics.structural_similarity` on grayscale images.

### Face Score

$$q_{\text{face}} = \text{clip}(\text{SSIM}_{\text{face}}, 0, 1)$$

**Interpretation:**
- SSIM > 0.9: Face is well-preserved
- SSIM 0.7-0.9: Some degradation
- SSIM < 0.7: Significant face blurring/distortion

## Metric 2: Structural Quality (Laplacian Variance)

### Laplacian Edge Detection

The Laplacian operator detects edges via second-order derivatives:

$$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$

Approximated by convolution with kernel:

$$L = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

### Edge Variance

$$v = \text{Var}(\nabla^2 I)$$

**Ideal range**: $v \in [80, 500]$

$$q_{\text{structure}} = \begin{cases} \frac{v}{80} & \text{if } v < 80 \text{ (too blurry)} \\ 1.0 & \text{if } 80 \leq v \leq 500 \text{ (good detail)} \\ \max\left(0.5, 1 - \frac{v - 500}{500}\right) & \text{if } v > 500 \text{ (too noisy/sharp)} \end{cases}$$

Low variance = blurry/smooth image (bad for garment detail). Very high variance = noise or artifacts.

## Metric 3: Artifact Detection

### Block-Based Uniformity

Divide image into 32 × 32 blocks. For each block, compute standard deviation:

$$\sigma_{\text{block}} = \text{std}(\text{pixels in block})$$

A block is "uniform" (potentially artifacted) if $\sigma_{\text{block}} < 5$.

$$\text{uniform\_ratio} = \frac{\text{count}(\sigma_{\text{block}} < 5)}{\text{total blocks}}$$

### Gradient Variance

Compute gradient magnitude using Sobel operators:

$$G = \sqrt{G_x^2 + G_y^2}$$

$$v_G = \text{Var}(G)$$

### Artifact Score

$$q_{\text{artifact}} = 1.0 - \text{uniform\_ratio} \times 0.5 - \begin{cases} 0.3 & \text{if } v_G < 50 \\ 0 & \text{otherwise} \end{cases}$$

High uniform ratio + low gradient variance = flat, featureless patches = likely artifacts.

## Metric 4: Image Quality — Brightness

$$\mu = \text{mean}(I_{\text{gray}})$$

Ideal range: $\mu \in [80, 180]$ (not too dark, not too bright).

$$q_{\text{brightness}} = \begin{cases} \mu / 80 & \text{if } \mu < 80 \\ 1.0 & \text{if } 80 \leq \mu \leq 180 \\ \max\left(0, 1 - \frac{\mu - 180}{75}\right) & \text{if } \mu > 180 \end{cases}$$

## Metric 5: Image Quality — Contrast

$$\sigma = \text{std}(I_{\text{gray}})$$

Ideal range: $\sigma \in [40, 80]$.

$$q_{\text{contrast}} = \begin{cases} \sigma / 40 & \text{if } \sigma < 40 \\ 1.0 & \text{if } 40 \leq \sigma \leq 80 \\ \max\left(0.5, 1 - \frac{\sigma - 80}{80}\right) & \text{if } \sigma > 80 \end{cases}$$

## Metric 6: Skin Tone Fidelity (Delta-E in LAB)

### Skin Pixel Detection

Same HSV-based detection as preprocessing:

$$\text{skin}(H, S, V) = (H < 0.1 \lor H > 0.9) \land S \in [0.08, 0.75] \land V \in [0.15, 0.95]$$

### LAB Color Space

Convert skin pixels from RGB → LAB:

$$\text{RGB} \xrightarrow{\text{linear}} \text{XYZ} \xrightarrow{D65} \text{LAB}$$

### Delta-E (CIE76)

For each pair of skin pixels (original $\mathbf{a}$, result $\mathbf{b}$):

$$\Delta E = \sqrt{(L_a - L_b)^2 + (a^*_a - a^*_b)^2 + (b^*_a - b^*_b)^2}$$

Mean Delta-E across all skin pixels:

$$\overline{\Delta E} = \frac{1}{|\mathcal{S}|} \sum_{p \in \mathcal{S}} \Delta E(p)$$

### Skin Score

$$q_{\text{skin}} = \max\left(0, 1 - \frac{\overline{\Delta E}}{20}\right)$$

**Interpretation:**
- $\overline{\Delta E} < 3$: Imperceptible difference (score > 0.85)
- $\overline{\Delta E} \in [3, 6]$: Noticeable on close inspection (score 0.7-0.85)
- $\overline{\Delta E} > 10$: Obvious color shift (score < 0.5)

## Score Interpretation

| Score Range | Quality | Action |
|-------------|---------|--------|
| 85-100 | Excellent | Show result directly |
| 70-84 | Good | Show with minor disclaimer |
| 50-69 | Fair | Apply additional postprocessing |
| 0-49 | Poor | Suggest retrying with different photo |

## Example Calculation

For a try-on result with:
- Face SSIM: 0.92
- Laplacian variance: 200
- Uniform ratio: 0.05, gradient variance: 120
- Mean brightness: 140
- Contrast std: 55
- Skin Delta-E: 4.5

$$Q = 100 \times (0.35 \times 0.92 + 0.25 \times 1.0 + 0.20 \times 0.975 + 0.10 \times 1.0 + 0.10 \times 1.0 + 0.15 \times 0.775)$$

$$Q = 100 \times (0.322 + 0.25 + 0.195 + 0.1 + 0.1 + 0.116) = 100 \times 1.083$$

Clamped to 100. (In practice, the weights summing > 1.0 means a perfect image naturally caps at 100.)
