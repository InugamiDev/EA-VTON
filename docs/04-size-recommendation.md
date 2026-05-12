# Size Recommendation Engine — Mathematical Analysis

## Overview

Two size recommendation approaches operate in the system:

1. **Formula-based**: Direct regression from (height, weight) → estimated measurements → size chart lookup
2. **ML-based**: GradientBoosting classifier from (height, weight, BMI, age, body_type, category) → size label

Both support explicit population modes. `us_women` is the default calibration,
`vietnamese` uses the EA-VTON Vietnamese adaptation, and `universal` is kept as a
legacy alias for the US-women profile.

## Population Profiles

The recommendation service centralizes anthropometric priors in
`services/recommendation/src/population_profiles.py`. This keeps visual sizing,
manual size estimation, and upper-body style ranking on the same assumptions.

| Population key | Current behavior |
|----------------|------------------|
| `us_women` | Default profile for US women's upper-body sizing and real-catalog style ranking |
| `vietnamese` | EA-VTON Vietnamese calibration, body-type classifier, and VN flattery rules |
| `universal` | Legacy input accepted by the API; resolves to the US-women profile |

### Measurement Regression

For formula-based sizing, each profile estimates body measurements from height
and weight before size-chart lookup. Inputs are height in cm and weight in kg;
outputs are estimated circumferences in cm.

US-women profile:

$$\hat{c}_{US} = 0.24h + 0.42w + 24.0$$

$$\hat{waist}_{US} = 0.20h + 0.50w + 10.0$$

$$\hat{hip}_{US} = 0.35h + 0.55w + 6.0$$

Vietnamese profile:

$$\hat{c}_{VN} = 0.485h + 0.22w - 2.5$$

$$\hat{waist}_{VN} = 0.30h + 0.38w - 4.0$$

$$\hat{hip}_{VN} = 0.42h + 0.24w + 3.0$$

The US-women constants are practical calibration coefficients for the current
demo. They are anchored to RTR/NHANES population statistics but should be
replaced by a trained measurement model when labeled bust/chest data is
available.

### Visual Upper-Body Calibration

The visual `/recommend-size/visual` flow does not ask for weight, so it estimates
visual weight from height plus normalized shoulder/hip and waist/hip ratios:

$$\hat{w}_{visual} = BMI_p(h / 100)^2 + (r_s - r_{s,p})\alpha_s + (r_w - r_{w,p})\alpha_w$$

where:

| Symbol | Meaning |
|--------|---------|
| $BMI_p$ | Population-specific visual BMI prior |
| $r_s$ | Observed shoulder-to-hip ratio |
| $r_w$ | Observed waist-to-hip ratio |
| $r_{s,p}$, $r_{w,p}$ | Source-specific population ratio anchors |
| $\alpha_s$, $\alpha_w$ | Source-specific ratio-to-weight calibration weights |

`world_3d` and `image_2d` use separate ratio anchors because projected 2D body
ratios are not numerically comparable to metric 3D ratios.

## Formula-Based Size Matching

### Size Chart Structure

Each garment has a size chart — a list of entries:

$$S = \{(s_k, c_k^{\min}, c_k^{\max})\}_{k=1}^K$$

where $s_k$ is the size label, $c_k^{\min}$ and $c_k^{\max}$ are the chest range bounds.

### Best Size Selection

Given estimated chest $\hat{c}$:

1. **Compute midpoints**: $m_k = (c_k^{\min} + c_k^{\max}) / 2$

2. **Compute distances**: $d_k = |\hat{c} - m_k|$

3. **Prefer in-range matches**: If $c_k^{\min} \leq \hat{c} \leq c_k^{\max}$ and $d_k < d_{\text{best}}$, select $k$

4. **Fallback to nearest**: If no in-range match, select $k^* = \arg\min_k d_k$

$$k^* = \begin{cases} \arg\min_{k: \hat{c} \in [c_k^{\min}, c_k^{\max}]} d_k & \text{if any in-range match exists} \\ \arg\min_k d_k & \text{otherwise} \end{cases}$$

### Confidence Scoring

**Case 1: In-range** ($c_{k^*}^{\min} \leq \hat{c} \leq c_{k^*}^{\max}$)

Compute off-center ratio:

$$\rho = \frac{|\hat{c} - m_{k^*}|}{(c_{k^*}^{\max} - c_{k^*}^{\min}) / 2}$$

- $\rho \in [0, 0.5)$: **high** confidence (near center of range)
- $\rho \in [0.5, 1.0]$: **medium** confidence (near edge of range)

**Case 2: Out-of-range**

Compute overshoot:

$$o = \min(|\hat{c} - c_{k^*}^{\min}|, |\hat{c} - c_{k^*}^{\max}|)$$

- $o \leq 2$ cm: **medium** confidence (borderline)
- $o > 2$ cm: **low** confidence (poor fit)

### Alternative Sizes

The system always suggests adjacent sizes:

$$\text{alternatives} = \{s_{k^*-1} \text{ (size down)}, s_{k^*+1} \text{ (size up)}\}$$

Each with contextual note: "Size down for a slimmer fit ($c_{k^*-1}^{\min}$–$c_{k^*-1}^{\max}$ cm)"

## ML-Based Size Recommendation

### Feature Engineering

Input feature vector for each sample:

$$\mathbf{x}_i = [h_i, w_i, b_i, a_i, \text{enc}(\text{body\_type}_i), \text{enc}(\text{category}_i)]$$

| Feature | Type | Preprocessing |
|---------|------|---------------|
| height_cm | continuous | raw |
| weight_kg | continuous | raw |
| bmi | continuous | fill NaN with 22.0 |
| age | continuous | fill NaN with 30 |
| body_type | categorical | LabelEncoder |
| category | categorical | LabelEncoder |

### Target Variable

RTR numeric sizes (0-20) are mapped to standard labels:

$$\text{size\_map}: \{0,1\} \to \text{XXS}, \{2,3\} \to \text{XS}, \{4,5,6\} \to \text{S}, ...$$

Then label-encoded for the classifier.

### Model: Gradient Boosting Classifier

$$F(\mathbf{x}) = \sum_{m=1}^{M} \gamma_m h_m(\mathbf{x})$$

where $h_m$ are regression trees fitted to residuals at each stage.

**Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 200 | Sufficient for convergence without overfitting |
| max_depth | 5 | Captures interactions (e.g., height × weight) |
| learning_rate | 0.1 | Standard; balances bias-variance |
| min_samples_leaf | 20 | Prevents overfitting to rare size/body combos |
| random_state | 42 | Reproducibility |

### Vietnamese Weighting

For the EA-VTON model, training uses importance weights (see [03-importance-weighted-sampling.md](./03-importance-weighted-sampling.md)):

$$\min_F \sum_{i=1}^{N} \tilde{r}_i \cdot L(y_i, F(\mathbf{x}_i))$$

where $\tilde{r}_i$ is the normalized importance weight and $L$ is the deviance loss.

This causes the model to focus on samples that look like Vietnamese body types — effectively learning "what size should a person with Vietnamese proportions wear?"

The served tuned GBM artifact is regenerated with:

```bash
python research/models/retrain_gbm_density_ratio.py
```

That script recomputes Copula+PSIS density ratios on the valid fit-only training
split, applies the selected tempering, rewrites the ignored pickle artifacts in
`research/models/variants/`, and refreshes `research/evaluation/tuned_results.json`.

Current Vietnamese-targeted Copula diagnostics for `gbm_copula_tempered_a075`:

| Metric | Value |
|--------|-------|
| Method | `copula_psis` |
| Tempering alpha | `0.75` |
| Raw Copula+PSIS ESS | 20,806 / 86,379 (24.1%) |
| Tempered ESS | 32,355 / 86,379 (37.5%) |
| PSIS k-hat | -0.001 |
| US train mean | 165.6 cm, 60.3 kg |
| VN target mean | 156.2 cm, 53.9 kg |

### Evaluation Metrics

**Top-1 Accuracy**: Fraction of exact size matches.

$$\text{Acc} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{1}[y_i = \hat{y}_i]$$

**Within-1 Accuracy**: Fraction where predicted size is at most 1 step from true.

Define size order: XXS=0, XS=1, S=2, M=3, L=4, XL=5, XXL=6.

$$\text{W1} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{1}[|\text{ord}(y_i) - \text{ord}(\hat{y}_i)| \leq 1]$$

### Observed Performance

| Model | Full Top-1 | Full Within-1 | VN-range Top-1 | VN-range Within-1 |
|-------|------------|---------------|----------------|-------------------|
| `gbm_uniform_current` | 49.0% | 83.4% | 50.7% | 70.1% |
| `gbm_indep_tempered_a05` | 48.9% | 83.1% | 51.5% | 70.9% |
| `gbm_copula_current` | 48.5% | 83.2% | 51.4% | 70.7% |
| `gbm_copula_tempered_a075` | 48.7% | 83.2% | 51.4% | 71.0% |

The EA-VTON weighted models have slightly lower overall accuracy because they are
optimized for a different population slice. The key metric is performance on
VN-range bodies, where the tuned Copula model improves within-1 accuracy from
70.1% to 71.0%.

Serving therefore selects the model by population objective and exact top-1
accuracy first: `gbm_uniform_current` for `us_women`/`universal` requests and
`gbm_indep_tempered_a05` for `vietnamese` requests. `gbm_copula_tempered_a075`
stays available because it has the best VN-range within-1 score.

## Comparison Mode

The comparison endpoint runs both models and produces:

$$\text{output} = \begin{cases}
\text{baseline}: & (\hat{c}_{\text{US}}, \hat{w}_{\text{US}}, \hat{p}_{\text{US}}, s_{\text{US}}) \\
\text{eavton}: & (\hat{c}_{\text{VN}}, \hat{w}_{\text{VN}}, \hat{p}_{\text{VN}}, s_{\text{VN}}, k^*_{\text{cluster}}) \\
\text{comparison}: & (\Delta c, \text{size\_changed}, \text{explanation})
\end{cases}$$

where:

$$\Delta c = \hat{c}_{\text{US}} - \hat{c}_{\text{VN}}$$

$$\text{size\_changed} = (s_{\text{US}} \neq s_{\text{VN}})$$

This side-by-side comparison is the core demonstration of EA-VTON's value proposition: the universal model oversizes because it was calibrated to US body proportions.
