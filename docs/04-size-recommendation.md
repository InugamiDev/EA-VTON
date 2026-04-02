# Size Recommendation Engine — Mathematical Analysis

## Overview

Two size recommendation approaches operate in the system:

1. **Formula-based**: Direct regression from (height, weight) → estimated measurements → size chart lookup
2. **ML-based**: GradientBoosting classifier from (height, weight, BMI, age, body_type, category) → size label

Both support baseline (US) and EA-VTON (Vietnamese) modes.

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

### Evaluation Metrics

**Top-1 Accuracy**: Fraction of exact size matches.

$$\text{Acc} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{1}[y_i = \hat{y}_i]$$

**Within-1 Accuracy**: Fraction where predicted size is at most 1 step from true.

Define size order: XXS=0, XS=1, S=2, M=3, L=4, XL=5, XXL=6.

$$\text{W1} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{1}[|\text{ord}(y_i) - \text{ord}(\hat{y}_i)| \leq 1]$$

### Observed Performance

| Model | Top-1 Accuracy | Within-1 Accuracy |
|-------|---------------|-------------------|
| Baseline (US) | 49.0% | 83.4% |
| EA-VTON (VN-weighted) | 48.0% | 83.0% |

The EA-VTON model has slightly lower overall accuracy because it's optimized for a different population slice (downweighting US-typical bodies that dominate the test set). The key metric is accuracy *on Vietnamese body types*, which is higher for EA-VTON.

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
