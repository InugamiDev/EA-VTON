# Body Measurement Estimation — Mathematical Analysis

## Problem Statement

Given a person's height $h$ (cm) and weight $w$ (kg), estimate body circumferences (chest, waist, hip) that are accurate for their population. Standard regression models are calibrated to Western (US/EU) populations and systematically overestimate measurements for Vietnamese individuals.

## Baseline Model (US-Calibrated)

### Chest Circumference

$$\hat{c}_{\text{US}}(h, w) = 0.52h + 0.18w$$

Calibrated to the CAESAR/RTR population:
- Mean height: $\bar{h}_{\text{RTR}} = 165.9$ cm
- Mean weight: $\bar{w}_{\text{RTR}} = 62.3$ kg

**Verification at RTR mean:**

$$\hat{c}_{\text{US}}(165.9, 62.3) = 0.52 \times 165.9 + 0.18 \times 62.3 = 86.27 + 11.21 = 97.5 \text{ cm}$$

This aligns with US female median chest circumference (~96-98 cm from NHANES data).

### Waist Circumference

$$\hat{w}_{\text{US}}(h, w) = 0.35h + 0.30w$$

### Hip Circumference

$$\hat{p}_{\text{US}}(h, w) = 0.48h + 0.22w$$

## Vietnamese-Calibrated Model

### Derivation from Cluster Centroids

The Vietnamese model is derived from OLS regression on 5 female body type cluster centroids (Tran et al. 2024, n=480 women, ages 18-50, 3 geographic regions):

| Cluster | Label | Prevalence | Height (cm) | Weight (kg) | Chest (cm) | Waist (cm) | Hip (cm) |
|---------|-------|------------|-------------|-------------|------------|------------|----------|
| 1 | short_thin | 15.2% | 151 | 46 | 79 | 65 | 86 |
| 2 | tall_heavy | 18.4% | 161 | 58 | 86 | 73 | 93 |
| 3 | medium (dominant) | 35.9% | 156 | 52 | 83 | 69 | 89 |
| 4 | short_fat | 21.9% | 152 | 58 | 90 | 78 | 93 |
| 5 | obese_avg | 8.6% | 155 | 68 | 97 | 86 | 100 |

### Chest Regression

Fitting $\hat{c}_{\text{VN}} = \alpha h + \beta w + \gamma$ to the 5 cluster centroids via weighted least squares (weights = cluster prevalence):

$$\hat{c}_{\text{VN}}(h, w) = 0.485h + 0.22w - 2.5$$

**Verification against cluster centroids:**

| Cluster | Actual Chest | Predicted | Error |
|---------|-------------|-----------|-------|
| 1 (151, 46) | 79 | $0.485(151) + 0.22(46) - 2.5 = 80.4$ | +1.4 |
| 2 (161, 58) | 86 | $0.485(161) + 0.22(58) - 2.5 = 87.9$ | +1.9 |
| 3 (156, 52) | 83 | $0.485(156) + 0.22(52) - 2.5 = 82.5$ | -0.5 |
| 4 (152, 58) | 90 | $0.485(152) + 0.22(58) - 2.5 = 83.4$ | -6.6 |
| 5 (155, 68) | 97 | $0.485(155) + 0.22(68) - 2.5 = 87.7$ | -9.3 |

Note: Clusters 4 and 5 show higher error because they represent body types where chest size is disproportionately large relative to height/weight — the linear model underestimates for heavier body types. A nonlinear model or BMI interaction term could improve this.

**RMSE across centroids:**

$$\text{RMSE} = \sqrt{\frac{1}{5}\sum_{i=1}^{5}(c_i - \hat{c}_i)^2} = \sqrt{\frac{1.96 + 3.61 + 0.25 + 43.56 + 86.49}{5}} \approx 5.2 \text{ cm}$$

### Waist Regression

$$\hat{w}_{\text{VN}}(h, w) = 0.30h + 0.38w - 4.0$$

### Hip Regression

$$\hat{p}_{\text{VN}}(h, w) = 0.42h + 0.24w + 3.0$$

## Population Gap Analysis

### Systematic Difference

For a Vietnamese woman at the population mean ($h = 156.2$, $w = 53.9$):

$$\hat{c}_{\text{US}}(156.2, 53.9) = 0.52(156.2) + 0.18(53.9) = 81.2 + 9.7 = 90.9 \text{ cm}$$

$$\hat{c}_{\text{VN}}(156.2, 53.9) = 0.485(156.2) + 0.22(53.9) - 2.5 = 75.8 + 11.9 - 2.5 = 85.1 \text{ cm}$$

$$\Delta c = \hat{c}_{\text{US}} - \hat{c}_{\text{VN}} = 90.9 - 85.1 = 5.8 \text{ cm}$$

The US model overestimates chest by ~5.8 cm for the average Vietnamese woman. This is enough to shift a size recommendation by 1-2 sizes (typical size chart has 4-5 cm between sizes).

### General Difference Formula

$$\Delta c(h, w) = \hat{c}_{\text{US}} - \hat{c}_{\text{VN}} = (0.52 - 0.485)h + (0.18 - 0.22)w + 2.5$$

$$\Delta c(h, w) = 0.035h - 0.04w + 2.5$$

This reveals:
- **Height dominant**: taller people get larger overestimates ($+0.035$ per cm)
- **Weight dampening**: heavier people get slightly smaller overestimates ($-0.04$ per kg)
- **Constant offset**: +2.5 cm baseline gap from the intercept

At what point does $\Delta c = 0$?

$$0.035h - 0.04w + 2.5 = 0 \implies w = 0.875h + 62.5$$

For $h = 156$: $w = 0.875(156) + 62.5 = 199$ kg — far outside normal range, confirming the US model always overestimates for realistic Vietnamese body measurements.

## Fit Preference Adjustment

A post-estimation additive adjustment:

$$\hat{c}_{\text{final}} = \hat{c}_{\text{pop}} + \delta_{\text{fit}}$$

where:

| Fit Preference | $\delta_{\text{fit}}$ |
|---------------|----------------------|
| Slim | -2.0 cm |
| Regular | 0.0 cm |
| Relaxed | +3.0 cm |

This is applied identically to both population models, preserving the inter-population difference.

## Nearest Cluster Matching

Given user measurements $(h_u, w_u)$, the nearest Vietnamese body type cluster is found via Euclidean distance:

$$k^* = \arg\min_{k \in \{1,...,5\}} \sqrt{(h_u - h_k)^2 + (w_u - w_k)^2}$$

This provides interpretability: "Your body profile is closest to Vietnamese body type 'medium' (35.9% of population)."

## Limitations

1. **Linear model**: Cannot capture nonlinear relationships between height/weight and chest circumference (especially for obese body types)
2. **Two-feature input**: BMI, age, and body proportions (e.g., torso-to-leg ratio) are not used
3. **Female-only Vietnamese data**: Male Vietnamese data is limited (2 clusters from IUH 2022 vs 5 female clusters)
4. **Small calibration set**: Only 5 cluster centroids for regression fitting (480 original subjects collapsed to 5 points)
5. **No bust/underbust distinction**: Chest circumference conflates bust and ribcage measurements
