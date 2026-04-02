# Datasets & Data Pipeline

## Data Sources

### 1. Rent the Runway (RTR) Dataset

**Source**: McAuley Lab, UCSD
**Size**: 192,311 rows
**Population**: US women (self-reported body measurements + fit feedback)

| Field | Description | Coverage |
|-------|-------------|----------|
| height_cm | Self-reported height | ~90% |
| weight_kg | Self-reported weight | ~85% |
| bmi | Computed: $w / (h/100)^2$ | derived |
| size | Numeric size 0-20 | 100% |
| fit | "fit" / "small" / "large" | 100% |
| body_type_norm | Normalized body type label | ~70% |
| category | Garment category | 100% |
| age | Self-reported age | ~80% |

**Population statistics:**
- Height: $\bar{h} = 165.9$ cm, $\sigma_h \approx 7.0$ cm
- Weight: $\bar{w} = 62.3$ kg, $\sigma_w \approx 12.0$ kg
- BMI: $\bar{b} \approx 22.7$, $\sigma_b \approx 4.0$

**Limitations**: No ethnicity data. Biased toward US urban demographic (rental clothing users). Self-reported measurements may have systematic error.

### 2. Vietnamese Female Anthropometric Data (Tran et al. 2024)

**Source**: "Anthropometric characteristics and body shape classification of Vietnamese women" — peer-reviewed study
**Sample**: 480 women, ages 18-50, 3 geographic regions of Vietnam
**Method**: Professional body scanning + K-means clustering

**5 Clusters:**

| Cluster | Label | % | H (cm) | W (kg) | Chest | Waist | Hip |
|---------|-------|---|--------|--------|-------|-------|-----|
| 1 | short_thin | 15.2 | 151 | 46 | 79 | 65 | 86 |
| 2 | tall_heavy | 18.4 | 161 | 58 | 86 | 73 | 93 |
| 3 | medium | 35.9 | 156 | 52 | 83 | 69 | 89 |
| 4 | short_fat | 21.9 | 152 | 58 | 90 | 78 | 93 |
| 5 | obese_avg | 8.6 | 155 | 68 | 97 | 86 | 100 |

**National estimates** (population-weighted):
- Female: $\bar{h} = 156.2 \pm 5.5$ cm, $\bar{w} = 53.9 \pm 8.0$ kg

### 3. Vietnamese Male Anthropometric Data (IUH 2022)

**Source**: Industrial University of Ho Chi Minh City study
**Sample**: 1,106 men

**2 Clusters:**

| Cluster | Label | % | H (cm) | W (kg) |
|---------|-------|---|--------|--------|
| 1 | thin_short_upper | 60.4 | 166 | 60 |
| 2 | heavy_tall | 39.6 | 170 | 72 |

**National estimates:**
- Male: $\bar{h} = 168.1 \pm 6.0$ cm, $\bar{w} = 63.0 \pm 9.5$ kg

## Data Pipeline

### Step 1: Build Vietnamese Prior (`build_vn_prior.py`)

Input: Literature values (hardcoded from papers)
Output: `vn_female_prior.json`

```json
{
  "clusters": [...],
  "national_stats": {
    "female": {"height_cm": {"mean": 156.2, "std": 5.5}, ...},
    "male": {"height_cm": {"mean": 168.1, "std": 6.0}, ...}
  },
  "rtr_gap": {
    "height_cm": {"rtr_mean": 165.9, "vn_mean": 156.2, "diff": 9.7}
  }
}
```

### Step 2: Adapt RTR to Vietnamese (`adapt_rtr_to_vn.py`)

Input: `rtr_body_fit.parquet` + `vn_female_prior.json`
Output:
- `rtr_vn_adapted.parquet` (192K rows + importance weights, 2.9 MB)
- `rtr_vn_sampled.parquet` (20K resampled rows, 311 KB)

**Process**: See [03-importance-weighted-sampling.md](./03-importance-weighted-sampling.md)

### Step 3: Train Models (`train_size_rec.py`)

Input: `rtr_body_fit.parquet`
Output:
- `baseline_size_model.pkl` — US-population GradientBoosting
- `eavton_size_model.pkl` — Vietnamese-weighted GradientBoosting
- `model_config.json` — metadata, encoder classes, metrics

**Process**: See [04-size-recommendation.md](./04-size-recommendation.md)

## Population Gap

The core motivation for EA-VTON, visualized:

```
Height Distribution:
RTR (US):     ▁▂▃▅▇█▇▅▃▂▁     μ=165.9
Vietnam:  ▁▂▃▅▇█▇▅▃▂▁         μ=156.2
              ←── 9.7 cm gap ──→

Weight Distribution:
RTR (US):       ▁▂▃▅▇█▇▅▃▂▁   μ=62.3
Vietnam:   ▁▂▃▅▇█▇▅▃▂▁        μ=53.9
              ←── 8.4 kg gap ──→
```

The overlap between distributions determines the effective sample size after importance weighting. With a 9.7 cm height gap (>1σ for both distributions), the overlap is moderate — hence ESS of 12.6%.

## Data Quality Considerations

1. **Self-reported measurements (RTR)**: May have systematic bias. Height tends to be over-reported, weight under-reported. This affects both populations equally if the bias is consistent.

2. **Small Vietnamese sample**: 480 women clustered into 5 groups. The cluster centroids are stable (validated by silhouette scores in the original paper) but don't capture the full distribution.

3. **No garment-fit data for Vietnamese**: We have body measurements but not fit feedback. The importance weighting assumes the body→size relationship transfers across populations — a reasonable assumption for standardized garments but potentially inaccurate for culturally-specific cuts.

4. **Age distribution mismatch**: RTR skews younger (fashion rental demographic). Vietnamese data covers 18-50 evenly. Age affects body proportions, so this could introduce bias.
