# Benchmark B1 — Size Prediction

Predict a garment size band `s ∈ {XXS, XS, S, M, L, XL, XXL}` from observable user features.

## Task

**Input** (per row):
- `height_cm` (float)
- `weight_kg` (float, optional — present in RTR, absent in ModCloth)
- `bmi` (float, derivable)
- `age` (int, optional)
- `body_type_cluster` (int ∈ {1..5} for VN; population-specific clusters for US/KR)
- `garment_category` (str — one of: top, blouse, shirt, jumpsuit, blazer, …)

**Output**: a probability distribution over the 7 size bands.

## Datasets

| Dataset | Population | Rows (female upper, fit='fit') | Has weight? |
|---|---|---|---|
| RTR (primary) | US-rental users | 10,938 | yes |
| ModCloth (external validity) | US-mainstream users | 15,564 | no |

Held-out test split: **20% stratified by size band, seed=42** (fixed). Train: 80%.

## Metrics

| Metric | Description | Aggregation |
|---|---|---|
| **Exact** | `1[y_pred == y_true]` | mean |
| **Within-1** | `1[\|y_pred − y_true\| ≤ 1]` | mean — primary deploy metric (1 size off is acceptable; 2 off is a return) |
| **MAE** | `\|y_pred − y_true\|` in band units | mean |

Sub-population metrics (VN-range subset of RTR test, `h < 160 ∧ w < 55`, n ≈ 169): same three metrics computed on the subset.

## Baselines reported in v2 paper

| Variant | Tempering | Exact | Within-1 | MAE | VN-Exact | VN-W1 |
|---|---|---:|---:|---:|---:|---:|
| uniform | — | 0.516 | 0.698 | 1.004 | 0.497 | 0.734 |
| independence | α=1.0 | 0.517 | 0.696 | 1.014 | 0.521 | 0.722 |
| copula | α=1.0 | 0.523 | 0.705 | 0.998 | 0.533 | 0.734 |
| copula+PSIS | α=1.0 | 0.516 | 0.700 | 1.014 | 0.497 | 0.722 |
| **copula+PSIS α=0.75** (default) | α=0.75 | 0.522 | **0.707** | **0.990** | 0.509 | **0.740** |
| copula+PSIS α=0.50 | α=0.50 | 0.518 | 0.699 | 1.011 | 0.497 | 0.710 |

Honest interpretation: the reweighting gains are small (+0.6 pp exact, +0.9 pp within-1 over uniform), and PSIS k̂ = −0.09 means the tails were not heavy — PSIS is essentially a no-op on this data. Reported as a negative finding for principled covariate-shift correction in this specific setup.

**ModCloth external-validity row** (h+category only, apples-to-apples vs the camera-only deployment scenario): Exact 0.296 / W1 0.602 — confirms the task is non-trivial without weight/BMI signal.

**Ceiling row** (ModCloth, h+bust+waist+hip+category, "if you had a tape measure"): Exact 0.991 / W1 0.996 — the upper bound when circumferences are directly measured.

## Reproducing the numbers

```bash
# Reproduces the 6-row ablation table
python research/models/ablation_female_upper.py

# Reproduces the ModCloth rows
python research/datasets/scripts/process_modcloth.py
python research/models/train_modcloth_upper_size.py

# Evaluator (consumes results from above + the parquet files)
python benchmarks/size_pred/evaluate.py
```

## Files in this directory

- `evaluate.py` — runs all 8 rows and writes `results.json`.
- `baselines.json` — paper-reported baseline numbers for cross-checking.
- `splits.json` — split spec (test_size=0.2, random_state=42, stratify="size_label").
