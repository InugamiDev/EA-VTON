# PRISM-UB Benchmark Suite

Three reusable benchmarks built on the PRISM-UB (208,069-item upper-body fashion catalog with face-redaction-first labels) dataset.

| ID | Task | Primary metric | Status |
|---|---|---|---|
| **B1** | Size Prediction | Exact accuracy + Within-1 + MAE | `size_pred/` |
| **B2** | Body-Shape-Aware Item Ranking | NDCG@K | `ffm_rank/` |
| **B3** | Cold-Start Calibration | Visual alignment + Recall@K + Kendall's τ | `coldstart_cal/` |

Each benchmark directory contains:
- `README.md` — task definition, splits, metrics, baselines.
- `evaluate.py` — single-file evaluator. Anyone with the dataset bundle can run this.
- `baselines.json` — reference numbers from the paper.

## Running a benchmark

```bash
# 1. Get the dataset bundle (CC BY 4.0)
cd research/datasets/processed/dataset_bundle_v1

# 2. Run a benchmark — example: size prediction
python ../../../../benchmarks/size_pred/evaluate.py --variant copula_psis_a075
```

All evaluators print a markdown-formatted table to stdout AND write a JSON sidecar so leaderboard maintenance is automatable.

## Dependencies

- Python ≥ 3.11
- numpy ≥ 2.0
- pandas ≥ 2.2
- scikit-learn ≥ 1.5
- duckdb ≥ 1.5
- scipy ≥ 1.14

A `requirements.txt` is included per benchmark.

## Submission protocol

Submitting new numbers? Open a PR adding a row to `<benchmark>/leaderboard.md` and attach:
- The evaluator's JSON output
- A pointer to your training code (any license)
- A 1-paragraph method description

We do not require code or weights to be public, but the JSON output and method description must be public.

## License

Benchmark protocol files, baseline code, and evaluators in this directory are CC BY 4.0 (same as the dataset bundle's derivative labels). Source-dataset images keep their upstream licenses — see `LICENSE.SOURCES.md` in the bundle for the per-source matrix.
