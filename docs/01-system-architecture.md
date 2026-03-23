# EA-VTON System Architecture

## Overview

EA-VTON (Ethnicity-Aware Virtual Try-On) is a framework that combines population-specific anthropometric modeling with virtual garment try-on to produce accurate size recommendations and realistic clothing visualizations for underrepresented body types — specifically Vietnamese populations.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                       │
│  /catalog  /closet  /try-on  /profile  /product/[id]           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST API
┌──────────────────────────▼──────────────────────────────────────┐
│                     Backend (FastAPI)                            │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   Routers   │  │   Services   │  │      Pipeline          │ │
│  │             │  │              │  │                        │ │
│  │ /garments   │  │ ea_size_     │  │ Preprocessor           │ │
│  │ /tryon      │──│ engine.py    │  │   ↓                    │ │
│  │ /size_rec   │  │              │  │ Inference Backend      │ │
│  │ /upload     │  │              │  │   (FASHN / Composite)  │ │
│  │             │  │              │  │   ↓                    │ │
│  └─────────────┘  └──────────────┘  │ Postprocessor          │ │
│                                     │   (Quality Scoring)    │ │
│                                     └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    ML / Data Layer                               │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐ │
│  │  Training        │  │  Datasets                            │ │
│  │                  │  │                                      │ │
│  │  train_size_     │  │  RTR (192K rows, US population)     │ │
│  │  rec.py          │  │  rtr_vn_adapted.parquet (weighted)  │ │
│  │                  │  │  rtr_vn_sampled.parquet (20K)       │ │
│  │  baseline.pkl    │  │  vn_female_prior.json               │ │
│  │  eavton.pkl      │  │                                      │ │
│  └──────────────────┘  └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Size Recommendation Engine (`ea_size_engine.py`)

The heart of the system. Provides two estimation modes:

- **Universal (US-calibrated)**: Regression coefficients fitted to CAESAR/RTR population
- **Vietnamese-calibrated**: Coefficients fitted to Tran et al. (2024) cluster centroids

Outputs a recommended size with confidence level and alternatives, plus a comparison mode showing both predictions side-by-side.

→ Full math: [02-body-measurement-estimation.md](./02-body-measurement-estimation.md)

### 2. Importance-Weighted Sampling (`adapt_rtr_to_vn.py`)

Adapts the Western RTR dataset to simulate Vietnamese body proportions using density ratio estimation. Produces importance weights and a resampled subset.

→ Full math: [03-importance-weighted-sampling.md](./03-importance-weighted-sampling.md)

### 3. Size Recommendation ML Model (`train_size_rec.py`)

GradientBoosting classifier trained on RTR with optional Vietnamese importance weights. Predicts size label from body measurements.

→ Full math: [04-size-recommendation.md](./04-size-recommendation.md)

### 4. Virtual Try-On Pipeline

Three-stage pipeline:
- **Preprocessing**: Person detection, skin segmentation, garment extraction
- **Inference**: VTON model (FASHN v1.5 or local composite fallback)
- **Postprocessing**: Quality scoring, color correction, sharpening

→ Full math: [05-virtual-tryon-pipeline.md](./05-virtual-tryon-pipeline.md), [06-quality-scoring.md](./06-quality-scoring.md)

### 5. Vietnamese Anthropometric Priors (`build_vn_prior.py`)

Population-level body statistics from peer-reviewed studies, structured as cluster centroids with prevalence percentages.

→ Full details: [07-datasets.md](./07-datasets.md)

## Data Flow

```
User Input (height, weight, fit_preference)
    │
    ├─→ Universal Formula ──→ Baseline Size + Confidence
    │
    ├─→ Vietnamese Formula ──→ EA-VTON Size + Confidence
    │
    └─→ Comparison ──→ {size_changed, chest_diff, explanation, nearest_cluster}

User Upload (person photo + garment selection)
    │
    ├─→ Preprocessor ──→ Normalized person (768×1024) + garment mask
    │
    ├─→ VTON Inference ──→ Raw try-on result
    │
    └─→ Postprocessor ──→ Quality-scored result (0-100)
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Population-specific regression (not one-size-fits-all) | 9.7cm mean height gap between RTR and Vietnamese populations causes systematic oversizing |
| Importance weighting (not separate VN dataset) | No large-scale Vietnamese clothing fit dataset exists; reweighting RTR is practical |
| Cluster-based body typing | Tran et al. 2024 provides validated cluster centroids; more interpretable than raw statistics |
| Confidence scoring with alternatives | Users need actionable guidance, not just a label |
| FASHN v1.5 (Apache 2.0) | Eliminates NC license dependencies from SMPL, IDM-VTON, CatVTON |
