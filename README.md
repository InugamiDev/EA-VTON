# EA-VTON: Ethnicity-Aware Virtual Try-On

**EA-VTON** takes a standard virtual try-on backbone (FASHN VTON v1.5) and makes it **population-aware and fit-aware** — so users don't just see "how it looks" but "how it fits."

> Research project for the paper: *EA-VTON: Ethnicity-Aware Virtual Try-On with Anthropometric Fit Prediction*

---

## What This Does

| Feature | Status | Description |
|---------|--------|-------------|
| **Image VTON** | Working | Upload person photo + garment image → try-on result |
| **Size Recommendation** | Working | Height + weight → predicted size (US and Vietnamese models) |
| **Body Measurement** | Working | Webcam/photo + height/weight → chest, waist, hip, shoulder estimates |
| **Fit-Aware Rendering** | Not built | Predict fit level (tight/normal/loose) and visually reflect it in VTON output |
| **Video VTON** | Experimental | Keyframe + optical flow warping — needs GPU infrastructure |

## Architecture

```
Person Photo + Garment Image + Body Info (h, w)
        |              |              |
        v              v              v
  ┌──────────┐  ┌──────────┐  ┌─────────────────┐
  │ Preprocess│  │ Preprocess│  │ Body Estimator  │
  │ Person    │  │ Garment  │  │ (MediaPipe Pose) │
  └─────┬─────┘  └─────┬────┘  └────────┬────────┘
        |              |                 |
        v              v                 v
  ┌─────────────────────────┐   ┌────────────────┐
  │    FASHN VTON v1.5      │   │  EA Size Engine │
  │    (972M params, MMDiT) │   │  (US + VN model)│
  │    20 denoise steps     │   │  → Size + Fit   │
  └───────────┬─────────────┘   └────────┬───────┘
              |                          |
              v                          v
  ┌─────────────────────────────────────────────┐
  │           Garment Composite (Stage 5)        │
  │                                              │
  │  1. Auto-detect garment mask (semantic)      │
  │  2. Sharpen garment (frequency-separated)    │
  │  3. Transfer scene lighting onto garment     │
  │  4. Composite: original × (1-mask) +         │
  │     enhanced_vton × mask                     │
  │  → Face, skin, hair, background = untouched  │
  └──────────────────┬──────────────────────────┘
                     v
              Result Image + Size Recommendation
```

## Project Structure

```
research-try-out/
├── backend/                    # FastAPI (Python)
│   ├── main.py                 # App entrypoint
│   ├── routers/
│   │   ├── tryon.py            # Full try-on (catalog-based)
│   │   ├── tryon_simple.py     # Simple try-on (upload 2 images)
│   │   ├── body_measurement.py # Photo-based body estimation
│   │   ├── size_recommendation.py
│   │   ├── garments.py
│   │   ├── upload.py
│   │   └── video_tryon.py      # Experimental
│   ├── services/
│   │   ├── pipeline/           # 5-stage VTON pipeline
│   │   │   ├── backends/       # FASHN HF, Replicate, local composite
│   │   │   ├── postprocessing/ # Garment composite, detail enhance, face restore
│   │   │   ├── preprocessor.py
│   │   │   └── postprocessor.py
│   │   ├── body_estimator.py   # MediaPipe Pose → measurements
│   │   ├── ea_size_engine.py   # Ethnicity-aware sizing
│   │   └── video_pipeline/     # Experimental video VTON
│   └── test_results/           # FASHN inference outputs + step comparisons
│
├── frontend/                   # Next.js 15 (TypeScript)
│   └── src/app/
│       ├── simple/             # Upload 2 images → try-on
│       ├── measure/            # Webcam body measurement
│       ├── try-on/             # Full try-on wizard
│       ├── catalog/            # Garment browse
│       └── ...
│
├── ea-vton/                    # Research artifacts
│   ├── datasets/               # BodyM, RTR, Vietnamese priors
│   │   ├── scripts/            # Processing & adaptation scripts
│   │   └── processed/          # Vietnamese population priors (JSON)
│   ├── models/                 # Trained size models + training script
│   └── paper/                  # LaTeX paper draft
│
├── docs/                       # Technical documentation
│   ├── 01-system-architecture.md
│   ├── 02-body-measurement-estimation.md
│   ├── 03-importance-weighted-sampling.md
│   ├── 04-size-recommendation.md
│   ├── 05-virtual-tryon-pipeline.md
│   ├── 06-quality-scoring.md
│   ├── 07-datasets.md
│   ├── 08-benchmarks.md
│   ├── 09-video-tryon-pipeline.md
│   └── 10-research-plan.md
│
└── reports/                    # Weekly progress reports (HTML slides)
```

## Quick Start

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API docs at `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

App at `http://localhost:3000`

### Try It

1. **Simple Try-On**: `http://localhost:3000/simple` — upload person + garment → result
2. **Body Measurement**: `http://localhost:3000/measure` — webcam + height/weight → size recommendation
3. **API directly**: `POST http://localhost:8000/api/tryon/simple` with `person` + `garment` files

## Novel Research Contributions

| # | Contribution | Status |
|---|-------------|--------|
| C1 | Vietnamese anthropometric model (Tran et al. 2024, 5 body clusters) | Done |
| C2 | Importance-weighted population adaptation (RTR → Vietnamese, ESS 12.6%) | Done |
| C3 | Fit-aware VTON rendering (tight/normal/loose visual adjustment) | TODO |
| C4 | Photo-based body estimation (MediaPipe + ellipse model) | Done |
| C5 | Fit-accuracy evaluation metrics (SA, FVC, PF) | TODO |

### Key Finding

The US-calibrated model **overestimates Vietnamese chest circumference by ~5.8 cm** at population mean. This causes wrong size recommendations for ~30% of Vietnamese users. The Vietnamese-calibrated model corrects this using population-specific regression coefficients.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| VTON Model | FASHN VTON v1.5 (Apache 2.0, 972M params, MMDiT) |
| Body Estimation | MediaPipe PoseLandmarker + Ramanujan's ellipse formula |
| Backend | FastAPI, Python 3.11+ |
| Frontend | Next.js 15, TypeScript, Tailwind CSS |
| Datasets | BodyM (Amazon), RTR (192K rows), Vietnamese female clusters |

## Hardware Notes

| Task | CPU (Apple M2) | GPU (A100) |
|------|----------------|------------|
| VTON inference (20 steps) | ~16 min | ~5 sec |
| Body estimation | ~2 sec | ~0.5 sec |
| Garment composite | ~3 sec | ~1 sec |
| Video VTON (Plan A) | ~5 FPS offline | ~15 FPS |

The image VTON pipeline runs on CPU but is slow. For practical use, a GPU is recommended.

## Documentation

Full technical docs in `docs/`. Start with:
- `docs/01-system-architecture.md` — system overview
- `docs/10-research-plan.md` — paper roadmap and implementation plan
- `docs/05-virtual-tryon-pipeline.md` — VTON pipeline details

## License

Research project. FASHN VTON v1.5 is Apache 2.0.
