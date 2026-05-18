# Collaborator Setup — A100 Training Handoff

This doc gets you from `git clone` to a trained model + paper-ready metrics. Target: **45 min** clone → first epoch printing, **3-4 hours** to full results.

The primary deliverable is the **BodyM vision encoder** — a CNN that maps a person photo + height → body measurements → predicted size band. If end-to-end size accuracy beats our current scalar-only GBM baseline (52.2% exact / 70.7% within-1 on RTR female-upper), it's a +5-10 percentage-point contribution to the paper.

A secondary task — **CLIP-feature attribute classifier** — is also runnable as a 1-day side experiment.

---

## 1. Clone + Python environment

```bash
git clone https://github.com/<lead>/research-try-out.git
cd research-try-out

python3.11 -m venv .venv
source .venv/bin/activate

pip install \
  torch>=2.5 torchvision>=0.20 \
  numpy>=2.0 pandas>=2.2 scipy>=1.14 scikit-learn>=1.5 \
  duckdb>=1.5 pyarrow>=17 \
  open-clip-torch>=2.30 Pillow>=10.4 mediapipe>=0.10 \
  tqdm
```

Verify the GPU:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: True NVIDIA A100-...
```

---

## 2. What you get from `git clone` (all in repo, ~few MB)

- All training scripts and benchmark evaluators
- Per-source labeled parquets:
  - `research/datasets/processed/deepfashion2/items_upper_labeled.parquet` (~140k rows, ~30 MB)
  - `research/datasets/processed/fashionpedia/items_upper_labeled.parquet` (~38k rows, ~8 MB)
  - `research/datasets/processed/deepfashion_with_masks/items_upper_labeled.parquet` (~25k rows)
  - `research/datasets/processed/fashion_products_small/items_upper_labeled.parquet` (~5k rows)
- Body anthropometry data:
  - `research/datasets/processed/bodym_measurements.parquet` (2,507 subjects)
  - `research/datasets/processed/rtr_body_fit.parquet` (192,311 size-feedback rows)
  - `research/datasets/processed/modcloth_body_fit.parquet` (23,467 rows)
- Trained baselines for comparison:
  - `research/models/variants/gbm_female_upper_vn.pkl` — 52.2% exact / 70.7% within-1 (the current best size GBM)
  - `research/models/variants/gbm_modcloth_upper.pkl` — used by the photo→size chain at eval time
- VN/US/KR body-rule JSONs in `research/datasets/processed/dataset_bundle_v1/`
- MediaPipe pose-landmarker model in `services/feature-service/models/pose_landmarker_heavy.task` (50 MB)
- Paper outlines + audit history in `reports/paper_outline_v3.md`

---

## 3. Data you need separately (not in git)

| Asset | Size | Source | Needed for |
|---|---|---|---|
| BodyM photos | ~2 GB | **You download from Meta's public release** (see step 4) | Primary training task |
| `dataset_bundle_v1/labels.parquet` | 640 MB | rsync / Drive from your lead | Secondary task (CLIP-feature classifier) |
| `combined/images_redacted/` | ~13 GB | rsync from lead | Optional vision tasks on the catalog |

For the primary BodyM task, the only external download is **BodyM photos**. Everything else is in the repo or comes from your lead.

---

## 4. Download BodyM photos (~2 GB, ~10 min)

The repo already has the subject-to-photo mapping CSVs:
```
research/datasets/raw/bodym/train/subject_to_photo_map.csv
research/datasets/raw/bodym/testA/subject_to_photo_map.csv
research/datasets/raw/bodym/testB/subject_to_photo_map.csv
```

Each CSV maps `subject_id → photo_id` (4 photos per subject). You need to download the actual JPEG files into:
```
research/datasets/raw/bodym/train/photos/<photo_id>.jpg
research/datasets/raw/bodym/testA/photos/<photo_id>.jpg
research/datasets/raw/bodym/testB/photos/<photo_id>.jpg
```

Source: <https://github.com/facebookresearch/bodym> — follow their download instructions (BodyM is publicly released by Meta under CC BY-NC 4.0; license matters for *commercial* use but is fine for academic).

---

## 5. Sanity-check photo paths resolve

```bash
python3 -c "
from research.models.train_body_vision_encoder import build_photo_index, find_photo_path
photos = build_photo_index()
sample = photos.head(20)
ok = sum(find_photo_path(r.split, r.photo_id) is not None for _, r in sample.iterrows())
print(f'{ok}/20 sample photos resolved')
"
```

Expect at least 1/20 resolved. If 0/20, double-check the photo directory naming — the script accepts `photos/`, `photo_front/`, `photo_left/`, or files directly under the split directory.

---

## 6. Smoke train (3 epochs, ~15 min)

This validates the pipeline end-to-end before committing to the long run:

```bash
python3 research/models/train_body_vision_encoder.py \
    --epochs 3 --batch-size 32 --workers 8 --fp16 --gender female
```

Expect:
- Training loss decreases each epoch (`train_mse(z)` drops from ~0.5 → ~0.3 by epoch 3)
- testA / testB MAE in the 3-6 cm range
- A checkpoint saved at `research/models/variants/body_vision_encoder.pt`

If anything breaks, fix here. The full run won't fix bugs the smoke run reveals.

---

## 7. Full training run (~3-4 hours on A100)

```bash
python3 research/models/train_body_vision_encoder.py \
    --epochs 25 --batch-size 32 --workers 8 --fp16 --gender female
```

Output files when done:
- `research/models/variants/body_vision_encoder.pt` — state_dict + target normalization
- `research/eval/body_vision_encoder_results.json` — full metrics history + per-epoch + final chain-to-size eval

Per-epoch log format:
```
ep 15/25  train_mse(z)=0.142  testA MAE=2.81cm  testB MAE=3.04cm
```

Target by epoch 25:
- Train MSE (z-normalized) should drop to **~0.10**
- **testB MAE < 2.5 cm avg** — that beats the BodyM paper's scalar baseline by ~30%

Final size-chain log:
```
size chain testA: {'exact': 0.68, 'within1': 0.91, 'n': 88, ...}
size chain testB: {'exact': 0.65, 'within1': 0.89, 'n': 401, ...}
```

This is the **headline number for the paper**: photo → measurements → size band accuracy.

---

## 8. Decision gate

After the full run, look at `body_vision_encoder_results.json` → `final.size_chain_testB.within1`:

| Result | Action |
|---|---|
| ≥ 0.90 | **Strong win.** Add to paper as primary contribution. Ping the lead. |
| 0.80 – 0.89 | Reportable. Likely +3-5 pp paper bump. |
| < 0.80 | Investigate: photo quality, augmentation strength, try ViT-B/16 backbone (`--backbone vit_b_16` if added) |

Comparison points already on file:
- **Current GBM scalar-only**: 52.2% exact / 70.7% within-1 (on RTR female-upper)
- **ModCloth tape-measure ceiling**: 99.1% exact / 99.6% within-1 (when bust+waist+hip are directly known)
- **ModCloth height-only**: 29.6% exact / 60.2% within-1

The photo encoder's chain-to-size should fall between these. Higher = stronger paper claim.

---

## 9. Common breakages and fixes

| Error | Fix |
|---|---|
| `missing photo: split=train, photo_id=...` | BodyM photos not downloaded; see step 4 |
| `CUDA out of memory` | drop to `--batch-size 16`, then drop `--fp16` if it still OOMs |
| `ModCloth size model missing at .../gbm_modcloth_upper.pkl` | run `python3 research/models/train_modcloth_upper_size.py` once first |
| `expansion produced 0 rows` | photo CSV subject_ids don't match the measurements parquet; re-export the photo map matching BodyM's release |
| `0/20 sample photos resolved` | photo directory naming mismatch; check that files live under `train/photos/`, not `train/` directly |
| `cuDNN version mismatch` | reinstall torch matching your CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |

---

## 10. Reporting back

When the full run is done, drop these in our shared channel:

1. **`research/eval/body_vision_encoder_results.json`** (full metrics)
2. **`research/models/variants/body_vision_encoder.pt`** via Drive (gitignored, too big for PR)
3. **A 2-paragraph note**:
   - What the chain accuracy says (exact + within1 on testA + testB)
   - Whether it beats the GBM scalar baseline (52.2% exact / 70.7% within-1)
   - What you'd change about the setup if you had another 24 hours of A100

Lead runs the user study in parallel; we merge both result streams when you're done.

---

## 11. Optional secondary task — CLIP-feature attribute classifier (1 day)

If the BodyM run is going well and you have spare A100 cycles, run the redaction-invariant attribute classifier on the 208k catalog. Needs the bundle's `labels.parquet` (640 MB, get from the lead) + the redacted images directory (~13 GB).

```bash
# Stage 1 (~25-35 min): extract CLIP features for paired images
python3 research/models/redaction_invariant/01_extract_features.py \
  --batch-size 256 --num-workers 8 --device cuda --fp16

# Stage 2 (~5-10 min): train attribute heads
python3 research/models/redaction_invariant/02_train_heads.py \
  --epochs 20 --batch-size 512 --lr 1e-3

# Stage 3: emit decision-gate verdict
python3 research/models/redaction_invariant/03_evaluate.py
```

Output: `research/models/redaction_invariant/RESULTS.md` with STRONG_WIN / PARTIAL_WIN / NEGATIVE_RESULT verdict.

---

## 12. Conventions and gotchas

- **Do NOT write to `research/datasets/processed/combined/items_upper_combined.parquet`** during testing. There was a prior bug where a smoke-test overwrote the 643MB source with a 50-row subset. The filter scripts now use `*_flags.parquet` sidecars; mirror that pattern if you write your own scripts that touch this file.
- All random seeds default to **42**. Stratify size-band splits by `size_label`.
- CLIP backbone: **`ViT-B-32 / laion2b_s34b_b79k`** (OpenCLIP). Don't accidentally switch to `openai` weights — embeddings differ.
- VN body taxonomy is from Trieu et al. 2024 (*Vlákna a Textil*); 5 clusters. US (Lee 2007 FFIT) and KR (Size Korea) rule files at `research/datasets/processed/dataset_bundle_v1/body_rules_{vn,us,kr}.json`.
- Attribute label confidence in the catalog is low (~18-30%) because the CLIP prompts are highly correlated. Filter by argmax label, not by confidence threshold.

---

## 13. Where to read context before you start training

| File | Why read it |
|---|---|
| `reports/paper_outline_v3.md` | Current paper plan + 5-pass audit trail. The audit's verdict (45-58% cumulative A* probability with user study) is the operating estimate. |
| `research/models/train_body_vision_encoder.py` | The script itself — read the docstring header for architecture choices. |
| `research/models/train_body_regression.py` | The scalar-only baseline you're trying to beat with vision input. |
| `research/models/train_modcloth_upper_size.py` | The ModCloth size predictor you're chaining into. |
| `research/user_study/PROTOCOL.md` | The pre-registered user study running in parallel. |
| `benchmarks/{size_pred,ffm_rank,coldstart_cal}/README.md` | The three benchmark protocols your model should report against. |

Good luck — ping the lead with any blockers.
