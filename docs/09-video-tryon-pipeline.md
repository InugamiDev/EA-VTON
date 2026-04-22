# RT-VTON: Real-Time Video Virtual Try-On

> **Document type**: Architecture specification & research contribution outline
> **Last updated**: 2026-04-06
> **Status**: Plan A implemented, Plan B designed (pending GPU infrastructure)

---

## Overview

RT-VTON extends the image-based VTON pipeline (FASHN VTON v1.5 + garment-masked compositing) to **video**. Two phases:

| Phase | Description | Status | Target FPS |
|-------|-------------|--------|------------|
| **Plan A** | Keyframe VTON + optical flow warping | **Implemented (demo)** | 5–15 offline |
| **Plan B** | Distilled lightweight model, single forward pass | Designed, pending infra | 30 real-time |

**Plan A is a demo/proof-of-concept pipeline.** It produces research-quality results for benchmarking and paper evaluation but is not designed for production real-time use. It processes video offline — VTON inference on sparse keyframes (~1 in 10 frames), optical flow warping for the rest.

**Plan B is the production target.** A distilled student model trained on Plan A's output, targeting 30 FPS single-pass inference suitable for deployment.

Both phases are paper candidates. Plan A contributes novel techniques (pose-aware keyframe selection, garment-masked flow warping). Plan B contributes a self-supervised distillation methodology.

---

## Plan A — Keyframe + Optical Flow Warping

### Architecture

```
VideoInput → FrameExtractor → PoseTracker → KeyframeDetector
                                              ↓
                            AsyncVTON (background, keyframes only, ~0.2 FPS)
                                              ↓
                            OpticalFlowWarper (every frame, ~10 FPS)
                                              ↓
                            GarmentCompositor → TemporalSmoother → VideoOutput
```

**Core insight**: VTON runs on ~1 in 10 frames (keyframes). For the other 9, warp the garment using optical flow. Face/skin/background are always from the original frame — only the garment region is synthesized or warped.

### Pipeline Stages

The pipeline executes 7 sequential stages:

#### Stage 1 — Frame Extraction

Extract frames from input video, normalizing to a target FPS.

- **Input**: Video file (MP4/MOV/AVI/WebM)
- **Method**: `cv2.VideoCapture` → generator pattern (no full-video memory load)
- **Output**: Sequence of `(frame_idx, PIL.Image, timestamp_ms)`
- **FPS normalization**: Skip source frames to match target FPS

$$\text{frame\_interval} = \max\left(1, \frac{\text{fps}_{\text{source}}}{\text{fps}_{\text{target}}}\right)$$

#### Stage 2 — Pose Tracking

Detect 33 body landmarks per frame using MediaPipe PoseLandmarker.

- **Model**: `pose_landmarker_lite.task` (5.8 MB, ~15 FPS on M2 CPU)
- **Output**: `(33, 3)` array per frame — `[x, y, visibility]` in normalized coordinates
- **Garment subset**: 8 landmarks relevant to garment fitting:

| Index | Landmark | Role |
|-------|----------|------|
| 11 | Left shoulder | Garment width |
| 12 | Right shoulder | Garment width |
| 13 | Left elbow | Sleeve position |
| 14 | Right elbow | Sleeve position |
| 15 | Left wrist | Sleeve endpoint |
| 16 | Right wrist | Sleeve endpoint |
| 23 | Left hip | Garment hem |
| 24 | Right hip | Garment hem |

#### Stage 3 — Keyframe Detection

Determine which frames need full VTON inference based on pose changes.

**Decision rules** (evaluated in order):

1. **Frame 0** → always keyframe
2. **Pose lost** (no detection) → force keyframe
3. **Cooldown**: skip if `frames_since_last < min_interval`
4. **Max interval**: force if `frames_since_last ≥ max_interval`
5. **Pose delta**: trigger if garment landmark displacement exceeds threshold

$$\delta_{\text{pose}} = \frac{1}{8} \sum_{i \in \text{garment\_ids}} \left\| \mathbf{p}_i^{(t)} - \mathbf{p}_i^{(t_k)} \right\|_2$$

where $\mathbf{p}_i^{(t)}$ is the $(x, y)$ position of landmark $i$ at frame $t$, and $t_k$ is the last keyframe.

6. **Torso rotation**: trigger if shoulder angle change exceeds rotation threshold

$$\theta = \arctan2\left(\Delta y_{\text{shoulders}}, \Delta x_{\text{shoulders}}\right)$$

$$\delta_{\text{rot}} = |\theta^{(t)} - \theta^{(t_k)}|$$

**Expected density**: ~1 keyframe per 5–15 frames (6–20% of frames).

#### Stage 4 — Keyframe VTON Inference

Run full FASHN VTON v1.5 inference on keyframe frames only.

- Reuses the existing image pipeline's `huggingface_vton.py` backend
- ~5 seconds per keyframe on CPU, ~1s on GPU
- After inference, detects garment mask on the result using `FashnHumanParser`
- Stores: `(result_image, garment_mask)` per keyframe

#### Stage 5 — Optical Flow Warping

For each inter-frame, warp the nearest keyframe's garment onto the current frame.

**Flow estimation** — RAFT-Small (`torchvision.models.optical_flow.raft_small`):
- Input: two frames normalized to $[-1, 1]$, resolution $512 \times 384$ (divisible by 8)
- Output: $(2, H, W)$ flow field in pixel displacement units
- Speed: ~8–12 FPS on M2 CPU

**Occlusion detection** — Forward-backward consistency:

For each pixel $\mathbf{x}$, check:

$$\left\| \mathbf{f}_{\text{fwd}}(\mathbf{x}) + \mathbf{f}_{\text{bwd}}\left(\mathbf{x} + \mathbf{f}_{\text{fwd}}(\mathbf{x})\right) \right\|_2 > \tau_{\text{occ}}$$

where $\tau_{\text{occ}} = 2.0$ pixels. Pixels exceeding this threshold are marked occluded.

**Backward warping** — `torch.nn.functional.grid_sample(mode='bilinear')`:

$$I_{\text{warped}}(\mathbf{x}) = I_{\text{keyframe}}\left(\mathbf{x} + \mathbf{f}(\mathbf{x})\right)$$

**Garment-masked warping**:
1. Warp keyframe garment pixels using flow field
2. Warp keyframe garment mask using same flow field
3. Zero out occluded regions → fall back to original frame pixels
4. Composite: `current_frame * (1 - mask) + warped_garment * mask`

#### Stage 6 — Temporal Smoothing

Blend outgoing and incoming keyframe results near keyframe boundaries.

Within a blend window of $N$ frames after a new keyframe:

$$\alpha(t) = \frac{t - t_k}{N}$$

$$I_{\text{blended}} = (1 - \alpha) \cdot I_{\text{outgoing}} + \alpha \cdot I_{\text{incoming}}$$

Only applied within the garment mask — non-garment pixels are always from the original frame.

#### Stage 7 — Video Assembly

Assemble processed frames into MP4 output.

- `cv2.VideoWriter` with H.264 codec
- Audio preserved via ffmpeg mux from source video
- Output at target FPS

### Configuration

```python
@dataclass
class VideoPipelineConfig:
    vton_config: PipelineConfig          # Reuse image pipeline config
    target_fps: int = 15
    keyframe_pose_threshold: float = 0.05
    keyframe_min_interval: int = 5
    keyframe_max_interval: int = 30
    flow_model: str = "raft_small"
    blend_window: int = 3
    occlusion_threshold: float = 2.0
    max_duration_sec: float = 30.0       # Limit for demo
    output_format: str = "mp4"
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/video-tryon` | Upload video + garment_id → job_id |
| `GET` | `/api/video-tryon/{job_id}` | Poll status (7 stages tracked) |
| `GET` | `/api/video-tryon/{job_id}/video` | Download result MP4 |

### Performance Targets (M2 CPU — projected, not measured)

The pipeline below has been designed and individual components exist, but an end-to-end offline run on our reference hardware has not been benchmarked. The numbers are **design targets**, derived by composing published per-component costs of the individual libraries used. Actual end-to-end throughput will be filled in once we run the benchmark in `backend/tests/test_video_pipeline.py` on the reference machine.

| Stage | FPS (target) | Basis |
|-------|--------------|-------|
| Frame extraction | >60 | Trivial I/O, `cv2.VideoCapture` |
| Pose tracking | ~15 | MediaPipe CPU reference |
| VTON keyframe | ~0.2 | ~5s per keyframe observed for image pipeline |
| Optical flow | ~8–12 | RAFT-Small torchvision reference at 512×384 |
| Garment composite | ~30 | NumPy/cv2 ops |
| **Overall offline** | **~5–10 (projected)** | Flow expected to dominate |

### File Structure

```
services/video_pipeline/
├── __init__.py              # VideoPipelineConfig, run_video_pipeline()
├── frame_extractor.py       # cv2.VideoCapture → frame generator
├── pose_tracker.py          # MediaPipe PoseLandmarker → 33 landmarks
├── keyframe_detector.py     # Pose-delta triggers keyframe VTON
├── optical_flow.py          # RAFT-Small wrapper + backward warping
├── garment_warper.py        # Warp garment mask+pixels via flow field
├── temporal_smoother.py     # Blend near keyframe transitions
├── async_vton_worker.py     # Background VTON on keyframes
├── video_assembler.py       # Frames → MP4 (cv2/ffmpeg)

apps/api/src/routes/
├── video_tryon.py           # POST/GET endpoints

backend/tests/
├── test_video_pipeline.py   # 18 unit + integration tests
```

### Limitations (Plan A)

- **Offline only**: Processes entire video before producing output. Not suitable for live/streaming use.
- **Flow drift**: Warping accumulates error over many frames between keyframes. Mitigated by max_interval forcing periodic keyframes.
- **Garment tearing**: At extreme limb movement or self-occlusion, warped garment can tear. Occlusion mask zeros these out, falling back to original pixels.
- **Single person**: Assumes one person in frame. Multi-person would require per-person tracking and masking.
- **CPU bottleneck**: RAFT-Small on CPU is the main throughput limiter. We have not yet benchmarked it end-to-end on our hardware — the FPS numbers in this document are design targets extrapolated from the torchvision RAFT-Small reference and from SEA-RAFT's published GPU numbers (SEA-RAFT is a different model). Treat them as targets, not measurements.

---

## Plan B — Distilled Real-Time Model (Future)

> **Status**: Designed. Requires GPU training infrastructure (4–8× A100, ~3–5 days).

### Goal

True 30 FPS (33ms/frame), single forward pass per frame, no keyframe+warp architecture needed. Suitable for production deployment and live streaming.

### Architecture

```
PersonFrame + Garment → StudentModel → TryOnFrame
                           ↓
                    TemporalState (ConvGRU hidden state between frames)
```

### Approach

#### Phase B1 — Self-Supervised Data Generation

Use Plan A to generate unlimited training pairs:

- **Input**: `(person_frame, garment_image)` from diverse video sources
- **Target**: VTON result from Plan A's teacher pipeline (FASHN VTON v1.5)
- **Scale**: Generate 100K+ pairs from diverse videos, poses, garments, lighting conditions
- **Key advantage**: No manual labeling needed — Plan A generates ground truth automatically

This is a novel contribution: **the demo pipeline generates its own training data for the production model**.

#### Phase B2 — Student Model Architecture (< 20M params)

```
┌─────────────────────────────────────────────────────┐
│                   Student Model                      │
│                                                      │
│  Person Frame ──→ MobileNetV3 Encoder (~5ms)         │
│                        ↓                             │
│  Garment Image ──→ TPS Warping Module                │
│                        ↓                             │
│                   Fusion Decoder                     │
│                   (upsampling + residual blocks)      │
│                        ↓                             │
│  Prev Hidden ──→ ConvGRU Temporal Module ──→ Hidden  │
│                        ↓                             │
│                   Output Frame                       │
└─────────────────────────────────────────────────────┘
```

| Component | Purpose | Param count |
|-----------|---------|-------------|
| MobileNetV3 encoder | Fast feature extraction from person frame | ~5M |
| TPS warping module | Geometric garment deformation (Thin-Plate Spline) | ~2M |
| Fusion decoder | Merge warped garment with person features | ~10M |
| ConvGRU temporal module | Maintain temporal coherence between frames | ~3M |
| **Total** | | **~20M** |

**Target inference**: <33ms per frame on GPU (30 FPS).

#### Phase B3 — Training

**Losses**:

| Loss | Weight | Formula |
|------|--------|---------|
| L1 reconstruction | 1.0 | $\mathcal{L}_1 = \|I_{\text{student}} - I_{\text{teacher}}\|_1$ |
| Perceptual (VGG) | 0.5 | $\mathcal{L}_{\text{perc}} = \sum_l \|\phi_l(I_s) - \phi_l(I_t)\|_2^2$ |
| GAN adversarial | 0.1 | $\mathcal{L}_{\text{adv}} = -\log D(I_s)$ (PatchGAN) |
| Temporal consistency | 0.3 | $\mathcal{L}_{\text{temp}} = \|W(I_s^{t-1}, f_{t-1 \to t}) - I_s^t\|_1$ |

**Total**: $\mathcal{L} = \mathcal{L}_1 + 0.5 \mathcal{L}_{\text{perc}} + 0.1 \mathcal{L}_{\text{adv}} + 0.3 \mathcal{L}_{\text{temp}}$

**Hardware**: 4–8× A100 40GB, ~3–5 days training.

#### Phase B4 — Export & Deployment

| Target platform | Export format | Expected FPS |
|-----------------|---------------|--------------|
| NVIDIA GPU | TensorRT FP16 | ~30–60 |
| Apple M-series | CoreML | ~30 |
| Edge devices | ONNX Runtime | ~15–20 |

### File Structure (Future)

```
services/distillation/
├── __init__.py
├── teacher_generator.py      # Generate training pairs from Plan A
├── student_model.py          # MobileNetV3 + TPS + fusion decoder + ConvGRU
├── training_loop.py          # Knowledge distillation training
├── export.py                 # TensorRT/CoreML export
```

### Alternative: Optimize SwiftTry

Instead of distilling from scratch, optimize an existing model:

| Optimization | SwiftTry FPS | Cumulative |
|-------------|-------------|------------|
| Baseline (512×384) | 2.27 | 2.27 |
| + TensorRT FP16 | ~2× | ~4.5 |
| + Reduced resolution (384×256) | ~1.5× | ~7 |
| + Plan A keyframe strategy | ~2–3× | ~15–20 |

Less novel but faster to implement. Could serve as a comparison baseline.

---

## Paper Contribution (RT-VTON)

### Novel Contributions

1. **Pose-aware keyframe selection for VTON** — Uses garment-relevant landmark subset (shoulders, elbows, hips) instead of generic scene change detection. Triggers VTON only when garment-relevant body geometry changes.

2. **Garment-masked optical flow warping** — Flow applied only within semantic garment mask, extending the two-mask compositing approach to the temporal domain. Non-garment regions (face, skin, background) are always original.

3. **Occlusion-aware fallback** — Forward-backward consistency check detects flow inconsistencies. Occluded regions gracefully fall back to original pixels instead of producing artifacts.

4. **Temporal VTON quality benchmark** — First systematic FPS vs quality tradeoff curve (SSIM, LPIPS, FID) across keyframe intervals for video VTON.

5. **Self-supervised VTON data generation** — Plan A generates unlimited labeled training pairs for Plan B distillation, eliminating manual data curation.

### Comparison Baselines

| Method | FPS | Quality | Notes |
|--------|-----|---------|-------|
| Per-frame FASHN VTON | 0.2 | Highest | Upper bound |
| SwiftTry per-frame | 2.27 | High | BSD-3 open weights |
| Keyframe only (no warp) | ~3 | Popping artifacts | No temporal coherence |
| Keyframe + naive interpolation | ~10 | Blurry transitions | Pixel-space blending |
| **Keyframe + flow warp (Plan A)** | **~10** | **Near-keyframe** | Our method |
| Distilled student (Plan B) | **30** | TBD | Future |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| RAFT-Small too slow on CPU (~5 FPS) | Below target | Reduce to 384×256 or use FastFlowNet |
| Garment tearing at limb boundaries | Visual artifacts | Occlusion mask zeros out; fallback to original |
| Pose tracking loses subject | No keyframe trigger | Force keyframe when pose=None |
| VTON quality varies between keyframes | Temporal flicker | Fixed seed + temporal blending window |
| MPS not supported by RAFT | Can't use M2 GPU | RAFT-Small designed for CPU efficiency |
| Plan B training data quality | Student ceiling | Curate training set; filter low-confidence Plan A results |

---

## Verification Criteria

| Test | Method | Target |
|------|--------|--------|
| Flow accuracy | Synthetic translation, verify pixel error | < 1px mean error |
| Keyframe detection | Synthetic pose sequences (still, slow, fast, lost) | Correct triggers |
| Integration | 3-second test video end-to-end | No black frames, correct count |
| Visual quality | Garment stays plausible during turns | No tearing |
| Temporal coherence | Consecutive frame SSIM in garment region | > 0.85 |
| Performance | End-to-end on M2 CPU | 5–10 FPS offline |
