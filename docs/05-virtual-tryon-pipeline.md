# Virtual Try-On Pipeline — Mathematical Analysis

## Pipeline Architecture

```
Person Image  ──→  Preprocessor  ──→  Inference Backend  ──→  Postprocessor  ──→  Result
Garment Image ──→  Preprocessor  ──/                          (Quality Score)
```

Three stages, each with specific mathematical operations.

## Stage 1: Preprocessing

### 1.1 Person Image Processing

#### EXIF Correction
Apply rotation matrix based on EXIF orientation tag to ensure consistent image orientation.

#### Skin Detection (HSV Space)

Convert RGB → HSV. A pixel $(H, S, V)$ is classified as skin if:

$$\text{skin}(H, S, V) = \begin{cases} 1 & \text{if } (H < 0.1 \lor H > 0.9) \land S \in [0.08, 0.75] \land V \in [0.15, 0.95] \\ 0 & \text{otherwise} \end{cases}$$

Where $H, S, V \in [0, 1]$. The hue constraint captures the red-yellow range wrapping around $H=0$.

**Skin ratio**: $\rho_{\text{skin}} = \frac{\sum \text{skin}(H_i, S_i, V_i)}{W \times H}$

Used for auto-cropping: find the vertical band containing the most skin pixels to isolate the upper body.

#### Auto-Crop Upper Body

1. Divide image into vertical thirds
2. Find the third with highest skin ratio → this is the torso region
3. Crop to upper 60% of image height, centered on the detected region
4. Heuristic: if skin ratio < 0.05, skip cropping (likely no person detected)

#### Resize to Target Resolution

Target: 768 × 1024 pixels (3:4 aspect ratio, standard for VTON models).

Resize with aspect-ratio preservation + padding:

$$s = \min\left(\frac{768}{W_{\text{src}}}, \frac{1024}{H_{\text{src}}}\right)$$

$$W_{\text{new}} = \lfloor W_{\text{src}} \cdot s \rfloor, \quad H_{\text{new}} = \lfloor H_{\text{src}} \cdot s \rfloor$$

Padding: centered on neutral gray (#808080) canvas.

#### Brightness Normalization

Target mean pixel value: $\mu_{\text{target}} = 130$ (mid-range).

$$\alpha = \frac{\mu_{\text{target}}}{\mu_{\text{image}} + \epsilon}$$

Clamped to $\alpha \in [0.7, 1.4]$ to prevent extreme correction. Applied as:

$$I_{\text{norm}} = \text{clip}(\alpha \cdot I_{\text{src}}, 0, 255)$$

### 1.2 Garment Image Processing

#### Background Removal

Edge-based approach using Sobel gradients:

1. **Edge detection**: $G_x = I * S_x$, $G_y = I * S_y$ where $S_x, S_y$ are Sobel kernels
2. **Edge magnitude**: $G = \sqrt{G_x^2 + G_y^2}$
3. **Corner sampling**: Average color of 4 corners (each 10% of image dimensions)
4. **Color similarity**: $\text{bg}(p) = \|I(p) - c_{\text{corner}}\|_2 < \theta$
5. **Morphological cleanup**: Erosion → dilation to remove noise from mask

#### Garment Mask Generation

Binary mask $M$ where:

$$M(p) = \begin{cases} 1 & \text{if pixel is garment (foreground)} \\ 0 & \text{if pixel is background} \end{cases}$$

Used by VTON models to know where the garment is in the source image.

## Stage 2: Inference

### Backend Options

1. **FASHN VTON v1.5** (target): MMDiT architecture, 972M params, pixel-space, maskless
2. **Local Composite** (fallback): Alpha-blended overlay with perspective transform

#### Local Composite Pipeline

When no VTON model is available, the system creates a basic composite:

1. **Resize garment** to fit detected body region
2. **Perspective transform** to approximate body curvature
3. **Alpha blend**: $I_{\text{result}} = \alpha \cdot I_{\text{garment}} + (1 - \alpha) \cdot I_{\text{person}}$
4. **Edge feathering**: Gaussian blur on mask edges (σ = 3px)

Confidence for composite: 0.65 (lower than model-based inference).

## Stage 3: Postprocessing

### Color Correction

Per-channel mean matching with conservative blending:

For each channel $c \in \{R, G, B\}$:

$$\mu_c^{\text{orig}} = \text{mean}(I_{\text{original}}^c), \quad \mu_c^{\text{result}} = \text{mean}(I_{\text{result}}^c)$$

$$r_c = \text{clip}\left(\frac{\mu_c^{\text{orig}}}{\mu_c^{\text{result}} + \epsilon}, 0.8, 1.2\right)$$

$$I_{\text{corrected}}^c = \text{clip}(0.7 \cdot r_c \cdot I_{\text{result}}^c + 0.3 \cdot I_{\text{result}}^c, 0, 255)$$

The 0.7/0.3 blend (30% original, 70% corrected) prevents over-correction while reducing color cast from the VTON model.

### Sharpening

Applied via PIL `ImageEnhance.Sharpness` with factor 1.15:

$$I_{\text{sharp}} = 1.15 \cdot I_{\text{corrected}} - 0.15 \cdot \text{blur}(I_{\text{corrected}})$$

Mild sharpening to counteract the softness typical of diffusion model outputs.

### Quality Scoring

See [06-quality-scoring.md](./06-quality-scoring.md) for full mathematical details.

## End-to-End Latency Budget

| Stage | Target | Notes |
|-------|--------|-------|
| Preprocessing | < 1s | CPU-bound, PIL/numpy operations |
| Inference (FASHN) | < 10s | GPU with LCM-LoRA (4-8 steps) |
| Inference (composite) | < 0.5s | CPU-only fallback |
| Postprocessing | < 1s | Quality scoring + corrections |
| **Total** | **< 12s** | End-to-end with VTON model |
