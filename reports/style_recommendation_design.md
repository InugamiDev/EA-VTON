# Style Recommendation — Design Document

Status: design draft, not yet implemented
Date: 2026-05-03
Owner: Lê Thành Danh
Purpose: Reference for user flow / idealization work

---

## 1. Where the current system stops

What we have today:
- Webcam OR uploaded photo → MediaPipe pose (33 landmarks, 2D + 3D world)
- Body ratios extracted client-side (S/H, W/H, S/T, T/L, A/T)
- Weight estimated from height + ratios
- GBM model predicts size (XXS–XXL) with probability distribution
- Size pivot table maps VN size → US equivalent → real cm (bust/waist/hip ranges)

What's missing:
- We tell the user **what size**, not **what specific garment to wear**
- No body shape categorization beyond "balanced / hourglass / pear / rectangle / inverted_triangle"
- **No catalog** — without items to recommend, we can only output abstract attribute tags
- No outfit suggestion, no color guidance, no occasion context
- No personalization signal (zero user accounts, zero clicks, zero purchases)

**The end goal**: input a person → output a ranked list of real, buyable garments
that fit their body, suit their skin tone, and match the occasion. Each step
below contributes a feature that filters/scores the catalog toward this output.

---

## 2. Inputs available (ranked by reliability)

| Signal | Source | Reliability | Privacy | Already have? |
|---|---|---|---|---|
| Height | User input | High | Trivial | Yes |
| Body ratios (5) | MediaPipe world landmarks | High (3D) / Med (2D) | Local-only | Yes |
| Predicted size | GBM model | High (~83% within-1) | n/a | Yes |
| Body type cluster | Derive from ratios via VN study centroids | Medium | Local | **To build** |
| Skin tone | Face landmark sample → LAB | Medium (lighting-dependent) | Local | **To build** |
| Face shape | MediaPipe Face Mesh (468 pts) | Medium | Local | **To build** |
| Occasion | User picker | High | n/a | **To build** |
| Personal preferences | Click / save / purchase history | Highest (long-term) | DB | **To build (logging)** |
| Color season confirmation | User picker | High | n/a | **To build** |

---

## 3. Outputs we should target (in priority order)

### 3a. Size + size confidence (have it)
Already shown. Keep it.

### 3b. Fit critique (cheap, high-trust)
Compare user measurements against pivoted cm bands.
Example: "Predicted M, but hip estimate at L threshold — size up if M feels tight."

### 3c. Style attribute tags (intermediate signal — used as a filter, not the user-facing output)
Garment-level guidance derived from body type. Used internally to query/filter the catalog.
Example internal representation:
```
recommend:
  - silhouette: A-line, fit-and-flare
  - neckline: V-neck, sweetheart
  - length: knee-length
  - color family: warm earth tones
avoid:
  - oversized boxy cuts
  - high necklines
  - horizontal stripes at hip
```

### 3d. Color palette suggestion
5–8 hex codes derived from skin tone seasonal classification.
User can override; we filter catalog by these.

### 3e. **Concrete garment / catalog items (the actual user-facing output)**
**This is what the user sees.** Top-K real items from a catalog, ranked, with the size already pre-selected.
Example output:
```json
[
  {
    "item_id": "vn_dress_0421",
    "title": "Áo midi cổ V eo cao - tone be",
    "brand": "Local VN brand",
    "image_url": "...",
    "price_vnd": 590000,
    "available_sizes": ["S","M","L"],
    "recommended_size_for_user": "M",
    "match_score": 0.87,
    "match_reasons": [
      "A-line silhouette suits Group 3 medium build",
      "V-neck (matches body-type recommendation)",
      "Warm beige in your seasonal palette",
      "Available in your predicted size M"
    ]
  }
]
```

This is the deliverable. Sections 3a–3d are *features that build up to this*, not standalone outputs.

### 3f. Outfit completion (long-term)
"If she likes this top, suggest these bottoms." Requires labeled outfit data we don't have.

---

## 4. The five approaches (ranked by effort/impact)

### Approach 1 — Body type × occasion rules engine

**Effort**: 3–5 days
**Data needed**: zero (rules from styling literature + Trieu et al. 2024 cluster definitions)
**Output**: structured tags (recommend/avoid lists)

Implementation:
- `body_type_vn.py` — KNN classifier on 5 ratios → one of 5 VN body type clusters
- `data/style_rules_vn.json` — 5 groups × 4 occasions × attribute lists
- Endpoint: `POST /recommend-style` returns tags + one-line "why"

Limit: rules are generic; not personalized.

---

### Approach 2 — CLIP catalog matching

**Effort**: 1 week (after Approach 1)
**Data needed**: a catalog of items (we don't have one yet)
**Output**: top-K item recommendations

Implementation:
- One-time: compute CLIP embeddings for catalog items (~30s per 1k items on M2 CPU)
- Manual: curate 5–10 "exemplar" items per body type cluster
- Runtime: classify body type → cosine similarity to exemplars → top-K
- No training required (pretrained CLIP)

Limit: needs catalog + exemplar curation. Cold-start problem if catalog small.

---

### Approach 3 — Skin tone color recommendation

**Effort**: 2–3 days, parallel track
**Data needed**: zero (seasonal palette tables from color theory)
**Output**: 5–8 hex colors that flatter user

Implementation:
- `skin_tone.py` — sample face landmarks (cheek + forehead pixels) → average LAB
- Map to 4 seasonal palettes (warm/cool × light/deep)
- UI: color swatches next to size prediction; can filter catalog by these

Limit: webcam lighting kills accuracy. Always show with "confirm or override" UI.

---

### Approach 4 — Fit critique from size pivot

**Effort**: 2 days
**Data needed**: the pivot table (already built)
**Output**: per-region fit warnings

Implementation:
- For predicted size, look up cm bands (bust/waist/hip)
- Compare against estimated user measurements
- Generate badges: "Tight at hip," "Loose at waist," etc.

Limit: based on estimated weight → estimated measurements; ~3–5 cm error compounds.

---

### Approach 5 — Two-tower implicit feedback model

**Effort**: 1+ months (needs user data first)
**Data needed**: ~1k+ user interactions (clicks, saves, dwell time)
**Output**: personalized item ranking

Implementation:
- Body embedding tower (CNN on body silhouette OR concat of ratios + size + skin tone)
- Item embedding tower (CLIP image features)
- Train on (user, item, interaction) triples
- Already scaffolded in `services/recommendation/src/item_recommender.py`

Limit: zero training data right now. **The ONLY path forward is to start logging now.**

---

## 4.5 Garment recommendation pipeline (the actual user-facing flow)

This is how the pieces above compose into "give me items I should wear." Each
upstream feature is a **filter or score** applied to the catalog, ending in a
ranked list of real items.

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT                                                      │
│  - Photo (webcam / upload)                                  │
│  - Height                                                   │
│  - Occasion (work / casual / formal / party)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  EXTRACT (per-user signals)                                 │
│  - 5 body ratios (MediaPipe)                                │
│  - VN body type (1..5) from cluster classifier              │
│  - Predicted size (GBM) + cm bands (size pivot)             │
│  - Skin tone (LAB → seasonal palette)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  DERIVE (style attributes for filtering)                    │
│  - Recommended silhouettes / necklines / lengths            │
│  - Avoided silhouettes / necklines                          │
│  - Color palette hex codes                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  CATALOG QUERY (filter)                                     │
│  - Available in user's size                                 │
│  - Tagged with at least 1 recommended attribute             │
│  - Not tagged with any "avoid" attribute                    │
│  - Color in palette (or palette-adjacent)                   │
│  - Price range / brand / occasion filters from user         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  SCORE (rank what's left)                                   │
│  Score = w1·attribute_match + w2·color_match                │
│        + w3·CLIP_sim_to_body_exemplar + w4·popularity       │
│  (Phase 4: + w5·learned_preference from two-tower)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT                                                     │
│  Top-K items with:                                          │
│   - Image, title, price, available sizes                    │
│   - Recommended_size_for_user                               │
│   - 2–4 plain-language match reasons                        │
└─────────────────────────────────────────────────────────────┘
```

### Catalog data model (minimum required per item)

```json
{
  "item_id": "uuid",
  "title": "string",
  "brand": "string",
  "image_url": "string",
  "price_vnd": "int",
  "category": "dress | top | bottom | outerwear | shirt",
  "available_sizes": ["S", "M", "L", "XL"],
  "size_chart_cm": {
    "S": {"bust": 84, "waist": 66, "hip": 91},
    "M": {"bust": 90, "waist": 71, "hip": 97}
  },
  "attributes": {
    "silhouette": "A-line | bodycon | shift | wrap | fit-and-flare",
    "neckline": "V-neck | crew | scoop | sweetheart | high-neck | off-shoulder",
    "length": "mini | knee | midi | maxi",
    "sleeve": "sleeveless | short | three-quarter | long",
    "pattern": "solid | stripe | floral | print"
  },
  "primary_color_hex": "#bc8f6e",
  "secondary_colors_hex": ["#fff", "#000"],
  "occasion_tags": ["work", "casual", "party", "formal", "date"],
  "clip_embedding": "[512-dim float vector]"
}
```

Without this metadata on each catalog item, garment recommendation cannot
work — even with a perfect body model. **Catalog ingestion is the gating
dependency for everything past Phase 2.**

### Three sources for the catalog (pick one, ranked easiest → hardest)

1. **Synthetic / scraped seed catalog** (~200 items, 1–2 days)
   Scrape a few VN brand sites (Coolmate, Ivy Moda, Dottie). Manually tag attributes for ~200 items. Enough to demo the flow.
2. **Brand partner CSV** (~thousands of items, depends on partner)
   One brand provides their full SKU list with attributes. Cleanest data, but needs a deal.
3. **Open dataset stitching** (~10k items, 1 week)
   DeepFashion / DeepFashion2 / Fashionpedia have images + attributes but
   not VN brands or VN sizing. Useful for prototyping the pipeline; not for
   shipping to VN users.

For research / demo: option 1. For real product: option 2.

### What the user sees (end-to-end UI flow)

1. Open app → see camera/upload + height field
2. Capture pose → size prediction appears (existing flow)
3. Pick occasion (chip selector: Work / Casual / Formal / Party)
4. **Recommendations panel populates** with 6–12 item cards
5. Each card: photo, title, price, "M (your size)", 2–3 match reason chips
6. Tap a card → detail view + "Save to favorites" / "Buy on brand site"
7. (Logged: every shown item, every tap, every save → feeds Phase 4)

---

## 4.6 ML / techniques / math per stage

This is the math that makes each pipeline stage work. Every choice listed here
runs on M2 CPU with no GPU, no API costs, no training data we don't already have.

### Stage 1 — Pose extraction (already done)
- **Method**: MediaPipe PoseLandmarker (lite, INT8 GPU/CPU delegate)
- **Math under the hood**: BlazePose architecture — heatmap regression for
  detector + offset regression for tracker. Outputs 33 (x, y, z, visibility).
- **3D world landmarks**: solved via **weak-perspective projection** with a
  prior over body proportions; output in metric coordinates relative to hip
  midpoint.
- **Why this**: only client-side option that gives meter-scale 3D for free.

### Stage 2 — Body type classification (5 VN clusters)
- **Method**: **K-Nearest Neighbors (k=3 or k=5) with Mahalanobis distance**
  in the 5-dim ratio space (S/H, W/H, S/T, T/L, A/T).
- **Math**:
  ```
  d_M(x, μ_c) = √( (x − μ_c)ᵀ Σ⁻¹ (x − μ_c) )
  cluster = argmin_c  d_M(x, μ_c)
  confidence = softmax(−d_M / τ) for temperature τ
  ```
  Σ is the within-cluster pooled covariance estimated from the Trieu et al. 2024
  cluster statistics. Mahalanobis (not Euclidean) because the ratios have
  different scales and are correlated.
- **Alternative considered**: small MLP — overkill for 5 classes from 5 features,
  needs more data than we have.
- **Output**: cluster ∈ {1..5} + confidence ∈ [0, 1].

### Stage 3 — Weight / measurement estimation (already done, refine)
- **Method**: **piecewise linear regression** with separate calibration per
  landmark source (3D world vs 2D image).
- **Math**:
  ```
  w_est = a·h² + b·(S/H − μ_SH) + c·(W/H − μ_WH) + ε
  ```
  Coefficients (a, b, c) and baselines (μ_SH, μ_WH) differ between
  source ∈ {world_3d, image_2d}.
- **Refinement**: replace with **ridge regression** trained on synthetic data
  generated by sampling the VN STEPS Survey marginals and the 5 cluster
  centroids — gives proper coefficients instead of hand-tuned ones.

### Stage 4 — Size prediction (already done)
- **Method**: **Gradient Boosted Trees** (LightGBM / sklearn GBM), 7-class
  classifier with **CORN ordinal loss** option (already implemented in MLP variant).
- **Importance weighting math** (already implemented):
  ```
  w(x) = P_VN(h, w) / P_US(h, w)        ← copula density ratio
  w_PSIS = PSIS_smooth(w)                ← Pareto-smoothed tail
  w_tempered = w_PSIS^α   with α=0.75    ← partial transfer
  loss = Σ w_tempered · cross_entropy(ŷ, y)
  ```
- **CORN loss** for ordinal targets:
  ```
  P(y > k) = σ(z_k)
  P(y = k) = ∏_{j<k} σ(z_j) · (1 − σ(z_k))
  ```
  Penalizes off-by-2 predictions more than off-by-1 — appropriate for size.

### Stage 5 — Skin tone → seasonal palette
- **Method**: **K-Means in CIE LAB color space** with k=4 fixed seasonal centroids.
- **Pipeline**:
  ```
  RGB → linear RGB → XYZ (D65) → LAB
  sample = mean LAB across (cheek, forehead, jaw) face landmarks
  season = argmin_s  || sample − centroid_s ||₂   in LAB
  ```
- **Why LAB not RGB**: LAB is perceptually uniform — Euclidean distance ≈ how
  different colors look to humans. RGB distance is meaningless for color
  similarity.
- **Output**: season ∈ {warm-light, warm-deep, cool-light, cool-deep} →
  lookup table of 5–8 hex codes.

### Stage 6 — Style attribute derivation (rules)
- **Method**: deterministic **lookup table** keyed by (body_type, occasion).
- **Math**: none. Pure if/else from styling literature. Honest about it.
- **Why not ML here**: no training data linking body type to "successful" outfits.
  ML would be a black box wrapping the same domain knowledge.

### Stage 7 — Catalog filtering (hard constraints)
- **Method**: **boolean predicate composition** — SQL or in-memory filter.
  ```
  candidates = catalog.filter(item =>
       item.available_sizes.includes(predicted_size)
    && item.attributes ∩ recommend_set ≠ ∅
    && item.attributes ∩ avoid_set     = ∅
    && color_distance(item.color, palette) < threshold
    && item.occasion_tags.includes(occasion)
  )
  ```
- **Color distance**: ΔE76 in LAB space:
  ```
  ΔE = √( (L₁−L₂)² + (a₁−a₂)² + (b₁−b₂)² )
  ```
  ΔE < 25 ≈ "same color family." More accurate variants (CIEDE2000) exist if
  needed.

### Stage 8 — Catalog ranking (scoring)
This is where the math compounds. Three sub-techniques work together:

#### 8a. Content-based scoring (CLIP embeddings, no training)
- **Method**: pretrained **CLIP image encoder** (ViT-B/32 runs on CPU).
- **Math**:
  ```
  e_item = CLIP_encode(item.image)         ← 512-dim vector
  e_user = mean(CLIP_encode(exemplars[body_type]))   ← centroid of curated examples
  score_clip = cos(e_user, e_item) = (e_user · e_item) / (‖e_user‖·‖e_item‖)
  ```
- **Why CLIP**: it already encodes "fitted blouse with V-neck" semantically
  from billions of (image, caption) pairs. We don't need to train.
- **Math optimization**: pre-normalize embeddings (‖e‖=1) and
  store as float16 → cosine becomes a single dot product, ~1ms for 10k items.

#### 8b. Attribute / color match scoring
- **Method**: **weighted Jaccard similarity** + color ΔE.
  ```
  score_attr = |item.attributes ∩ recommend_set| / |recommend_set|
  score_color = exp(−ΔE_min / σ)    with σ ≈ 15
  ```
- σ controls how strict color matching is. Lower σ = stricter.

#### 8c. Final score (linear combination)
- **Math**:
  ```
  score(item) = w₁·score_attr + w₂·score_color + w₃·score_clip + w₄·log(1 + popularity)
  ```
- **Initial weights** (no data): w₁=0.40, w₂=0.20, w₃=0.30, w₄=0.10.
- **Future**: learn weights via **logistic regression on click-through events**
  (see Stage 10).
- **Output**: rank items by score, return top-K.

### Stage 9 — Diversity reranking (so top-K isn't 10 near-duplicates)
- **Method**: **Maximal Marginal Relevance (MMR)**.
  ```
  MMR(item) = λ·score(item) − (1−λ)·max_{j∈selected} cos(e_item, e_j)
  ```
  Greedy selection: pick highest-MMR item, add to selected, repeat.
- **λ ≈ 0.7** balances relevance vs diversity.
- **Why this matters**: without MMR, top-10 from a fashion catalog is "10 black
  V-neck dresses." MMR forces variety.

### Stage 10 — Personalization (Phase 4, after data)
- **Method**: **Two-tower neural network** with **BPR (Bayesian Personalized
  Ranking) loss** on implicit feedback.
- **Math** (BPR):
  ```
  L = − Σ_{(u, i_pos, i_neg)}  log σ( u·i_pos − u·i_neg )
  ```
  For each positive interaction (clicked, saved), sample a negative (not
  clicked) and push the positive item embedding closer to the user embedding
  in dot-product space.
- **User tower input**: concat( body_ratios, body_type one-hot, predicted_size one-hot,
  skin_tone_LAB, occasion one-hot ) → 2-layer MLP → 64-dim user embedding.
- **Item tower input**: concat( CLIP embedding, attribute multi-hot, color LAB,
  price bucket ) → 2-layer MLP → 64-dim item embedding.
- **Inference**: dot product, sort, top-K.
- **Why BPR not pointwise**: implicit feedback has no negatives — only
  "interacted" vs "didn't see." Pairwise ranking is the right objective.

### Stage 11 — Feedback loop / online learning
- **Method**: **Thompson sampling on a contextual bandit**.
- **Math**: maintain a Beta(α, β) posterior per (body_type, attribute_bucket)
  pair. On click: α += 1. On no-click after impression: β += 1.
  At serve time:
  ```
  for each candidate item:
    sample θ ~ Beta(α, β)
    score = θ · base_score
  ```
- **Why this**: gradually learns which attribute combinations actually convert
  per body type, without a full retrain. Self-corrects rules-engine guesses
  using real data. Cheap to run, cheap to update.

### Stage 12 — Evaluation metrics (how we know it works)
- **Offline (during dev)**:
  - **NDCG@10** — ranking quality assuming we have ground-truth relevance
  - **Recall@K** — fraction of ground-truth positives in top-K
  - **Coverage** — fraction of catalog ever shown across users (avoid filter bubbles)
- **Online (after launch)**:
  - **CTR@K** — clicks ÷ impressions on recommendation card
  - **Save rate** — saves ÷ impressions
  - **Time-to-first-save** — proxy for relevance speed
  - **Size return rate** — bought-then-returned-due-to-fit (gold metric for size accuracy)

### Summary table — what runs where, what costs what

| Stage | Method | Where | Trains on | Latency |
|---|---|---|---|---|
| Pose | BlazePose | Browser (WASM+GPU) | Pretrained | ~15 ms |
| Body type | Mahalanobis kNN | Browser or server | VN cluster centroids | <1 ms |
| Weight est | Piecewise linear | Server | Hand-calibrated | <1 ms |
| Size | LightGBM + copula weighting + CORN | Server | RTR 82k samples | ~5 ms |
| Skin tone | K-Means in LAB | Browser | 4 fixed centroids | ~2 ms |
| Style rules | Lookup table | Server | Styling literature | <1 ms |
| Filter | Boolean predicates + ΔE | Server (SQL or Python) | n/a | ~10 ms / 10k items |
| Score | CLIP cosine + Jaccard + ΔE | Server | Pretrained CLIP | ~50 ms / 10k items |
| Rerank | MMR | Server | n/a | ~5 ms |
| Personalize | Two-tower + BPR | Server | Click logs (Phase 4) | ~10 ms |
| Online | Thompson sampling | Server | Live impressions | <1 ms |

Total inference budget: **~100 ms** end-to-end on M2 CPU with no GPU. Well
inside an interactive UX target (<200 ms feels instant).

---

## 4.7 The Differentiation — what no one else does (the actual contribution)

Before deciding the build, we audited the prior art. Here's what every existing
product/paper does, what each is missing, and where the actual gap is.

### Prior art audit

| Product / paper | What it does well | What it does not do |
|---|---|---|
| **Stitch Fix** | Human stylist + ML on order history | Needs purchase data; vague "we picked this because..." |
| **True Fit / Fit Analytics (Amazon)** | Predict size from prior brand purchases | Needs purchase history; size only, not style |
| **3DLOOK / TrueToForm / Mirrorsize** | Real cm from 2 photos via 3D mesh | Stops at measurement; no style/recommendation layer |
| **Zalando Algorithmic Fashion Companion** | Outfit completion (top → bottom) | Ignores user body; trained on celebrity outfits |
| **Pinterest Lens / Amazon StyleSnap** | Visual search ("find similar to this") | No body-fit awareness; no "what suits YOU" |
| **Glance AI** | Generative outfit on user photo | Black-box "AI styling"; no reasoning shown |
| **Polyvore / POG (Alibaba) / FashionBERT** | Compatibility prediction across items | Body shape ignored entirely |
| **VITON / VITON-HD / DM-VTON / CatVTON** | Render garment on body | Rendering only — not a recommender |
| **CountER / CausalX (counterfactual rec)** | Explainable recs via causal graphs | Generic recs, never applied to fashion + body fit |
| **DeepFashion / DeepFashion2 benchmarks** | Attribute classification, retrieval | Catalog focus, not user focus |
| **ACM Computing Surveys 2024 (modern fashion rec)** | "Body shape personalization is undeveloped due to lack of diverse datasets" | Confirms the gap |

### The five things nobody combines

Each of these exists individually in someone's research; **no shipped product
or paper combines them**. That's our defensible angle.

#### 1. Approximate biomechanical fit simulation (not statistical)

**What everyone does**: predict fit from past purchases ("ran small per N reviews").
**Problem**: requires mountains of review data; cold-start kills it.
**What we do instead**: simulate garment on body geometrically.
- User's MediaPipe 33-landmark + size pivot → approximate body silhouette polygon
- Garment metadata (silhouette type, length category, ease allowance) → parametric drape model:
  - Top: cylinder around chest with ease parameter
  - A-line skirt: truncated cone with hip-anchor + flare angle
  - Pants: two cylinders (thigh + calf) with rise parameter
- For each (item, user) pair compute **stress map** at 6 control points (shoulder, bust, waist, hip, thigh, calf)
- Output: per-region tightness score ∈ [too tight, snug, fitted, loose, too loose]

**Math**:
```
stress(point p) = (body_circ(p) − garment_circ(p) + ease(p)) / body_circ(p)
```
Negative = too tight (garment smaller than body − ease), positive = loose.

**Why this is novel**: physics-grounded fit prediction without training data.
3DLOOK does measurement extraction, not fit simulation. Stitch Fix does
data-driven fit, not geometric. Nobody bridges them on a consumer app.

#### 2. Reasoning DAG visible to the user (radical transparency)

**What everyone does**: "We picked this for you" + nothing, OR a single sentence.
**Problem**: zero trust, no learning loop for the user.
**What we do instead**: every recommendation card opens to its full causal graph.

```
                    [Item: Áo midi terracotta]
                           ▲
        ┌──────────────────┼──────────────────┐
        │                  │                  │
[Body type Group 3]  [Skin tone:warm-deep] [Size M available]
        │                  │                  │
[A-line silhouette]  [Terracotta in palette] [Bust 90-93cm fit]
        │                  │                  │
[Trieu et al. 2024]  [Color theory ref]   [Pivot table]
```

User clicks any node → see source, evidence, confidence. Every claim auditable.

**Why this is novel**: counterfactual explanation papers (CountER, CausalX)
do this for movies/products in research. **Nobody ships it for fashion.**
Reason: it forces honesty. Big platforms can't expose their bias-laden ranking;
we can because we have nothing to hide.

#### 3. Counterfactual try-on at the point of recommendation

**What everyone does**: VITON exists as a separate "try this on" feature, OR
recommendation exists as a separate carousel. They don't compose.
**Problem**: user gets recs but can't visualize → trust gap → no purchase.
**What we do instead**: top-K recommendations come pre-rendered on the user.

- Already have CatVTON / DM-VTON in the stack
- For top 3-5 items, generate try-on render at recommendation time (cached)
- User sees: "Here's the rec | Here's the why | Here's you wearing it"

**Why this is novel**: VITON papers focus on render quality. Fashion rec
papers focus on ranking. **Nobody bundles them in one UI surface.** This is
literally the highest-conversion thing we could ship — "see yourself in it
before buying" — and our existing pipeline is one wiring step away.

#### 4. Steerable style axes (user controls the recommender)

**What everyone does**: implicit feedback ("we'll learn from clicks").
**Problem**: cold start; user has no agency; black box.
**What we do instead**: expose 4-5 sliders that re-rank live.

```
Conservative ●━━━━━━━━━━━━━━━ Bold
Comfortable  ━━━━●━━━━━━━━━━━ Statement
Cool tones   ━━━━━━━━━━●━━━━━ Warm tones
Slim cut     ━━━━━━━━━●━━━━━━ Loose cut
Cover        ━━●━━━━━━━━━━━━━ Show
```

Each slider modifies the score weights:
```
score = base_score
      + α·boldness · novelty(item)
      + β·warmth · color_temperature(item)
      + γ·looseness · ease_amount(item)
```

Move slider → instant re-rank (we have the precomputed scores already).

**Why this is novel**: every recommender treats personalization as something
the system learns about you. **Nobody flips it: let the user steer.** Stitch
Fix has rough "style profile" sliders at signup but they're static. Live
re-rank with these axes is a research/product gap.

#### 5. Population-honest models, exposed as a feature

**What everyone does**: train on whatever data they have (mostly Western),
silently apply globally.
**Problem**: a "Medium" recommended to a Vietnamese user from a US-trained model
is systematically wrong (we proved this — old GBM stuck at XXL).
**What we do instead**: copula-tempered density ratio reweighting for VN→US
transfer, **and we expose it in the UI**.

```
"Trained on 82k US/UK samples, reweighted toward Vietnamese body distribution
(α=0.75 partial transfer). Within-1-size accuracy on VN test users: 83.4%.
[See methodology]"
```

**Why this is novel**: this is the standard-but-not-shipped technique. ACM
survey (2024) explicitly calls out the "lack of diverse datasets" gap.
Showing the methodology IS the differentiator — turns a transparency liability
(small data) into a trust asset (honest method).

### The combined product positioning

Not "AI personal stylist" (overpromised, black box, has 50 competitors).
Instead: **"the only fit assistant that shows its work, simulates clothes on
your actual body, and was honestly built for Vietnamese proportions."**

Tagline candidates:
- "Sizing that explains itself"
- "Try it on before you trust it"
- "Built for you, not just trained on you"

### Why this is defensible against bigger players

| They have | We have | Why we win on this axis |
|---|---|---|
| Millions of purchases | Geometric fit math | They optimize CTR, we optimize fit accuracy |
| Black-box deep models | Reasoning DAG | They can't expose their bias; we can |
| Generic global models | VN-tuned via copula | They serve "Asian" as one bucket; we serve VN specifically |
| Generative styling | Personalized try-on bundled with rec | They show outfits; we show YOU in outfits |
| Implicit personalization | Steerable axes | Faster cold-start, user feels in control |

Big players won't copy because:
- **Stitch Fix** can't run try-on per recommendation (cost) and won't expose reasoning (legal/PR)
- **Amazon** optimizes for purchase volume — fit-first hurts impressions/click
- **Zalando** is European; copula transfer to VN is not their priority
- **3DLOOK** stops at measurement — adding a recommender means becoming a
  competitor to their B2B brand customers

### Roadmap delta (what changes in our phased rollout)

The Phase 1-4 plan stays, but **add these 5 differentiators inline**:

| Phase | Add this differentiator |
|---|---|
| Phase 1 | #2 Reasoning DAG — every rule fires with traceable source |
| Phase 2 | #5 Population-honest UI badge — show methodology link |
| Phase 3 | #1 Geometric fit simulation + #4 Steerable axes (alongside catalog launch) |
| Phase 4 | #3 Bundled try-on render for top-K (uses existing CatVTON pipeline) |

### Research paper potential

This combination is publishable. Suggested paper title:
> *"Geometric Fit Prediction with Counterfactual Try-On and Causal Explanation
> Graphs for Style Recommendation in Under-Represented Populations"*

Targets: RecSys, SIGIR, or KDD applied track. Contributions:
1. Novel geometric fit predictor without training data (vs statistical)
2. Causal DAG explanation framework applied to fashion (vs movies/books)
3. Bundled try-on as recommendation evidence (UX contribution)
4. Steerable axes for cold-start personalization
5. Reproducible domain transfer pipeline for under-served populations

### What this costs us

- Geometric fit math: ~1 week of engineering (no ML training)
- Reasoning DAG: ~3 days of UI + tracking instrumentation in rules engine
- Try-on bundling: ~3 days (we have the model; just precompute + cache)
- Steerable axes: ~2 days (re-rank is already cheap)
- Honesty UI: ~1 day (it's a modal + a markdown explainer)

**Total: ~3 weeks** beyond the baseline Phase 1-3. All five differentiators
fit inside our existing infrastructure (M2 CPU, no GPU, no new training data).

---

## 5. Recommended phased rollout

### Phase 1 — Foundation (Week 1)
- Body type classifier (5-cluster VN study) → add to `/recommend-size/visual` response
- Rules engine + JSON → new `POST /recommend-style` endpoint
- UI: chip-style "recommend" / "avoid" tags below size prediction
- Start logging: every recommendation shown, every interaction (TODO: pick a logging stack — Postgres event table, Loki, or just JSON files for now)

### Phase 2 — Color + fit (Week 2)
- Skin tone extraction from face landmarks
- Seasonal palette suggestion + override UI
- Fit critique badges using size pivot table

### Phase 3 — Catalog (Week 3, if we have catalog data)
- Ingest catalog (CSV / Shopify export / scrape)
- Compute CLIP embeddings
- Hand-curate body type exemplars
- Top-K item ranking

### Phase 4 — Personalization (Month 2+)
- Two-tower model on accumulated interaction data
- Reranker on top of rules engine
- A/B test rules-only vs rules+ML

---

## 6. Data we need to start logging NOW

(Cost of not logging today = "we still have zero data in 2 months")

Minimum event schema:
```
event_id          uuid
user_id           uuid (anon if no account)
session_id        uuid
timestamp         timestamptz
event_type        enum (size_predicted, style_shown, item_clicked, item_saved, item_purchased, feedback_given)
context: {
  source          (webcam | upload)
  height_cm       float
  body_ratios     { sh, wh, st, tl, at }
  body_type_vn    int (1..5) | null
  predicted_size  string
  occasion        string | null
  skin_tone_lab   { L, a, b } | null
  shown_items     [item_id, ...] | null
  clicked_item_id uuid | null
}
```

This is enough to retroactively train any of the future ML approaches.

---

## 7. What to skip (and why)

| Idea | Why skip |
|---|---|
| Custom CNN body encoder trained from scratch | No labels, no GPU, ~100h work for worse-than-CLIP results |
| Outfit completion ("if she wears X, suggest Y") | Needs labeled outfit pairs we don't have |
| Trend prediction (what's "in" right now) | Needs time-series purchase data |
| Generative styling ("AI designs an outfit for you") | Different problem; needs diffusion model + GPU |
| Promising "AI personalization" in UI copy | Until we have feedback data, it's rule-based — claiming AI is dishonest |
| Multi-brand size translation | Each brand's chart deviates ±3cm from ASTM; would need per-brand calibration |

---

## 8. Open questions for the user flow design

These need decisions before user flow can be drawn:

1. **Onboarding shape** — do we ask occasion first (top-down: "going to a wedding") or body first (bottom-up: "here's me, what fits")?
2. **One-shot vs ongoing** — is each session standalone, or do we build a saved profile (closet, preferences)?
3. **Anonymous vs account** — can someone use the full flow without signing up? Where does "save look" force a signup wall?
4. **Override authority** — when our rec contradicts user picks (e.g. they want oversized but body type says "avoid"), do we hide the conflict or surface it?
5. **Style discovery vs validation** — does the UI lead with "here's what suits you" (we decide) or "browse and we'll filter" (they decide)?
6. **Feedback capture** — thumbs up/down per recommendation? Save-to-favorites only? Open-ended "tell us why"?
7. **Multi-photo** — do we let users upload multiple photos (front + side, multiple outfits) and aggregate? Or strict one-shot?
8. **Locale switching** — VN / US / UK toggle for size labels in the output? (We have the pivot table for this.)
9. **Body type confidence** — when classifier is unsure (between two clusters), show both? Show closest with low confidence? Force a manual pick?
10. **Catalog source** — are we recommending from our own catalog, or are we a meta-tool that links out to brands? Changes everything about the flow.

---

## 9. Files referenced / to create

Existing:
- `services/recommendation/src/app.py` — `/recommend-size/visual` endpoint
- `services/recommendation/src/research_models.py` — GBM loader
- `apps/web/src/hooks/use-body-ratios.ts` — ratio extraction
- `apps/web/src/hooks/use-pose-detector.ts` — MediaPipe wrapper
- `apps/web/src/hooks/use-image-pose.ts` — one-shot upload pose
- `reports/size_rosetta_stone.png` — VN→US→cm pivot visualization
- `reports/size_rosetta_stone.py` — pivot table source

To create (Phase 1):
- `services/recommendation/src/body_type_vn.py` — 5-cluster classifier
- `services/recommendation/src/style_rules.py` — rules engine
- `services/recommendation/data/style_rules_vn.json` — rules table
- `services/recommendation/data/vn_body_type_centroids.json` — cluster centroids from Trieu et al. 2024
- `apps/api/src/routes/style.js` — gateway proxy
- `apps/web/src/lib/api.ts` — `getStyleRecommendation()` client
- `apps/web/src/components/style-tags.tsx` — recommend/avoid chip UI

To create (Phase 2):
- `services/recommendation/src/skin_tone.py`
- `services/recommendation/data/seasonal_palettes.json`
- `apps/web/src/components/color-palette.tsx`
- `apps/web/src/components/fit-critique.tsx`

To create (logging, immediate):
- DB migration for `style_events` table (or use a simpler JSON log if no DB)
- `apps/web/src/lib/analytics.ts` — event firing helper
- `apps/api/src/routes/events.js` — ingest endpoint

---

## 10. Reference data sources

- Vietnam STEPS Survey 2009/2015 — population height/weight/waist/hip averages
- Trieu et al. 2024 — 5 body type clusters from N=480 VN women (TUL DSpace)
- Huyen et al. — N=593 3D Vietnamese female scans (Hanoi UST, not public)
- ASTM D5585 — US women's body measurements per numeric size
- ASTM D6240 — US men's body measurements
- User-provided VN ↔ US ↔ UK conversion tables (used in pivot)

---

## 11. Honest summary for the user flow designer

The system can today:
- Extract body shape and predict a size from a photo
- Tell you what cm range that size means in real-world terms

The system cannot today:
- Tell you what to wear specifically
- Show you items (no catalog)
- Personalize beyond hardcoded rules

Three weeks of work + a small catalog gets us to a credible style assistant. The user flow should assume this — design for the world where rules + body type + skin tone + size pivot all exist, but personalization is still ~6 months out.

The biggest UX question: do we frame ourselves as **"size + fit assistant"** (low ambition, high trust) or **"AI stylist"** (high ambition, currently overpromising)? My recommendation: position as the former, evolve into the latter once the logging data accumulates.

---

## 12. Implementation Plan v1 — Upper Body, Vietnamese Women (small scope)

This is the first concrete build. All other categories (bottoms, dresses,
outerwear, men's) are deferred until v1 ships and we validate the
Fit-Flatter-Match decomposition empirically.

### 12.1 Scope (intentionally narrow)

| Axis | v1 scope | Out of scope |
|---|---|---|
| Garment category | **Upper body only** (top, blouse, shirt, t-shirt) | Dresses, bottoms, outerwear, accessories |
| Population | **Vietnamese women** | Men, other markets |
| Age | 18–50 (matches Trieu et al. 2024) | Children, seniors |
| Sizes | XS–XXL (covers ~95% of VN distribution) | Petites, plus-plus |
| Inputs | Height + 5 body ratios + occasion + sliders | Skin tone (parallel track), face shape |

Why upper body first:
- Fewer landmarks needed (shoulder, bust, waist — already extracted)
- Drape model is simpler (cylinder + cone, no leg geometry)
- Largest VN apparel market segment
- Validates the math before extending

### 12.2 The three-component scoring (formal)

$$
s(u, i, c) = \alpha \cdot \text{Fit}(b_u, i) + \beta \cdot \text{Flatter}(b_u, a_i) + \gamma \cdot \text{Match}(a_i, c)
$$

Initial weights (no data yet): $\alpha=0.4, \beta=0.3, \gamma=0.3$.
Hard constraint: $\text{Fit} > 0.4$ and item-size availability.

#### Fit (geometric, no training)

Per body section $k \in \{\text{shoulder}, \text{bust}, \text{waist}\}$:
$$
\text{stress}_k = \frac{C_{u,k} - (g_{i,k} + \text{ease}_k) \cdot (1 + \text{stretch}_i)}{C_{u,k}}
$$

User circumferences $C_{u,k}$ from VN size pivot lookup (predicted size → cm bands).
Garment $g_{i,k}$ from item's size chart for the user's predicted size.

$$
\text{Fit} = \exp\left(-\frac{1}{|K|} \sum_{k} w_k \cdot \max(0, |\text{stress}_k - \tau_k|)^2 / \sigma^2\right)
$$

with intent-aware setpoint $\tau_k$ depending on garment drape class:
- bodycon: $\tau = -0.05$ (snug)
- fitted: $\tau = 0.00$
- semi-fitted: $\tau = +0.05$
- A-line/loose: $\tau = +0.10$
- oversized: $\tau = +0.20$

#### Flatter (rules-based with VN cluster prior)

5 body type clusters from Trieu et al. 2024:

| Group | % | Description | Recommend (upper body) | Avoid |
|---|---|---|---|---|
| 1 | 15% | Short, thin, small-shouldered | Structured shoulders, puff sleeves, light colors, horizontal stripes | Drop-shoulder, oversized, deep V |
| 2 | 18% | Tall, slim, large-shoulders | V-neck, soft shoulders, vertical lines | Boat neck, puff sleeve, horizontal stripes |
| 3 | 36% | Medium build (modal) | Most styles work; fitted shows shape | Few constraints |
| 4 | 22% | Short, fuller, medium-shoulders | V-neck, vertical seams, dark tones, three-quarter sleeve | Boat neck, crop top, horizontal stripes at bust |
| 5 | 9% | Heavy, big shoulders | Deep V, soft fabrics, vertical stripes, dark/cool tones | Boat neck, puff sleeve, light bright colors at shoulder |

Score:
$$
\text{Flatter}_{\text{rule}} = \frac{1}{|R|}\sum_{a \in a_i \cap R^+} +1 + \sum_{a \in a_i \cap R^-} -1
$$

normalized to $(0, 1]$ via shifted sigmoid.

#### Match (occasion + steerable)

Hard filter on occasion tags. Soft score from style sliders projected onto item attributes:
$$
\text{Match} = \mathbb{1}[c.\text{occ} \in i.\text{occ\_tags}] \cdot \left(0.5 + 0.5 \cdot \langle \theta, \phi_i \rangle / k\right)
$$

### 12.3 Files to ship

| Path | Purpose | Status |
|---|---|---|
| `services/recommendation/data/vn_body_centroids_v1.json` | 5-cluster centroids in (height, ratios) space | **TO CREATE** |
| `services/recommendation/data/upper_body_rules_vn_v1.json` | Cluster × attribute polarity matrix | **TO CREATE** |
| `services/recommendation/data/upper_catalog_seed_v1.json` | ~10-item seed catalog for testing | **TO CREATE** |
| `services/recommendation/src/body_type_vn.py` | Mahalanobis kNN classifier | **TO CREATE** |
| `services/recommendation/src/fit_upper.py` | Parametric fit math (upper body) | **TO CREATE** |
| `services/recommendation/src/style_upper.py` | Composition + explanation DAG | **TO CREATE** |
| `services/recommendation/src/app.py` | Add `POST /recommend-style/upper` endpoint | UPDATE |

### 12.4 API contract

**Request**
```json
{
  "height_cm": 158,
  "predicted_size": "M",
  "ratios": {"sh": 1.10, "wh": 0.78, "st": 0.65, "tl": 0.45, "at": 1.10},
  "context": {
    "occasion": "work",
    "sliders": {"bold": 0.0, "loose": 0.2, "warm": 0.5, "cover": 0.5}
  },
  "top_k": 10,
  "catalog": "seed_v1"
}
```

**Response**
```json
{
  "user_state": {
    "body_type_vn": {"group": 3, "label": "medium build", "confidence": 0.81},
    "size_bands_cm": {"shoulder": 38, "bust": 90, "waist": 71}
  },
  "items": [
    {
      "item_id": "...",
      "title": "...",
      "score": 0.82,
      "rank": 1,
      "explanation": {
        "fit": {"score": 0.88, "sections": [...]},
        "flatter": {"score": 0.75, "rules_fired": [...]},
        "match": {"score": 0.85, "occasion_ok": true, "slider_proj": 0.6}
      }
    }
  ]
}
```

### 12.5 Acceptance criteria (when v1 is "done")

- [ ] Endpoint returns in <200ms for 10-item catalog on M2 CPU
- [ ] Each item has full explanation DAG (fit + flatter + match breakdown)
- [ ] Body type classifier confidence on synthetic users matches expected cluster ≥80% of the time
- [ ] Fit score correlates with actual size chart deviation (manual spot-check on 5 items)
- [ ] No item with `Fit < 0.4` appears in top-K
- [ ] Slider movement causes re-rank (verify via curl with different slider values)

### 12.6 What's deferred to v2+

- Skin tone integration → after upper-body v1 ships
- Try-on rendering bundling → after we wire CatVTON to recommendation flow
- Frontend UI for style results → after backend math is validated
- FM correction term for Flatter → after we have ≥1k logged interactions
- Bottoms / dresses → after upper-body validates the decomposition
- Men's track → after women's v1 has users

---
