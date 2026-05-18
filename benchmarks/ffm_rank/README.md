# Benchmark B2 — Body-Shape-Aware Item Ranking

Given a user `(body_cluster, predicted_size, occasion)` and a 500-item candidate pool, produce a ranked list. Higher = better suited to the user's body shape.

## Task

**Input** per user:
- `body_cluster_id` ∈ {1..5} (Vietnamese taxonomy by default; multi-population if `population_id` provided)
- `predicted_size` ∈ {XS, S, M, L, XL, XXL}
- `occasion` ∈ {casual, work, date, party, formal}

**Output**: ordered list of item_ids from the 500-item pool.

## Silver labels (independent signal source)

**Why independent?** Earlier draft used the body-rule polarities directly as the silver label. That made FFM (the proposed system, which scores via the *same* body rules) tautologically tie the oracle at NDCG=1.000. An independent auditor flagged this self-evaluation issue.

**Current design**: silver labels come from a different signal source — **CLIP text-image cosine similarity** between a natural-language cluster archetype prompt and each item's image embedding. For example, for VN cluster 1 ("short_thin_small_shoulder"), the archetype prompt is:

> "a flattering top for a petite woman with narrow shoulders and a small frame"

We encode this with OpenCLIP ViT-B-32 (the same text encoder used to derive the item embeddings) and compute cosine similarity per item.

For each user, items are bucketed into 3 relevance bins by percentile rank of their silver score:
- Top 25% → relevance 2 (positive)
- Middle 50% → relevance 1 (neutral)
- Bottom 25% → relevance 0 (negative)

This means FFM and the CLIP-oracle baseline are now operating on **two genuinely different signal sources** (body-rule polarity vs. CLIP-text alignment). They can correlate or diverge; whichever, the comparison is non-tautological.

## Metrics

| Metric | Description |
|---|---|
| **NDCG@10** | Normalized Discounted Cumulative Gain at K=10 |
| **MRR** | Mean Reciprocal Rank of the first positive item |
| **Recall@10** | Fraction of positive items captured in top-10 |
| **Spearman ρ(FFM, silver)** | Per-user rank correlation, averaged. Measures agreement between body rules and CLIP archetypes. |

## Baselines

| Baseline | Description |
|---|---|
| **random** | Uniform shuffle of the 500 candidates |
| **color_only** | Score by Euclidean LAB distance to a per-user palette anchor |
| **FFM (this paper)** | Body-rule polarity sum from population-specific JSON |
| **CLIP-archetype oracle** | Direct silver ordering — ceiling (1.000 by construction) |

## Baseline numbers (v3 paper, n_users=100, pool=500, seed=42)

| Baseline                   | NDCG@10 |   MRR  | Recall@10 |
|----------------------------|--------:|-------:|----------:|
| random                     | 0.419   | 0.432  | 0.021     |
| color_only                 | 0.431   | 0.515  | 0.021     |
| **FFM (this paper)**       | **0.480** | **0.525** | **0.026** |
| CLIP-archetype oracle (ceiling) | 1.000 | 1.000 | 0.080 |

**Mean Spearman ρ(FFM, silver) = 0.157 ± 0.059** — body rules and CLIP archetypes have small but non-trivial agreement.

## Honest reading

- **There is real headroom.** FFM at 0.480 NDCG@10 is well below the 1.000 ceiling; future learned methods have somewhere to go.
- **FFM beats random by +0.06 NDCG@10 and +24% Recall@10** (0.026 vs 0.021).
- **The body rules ≠ CLIP archetypes.** ρ=0.157 means the two signal sources don't trivially agree — a useful empirical finding that the hand-curated body rules encode different information from CLIP's vision-language alignment.

## Reproducing

```bash
python benchmarks/ffm_rank/evaluate.py
```

First run computes 5 cluster archetype prompt embeddings via OpenCLIP and caches them to `cluster_text_embeds.npy` (~10 KB). Subsequent runs use the cache and complete in <30s.

## Files in this directory

- `evaluate.py` — runs the four baselines + silver-label derivation.
- `cluster_text_embeds.npy` — cached CLIP text embeddings for the 5 cluster prompts (committed for reproducibility).
- `results.json` — output of `evaluate.py`.
- `README.md` — this file.
