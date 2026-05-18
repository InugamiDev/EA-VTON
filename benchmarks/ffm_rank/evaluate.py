"""B2 — Body-Shape-Aware Item Ranking evaluator (v2 spec).

CRITICAL CHANGE vs the first draft: the silver label is now derived from a
DIFFERENT signal source than what FFM uses, eliminating the self-evaluation
issue the first version had.

Silver-label source (independent of FFM):
    For each cluster, we define a natural-language archetype prompt
    describing the body shape. We then score each item using CLIP-text-image
    cosine similarity between (archetype_prompt, item.clip_embedding).
    Higher cosine = "CLIP judges this item to suit the archetype".

FFM (the system under test) computes its score from the body-rule polarities,
which are hand-curated from styling literature (Hidayati et al. MM 2018 etc.)
— a DIFFERENT signal source than CLIP text-image alignment. The two signals
can correlate or diverge; whichever, B2 now reports a non-tautological
agreement measure.

Metrics:
    NDCG@10 — relevance bins from silver percentile (top 25% = +2, next 25% = +1, rest = 0)
    Spearman correlation between FFM score and silver score
    Random / color-only / FFM are evaluated against the silver labels
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import duckdb
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "research/datasets/processed/deepfashion2/style_catalog.duckdb"
BUNDLE = ROOT / "research/datasets/processed/dataset_bundle_v1"
OUT = Path(__file__).parent / "results.json"

POOL_SIZE = 500
N_USERS_PER_CLUSTER = 20
RNG = np.random.default_rng(42)

ATTRS = ["neckline", "silhouette", "sleeve", "pattern",
         "color_temperature", "color_value"]


# ── CLIP-archetype prompts per VN cluster ──────────────────────────────────
# These are NATURAL-LANGUAGE descriptions of each cluster — they encode the
# *body shape*, NOT the body-rule polarities. CLIP text-encoding of these
# prompts gives an independent signal source from the body rules.
CLUSTER_ARCHETYPE_PROMPTS = {
    "1": "a flattering top for a petite woman with narrow shoulders and a small frame",
    "2": "a flattering top for a tall slim woman with broad shoulders",
    "3": "a flattering top for a woman with a balanced average body shape",
    "4": "a flattering top for a petite curvy woman with fuller bust and waist",
    "5": "a flattering top for a fuller-figured woman with broad shoulders",
}


def load_rules(population: str) -> dict:
    name = {
        "vn_female": "body_rules_vn.json",
        "us_female": "body_rules_us.json",
        "kr_female": "body_rules_kr.json",
    }[population]
    return json.loads((BUNDLE / name).read_text())


# Silver label: CLIP text-image cosine. We precompute the cluster text vectors
# offline (cached) so the evaluator stays CPU-only.
CACHED_TEXT_EMBEDS = Path(__file__).parent / "cluster_text_embeds.npy"


def get_or_compute_text_embeds() -> dict[str, np.ndarray]:
    if CACHED_TEXT_EMBEDS.exists():
        d = np.load(CACHED_TEXT_EMBEDS, allow_pickle=True).item()
        return d
    # Compute via OpenCLIP — only on first run
    print("  computing cluster archetype CLIP text embeddings (first run)…")
    try:
        import open_clip
        import torch
    except ImportError:
        sys.exit(
            "!! open_clip + torch needed to bootstrap cluster text embeddings. "
            "Install once: pip install open-clip-torch torch"
        )
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    embeds = {}
    with torch.no_grad():
        for cid, prompt in CLUSTER_ARCHETYPE_PROMPTS.items():
            tokens = tokenizer([prompt])
            te = model.encode_text(tokens).squeeze(0).numpy()
            te = te / (np.linalg.norm(te) + 1e-8)
            embeds[cid] = te.astype(np.float32)
    np.save(CACHED_TEXT_EMBEDS, embeds, allow_pickle=True)
    print(f"  cached → {CACHED_TEXT_EMBEDS}")
    return embeds


def silver_clip(item_emb: np.ndarray, cluster_text_emb: np.ndarray) -> float:
    """Cosine similarity between item CLIP image emb and cluster text emb."""
    inorm = np.linalg.norm(item_emb) + 1e-8
    return float(np.dot(item_emb, cluster_text_emb) / inorm)


def ffm_shape_score(rules: dict, cluster: str, item: dict) -> float:
    """Mean polarity over the 6 attributes — the SYSTEM-UNDER-TEST signal."""
    r = rules["rules"][cluster]
    scores = []
    for attr in ATTRS:
        v = item.get(attr)
        if v is None or attr not in r:
            continue
        pol = r[attr].get(v, 0)
        scores.append(pol)
    return float(np.mean(scores)) if scores else 0.0


def relevance_bin(silver: float, p25: float, p75: float) -> int:
    """Percentile-based bin: top 25% = +2, mid 50% = +1, bottom 25% = 0."""
    if silver >= p75:
        return 2
    if silver >= p25:
        return 1
    return 0


def dcg(rel: list[int]) -> float:
    return sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rel))


def ndcg_at_k(rel_ordered: list[int], k: int) -> float:
    rel = rel_ordered[:k]
    ideal = sorted(rel_ordered, reverse=True)[:k]
    d = dcg(rel)
    di = dcg(ideal)
    return d / di if di > 0 else 0.0


def mrr(rel_ordered: list[int]) -> float:
    for i, r in enumerate(rel_ordered):
        if r == 2:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(rel_ordered: list[int], k: int) -> float:
    pos_total = sum(1 for r in rel_ordered if r == 2)
    if pos_total == 0:
        return 0.0
    hits = sum(1 for r in rel_ordered[:k] if r == 2)
    return hits / pos_total


def main() -> None:
    if not DB.exists():
        sys.exit(f"!! catalog DB not found at {DB}")
    con = duckdb.connect(str(DB), read_only=True)
    rules = load_rules("vn_female")
    cluster_ids = list(rules["rules"].keys())
    cluster_text_embeds = get_or_compute_text_embeds()

    # Pool
    rows = con.execute(f"""
        SELECT garment_id, neckline, silhouette, sleeve, pattern,
               color_temperature, color_value, primary_color_lab,
               best_season, clip_embedding
        FROM items
        WHERE clip_embedding IS NOT NULL
        ORDER BY hash(garment_id || 'b2pool42')
        LIMIT {POOL_SIZE}
    """).fetchall()
    cols = [d[0] for d in con.description]
    pool = [dict(zip(cols, r)) for r in rows]
    pool_embs = np.stack([np.asarray(it["clip_embedding"], dtype=np.float32) for it in pool])
    print(f"  pool size: {len(pool)}")

    users = []
    for cl in cluster_ids:
        for i in range(N_USERS_PER_CLUSTER):
            anchor_item = pool[(int(cl) * 137 + i * 31) % len(pool)]
            users.append({
                "cluster": cl,
                "anchor_lab": anchor_item["primary_color_lab"],
            })
    print(f"  users: {len(users)} ({len(cluster_ids)} clusters × {N_USERS_PER_CLUSTER})")

    # ── Silver (independent signal): CLIP text-image cosine per user ──
    silver_matrix = np.zeros((len(users), len(pool)), dtype=np.float32)
    for ui, u in enumerate(users):
        text_emb = cluster_text_embeds[u["cluster"]]
        for ii in range(len(pool)):
            silver_matrix[ui, ii] = silver_clip(pool_embs[ii], text_emb)

    # Build per-user relevance bins by percentile rank
    rel_matrix = np.zeros_like(silver_matrix, dtype=int)
    for ui in range(len(users)):
        sm = silver_matrix[ui]
        p25, p75 = np.percentile(sm, 25), np.percentile(sm, 75)
        rel_matrix[ui] = np.where(sm >= p75, 2, np.where(sm >= p25, 1, 0))

    # ── FFM (system under test): body-rule polarity sum ──
    ffm_matrix = np.zeros_like(silver_matrix, dtype=np.float32)
    for ui, u in enumerate(users):
        for ii, item in enumerate(pool):
            ffm_matrix[ui, ii] = ffm_shape_score(rules, u["cluster"], item)

    # Spearman correlation between FFM and silver (per user, then averaged)
    spear_rhos = []
    for ui in range(len(users)):
        rho, _ = spearmanr(ffm_matrix[ui], silver_matrix[ui])
        if not np.isnan(rho):
            spear_rhos.append(float(rho))

    # ── Baselines ──
    def rank_random(u, ui):
        return list(RNG.permutation(len(pool)))

    def rank_color_only(u, ui):
        anchor = np.array(u["anchor_lab"])
        dists = []
        for it in pool:
            lab = it.get("primary_color_lab")
            d = float(np.linalg.norm(np.array(lab) - anchor)) if lab else 100.0
            dists.append(d)
        return list(np.argsort(dists))

    def rank_clip_oracle(u, ui):
        # Direct silver ordering — the ceiling on this benchmark
        return list(np.argsort(-silver_matrix[ui]))

    def rank_ffm(u, ui):
        # Body-rule polarity sum (this paper's method)
        return list(np.argsort(-ffm_matrix[ui]))

    baselines = [
        ("random",                  rank_random),
        ("color_only",              rank_color_only),
        ("FFM (this paper)",        rank_ffm),
        ("CLIP-archetype oracle",   rank_clip_oracle),  # ceiling
    ]

    results = {}
    for name, ranker in baselines:
        ndcgs, mrrs, recalls = [], [], []
        for ui, u in enumerate(users):
            order = ranker(u, ui)
            rel_ordered = [int(rel_matrix[ui, i]) for i in order]
            ndcgs.append(ndcg_at_k(rel_ordered, 10))
            mrrs.append(mrr(rel_ordered))
            recalls.append(recall_at_k(rel_ordered, 10))
        results[name] = {
            "ndcg@10":   float(np.mean(ndcgs)),
            "mrr":       float(np.mean(mrrs)),
            "recall@10": float(np.mean(recalls)),
        }

    results["_diagnostics"] = {
        "ffm_silver_spearman_rho_mean": float(np.mean(spear_rhos)) if spear_rhos else None,
        "ffm_silver_spearman_rho_std": float(np.std(spear_rhos)) if spear_rhos else None,
        "_doc": (
            "Spearman ρ between FFM score and CLIP-archetype silver score per user, "
            "averaged across users. High ρ = body rules agree with CLIP's notion of "
            "shape-suitability. Low ρ = the two signals diverge meaningfully."
        ),
    }

    OUT.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  wrote {OUT}")

    print()
    print("## B2 — Body-Shape-Aware Item Ranking (CLIP-archetype silver labels)")
    print()
    print(f"Mean Spearman ρ(FFM, silver) = {np.mean(spear_rhos):.3f} ± {np.std(spear_rhos):.3f}")
    print()
    print("| Baseline                   | NDCG@10 |   MRR  | Recall@10 |")
    print("|----------------------------|--------:|-------:|----------:|")
    for name, m in results.items():
        if name.startswith("_"):
            continue
        print(f"| {name:<26s} | {m['ndcg@10']:.4f}  | {m['mrr']:.3f}  | {m['recall@10']:.4f}    |")


if __name__ == "__main__":
    main()
