"""Cross-population transfer experiment for the multi-population body rules.

The v3 paper claims the multi-population layer (VN/US/KR body-rule JSONs) is
operationally meaningful. This experiment quantifies that:

For each population P_train and P_eval pair, we:
  1. Score 500 items using P_train's body rules (FFM)
  2. Compute CLIP-archetype silver labels for users of population P_eval
  3. Report NDCG@10 + Spearman ρ between FFM(P_train) and silver(P_eval)

If matched (P_train == P_eval) outperforms mismatched populations, the
multi-population schema is operationally meaningful — different populations
need different rules. If they all look the same, the schema is window-dressing.

Output: research/eval/crosspop_transfer.json + a 3×3 transfer-matrix markdown table.
"""

# intent: prove the multi-population body-rule schema is empirically meaningful
# status: done
# next: report results in §5 of the paper
# confidence: high

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
OUT = ROOT / "research/eval/crosspop_transfer.json"
CACHE = ROOT / "benchmarks/ffm_rank/cluster_text_embeds.npy"

POOL_SIZE = 500
N_USERS_PER_CLUSTER = 20
RNG = np.random.default_rng(42)

ATTRS = ["neckline", "silhouette", "sleeve", "pattern",
         "color_temperature", "color_value"]

# Archetype prompts per population per cluster, encoded once.
# Per-population text prompts so cross-pop silver labels are population-specific.
ARCHETYPES = {
    "vn_female": {
        "1": "a flattering top for a petite woman with narrow shoulders and a small frame",
        "2": "a flattering top for a tall slim woman with broad shoulders",
        "3": "a flattering top for a woman with a balanced average body shape",
        "4": "a flattering top for a petite curvy woman with fuller bust and waist",
        "5": "a flattering top for a fuller-figured woman with broad shoulders",
    },
    "us_female": {
        "1": "a flattering top for a woman with an hourglass figure and defined waist",
        "2": "a flattering top for a woman with a smaller bust and fuller hips",
        "3": "a flattering top for a woman with a narrow waist and prominent hips",
        "4": "a flattering top for a woman with a straight rectangular figure",
        "5": "a flattering top for a woman with broad shoulders and narrower hips",
    },
    "kr_female": {
        "1": "a flattering top for a petite woman with a straight rectangular frame",
        "2": "a flattering top for a petite woman with a defined hourglass waist",
        "3": "a flattering top for a petite woman with narrow upper body and fuller lower",
        "4": "a flattering top for a petite woman with broader shoulders",
    },
}


def load_rules(population: str) -> dict:
    name = {
        "vn_female": "body_rules_vn.json",
        "us_female": "body_rules_us.json",
        "kr_female": "body_rules_kr.json",
    }[population]
    return json.loads((BUNDLE / name).read_text())


def encode_archetypes(populations: list[str]) -> dict[str, dict[str, np.ndarray]]:
    print("  encoding archetype prompts per (population, cluster)…")
    try:
        import open_clip
        import torch
    except ImportError:
        sys.exit("!! pip install open-clip-torch torch")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    out = {}
    with torch.no_grad():
        for pop in populations:
            out[pop] = {}
            for cid, prompt in ARCHETYPES[pop].items():
                tokens = tokenizer([prompt])
                e = model.encode_text(tokens).squeeze(0).numpy()
                e = e / (np.linalg.norm(e) + 1e-8)
                out[pop][cid] = e.astype(np.float32)
    return out


def ffm_score(rules: dict, cluster_id: str, item: dict) -> float:
    """Body-rule polarity sum."""
    r = rules["rules"].get(cluster_id)
    if r is None:
        return 0.0
    scores = []
    for attr in ATTRS:
        v = item.get(attr)
        if v is None or attr not in r:
            continue
        scores.append(r[attr].get(v, 0))
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(rel_ordered: list[int], k: int) -> float:
    def dcg(r):
        return sum((2 ** x - 1) / np.log2(i + 2) for i, x in enumerate(r))
    rel = rel_ordered[:k]
    ideal = sorted(rel_ordered, reverse=True)[:k]
    d, di = dcg(rel), dcg(ideal)
    return d / di if di > 0 else 0.0


def main() -> None:
    if not DB.exists():
        sys.exit(f"!! catalog DB not found at {DB}")
    con = duckdb.connect(str(DB), read_only=True)
    populations = ["vn_female", "us_female", "kr_female"]
    rules_by_pop = {p: load_rules(p) for p in populations}
    text_embeds = encode_archetypes(populations)

    rows = con.execute(f"""
        SELECT garment_id, neckline, silhouette, sleeve, pattern,
               color_temperature, color_value, primary_color_lab,
               clip_embedding
        FROM items
        WHERE clip_embedding IS NOT NULL
        ORDER BY hash(garment_id || 'crosspop42')
        LIMIT {POOL_SIZE}
    """).fetchall()
    cols = [d[0] for d in con.description]
    pool = [dict(zip(cols, r)) for r in rows]
    pool_embs = np.stack([np.asarray(it["clip_embedding"], dtype=np.float32) for it in pool])
    print(f"  pool size: {len(pool)}")

    # Build per-population synthetic users: for each cluster of P_eval, 20 users
    eval_users_by_pop: dict[str, list[dict]] = {}
    for pop in populations:
        cluster_ids = list(rules_by_pop[pop]["rules"].keys())
        users = []
        for cl in cluster_ids:
            for i in range(N_USERS_PER_CLUSTER):
                users.append({"cluster": cl})
        eval_users_by_pop[pop] = users
        print(f"  {pop}: {len(users)} eval users across {len(cluster_ids)} clusters")

    # Silver-label matrix per (eval_pop, user_idx, item_idx)
    silver_by_pop: dict[str, np.ndarray] = {}
    for pop in populations:
        users = eval_users_by_pop[pop]
        M = np.zeros((len(users), len(pool)), dtype=np.float32)
        for ui, u in enumerate(users):
            text_emb = text_embeds[pop][u["cluster"]]
            for ii in range(len(pool)):
                inorm = np.linalg.norm(pool_embs[ii]) + 1e-8
                M[ui, ii] = float(np.dot(pool_embs[ii], text_emb) / inorm)
        silver_by_pop[pop] = M

    # Build per-population relevance bins
    rel_by_pop = {}
    for pop, M in silver_by_pop.items():
        R = np.zeros_like(M, dtype=int)
        for ui in range(M.shape[0]):
            p25, p75 = np.percentile(M[ui], 25), np.percentile(M[ui], 75)
            R[ui] = np.where(M[ui] >= p75, 2, np.where(M[ui] >= p25, 1, 0))
        rel_by_pop[pop] = R

    # For each (train, eval) pair we need to map an eval-cluster to the most
    # similar train-cluster. The previous version used an INDEX-based mapping,
    # which conflates clusters that share an index but differ anthropometrically
    # (e.g., "VN cluster 3 = balanced average" ↔ "US cluster 3 = spoon = narrow
    # waist, prominent hips"). That bias inflated mismatched scores and produced
    # the eval-VN NDCG anomaly the auditor flagged.
    #
    # Fixed mapping: cluster X (eval) → cluster Y (train) where Y's archetype CLIP
    # text embedding has highest cosine similarity to X's. This is the principled
    # cross-population alignment: clusters map by *shape similarity*, not by index.
    def best_cluster_map(eval_pop: str, train_pop: str) -> dict[str, str]:
        eval_clusters = list(rules_by_pop[eval_pop]["rules"].keys())
        train_clusters = list(rules_by_pop[train_pop]["rules"].keys())
        mapping: dict[str, str] = {}
        for ec in eval_clusters:
            ev = text_embeds[eval_pop][ec]
            best = None
            best_sim = -2.0
            for tc in train_clusters:
                tv = text_embeds[train_pop][tc]
                sim = float(np.dot(ev, tv))  # already L2-normed
                if sim > best_sim:
                    best_sim = sim
                    best = tc
            mapping[ec] = best
        return mapping

    # Precompute mappings once per (train, eval) pair
    cluster_maps: dict[tuple[str, str], dict[str, str]] = {}
    for train_pop in populations:
        for eval_pop in populations:
            cluster_maps[(train_pop, eval_pop)] = best_cluster_map(eval_pop, train_pop)
            if train_pop != eval_pop:
                m = cluster_maps[(train_pop, eval_pop)]
                print(f"  map {eval_pop} → {train_pop}: " +
                      ", ".join(f"{e}→{m[e]}" for e in m))

    def map_cluster(eval_pop: str, train_pop: str, eval_cluster: str) -> str:
        return cluster_maps[(train_pop, eval_pop)][eval_cluster]

    transfer_matrix = {}
    for train_pop in populations:
        for eval_pop in populations:
            users = eval_users_by_pop[eval_pop]
            R = rel_by_pop[eval_pop]
            M = silver_by_pop[eval_pop]
            ndcgs, rhos = [], []
            for ui, u in enumerate(users):
                # FFM with P_train's rules, using mapped cluster
                mapped = map_cluster(eval_pop, train_pop, u["cluster"])
                ffm = np.array([ffm_score(rules_by_pop[train_pop], mapped, item) for item in pool])
                order = np.argsort(-ffm)
                rel_ordered = [int(R[ui, i]) for i in order]
                ndcgs.append(ndcg_at_k(rel_ordered, 10))
                rho, _ = spearmanr(ffm, M[ui])
                if not np.isnan(rho):
                    rhos.append(float(rho))
            transfer_matrix[f"{train_pop}__{eval_pop}"] = {
                "ndcg@10_mean": float(np.mean(ndcgs)),
                "ndcg@10_std":  float(np.std(ndcgs)),
                "spearman_rho_mean": float(np.mean(rhos)) if rhos else None,
                "spearman_rho_std":  float(np.std(rhos)) if rhos else None,
                "is_matched": train_pop == eval_pop,
            }

    OUT.write_text(json.dumps(transfer_matrix, indent=2, default=float))
    print(f"\n  wrote {OUT}")

    # Markdown 3×3 table
    print()
    print("## Cross-Population Transfer — NDCG@10 (eval-row × train-col)")
    print()
    print("|             |  trained on VN  |  trained on US  |  trained on KR  |")
    print("|-------------|----------------:|----------------:|----------------:|")
    for eval_pop in populations:
        cells = []
        for train_pop in populations:
            key = f"{train_pop}__{eval_pop}"
            v = transfer_matrix[key]
            marker = "**" if v["is_matched"] else ""
            cells.append(f"{marker}{v['ndcg@10_mean']:.3f}{marker}")
        print(f"| eval {eval_pop:<8s} | {cells[0]} | {cells[1]} | {cells[2]} |")
    print()
    print("(Bold = matched train/eval pair.)")

    print()
    print("## Spearman ρ (FFM_train vs silver_eval)")
    print()
    print("|             |  trained on VN  |  trained on US  |  trained on KR  |")
    print("|-------------|----------------:|----------------:|----------------:|")
    for eval_pop in populations:
        cells = []
        for train_pop in populations:
            key = f"{train_pop}__{eval_pop}"
            v = transfer_matrix[key]
            rho = v["spearman_rho_mean"]
            cells.append(f"{rho:.3f}" if rho is not None else "—")
        print(f"| eval {eval_pop:<8s} | {cells[0]} | {cells[1]} | {cells[2]} |")


if __name__ == "__main__":
    main()
