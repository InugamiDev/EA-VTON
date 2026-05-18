"""Ablation: does per-population identity ACTUALLY help the learned model?

The previous run showed learned_per_population.py hits NDCG@10 ≈ 1.000 on all
populations. That could mean either:
  (a) Per-population identity is genuinely informative for predicting suitability
  (b) The MLP is just memorizing the CLIP-text-image cosine structure, and the
      population+cluster one-hot inputs add nothing.

To distinguish (a) vs (b), we train an ABLATED model that excludes the population
one-hot but keeps the cluster one-hot:
  f(item_clip_emb, cluster_id) → suitability

If the ablation matches the full model, (b) is correct and the multi-population
claim is empty. If the ablation underperforms, (a) holds and the population
signal is real.

We also train a CLUSTER-AGNOSTIC ablation:
  f(item_clip_emb) → suitability

If this scores well, even cluster identity doesn't matter — the silver labels
were essentially population-cluster-invariant in this catalog.

The three numbers together tell the honest story about which signal sources
the silver labels actually contain.
"""

# intent: honest test of whether population+cluster identity helps the learned model
# status: done
# next: report results as the load-bearing fix for Contribution #2
# confidence: high

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import duckdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.models.learned_per_population import (  # noqa: E402
    ARCHETYPES, encode_archetypes, ndcg_at_k,
)

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "research/datasets/processed/deepfashion2/style_catalog.duckdb"
OUT = ROOT / "research/eval/learned_per_population_ablation.json"

POOL_SIZE = 2000
EPOCHS = 30
BATCH = 256
LR = 1e-3
RNG = np.random.default_rng(42)
torch.manual_seed(42)


class FullMLP(nn.Module):
    """clip + pop + cluster"""
    def __init__(self, clip_dim=512, n_pops=3, n_clusters=5, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim + n_pops + n_clusters, hidden),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, clip, pop_oh, cl_oh):
        return self.net(torch.cat([clip, pop_oh, cl_oh], dim=-1)).squeeze(-1)


class NoPopMLP(nn.Module):
    """clip + cluster (no population identity)"""
    def __init__(self, clip_dim=512, n_clusters=5, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim + n_clusters, hidden),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, clip, cl_oh):
        return self.net(torch.cat([clip, cl_oh], dim=-1)).squeeze(-1)


class NoIdentityMLP(nn.Module):
    """clip only — no population, no cluster"""
    def __init__(self, clip_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, clip, *_):
        return self.net(clip).squeeze(-1)


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    populations = ["vn_female", "us_female", "kr_female"]
    pop_idx = {p: i for i, p in enumerate(populations)}

    text_embeds = encode_archetypes()

    rows = con.execute(f"""
        SELECT garment_id, clip_embedding
        FROM items WHERE clip_embedding IS NOT NULL
        ORDER BY hash(garment_id || 'learnedpop42')
        LIMIT {POOL_SIZE}
    """).fetchall()
    item_embs = np.stack([np.asarray(r[1], dtype=np.float32) for r in rows])
    item_embs = item_embs / (np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-8)
    print(f"  items: {len(item_embs)}")

    # Build (pop, cluster, item, silver) pairs
    pairs = []
    for p in populations:
        for cid in ARCHETYPES[p]:
            te = text_embeds[p][cid]
            sims = item_embs @ te
            for i in range(len(item_embs)):
                pairs.append((i, p, cid, float(sims[i])))

    n_items = len(item_embs)
    item_perm = np.arange(n_items)
    RNG.shuffle(item_perm)
    train_items = set(item_perm[:int(0.8 * n_items)].tolist())
    test_items = set(item_perm[int(0.8 * n_items):].tolist())

    train_pairs = [t for t in pairs if t[0] in train_items]
    test_pairs  = [t for t in pairs if t[0] in test_items]
    print(f"  train pairs: {len(train_pairs):,}  test pairs: {len(test_pairs):,}")

    n_clusters_max = max(len(ARCHETYPES[p]) for p in populations)

    def build_tensors(p_list):
        X = np.stack([item_embs[i] for (i, *_) in p_list])
        pop_oh = np.zeros((len(p_list), len(populations)), dtype=np.float32)
        cl_oh = np.zeros((len(p_list), n_clusters_max), dtype=np.float32)
        y = np.zeros(len(p_list), dtype=np.float32)
        for i, (ii, pp, cc, ss) in enumerate(p_list):
            pop_oh[i, pop_idx[pp]] = 1.0
            cl_oh[i, int(cc) - 1] = 1.0
            y[i] = ss
        return (torch.from_numpy(X), torch.from_numpy(pop_oh),
                torch.from_numpy(cl_oh), torch.from_numpy(y))

    Xtr, Ptr, Ctr, Ytr = build_tensors(train_pairs)
    Xte, Pte, Cte, Yte = build_tensors(test_pairs)

    variants = {
        "full":       FullMLP(item_embs.shape[1], len(populations), n_clusters_max),
        "no_pop":     NoPopMLP(item_embs.shape[1], n_clusters_max),
        "no_identity": NoIdentityMLP(item_embs.shape[1]),
    }
    losses_by_variant: dict[str, float] = {}

    for name, model in variants.items():
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        ds = TensorDataset(Xtr, Ptr, Ctr, Ytr)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
        print(f"\n  training {name}…")
        for ep in range(EPOCHS):
            model.train()
            for xb, pb, cb, yb in loader:
                if name == "full":
                    pred = model(xb, pb, cb)
                elif name == "no_pop":
                    pred = model(xb, cb)
                else:
                    pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            if name == "full":
                pred_te = model(Xte, Pte, Cte)
            elif name == "no_pop":
                pred_te = model(Xte, Cte)
            else:
                pred_te = model(Xte)
            losses_by_variant[name] = float(loss_fn(pred_te, Yte))
        print(f"   final test MSE = {losses_by_variant[name]:.5f}")

    # NDCG@10 per population per variant
    EVAL_POOL = 500
    eval_idx = np.array(list(test_items))[:EVAL_POOL]
    eval_embs = torch.from_numpy(item_embs[eval_idx])

    ndcg_by_pop_by_variant: dict[str, dict] = {p: {} for p in populations}
    for p in populations:
        for name, model in variants.items():
            ndcgs = []
            for cid in ARCHETYPES[p]:
                te = text_embeds[p][cid]
                silver = item_embs[eval_idx] @ te
                p25, p75 = np.percentile(silver, 25), np.percentile(silver, 75)
                rel = np.where(silver >= p75, 2, np.where(silver >= p25, 1, 0))
                pop_oh = torch.zeros((len(eval_idx), len(populations)))
                pop_oh[:, pop_idx[p]] = 1
                cl_oh = torch.zeros((len(eval_idx), n_clusters_max))
                cl_oh[:, int(cid) - 1] = 1
                model.eval()
                with torch.no_grad():
                    if name == "full":
                        scores = model(eval_embs, pop_oh, cl_oh).numpy()
                    elif name == "no_pop":
                        scores = model(eval_embs, cl_oh).numpy()
                    else:
                        scores = model(eval_embs).numpy()
                order = np.argsort(-scores)
                rel_ordered = [int(rel[i]) for i in order]
                ndcgs.append(ndcg_at_k(rel_ordered, 10))
            ndcg_by_pop_by_variant[p][name] = float(np.mean(ndcgs))

    results = {
        "test_mse_by_variant": losses_by_variant,
        "ndcg_by_pop_by_variant": ndcg_by_pop_by_variant,
        "interpretation": (
            "Compare full vs no_pop vs no_identity NDCG@10. If full ≈ no_pop ≈ "
            "no_identity, the population+cluster identity adds NO signal — the "
            "MLP is just CLIP-cosine reconstruction. If full > no_pop > no_identity, "
            "population AND cluster identity each carry real signal."
        ),
    }
    OUT.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  wrote {OUT}")

    print()
    print("## Learned-Model Ablation — NDCG@10 (held-out items)")
    print()
    print("| eval pop    | full (clip+pop+cluster) | no_pop (clip+cluster) | no_identity (clip-only) | pop adds (full−no_pop) | cluster adds (no_pop−no_identity) |")
    print("|-------------|------------------------:|----------------------:|------------------------:|-----------------------:|----------------------------------:|")
    for p in populations:
        v = ndcg_by_pop_by_variant[p]
        pop_add = v["full"] - v["no_pop"]
        cl_add  = v["no_pop"] - v["no_identity"]
        print(f"| {p:<11s} | {v['full']:.3f}                  | {v['no_pop']:.3f}                | {v['no_identity']:.3f}                  | {pop_add:+.3f}                | {cl_add:+.3f}                           |")


if __name__ == "__main__":
    main()
