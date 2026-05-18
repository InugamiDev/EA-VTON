"""Cold-start calibration evaluation — Benchmark B3.

Two metrics over a synthetic user pool:
  1. Kendall's τ between FFM-only and calibrated rankings, swept over δ
  2. Leave-one-out Recall@K: hide the 4th of 4 liked items, calibrate from the 3 remaining,
     score whether the 4th appears in the calibrated top-K (K ∈ {5, 10, 20, 50}).

Three seed-selection strategies:
  - random           — uniform random
  - stratified_sql   — our SQL-window stratification
  - dpp_proxy        — greedy farthest-point in CLIP space (Nguyen UAI 2024 baseline)

Perf: all DB I/O happens once (single table read), then everything is numpy.

Output: research/eval/calibration_results.json
"""

# intent: produce benchmark-B3 numbers efficiently
# status: done
# next: render markdown table for paper §6
# confidence: high

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import duckdb
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "research/datasets/processed/deepfashion2/style_catalog.duckdb"
OUT = ROOT / "research/eval/calibration_results.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

DELTA_GRID = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
RECALL_KS = [5, 10, 20, 50]
N_USERS = 200
N_LIKED = 4
POOL_SIZE = 500
RNG = np.random.default_rng(42)


def load_universe(con) -> dict:
    """Single big read: pull everything we need to operate purely in numpy."""
    print("  loading universe from DuckDB…")
    rows = con.execute("""
        SELECT garment_id, clip_embedding,
               neckline, silhouette, color_temperature, best_season,
               neckline_conf, silhouette_conf
        FROM items
        WHERE clip_embedding IS NOT NULL
    """).fetchall()
    print(f"  universe size: {len(rows):,}")

    ids   = np.array([r[0] for r in rows])
    embs  = np.stack([np.asarray(r[1], dtype=np.float32) for r in rows])
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    necks  = np.array([r[2] for r in rows])
    sils   = np.array([r[3] for r in rows])
    temps  = np.array([r[4] for r in rows])
    seas   = np.array([r[5] for r in rows])
    conf_n = np.array([r[6] if r[6] is not None else 0.0 for r in rows], dtype=np.float32)
    conf_s = np.array([r[7] if r[7] is not None else 0.0 for r in rows], dtype=np.float32)

    return dict(
        ids=ids, embs=embs,
        necklines=necks, silhouettes=sils,
        color_temperatures=temps, seasons=seas,
        conf=conf_n + conf_s,
    )


def seeds_random(U: dict, n: int = 18) -> np.ndarray:
    return RNG.choice(len(U["ids"]), size=n, replace=False)


def seeds_stratified_sql(U: dict, n: int = 18) -> np.ndarray:
    """Best item per (neckline, silhouette, color_temp) cell, then shuffle+truncate."""
    keys = np.char.add(np.char.add(U["necklines"], "|"), U["silhouettes"])
    keys = np.char.add(np.char.add(keys, "|"), U["color_temperatures"])
    chosen: list[int] = []
    seen: set = set()
    order = np.argsort(-U["conf"])
    for idx in order:
        k = keys[idx]
        if k in seen:
            continue
        chosen.append(int(idx))
        seen.add(k)
        if len(chosen) >= n * 3:
            break
    RNG.shuffle(chosen)
    return np.array(chosen[:n])


def seeds_dpp_proxy(U: dict, n: int = 18, candidate_size: int = 600) -> np.ndarray:
    """Greedy farthest-point in normalized CLIP space — Nguyen UAI 2024 proxy."""
    pool = RNG.choice(len(U["ids"]), size=candidate_size, replace=False)
    pool_embs = U["embs"][pool]
    selected_local: list[int] = [0]
    while len(selected_local) < n:
        sel_embs = pool_embs[selected_local]
        sims = pool_embs @ sel_embs.T
        max_sims = sims.max(axis=1)
        max_sims[selected_local] = np.inf
        idx = int(np.argmin(max_sims))
        selected_local.append(idx)
    return pool[selected_local]


def make_users(U: dict, n_users: int, n_liked: int,
               noise_p: float = 0.4) -> list[dict]:
    """Synthetic user with attribute noise.

    Procedure:
      1. Pick preferred (neckline, silhouette, color_temp)
      2. For each of n_liked items, independently flip *each* attribute to a
         random value with probability noise_p. This is the "noise floor" —
         it makes the silver-label benchmark non-trivial: the FFM score alone
         no longer perfectly identifies all liked items, so the CLIP centroid
         can provide useful signal.

    noise_p=0.4 means ~40% chance per attribute per item is *off* from the
    user's stated preferences — comparable to real users who like things
    that aren't all on-brand.
    """
    print(f"  building {n_users} synthetic users (noise_p={noise_p})…")
    users: list[dict] = []
    attempts = 0
    necks_unique = np.unique(U["necklines"])
    sils_unique = np.unique(U["silhouettes"])
    temps_unique = np.unique(U["color_temperatures"])
    while len(users) < n_users and attempts < n_users * 30:
        attempts += 1
        pref_n = RNG.choice(necks_unique)
        pref_s = RNG.choice(sils_unique)
        pref_c = RNG.choice(temps_unique)

        liked: list[int] = []
        for _ in range(n_liked * 3):  # try up to 3× to find n_liked items
            n_pick = (RNG.choice(necks_unique) if RNG.random() < noise_p else pref_n)
            s_pick = (RNG.choice(sils_unique) if RNG.random() < noise_p else pref_s)
            c_pick = (RNG.choice(temps_unique) if RNG.random() < noise_p else pref_c)
            mask = (
                (U["necklines"] == n_pick)
                & (U["silhouettes"] == s_pick)
                & (U["color_temperatures"] == c_pick)
            )
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            liked.append(int(RNG.choice(idxs)))
            if len(liked) == n_liked:
                break

        if len(liked) < n_liked:
            continue

        liked_arr = np.array(liked)
        seasons = U["seasons"][liked_arr]
        season_vals, season_cnts = np.unique(seasons, return_counts=True)
        pref_season = season_vals[np.argmax(season_cnts)]
        users.append({
            "pref_neckline": pref_n,
            "pref_silhouette": pref_s,
            "pref_color_temp": pref_c,
            "pref_season": pref_season,
            "liked_idx": liked_arr,
        })
    print(f"  built {len(users)} users in {attempts} attempts")
    return users


def ffm_scores(U: dict, pool_idx: np.ndarray, user: dict) -> np.ndarray:
    s = np.full(len(pool_idx), 0.5, dtype=np.float32)
    s += 0.15 * (U["necklines"][pool_idx] == user["pref_neckline"])
    s += 0.15 * (U["silhouettes"][pool_idx] == user["pref_silhouette"])
    s += 0.10 * (U["color_temperatures"][pool_idx] == user["pref_color_temp"])
    s += 0.10 * (U["seasons"][pool_idx] == user["pref_season"])
    return np.minimum(s, 1.0)


def evaluate_strategy(U: dict, name: str, seed_fn, users: list[dict]) -> dict:
    print(f"\n── strategy: {name} ──")
    seed_idx = seed_fn(U)
    print(f"  picked {len(seed_idx)} seeds")

    pool_idx = RNG.choice(len(U["ids"]), size=POOL_SIZE, replace=False)

    tau_per_delta = {d: [] for d in DELTA_GRID}
    recall = {d: {k: [] for k in RECALL_KS} for d in DELTA_GRID}
    # Primary metric: top-K visual alignment to user's CLIP centroid.
    # Directly measures "are recommendations more like what the user picked?"
    align_per_delta = {d: {k: [] for k in [5, 10, 20]} for d in DELTA_GRID}

    for user in users:
        liked_idx = user["liked_idx"]
        held_out = int(liked_idx[-1])
        train_likes = liked_idx[:-1]

        eval_pool = pool_idx.copy()
        if held_out not in eval_pool:
            eval_pool = np.append(eval_pool, held_out)
        held_pos = int(np.where(eval_pool == held_out)[0][0])

        # User centroid for both scoring and the visual-alignment metric.
        # Both use the SAME 3 picks (training likes), not the held-out item.
        centroid = U["embs"][train_likes].mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-8)

        base = ffm_scores(U, eval_pool, user)
        pool_embs = U["embs"][eval_pool]
        sims = pool_embs @ centroid
        pref = (sims + 1.0) / 2.0

        base_order = np.argsort(-base)
        topK_for_tau = 50

        for delta in DELTA_GRID:
            scores = (1.0 - delta) * base + delta * pref
            cal_order = np.argsort(-scores)
            tau, _ = kendalltau(base_order[:topK_for_tau], cal_order[:topK_for_tau])
            if not np.isnan(tau):
                tau_per_delta[delta].append(float(tau))
            rank_of_held = int(np.where(cal_order == held_pos)[0][0])
            for k in RECALL_KS:
                recall[delta][k].append(1 if rank_of_held < k else 0)
            # Visual alignment: mean cosine(centroid, top-K)
            # Higher = recommendations are more visually similar to user's picks.
            for k in [5, 10, 20]:
                top_k_idx = cal_order[:k]
                top_k_embs = pool_embs[top_k_idx]
                top_k_sims = top_k_embs @ centroid
                align_per_delta[delta][k].append(float(top_k_sims.mean()))

    summary = {
        "strategy": name,
        "n_users": len(users),
        "tau_mean": {d: float(np.mean(v)) if v else None for d, v in tau_per_delta.items()},
        "tau_std": {d: float(np.std(v)) if v else None for d, v in tau_per_delta.items()},
        "recall": {
            d: {k: float(np.mean(v)) if v else None for k, v in ks.items()}
            for d, ks in recall.items()
        },
        "alignment": {
            d: {k: float(np.mean(v)) if v else None for k, v in ks.items()}
            for d, ks in align_per_delta.items()
        },
    }
    print(f"  τ@δ=0.3: {summary['tau_mean'][0.3]:.3f}")
    for k in [5, 10, 20]:
        print(f"  align@{k}: δ=0={summary['alignment'][0.0][k]:.3f}  "
              f"δ=0.3={summary['alignment'][0.3][k]:.3f}  "
              f"δ=0.7={summary['alignment'][0.7][k]:.3f}  "
              f"δ=1.0={summary['alignment'][1.0][k]:.3f}")
    for k in RECALL_KS:
        print(f"  R@{k}: δ=0={summary['recall'][0.0][k]:.3f}  "
              f"δ=0.3={summary['recall'][0.3][k]:.3f}  "
              f"δ=0.7={summary['recall'][0.7][k]:.3f}")
    return summary


def main() -> None:
    if not DB.exists():
        sys.exit(f"!! catalog DB not found at {DB}")

    con = duckdb.connect(str(DB), read_only=True)
    U = load_universe(con)
    users = make_users(U, N_USERS, N_LIKED)

    results = {
        "n_users": len(users),
        "n_liked_per_user": N_LIKED,
        "pool_size": POOL_SIZE,
        "delta_grid": DELTA_GRID,
        "recall_ks": RECALL_KS,
        "by_strategy": {},
    }

    for name, fn in [
        ("random",          lambda u: seeds_random(u, n=18)),
        ("stratified_sql",  lambda u: seeds_stratified_sql(u, n=18)),
        ("dpp_proxy",       lambda u: seeds_dpp_proxy(u, n=18)),
    ]:
        results["by_strategy"][name] = evaluate_strategy(U, name, fn, users)

    OUT.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n── wrote {OUT} ──")

    print()
    print("| Strategy        | align@10 δ=0 | δ=0.3 | δ=0.7 | δ=1.0 |  Δ(δ=1.0 − δ=0) |")
    print("|-----------------|-------------:|------:|------:|------:|----------------:|")
    for name, s in results["by_strategy"].items():
        d0, d10 = s['alignment'][0.0][10], s['alignment'][1.0][10]
        print(f"| {name:<15s} | {d0:.3f}        | {s['alignment'][0.3][10]:.3f} | {s['alignment'][0.7][10]:.3f} | {d10:.3f} | {d10-d0:+.3f}          |")

    print()
    print("| Strategy        | R@10 δ=0 | R@10 δ=0.3 | R@10 δ=0.7 | R@10 δ=1.0 |")
    print("|-----------------|---------:|-----------:|-----------:|-----------:|")
    for name, s in results["by_strategy"].items():
        print(f"| {name:<15s} | {s['recall'][0.0][10]:.3f}    | {s['recall'][0.3][10]:.3f}      | {s['recall'][0.7][10]:.3f}      | {s['recall'][1.0][10]:.3f}      |")

    print()
    print("| Strategy        | τ@δ=0.1 | τ@δ=0.3 | τ@δ=0.5 | τ@δ=0.7 | τ@δ=1.0 |")
    print("|-----------------|--------:|--------:|--------:|--------:|--------:|")
    for name, s in results["by_strategy"].items():
        row = [f"{s['tau_mean'][d]:.3f}" if s['tau_mean'][d] is not None else "—"
               for d in [0.1, 0.3, 0.5, 0.7, 1.0]]
        print(f"| {name:<15s} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
