"""Add person-type labels to every item in the combined catalog.

For each labeled garment we now compute:
  - suits_g1 .. suits_g5  — polarity-sum match score vs each VN body cluster
  - best_suits_cluster    — cluster id with highest score (1..5)
  - best_suits_score      — max suits score
  - best_season           — closest 12-season palette (CIEDE2000 in CIE LAB)
  - season_family         — Spring / Summer / Autumn / Winter
  - season_distance       — ΔE2000 to the matched season anchor
  - style_personality     — classic / romantic / sporty / dramatic / natural
                            (derived from attribute combinations)

These labels are pre-computed so the recommender can rank with a single SQL
filter and researchers can train classifiers on them directly.
"""

# intent: precompute person-categorization labels on every catalog row
# status: done
# next: rebuild dataset bundle to include these columns
# confidence: high

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
COMBINED = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
RULES_PATH = ROOT / "services/recommendation/data/upper_body_rules_vn_v1.json"
PALETTES_PATH = ROOT / "services/recommendation/data/seasonal_palettes.json"


def _delta_e2000(lab_a: list[float], lab_b: list[float]) -> float:
    """CIEDE2000 perceptual color distance."""
    L1, a1, b1 = lab_a
    L2, a2, b2 = lab_b
    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    Cbar = (C1 + C2) / 2.0
    G = 0.5 * (1 - math.sqrt(Cbar ** 7 / (Cbar ** 7 + 25 ** 7))) if Cbar > 0 else 0
    a1p, a2p = a1 * (1 + G), a2 * (1 + G)
    C1p = math.sqrt(a1p * a1p + b1 * b1)
    C2p = math.sqrt(a2p * a2p + b2 * b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if (a1p or b1) else 0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if (a2p or b2) else 0
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0
    else:
        diff = h2p - h1p
        if diff > 180: diff -= 360
        elif diff < -180: diff += 360
        dhp = diff
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lbarp = (L1 + L2) / 2
    Cbarp = (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbarp = h1p + h2p
    else:
        if abs(h1p - h2p) <= 180:
            hbarp = (h1p + h2p) / 2
        else:
            hbarp = (h1p + h2p + 360) / 2 if (h1p + h2p) < 360 else (h1p + h2p - 360) / 2
    T = (1 - 0.17 * math.cos(math.radians(hbarp - 30))
           + 0.24 * math.cos(math.radians(2 * hbarp))
           + 0.32 * math.cos(math.radians(3 * hbarp + 6))
           - 0.20 * math.cos(math.radians(4 * hbarp - 63)))
    SL = 1 + (0.015 * (Lbarp - 50) ** 2) / math.sqrt(20 + (Lbarp - 50) ** 2)
    SC = 1 + 0.045 * Cbarp
    SH = 1 + 0.015 * Cbarp * T
    RT = (-2 * math.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25 ** 7))
          * math.sin(math.radians(60 * math.exp(-((hbarp - 275) / 25) ** 2)))) if Cbarp > 0 else 0
    return math.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
                     + RT * (dCp / SC) * (dHp / SH))


def _score_body_type(item_attrs: dict, cluster_rules: dict) -> int:
    """Polarity-sum: +1 for each attribute the cluster recommends, -1 for avoid."""
    total = 0
    for dim, value in item_attrs.items():
        dim_rules = cluster_rules.get(dim)
        if isinstance(dim_rules, dict) and value in dim_rules:
            total += dim_rules[value]
    return total


# Style personality buckets — defined as garment-attribute signatures.
# Reference: traditional fashion taxonomy (classic, romantic, sporty, dramatic,
# natural). Maps any (neckline, silhouette, sleeve, pattern) combo to a label.
def _style_personality(neckline: str, silhouette: str, sleeve: str,
                       pattern: str, color_value: str) -> str:
    """Heuristic style-personality bucket from garment attributes."""
    nk = neckline or ""
    sil = silhouette or ""
    slv = sleeve or ""
    pat = pattern or ""

    # Dramatic — bold/statement combos
    if nk in ("deep_v_neck", "off_shoulder") or pat == "large_print":
        return "dramatic"
    # Romantic — soft, feminine, prints
    if nk == "sweetheart" or slv == "puff" or (pat == "small_print" and color_value == "light"):
        return "romantic"
    # Sporty — relaxed silhouettes, simple neck, no prints
    if sil in ("loose", "oversized") and pat == "solid" and nk in ("crew", "scoop"):
        return "sporty"
    # Classic — fitted, structured, simple neck, solid or stripes
    if sil in ("fitted", "semi_fitted") and nk in ("crew", "high_neck", "v_neck") and pat in ("solid", "vertical_stripe"):
        return "classic"
    # Natural — easy, flowing, neutral
    return "natural"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(COMBINED))
    ap.add_argument("--out", dest="out_path", default=str(COMBINED))
    args = ap.parse_args()

    with open(RULES_PATH, "r", encoding="utf-8") as f:
        rules_doc = json.load(f)
    rules_by_cluster = rules_doc["rules"]
    cluster_ids = sorted(int(k) for k in rules_by_cluster.keys())  # [1..5]

    with open(PALETTES_PATH, "r", encoding="utf-8") as f:
        palettes_doc = json.load(f)
    palettes = palettes_doc["palettes"]

    print(f"  reading {args.in_path}...")
    t = pq.read_table(args.in_path)
    n = len(t)
    print(f"  {n:,} rows")

    necks = t.column("neckline").to_pylist()
    sils = t.column("silhouette").to_pylist()
    slvs = t.column("sleeve").to_pylist()
    pats = t.column("pattern").to_pylist()
    cvals = t.column("color_value").to_pylist()
    ctemps = t.column("color_temperature").to_pylist()
    labs = t.column("primary_color_lab").to_pylist()

    # Output arrays
    suits = {cid: [] for cid in cluster_ids}
    best_cluster = []
    best_score = []
    best_season = []
    season_family = []
    season_distance = []
    style_personalities = []

    # Pre-compute per-cluster rule lookups
    cluster_rule_maps = {cid: rules_by_cluster[str(cid)] for cid in cluster_ids}

    # Pre-flatten season anchors for fast scan
    season_keys = list(palettes.keys())
    season_anchors = {k: palettes[k]["anchors"] for k in season_keys}
    season_families = {k: palettes[k]["family"] for k in season_keys}

    print("  scoring per item...")
    for i in range(n):
        attrs = {
            "neckline": necks[i],
            "silhouette": sils[i],
            "sleeve": slvs[i],
            "pattern": pats[i],
            "color_temperature": ctemps[i],
            "color_value": cvals[i],
        }
        scores = {}
        for cid in cluster_ids:
            scores[cid] = _score_body_type(attrs, cluster_rule_maps[cid])
            suits[cid].append(scores[cid])
        top_cid = max(scores, key=scores.get)
        best_cluster.append(top_cid)
        best_score.append(scores[top_cid])

        # Season: ΔE2000 from primary_color_lab to nearest anchor across all seasons
        lab = labs[i]
        if isinstance(lab, list) and len(lab) == 3:
            best_dist = float("inf")
            best_key = season_keys[0]
            for sk in season_keys:
                for anchor in season_anchors[sk]:
                    d = _delta_e2000(lab, anchor)
                    if d < best_dist:
                        best_dist = d
                        best_key = sk
            best_season.append(best_key)
            season_family.append(season_families[best_key])
            season_distance.append(round(best_dist, 2))
        else:
            best_season.append(None)
            season_family.append(None)
            season_distance.append(None)

        style_personalities.append(_style_personality(necks[i], sils[i], slvs[i], pats[i], cvals[i]))

        if (i + 1) % 25_000 == 0:
            print(f"    {i+1:,}/{n:,}")

    # Build patched table
    new_cols = {
        **{f"suits_g{cid}": suits[cid] for cid in cluster_ids},
        "best_suits_cluster": best_cluster,
        "best_suits_score": best_score,
        "best_season": best_season,
        "season_family": season_family,
        "season_distance_de2000": season_distance,
        "style_personality": style_personalities,
    }

    # Drop any of these columns if already present (idempotent)
    keep = [c for c in t.column_names if c not in new_cols]
    base = t.select(keep)
    additions = pa.table({k: pa.array(v) for k, v in new_cols.items()})
    out_table = pa.concat_tables([base, additions], promote_options="permissive") if False else pa.Table.from_arrays(
        list(base.itercolumns()) + [additions.column(c) for c in additions.column_names],
        names=list(base.column_names) + list(additions.column_names),
    )

    pq.write_table(out_table, args.out_path)
    sz = Path(args.out_path).stat().st_size / 1024 / 1024
    print(f"  wrote {args.out_path} ({sz:.1f} MB, {out_table.num_rows:,} rows)")

    # Summary
    print()
    print("  ── distributions ──")
    from collections import Counter
    bc = Counter(best_cluster)
    for cid in cluster_ids:
        print(f"    best_suits_cluster=g{cid:<2}  {bc.get(cid, 0):>7,}")
    print()
    bs = Counter(best_season)
    for k, n_ in bs.most_common():
        print(f"    best_season={k:<14}  {n_:>7,}")
    print()
    sp = Counter(style_personalities)
    for k, n_ in sp.most_common():
        print(f"    style_personality={k:<10}  {n_:>7,}")


if __name__ == "__main__":
    main()
