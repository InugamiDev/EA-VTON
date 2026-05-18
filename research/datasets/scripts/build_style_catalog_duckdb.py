"""Build a queryable DuckDB catalog over the COMBINED multi-source upper-body parquet.

DuckDB query-over-parquet means we don't ETL the data — we expose the parquet
as a SQL view and let the recommender filter/sort with normal SQL.

Source: research/datasets/processed/combined/items_upper_combined.parquet
        (208,069 rows across DeepFashion2 + Fashionpedia + DeepFashion-with-masks
         + fashion_products_small + VN brands, with face-redacted-image labels +
         person-categorization labels + CLIP embeddings)

This script:
  1. Verifies the combined parquet exists
  2. Creates research/datasets/processed/deepfashion2/style_catalog.duckdb
  3. Creates VIEW `items_raw` over the parquet (zero-copy, all columns)
  4. Creates VIEW `items` projecting the columns the recommender uses
  5. Prints sanity stats

Citation: Raasveldt & Mühleisen, "DuckDB: an Embeddable Analytical Database",
SIGMOD 2019. MIT license. Use freely in academic work.
"""

# intent: queryable DuckDB catalog wrapping the combined multi-source parquet
# status: done
# next: keep schema synced with downstream consumers (deepfashion2_catalog.py)
# confidence: high

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]
COMBINED = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
OUT = ROOT / "research/datasets/processed/deepfashion2"
DB = OUT / "style_catalog.duckdb"


def main() -> None:
    if not COMBINED.exists():
        print(f"!! {COMBINED} not found — run build_combined_catalog.py first", file=sys.stderr)
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()

    con = duckdb.connect(str(DB))

    # ── Raw view over the parquet (all columns, zero-copy) ──
    con.execute(f"""
        CREATE VIEW items_raw AS
        SELECT * FROM read_parquet('{COMBINED}')
    """)

    # ── Main view: aliased columns the recommendation service consumes ──
    # `image_local_path` (parquet) → `image_path` (view) so deepfashion2_catalog.py
    # can keep its existing path-resolution branch.
    con.execute("""
        CREATE VIEW items AS
        SELECT
            garment_id,
            source,
            split,
            image_id,
            image_local_path AS image_path,
            item_idx,
            category_id,
            category_name,
            bbox,
            occlusion,
            viewpoint,
            neckline, neckline_conf,
            silhouette, silhouette_conf,
            pattern, pattern_conf,
            sleeve, sleeve_conf,
            primary_color_lab,
            color_temperature,
            color_value,
            title,
            occasion_tags,
            clip_embedding,
            suits_g1, suits_g2, suits_g3, suits_g4, suits_g5,
            best_suits_cluster, best_suits_score,
            best_season, season_family, season_distance_de2000,
            style_personality,
            redaction_status
        FROM items_raw
    """)

    # ── Sanity stats ──
    total = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    print(f"  catalog has {total:,} garments")
    print()

    print("  ── per-source counts ──")
    for row in con.execute(
        "SELECT source, COUNT(*) FROM items GROUP BY source ORDER BY 2 DESC"
    ).fetchall():
        print(f"    {row[0]:32s} {row[1]:>7,}")

    print()
    print("  ── per-category counts ──")
    for row in con.execute(
        "SELECT category_id, category_name, COUNT(*) FROM items "
        "GROUP BY category_id, category_name ORDER BY category_id"
    ).fetchall():
        print(f"    {row[0]:2d} {row[1]:24s} {row[2]:>7,}")

    print()
    print("  ── person-label distributions ──")
    for col in ["best_suits_cluster", "best_season", "style_personality"]:
        rows = con.execute(
            f"SELECT {col}, COUNT(*) AS n FROM items GROUP BY {col} ORDER BY n DESC LIMIT 8"
        ).fetchall()
        top = ", ".join(f"{r[0]}={r[1]:,}" for r in rows)
        print(f"    {col:22s} {top}")

    print()
    print("  ── attribute distributions ──")
    for col in ["neckline", "silhouette", "pattern", "sleeve", "color_temperature", "color_value"]:
        rows = con.execute(
            f"SELECT {col}, COUNT(*) AS n FROM items GROUP BY {col} ORDER BY n DESC LIMIT 6"
        ).fetchall()
        top = ", ".join(f"{r[0]}={r[1]:,}" for r in rows)
        print(f"    {col:18s} {top}")

    print()
    print("  ── example: V-neck fitted warm-tone tops, sorted by confidence ──")
    sample = con.execute("""
        SELECT garment_id, source, neckline, silhouette, color_temperature,
               best_season, style_personality, neckline_conf + silhouette_conf AS conf
        FROM items
        WHERE neckline IN ('v_neck', 'deep_v_neck')
          AND silhouette IN ('fitted', 'semi_fitted')
          AND color_temperature = 'warm'
        ORDER BY conf DESC
        LIMIT 5
    """).fetchall()
    for row in sample:
        print(f"    {row}")

    print()
    print(f"  DB at: {DB}")
    print(f"  Size:  {DB.stat().st_size / 1024:.1f} KB (view-only over {COMBINED.stat().st_size/1024/1024:.0f} MB parquet)")

    con.close()


if __name__ == "__main__":
    main()
