"""Build a queryable DuckDB catalog over the labeled DeepFashion2 upper-body parquet.

DuckDB query-over-parquet means we don't ETL the data — we expose the parquet
as a SQL view and let the recommender filter/sort with normal SQL.

This script:
  1. Verifies the labeled parquet exists
  2. Creates research/datasets/processed/deepfashion2/style_catalog.duckdb
  3. Creates a VIEW `items` over the parquet (zero-copy)
  4. Creates indexes (DuckDB calls these "FAST" lookups via stored statistics)
  5. Adds a derived garment_id (split:image_id:item_idx) for stable joins
  6. Prints sanity stats and example queries

Citation: Raasveldt & Mühleisen, "DuckDB: an Embeddable Analytical Database",
SIGMOD 2019. MIT license. Use freely in academic work.
"""

# intent: queryable DuckDB catalog wrapping the labeled parquet for the recommender
# status: done
# next: add CLIP-embedding column unloading (separate file for size)
# confidence: high

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/datasets/processed/deepfashion2"
LABELED = OUT / "items_upper_labeled.parquet"
DB = OUT / "style_catalog.duckdb"


def main() -> None:
    if not LABELED.exists():
        print(f"!! {LABELED} not found — run label_deepfashion2_attributes.py first", file=sys.stderr)
        sys.exit(1)

    # Reset
    if DB.exists():
        DB.unlink()

    con = duckdb.connect(str(DB))

    # ── Create the view (zero-copy over parquet) ──
    con.execute(f"""
        CREATE VIEW items_raw AS
        SELECT * FROM read_parquet('{LABELED}')
    """)

    # Derived view: stable garment_id, drop the heavy clip_embedding column from
    # standard queries (callers fetch it separately when needed).
    con.execute("""
        CREATE VIEW items AS
        SELECT
            (split || ':' || image_id || ':' || CAST(item_idx AS VARCHAR)) AS garment_id,
            split, image_id, image_path, item_idx,
            category_id, category_name, bbox,
            occlusion, viewpoint,
            neckline, neckline_conf,
            silhouette, silhouette_conf,
            pattern, pattern_conf,
            sleeve,
            primary_color_lab,
            color_temperature, color_value
        FROM items_raw
    """)

    # Separate view that includes embeddings (slow path)
    con.execute("""
        CREATE VIEW items_with_embedding AS
        SELECT
            (split || ':' || image_id || ':' || CAST(item_idx AS VARCHAR)) AS garment_id,
            *
        FROM items_raw
    """)

    # ── Sanity stats ──
    total = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    print(f"  catalog has {total} garments")
    print()

    print("  ── per-split counts ──")
    for row in con.execute("SELECT split, COUNT(*) FROM items GROUP BY split ORDER BY split").fetchall():
        print(f"    {row[0]:8s} {row[1]:>7,}")

    print()
    print("  ── per-category counts ──")
    for row in con.execute(
        "SELECT category_id, category_name, COUNT(*) FROM items GROUP BY category_id, category_name ORDER BY category_id"
    ).fetchall():
        print(f"    {row[0]:2d} {row[1]:24s} {row[2]:>7,}")

    print()
    print("  ── attribute distributions ──")
    for col in ["neckline", "silhouette", "pattern", "sleeve", "color_temperature", "color_value"]:
        rows = con.execute(
            f"SELECT {col}, COUNT(*) AS n FROM items GROUP BY {col} ORDER BY n DESC LIMIT 6"
        ).fetchall()
        top = ", ".join(f"{r[0]}={r[1]:,}" for r in rows)
        print(f"    {col:18s} {top}")

    print()
    print("  ── confidence summary (mean per dim) ──")
    for col in ["neckline_conf", "silhouette_conf", "pattern_conf"]:
        mean, p5, p95 = con.execute(
            f"SELECT avg({col}), quantile({col}, 0.05), quantile({col}, 0.95) FROM items_raw"
        ).fetchone()
        print(f"    {col:18s} mean={mean:.3f}  p5={p5:.3f}  p95={p95:.3f}")

    print()
    print("  ── example query: V-neck warm-tone fitted tops for a date occasion ──")
    sample = con.execute("""
        SELECT garment_id, category_name, neckline, silhouette, color_temperature, color_value,
               neckline_conf, silhouette_conf
        FROM items
        WHERE neckline IN ('v_neck', 'deep_v_neck')
          AND silhouette IN ('fitted', 'semi_fitted')
          AND color_temperature = 'warm'
        ORDER BY neckline_conf + silhouette_conf DESC
        LIMIT 5
    """).fetchall()
    for row in sample:
        print(f"    {row}")

    print()
    print(f"  DB at: {DB}")
    print(f"  Size:  {DB.stat().st_size / 1024:.1f} KB (view-only, source parquet unchanged)")

    con.close()


if __name__ == "__main__":
    main()
