"""Visual inspection report — sample N labeled DeepFashion2 items per attribute
and render an HTML grid so you can verify the auto-labels by eye.

Output: reports/df2_label_inspection.html
Open with: `open reports/df2_label_inspection.html`
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import duckdb
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "research/datasets/processed/deepfashion2/style_catalog.duckdb"
RAW = ROOT / "research/datasets/raw/deepfashion2"
OUT = ROOT / "reports/df2_label_inspection.html"

PER_LABEL = 12         # successful cards per attribute value
CANDIDATE_POOL = 60    # rows to try per label (covers crop/render failures)
THUMB_SIZE = 220


def thumbnail_data_uri(image_path: Path, bbox: list[float] | None) -> str | None:
    try:
        img = Image.open(image_path).convert("RGB")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            mx, my = (x2 - x1) * 0.05, (y2 - y1) * 0.05
            w, h = img.size
            img = img.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)),
                            min(w, int(x2 + mx)), min(h, int(y2 + my))))
        img.thumbnail((THUMB_SIZE, THUMB_SIZE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def build_section(con, title: str, column: str) -> str:
    """Build one section of PER_LABEL successful items per unique value.

    We over-fetch (CANDIDATE_POOL rows) and accept the first PER_LABEL that
    render successfully. Uses ORDER BY hash() — deterministic and reliable
    (DuckDB's USING SAMPLE under-returns with selective filters).
    """
    values = [r[0] for r in con.execute(
        f"SELECT DISTINCT {column} FROM items ORDER BY {column}"
    ).fetchall()]

    html_blocks = [f'<h2>{title} ({column})</h2>']
    for v in values:
        # over-fetch, then keep first PER_LABEL successful thumbnails
        rows = con.execute(f"""
            SELECT image_id, item_idx, category_name, neckline, silhouette, sleeve,
                   pattern, color_temperature, color_value, image_path, bbox
            FROM items
            WHERE {column} = ?
            ORDER BY hash(image_id || CAST(item_idx AS VARCHAR) || ?)
            LIMIT {CANDIDATE_POOL}
        """, [v, v]).fetchall()

        total_in_db = con.execute(
            f"SELECT COUNT(*) FROM items WHERE {column} = ?", [v]
        ).fetchone()[0]

        cards = []
        tried = failed = 0
        for r in rows:
            if len(cards) >= PER_LABEL:
                break
            tried += 1
            iid, iidx, cat, nck, sil, slv, pat, ct, cv, ipath, bbox = r
            img_data = thumbnail_data_uri(RAW / ipath, list(bbox) if bbox else None)
            if not img_data:
                failed += 1
                continue
            label_html = (
                f"<div class='lbl'>{cat}</div>"
                f"<div class='attr'>{nck} · {sil} · {slv}</div>"
                f"<div class='attr'>{pat} · {ct}/{cv}</div>"
                f"<div class='id'>{iid}:{iidx}</div>"
            )
            cards.append(f"<div class='card'><img src='{img_data}'/>{label_html}</div>")

        print(f"    {column}={v}: {len(cards)} shown ({tried} tried, {failed} failed) — DB has {total_in_db:,}")

        meta = f"{len(cards)} shown · {total_in_db:,} in catalog"
        if failed:
            meta += f" · {failed} crops failed"
        html_blocks.append(f"<h3 class='val'>{v} <span class='meta'>({meta})</span></h3>")
        html_blocks.append(f"<div class='grid'>{''.join(cards)}</div>")

    return "\n".join(html_blocks)


def main() -> None:
    if not DB.exists():
        raise SystemExit(f"DB not found at {DB}; run build_style_catalog_duckdb.py first")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB), read_only=True)

    total = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    print(f"  catalog: {total:,} items")

    sections = []
    for title, col in [
        ("Neckline", "neckline"),
        ("Silhouette", "silhouette"),
        ("Pattern", "pattern"),
        ("Sleeve", "sleeve"),
        ("Color temperature", "color_temperature"),
        ("Color value", "color_value"),
    ]:
        print(f"  building section: {col}")
        sections.append(build_section(con, title, col))

    html = """<!doctype html>
<html><head><meta charset='utf-8'><title>DeepFashion2 label inspection</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1400px; margin: 24px auto; padding: 0 16px; color: #222; }
h1 { margin-top: 0; }
h2 { margin-top: 40px; border-bottom: 2px solid #eee; padding-bottom: 6px; }
h3.val { margin: 20px 0 8px; color: #555; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
.card { background: #fafafa; border: 1px solid #eee; border-radius: 6px; padding: 8px; text-align: center; }
.card img { width: 100%; height: 200px; object-fit: cover; border-radius: 4px; background: #ccc; }
.lbl { font-weight: 600; margin-top: 6px; font-size: 13px; }
.attr { font-size: 11px; color: #666; margin-top: 2px; }
.id { font-size: 10px; color: #999; margin-top: 4px; font-family: monospace; }
.meta { color: #666; font-size: 13px; }
</style></head><body>
<h1>DeepFashion2 auto-label inspection</h1>
<p class='meta'>"""
    html += f"{total:,} upper-body items labeled. Sampling {PER_LABEL} items per attribute value below. "
    html += "Each card shows the assigned (neckline · silhouette · sleeve), (pattern · temp/value), and the source image_id.</p>"
    html += "\n".join(sections)
    html += "</body></html>"

    OUT.write_text(html)
    print(f"  wrote {OUT}")
    print(f"  open: {OUT}")


if __name__ == "__main__":
    main()
