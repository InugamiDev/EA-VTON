"""Read-only catalog adapter over the DeepFashion2 DuckDB style_catalog.

The recommendation service uses this module to fetch a filtered candidate set
from the 140k-item labeled DeepFashion2 catalog at request time. Filtering is
done in SQL (sub-second) so we never load all 140k rows into Python.

Each row is translated into the same dict shape the rest of the
Fit-Flatter-Match pipeline expects (see style_upper._app_garment_to_style_item).

Data lifecycle:
  research/datasets/raw/deepfashion2/         ← raw images + annos (gitignored)
  research/datasets/processed/deepfashion2/
    items_upper_labeled.parquet               ← source of truth (40MB)
    style_catalog.duckdb                      ← view over parquet (read-only)

Citation: Raasveldt & Mühleisen, SIGMOD 2019 (DuckDB).
"""

# intent: adapt DeepFashion2 DuckDB rows to the Fit-Flatter-Match item shape
# status: done
# next: serve image bytes via a thumbnail route + add CLIP-embedding retrieval
# confidence: high

from __future__ import annotations

import os
import random
from pathlib import Path
from threading import Lock
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = REPO_ROOT / "research/datasets/processed/deepfashion2/style_catalog.duckdb"
DEFAULT_RAW = REPO_ROOT / "research/datasets/raw/deepfashion2"

# Size charts (target body cm, matching VN size pivot) per garment family.
# These are NOT in DeepFashion2 — we synthesize from VN body bands because
# DeepFashion2 doesn't carry brand size charts.
DEFAULT_SIZE_CHART_CM: dict[str, dict[str, dict[str, float]]] = {
    "top": {
        "XS": {"shoulder": 34, "bust": 80, "waist": 68},
        "S":  {"shoulder": 36, "bust": 85, "waist": 72},
        "M":  {"shoulder": 38, "bust": 90, "waist": 76},
        "L":  {"shoulder": 40, "bust": 96, "waist": 82},
        "XL": {"shoulder": 42, "bust": 102, "waist": 90},
    },
    "outwear": {
        "XS": {"shoulder": 36, "bust": 86, "waist": 74},
        "S":  {"shoulder": 38, "bust": 90, "waist": 78},
        "M":  {"shoulder": 40, "bust": 94, "waist": 82},
        "L":  {"shoulder": 42, "bust": 100, "waist": 88},
        "XL": {"shoulder": 44, "bust": 106, "waist": 94},
    },
}

# Ease (cm of design room over body) per silhouette label.
EASE_BY_SILHOUETTE: dict[str, dict[str, float]] = {
    "bodycon":     {"shoulder": 0, "bust": 0,  "waist": 0},
    "fitted":      {"shoulder": 0, "bust": 2,  "waist": 2},
    "semi_fitted": {"shoulder": 0, "bust": 6,  "waist": 8},
    "a_line":      {"shoulder": 0, "bust": 6,  "waist": 12},
    "loose":       {"shoulder": 1, "bust": 10, "waist": 14},
    "oversized":   {"shoulder": 3, "bust": 16, "waist": 20},
}

# Stretch coefficient by category (default fabric assumption).
STRETCH_BY_CATEGORY: dict[int, float] = {
    1: 0.10,  # short sleeve top — often jersey
    2: 0.05,  # long sleeve top
    3: 0.03,  # short sleeve outwear
    4: 0.03,  # long sleeve outwear
    5: 0.05,  # vest
    6: 0.05,  # sling
}

# Map DeepFashion2 category → our family + drape default.
CATEGORY_FAMILY: dict[int, str] = {1: "top", 2: "top", 3: "outwear", 4: "outwear", 5: "top", 6: "top"}

# Synthesize occasion tags from attributes (DeepFashion2 doesn't ship them).
def _infer_occasion_tags(neckline: str, silhouette: str, sleeve: str,
                        color_value: str, pattern: str) -> list[str]:
    tags: list[str] = ["casual"]  # baseline
    if silhouette in ("fitted", "semi_fitted") and color_value in ("dark", "medium"):
        tags.append("work")
    if neckline in ("deep_v_neck", "off_shoulder", "sweetheart"):
        tags.append("date")
        tags.append("party")
    if neckline in ("v_neck", "high_neck") and silhouette == "fitted" and color_value == "dark":
        tags.append("formal")
    if pattern in ("large_print", "horizontal_stripe") and color_value == "light":
        tags.append("party")
    return list(dict.fromkeys(tags))  # dedupe preserving order


# Hex color from LAB for UI display.
def _lab_to_hex(L: float, a: float, b: float) -> str:
    """LAB → sRGB hex. Inverse of the formula used in labeling."""
    import numpy as np

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    def finv(t):
        t3 = t ** 3
        return t3 if t3 > 0.008856 else (t - 16 / 116) / 7.787

    xn, yn, zn = 0.95047, 1.0, 1.08883
    X, Y, Z = finv(fx) * xn, finv(fy) * yn, finv(fz) * zn

    Mi = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])
    rgb_lin = np.array([X, Y, Z]) @ Mi.T
    rgb = np.where(rgb_lin > 0.0031308,
                   1.055 * np.clip(rgb_lin, 1e-9, 1) ** (1 / 2.4) - 0.055,
                   12.92 * rgb_lin)
    rgb = np.clip(rgb * 255, 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*rgb)


class DeepFashion2Catalog:
    """Singleton DuckDB-backed catalog."""

    _instance: "DeepFashion2Catalog | None" = None
    _lock = Lock()

    def __init__(self, db_path: Path | None = None) -> None:
        import duckdb

        path = db_path or DEFAULT_DB
        if not Path(path).exists():
            raise FileNotFoundError(
                f"DeepFashion2 catalog DB not found at {path}. "
                f"Run research/datasets/scripts/build_style_catalog_duckdb.py first."
            )
        # read_only=True is safe for concurrent workers
        self.con = duckdb.connect(str(path), read_only=True)
        self.db_path = path

    @classmethod
    def get_instance(cls) -> "DeepFashion2Catalog":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def total_count(self) -> int:
        return int(self.con.execute("SELECT COUNT(*) FROM items").fetchone()[0])

    def query_candidates(
        self,
        recommended_attrs: dict[str, list[str]] | None = None,
        occasion: str | None = None,
        min_attr_conf: float = 0.0,
        sample_size: int = 200,
        seed: int = 42,
    ) -> list[dict]:
        """Pull a candidate set filtered by attribute polarity hints.

        recommended_attrs: dict of {attribute_dim: [allowed_values]} from the
            flatter rules for the user's body type. Used as a hard pre-filter
            so we only score items the rules wouldn't immediately veto.
        """
        clauses: list[str] = []
        if recommended_attrs:
            for dim, allowed in recommended_attrs.items():
                if not allowed:
                    continue
                values = ", ".join(f"'{v}'" for v in allowed)
                clauses.append(f"{dim} IN ({values})")

        # Confidence floor disabled by default: our 8-way prompt set is highly
        # correlated, so softmax distributes mass uniformly across ~0.12-0.18.
        # Argmax label is still informative; keep the filter as an opt-in knob.
        if min_attr_conf > 0:
            clauses.append(f"neckline_conf > {min_attr_conf}")
            clauses.append(f"silhouette_conf > {min_attr_conf}")

        # Drop items with no resolved image path (broken extraction etc.)
        clauses.append("image_path IS NOT NULL")
        where = " AND ".join(clauses) if clauses else "1=1"

        # Deterministic shuffle via hash(garment_id + seed) — DuckDB has hash()
        query = f"""
            SELECT *
            FROM items
            WHERE {where}
            ORDER BY hash(garment_id || '{seed}')
            LIMIT {sample_size}
        """
        rows = self.con.execute(query).fetchall()
        cols = [d[0] for d in self.con.description]
        return [self._row_to_item(dict(zip(cols, r))) for r in rows]

    def get_by_garment_id(self, garment_id: str) -> dict | None:
        rows = self.con.execute(
            "SELECT * FROM items WHERE garment_id = ?", [garment_id]
        ).fetchall()
        if not rows:
            return None
        cols = [d[0] for d in self.con.description]
        return self._row_to_item(dict(zip(cols, rows[0])))

    def _row_to_item(self, row: dict) -> dict[str, Any]:
        """Translate a DuckDB row → the Fit-Flatter-Match item dict shape."""
        cid = int(row["category_id"])
        family = CATEGORY_FAMILY.get(cid, "top")
        size_chart = DEFAULT_SIZE_CHART_CM[family]

        silhouette = row["silhouette"] or "semi_fitted"
        ease = EASE_BY_SILHOUETTE.get(silhouette, EASE_BY_SILHOUETTE["semi_fitted"])
        stretch = STRETCH_BY_CATEGORY.get(cid, 0.05)

        lab = row["primary_color_lab"]
        if isinstance(lab, list) and len(lab) == 3:
            try:
                hex_code = _lab_to_hex(lab[0], lab[1], lab[2])
            except Exception:
                hex_code = "#888888"
        else:
            lab = [50.0, 0.0, 0.0]
            hex_code = "#888888"

        occasion_tags = _infer_occasion_tags(
            row["neckline"], silhouette, row["sleeve"],
            row["color_value"], row["pattern"],
        )

        # Image path resolution — combined parquet stores repo-relative paths
        # in image_path (e.g. "research/datasets/processed/combined/images/...");
        # DF2-only rows from the older schema had image_path like
        # "train/image/000026.jpg" relative to raw/deepfashion2/.
        ipath = row.get("image_path")
        source = row.get("source", "deepfashion2")
        if not ipath:
            local = None
        elif ipath.startswith("research/"):
            # repo-relative — combined catalog convention
            local = str(REPO_ROOT / ipath)
        else:
            # legacy DF2 path
            local = str(DEFAULT_RAW / ipath)

        # Brand label per source
        brand_label = {
            "deepfashion2": "DeepFashion2",
            "deepfashion_with_masks": "DeepFashion (with masks)",
            "fashionpedia": "Fashionpedia",
            "fashion_products_small": "Myntra (catalog)",
        }.get(source, source)

        # Title — use stored title if present (multi-source catalog), else synthesize
        title = row.get("title") or f"{row['category_name'].replace('_', ' ')} — {row['neckline']} {silhouette}"

        return {
            "item_id": row["garment_id"],
            "title": title,
            "category": "top" if family == "top" else "outwear",
            "target_gender": "female",
            "garment_family": "tops",
            "brand": brand_label,
            "image_url": f"/api/style/image/{row['garment_id']}",  # served lazily
            "image_local_path": local,
            "bbox": row["bbox"],
            "price_vnd": None,
            "available_sizes": list(size_chart.keys()),
            "size_chart_cm": size_chart,
            "ease_cm": ease,
            "stretch": stretch,
            "drape_class": silhouette,
            "attributes": {
                "neckline": row["neckline"],
                "silhouette": silhouette,
                "sleeve": row["sleeve"],
                "pattern": row["pattern"],
                "color_temperature": row["color_temperature"],
                "color_value": row["color_value"],
            },
            "primary_color_hex": hex_code,
            "primary_color_lab": list(lab),
            "occasion_tags": occasion_tags,
        }


def is_available() -> bool:
    """Whether the DB is built and ready to query."""
    return Path(DEFAULT_DB).exists()
