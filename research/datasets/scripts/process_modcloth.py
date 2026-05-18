"""Process ModCloth JSON → parquet for size-prediction benchmark (B1).

ModCloth (Misra+Wan+McAuley, RecSys 2018) is fit-feedback data from
modcloth.com — included as a second size-prediction dataset to triangulate the
RTR results in our B1 benchmark.

Source: https://mcauleylab.ucsd.edu/public_datasets/data/modcloth/modcloth_final_data.json.gz
Schema (per row): item_id, size (int), category, fit ∈ {small, fit, large},
                  height ("5ft 6in"), bust, waist, hips, bra_size, cup_size, ...

Crucial schema gap vs RTR: **no weight field**. Our copula+PSIS reweighting
relies on (h, w) joint density — without weight, we can only use:
  * uniform baseline
  * height-only independence weighting (loses information)

So ModCloth is used as a **second-dataset external validity test** for B1's
uniform baseline, NOT as a second validation of the copula reweighting. We
report this explicitly in the paper.

Output: research/datasets/processed/modcloth_body_fit.parquet
"""

# intent: produce ModCloth parquet matching the upper-body size benchmark schema
# status: done
# next: extend ablation_female_upper.py to load this as an alternate dataset
# confidence: high

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "research/datasets/raw/modcloth/modcloth_final_data.json.gz"
OUT = ROOT / "research/datasets/processed/modcloth_body_fit.parquet"

HEIGHT_RE = re.compile(r"(\d+)\s*ft\s*(\d*)\s*in?", re.IGNORECASE)


def parse_height_cm(h: str | None) -> float | None:
    """ '5ft 6in' or '5ft 6 in' or '5ft' → cm."""
    if not h or not isinstance(h, str):
        return None
    m = HEIGHT_RE.search(h.strip())
    if not m:
        return None
    feet = int(m.group(1))
    inches = int(m.group(2)) if m.group(2) else 0
    return round((feet * 12 + inches) * 2.54, 1)


def parse_inches(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ModCloth categories that map to our upper-body filter
UPPER_CATEGORIES = {
    "tops", "top", "blouse", "shirt", "sweater", "jacket", "coat",
    "outerwear", "blazer", "cardigan", "vest", "tank", "t-shirt",
    "knit", "hoodie", "buttondown",
    # Some product-level categories may be present:
    "tee", "tunic",
}


def map_size_to_label(size: int | None) -> str | None:
    """ModCloth uses integer sizes (often dress sizes 0-32). Bucket to letter
    bands matching the size head's target schema.

    The mapping is heuristic and documented in the paper appendix.
    """
    if size is None or pd.isna(size):
        return None
    s = int(size)
    if s <= 2:
        return "XS"
    if s <= 6:
        return "S"
    if s <= 10:
        return "M"
    if s <= 14:
        return "L"
    if s <= 18:
        return "XL"
    return "XXL"


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"!! raw file missing at {RAW} — run download step first")

    print(f"  reading {RAW.name} (~{RAW.stat().st_size/1024/1024:.1f} MB gzip)…")
    rows = []
    with gzip.open(RAW, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"  raw rows: {len(rows):,}")

    df = pd.DataFrame(rows)
    print(f"  columns: {list(df.columns)}")

    # Normalize
    df["height_cm"] = df["height"].apply(parse_height_cm)
    df["bust_in"] = df["bust"].apply(parse_inches) if "bust" in df else None
    df["waist_in"] = df["waist"].apply(parse_inches) if "waist" in df else None
    df["hips_in"] = df["hips"].apply(parse_inches) if "hips" in df else None
    df["bust_cm"] = df["bust_in"] * 2.54 if "bust_in" in df else None
    df["waist_cm"] = df["waist_in"] * 2.54 if "waist_in" in df else None
    df["hips_cm"] = df["hips_in"] * 2.54 if "hips_in" in df else None

    df["category_norm"] = df["category"].fillna("").str.lower().str.strip()

    # Bucket size to letter band (matching the size head's target schema)
    df["size_label"] = df["size"].apply(map_size_to_label)

    # Filter to fit='fit' rows with valid letter size and height
    df_fit = df[
        df["fit"].isin(["fit", "small", "large"])
        & df["height_cm"].notna()
        & df["size_label"].notna()
    ].copy()
    print(f"  valid-row count: {len(df_fit):,}")

    # Upper-body filter — ModCloth category field is fuzzier than RTR, so use
    # a permissive containment check.
    def is_upper(cat: str) -> bool:
        return any(token in cat for token in UPPER_CATEGORIES)

    df_upper = df_fit[df_fit["category_norm"].apply(is_upper)].copy()
    print(f"  female upper-body fit='fit'+'small'+'large': {len(df_upper):,}")

    df_fit_only = df_upper[df_upper["fit"] == "fit"].copy()
    print(f"  female upper-body fit='fit' only: {len(df_fit_only):,}")

    print()
    print("  size distribution (fit='fit'):")
    for s, n in df_fit_only["size_label"].value_counts().items():
        print(f"    {s:<5} {n:>5,}")

    print()
    print("  height distribution (cm):")
    print(f"    min={df_fit_only['height_cm'].min():.1f}  "
          f"median={df_fit_only['height_cm'].median():.1f}  "
          f"max={df_fit_only['height_cm'].max():.1f}  "
          f"mean={df_fit_only['height_cm'].mean():.1f}")

    # Save the full upper-body subset (fit + small + large) so the ablation
    # script can choose its own filter.
    out_cols = [
        "item_id", "user_id", "size", "size_label", "fit",
        "category", "category_norm",
        "height", "height_cm",
        "bust_cm", "waist_cm", "hips_cm",
        "quality", "length", "bra size", "cup size",
    ]
    out_cols = [c for c in out_cols if c in df_upper.columns]
    df_out = df_upper[out_cols].rename(columns={
        "category": "category_raw",
        "category_norm": "category",
    })
    df_out.to_parquet(OUT, index=False)
    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"\n  wrote {OUT}  ({size_mb:.1f} MB, {len(df_out):,} rows)")


if __name__ == "__main__":
    main()
