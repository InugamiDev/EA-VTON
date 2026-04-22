"""
Process Rent The Runway dataset for EA-VTON research.
Cleans, normalizes, and exports to compact Parquet format.

Source: McAuley Lab, UCSD — CC BY 4.0
Original: 192,544 rental reviews with self-reported body measurements.
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent / "raw" / "rtr_raw.json"
OUT_DIR = Path(__file__).parent.parent / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_height_inches(h: str | None) -> float | None:
    """Convert '5\\' 8\"' to cm."""
    if not h or not isinstance(h, str):
        return None
    m = re.match(r"(\d+)'\s*(\d+)\"?", h.strip())
    if not m:
        return None
    feet, inches = int(m.group(1)), int(m.group(2))
    return round((feet * 12 + inches) * 2.54, 1)


def parse_weight_kg(w: str | None) -> float | None:
    """Convert '137lbs' to kg."""
    if not w or not isinstance(w, str):
        return None
    m = re.match(r"(\d+)", w.strip())
    if not m:
        return None
    return round(int(m.group(1)) * 0.453592, 1)


def parse_bust(b: str | None) -> tuple[int | None, str | None]:
    """Split '34d' into band (int) and cup (str)."""
    if not b or not isinstance(b, str):
        return None, None
    m = re.match(r"(\d+)\s*([a-zA-Z]+)", b.strip())
    if not m:
        return None, None
    return int(m.group(1)), m.group(2).upper()


def parse_age(a) -> int | None:
    """Parse age, filter obviously wrong values."""
    if a is None:
        return None
    try:
        age = int(a)
        return age if 13 <= age <= 100 else None
    except (ValueError, TypeError):
        return None


def main():
    print("Loading RTR raw data...")
    records = []
    with open(RAW_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"  Raw records: {len(records):,}")
    df = pd.DataFrame(records)

    # Rename columns to snake_case
    df = df.rename(columns={
        "bust size": "bust_size_raw",
        "body type": "body_type",
        "rented for": "occasion",
        "review_text": "review_text",
        "review_summary": "review_summary",
        "review_date": "review_date",
    })

    # Parse measurements
    df["height_cm"] = df["height"].apply(parse_height_inches)
    df["weight_kg"] = df["weight"].apply(parse_weight_kg)
    df["age"] = df["age"].apply(parse_age)

    bust_parsed = df["bust_size_raw"].apply(parse_bust)
    df["bust_band"] = bust_parsed.apply(lambda x: x[0])
    df["bust_cup"] = bust_parsed.apply(lambda x: x[1])

    # Compute BMI where possible
    df["bmi"] = np.where(
        df["height_cm"].notna() & df["weight_kg"].notna() & (df["height_cm"] > 0),
        (df["weight_kg"] / (df["height_cm"] / 100) ** 2).round(1),
        np.nan,
    )

    # Normalize body type labels
    body_type_map = {
        "hourglass": "hourglass",
        "straight & narrow": "rectangle",
        "pear": "pear",
        "athletic": "athletic",
        "petite": "petite",
        "full bust": "full_bust",
        "apple": "apple",
    }
    df["body_type_norm"] = df["body_type"].str.lower().str.strip().map(body_type_map)

    # Normalize fit labels
    df["fit"] = df["fit"].str.lower().str.strip()

    # Select and order columns for output
    # Full dataset (all columns)
    full_cols = [
        "user_id", "item_id", "category", "size", "fit",
        "height_cm", "weight_kg", "bmi", "age",
        "body_type_norm", "bust_band", "bust_cup",
        "occasion", "rating",
        "review_text", "review_summary", "review_date",
    ]
    df_full = df[full_cols].copy()

    # Compact dataset (body measurements + fit only, no text)
    compact_cols = [
        "user_id", "item_id", "category", "size", "fit",
        "height_cm", "weight_kg", "bmi", "age",
        "body_type_norm", "bust_band", "bust_cup",
    ]
    df_compact = df[compact_cols].copy()

    # Filter: keep only rows with at least height or weight
    df_compact_valid = df_compact[
        df_compact["height_cm"].notna() | df_compact["weight_kg"].notna()
    ].copy()

    # Stats
    print(f"\n--- RTR Dataset Stats ---")
    print(f"  Total records:       {len(df_full):,}")
    print(f"  With height:         {df_full['height_cm'].notna().sum():,}")
    print(f"  With weight:         {df_full['weight_kg'].notna().sum():,}")
    print(f"  With age:            {df_full['age'].notna().sum():,}")
    print(f"  With body type:      {df_full['body_type_norm'].notna().sum():,}")
    print(f"  With bust size:      {df_full['bust_band'].notna().sum():,}")
    print(f"  With BMI:            {df_full['bmi'].notna().sum():,}")
    print(f"  Compact (has body):  {len(df_compact_valid):,}")
    print(f"\n  Unique users:  {df_full['user_id'].nunique():,}")
    print(f"  Unique items:  {df_full['item_id'].nunique():,}")
    print(f"\n  Fit distribution:")
    print(df_full["fit"].value_counts().to_string(header=False))
    print(f"\n  Body type distribution:")
    print(df_full["body_type_norm"].value_counts().to_string(header=False))
    print(f"\n  Category top 10:")
    print(df_full["category"].value_counts().head(10).to_string(header=False))

    # Height/weight stats
    print(f"\n  Height (cm): mean={df_full['height_cm'].mean():.1f}, "
          f"std={df_full['height_cm'].std():.1f}, "
          f"min={df_full['height_cm'].min():.1f}, "
          f"max={df_full['height_cm'].max():.1f}")
    print(f"  Weight (kg): mean={df_full['weight_kg'].mean():.1f}, "
          f"std={df_full['weight_kg'].std():.1f}, "
          f"min={df_full['weight_kg'].min():.1f}, "
          f"max={df_full['weight_kg'].max():.1f}")

    # Save
    full_path = OUT_DIR / "rtr_full.parquet"
    compact_path = OUT_DIR / "rtr_body_fit.parquet"
    csv_path = OUT_DIR / "rtr_body_fit.csv"

    df_full.to_parquet(full_path, index=False)
    df_compact_valid.to_parquet(compact_path, index=False)
    df_compact_valid.to_csv(csv_path, index=False)

    print(f"\n  Saved: {full_path.name} ({full_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Saved: {compact_path.name} ({compact_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Saved: {csv_path.name} ({csv_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
