"""
Process Amazon BodyM dataset for EA-VTON research.
Merges splits, joins metadata with measurements, exports compact Parquet.

Source: Amazon Science Open Data — CC BY-NC 4.0
Original: 2,505 subjects, 14 body measurements + height/weight/gender.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "raw" / "bodym"
OUT_DIR = Path(__file__).parent.parent / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "testA", "testB"]

# Standardize column names
MEASUREMENT_RENAME = {
    "ankle": "ankle_girth_cm",
    "arm-length": "arm_length_cm",
    "bicep": "bicep_girth_cm",
    "calf": "calf_girth_cm",
    "chest": "chest_girth_cm",
    "forearm": "forearm_girth_cm",
    "height": "height_body_cm",
    "hip": "hip_girth_cm",
    "leg-length": "leg_length_cm",
    "shoulder-breadth": "shoulder_breadth_cm",
    "shoulder-to-crotch": "shoulder_to_crotch_cm",
    "thigh": "thigh_girth_cm",
    "waist": "waist_girth_cm",
    "wrist": "wrist_girth_cm",
}


def load_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load metadata and measurements for a split."""
    split_dir = RAW_DIR / split
    meta = pd.read_csv(split_dir / "hwg_metadata.csv")
    meas = pd.read_csv(split_dir / "measurements.csv")
    meta["split"] = split
    return meta, meas


def main():
    print("Loading BodyM splits...")
    all_meta = []
    all_meas = []

    for split in SPLITS:
        meta, meas = load_split(split)
        print(f"  {split}: {len(meta)} subjects, {len(meas)} measurements")
        all_meta.append(meta)
        all_meas.append(meas)

    meta_df = pd.concat(all_meta, ignore_index=True)
    meas_df = pd.concat(all_meas, ignore_index=True)

    # Rename measurement columns
    meas_df = meas_df.rename(columns=MEASUREMENT_RENAME)

    # Merge metadata with measurements
    df = meta_df.merge(meas_df, on="subject_id", how="inner")

    # Compute BMI
    df["bmi"] = (df["weight_kg"] / (df["height_cm"] / 100) ** 2).round(1)

    # Compute waist-to-hip ratio
    df["whr"] = (df["waist_girth_cm"] / df["hip_girth_cm"]).round(3)

    # Compute shoulder-to-hip ratio
    df["shr"] = (df["shoulder_breadth_cm"] / df["hip_girth_cm"]).round(3)

    # Round all measurements to 1 decimal
    meas_cols = list(MEASUREMENT_RENAME.values())
    df[meas_cols] = df[meas_cols].round(1)

    # Stats
    print(f"\n--- BodyM Dataset Stats ---")
    print(f"  Total subjects:  {len(df):,}")
    print(f"  Gender split:    {df['gender'].value_counts().to_dict()}")
    print(f"\n  Height (cm):  mean={df['height_cm'].mean():.1f}, std={df['height_cm'].std():.1f}")
    print(f"  Weight (kg):  mean={df['weight_kg'].mean():.1f}, std={df['weight_kg'].std():.1f}")
    print(f"  BMI:          mean={df['bmi'].mean():.1f}, std={df['bmi'].std():.1f}")

    print(f"\n  Measurement ranges:")
    for col in meas_cols:
        print(f"    {col:25s}: {df[col].min():.1f} – {df[col].max():.1f}  (mean {df[col].mean():.1f})")

    # Gender-stratified stats
    for gender in ["female", "male"]:
        g = df[df["gender"] == gender]
        print(f"\n  {gender.upper()} (n={len(g)}):")
        print(f"    Height: {g['height_cm'].mean():.1f} ± {g['height_cm'].std():.1f} cm")
        print(f"    Weight: {g['weight_kg'].mean():.1f} ± {g['weight_kg'].std():.1f} kg")
        print(f"    Chest:  {g['chest_girth_cm'].mean():.1f} ± {g['chest_girth_cm'].std():.1f} cm")
        print(f"    Waist:  {g['waist_girth_cm'].mean():.1f} ± {g['waist_girth_cm'].std():.1f} cm")
        print(f"    Hip:    {g['hip_girth_cm'].mean():.1f} ± {g['hip_girth_cm'].std():.1f} cm")
        print(f"    WHR:    {g['whr'].mean():.3f} ± {g['whr'].std():.3f}")

    # Save
    out_parquet = OUT_DIR / "bodym_measurements.parquet"
    out_csv = OUT_DIR / "bodym_measurements.csv"

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print(f"\n  Saved: {out_parquet.name} ({out_parquet.stat().st_size / 1e3:.1f} KB)")
    print(f"  Saved: {out_csv.name} ({out_csv.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
