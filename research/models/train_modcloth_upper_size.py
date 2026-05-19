"""Train a size GBM on ModCloth's female-upper subset for the B1 benchmark.

ModCloth lacks a weight field, so:
  - copula+PSIS reweighting against the VN (h, w) prior is NOT applicable here
  - we report a uniform-baseline GBM only; this is the dataset's main role in
    the paper: an external-validity test for the size benchmark's protocol,
    not a second validation of the reweighting method

Features (no weight): height_cm, bust_cm, waist_cm, hips_cm, category_enc.
Bust/waist/hips are partially missing — fill with cohort median per size.
"""

# intent: ModCloth-side number for the B1 benchmark; second-dataset external validity
# status: done
# next: optionally add a height-only-independence row as a sensitivity check
# confidence: high

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research/datasets/processed/modcloth_body_fit.parquet"
VARIANTS = ROOT / "research/models/variants"
SIZE_ORDER = {"XXS": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5, "XXL": 6}


def main() -> None:
    df = pd.read_parquet(DATA)
    print(f"  ModCloth upper-body rows: {len(df):,}")

    # Keep only fit='fit' for size training (small/large rows used for fit-feedback,
    # not for size-truth labelling in this benchmark variant).
    df = df[df["fit"] == "fit"].copy()
    print(f"  fit='fit': {len(df):,}")

    # Reasonable adult height filter
    df = df[(df["height_cm"] >= 140) & (df["height_cm"] <= 200)].copy()
    print(f"  height ∈ [140, 200] cm: {len(df):,}")

    # Impute missing bust/waist/hips with cohort median (per size band) so we
    # don't drop large fractions of rows.
    for col in ["bust_cm", "waist_cm", "hips_cm"]:
        df[col] = df.groupby("size_label")[col].transform(
            lambda s: s.fillna(s.median())
        )
        df[col] = df[col].fillna(df[col].median())  # final fallback
        print(f"  imputed {col}: NA → cohort median; final NA={df[col].isna().sum()}")

    cat_enc = LabelEncoder().fit(df["category"])
    size_enc = LabelEncoder().fit(["XXS", "XS", "S", "M", "L", "XL", "XXL"])

    df["category_enc"] = cat_enc.transform(df["category"])
    df["size_enc"] = size_enc.transform(df["size_label"])

    # Two feature sets:
    #   "with_circ"  — height + bust + waist + hips + category  (rich, ceiling test)
    #   "h_only"     — height + category only  (apples-to-apples vs RTR's camera-only feats)
    # The h_only variant is the row reported in paper §4.
    feature_sets = {
        "with_circ": ["height_cm", "bust_cm", "waist_cm", "hips_cm", "category_enc"],
        "h_only":    ["height_cm", "category_enc"],
    }
    rows = {}
    best_model = None
    best_cols: list[str] = []
    best_metrics = None

    for fname, cols in feature_sets.items():
        X = df[cols].values
        y = df["size_enc"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )
        print(f"\n  ── feature set: {fname}  ({cols}) ──")
        print(f"    train n={len(X_train):,}  test n={len(X_test):,}")
        m = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42,
        )
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        exact = accuracy_score(y_test, y_pred)
        within1 = float(np.mean(np.abs(y_test - y_pred) <= 1))
        mae = float(np.mean(np.abs(y_test - y_pred)))
        print(f"    exact:    {exact:.4f}")
        print(f"    within-1: {within1:.4f}")
        print(f"    MAE:      {mae:.4f}")
        rows[fname] = {
            "feature_cols": cols,
            "exact": exact, "within1": within1, "mae": mae,
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        }
        if fname == "h_only":
            h_only_model = m
            h_only_cols = cols
            h_only_metrics = (exact, within1, mae)
        elif fname == "with_circ":
            with_circ_model = m
            with_circ_cols = cols
            with_circ_metrics = (exact, within1, mae)

    # Save BOTH variants — h_only for the paper's apples-to-apples baseline,
    # with_circ for the photo→measurements→size chain (Path A).
    assert h_only_model is not None and with_circ_model is not None

    h_only_bundle = {
        "model": h_only_model,
        "cat_enc": cat_enc,
        "size_enc": size_enc,
        "feature_cols": h_only_cols,
        "results": {
            "exact": h_only_metrics[0], "within1": h_only_metrics[1], "mae": h_only_metrics[2],
            "rows_by_feature_set": rows,
            "dataset": "modcloth", "scope": "female_upper_body_fit_only",
            "reweighting": "uniform (ModCloth has no weight field)",
            "note": "h_only is the apples-to-apples comparison vs RTR (which lacks bust/waist/hips).",
        },
    }
    out = VARIANTS / "gbm_modcloth_upper.pkl"
    out.write_bytes(pickle.dumps(h_only_bundle))
    print(f"\n  wrote {out} ({out.stat().st_size/1024/1024:.1f} MB)")

    with_circ_bundle = {
        "model": with_circ_model,
        "cat_enc": cat_enc,
        "size_enc": size_enc,
        "feature_cols": with_circ_cols,
        "results": {
            "exact": with_circ_metrics[0], "within1": with_circ_metrics[1], "mae": with_circ_metrics[2],
            "dataset": "modcloth", "scope": "female_upper_body_fit_only",
            "feature_set": "with_circ",
            "note": "with_circ is the ceiling — uses bust+waist+hip directly. Used by the BodyM photo→measurements→size chain.",
        },
    }
    out2 = VARIANTS / "gbm_modcloth_upper_with_circ.pkl"
    out2.write_bytes(pickle.dumps(with_circ_bundle))
    print(f"  wrote {out2} ({out2.stat().st_size/1024/1024:.1f} MB)")

    # Append to a comparison JSON so the paper's Table 4.2 lands cleanly
    cmp_path = VARIANTS / "b1_comparison.json"
    cmp = json.loads(cmp_path.read_text()) if cmp_path.exists() else {}
    cmp["modcloth_uniform_upper"] = h_only_bundle["results"]
    cmp["modcloth_uniform_upper_with_circ"] = with_circ_bundle["results"]
    cmp_path.write_text(json.dumps(cmp, indent=2))
    print(f"  appended to {cmp_path}")


if __name__ == "__main__":
    main()
