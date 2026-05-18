"""Full 6-row ablation table for the size head — female upper-body subset.

Trains six GBM variants on the exact same train split (female upper-body RTR,
fit='fit' rows, n_train ≈ 8,750) and evaluates on the exact same test split
(n_test ≈ 2,188) so the numbers are directly comparable for the paper §4 table.

Variants:
    1. uniform           (no reweighting)
    2. independence      (marginal density ratios, no copula coupling)
    3. copula            (Gaussian copula, no PSIS)
    4. copula+PSIS       (Pareto-smoothed tail, α=1.0)
    5. copula+PSIS α=0.75 (chosen — the paper's main system)
    6. copula+PSIS α=0.50 (extra tempering)

Also reports the VN-range subset (h<160, w<55) metrics for each row.

Writes:
    research/models/variants/ablation_female_upper.json
"""

# intent: produce the 6-row ablation table for paper §4
# status: in_progress
# next: render as a markdown table for the paper appendix
# confidence: high

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from research.weighting.copula_psis import (  # noqa: E402
    compute_weights_copula,
    compute_weights_independent,
)

DATA_DIR = ROOT / "research/datasets/processed"
VARIANTS_DIR = ROOT / "research/models/variants"

SIZE_MAP = {
    0: "XXS", 1: "XXS", 2: "XS", 3: "XS", 4: "S", 5: "S", 6: "S",
    7: "M", 8: "M", 9: "M", 10: "L", 11: "L", 12: "L", 13: "L",
    14: "XL", 15: "XL", 16: "XL", 17: "XXL", 18: "XXL", 19: "XXL", 20: "XXL",
}
SIZE_ORDER = {"XXS": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5, "XXL": 6}

UPPER_CATEGORIES = {
    "top", "blouse", "shirt", "jumpsuit", "romper",
    "jacket", "blazer", "cardigan", "sweater",
    "vest", "tank", "t-shirt", "tunic", "sweatershirt",
    "buttondown", "crewneck",
}


def load_female_upper() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "rtr_body_fit.parquet")
    df = df[df["category"].str.lower().isin(UPPER_CATEGORIES)].copy()
    df = df[
        df["height_cm"].notna()
        & df["weight_kg"].notna()
        & df["fit"].isin(["fit", "small", "large"])
        & df["body_type_norm"].notna()
    ].copy()
    df = df[df["fit"] == "fit"].copy()
    df["size_label"] = df["size"].map(SIZE_MAP)
    df = df[df["size_label"].notna()].copy()
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def evaluate(model, X_test, y_test, label_to_idx):
    y_pred = model.predict(X_test)
    exact = accuracy_score(y_test, y_pred)
    within1 = float(np.mean(np.abs(y_test - y_pred) <= 1))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    return {"exact": exact, "within1": within1, "mae": mae}


def evaluate_vn_subset(model, X_test, y_test):
    vn_mask = (X_test[:, 0] < 160) & (X_test[:, 1] < 55)
    n = int(vn_mask.sum())
    if n < 20:
        return {"n": n, "exact": None, "within1": None, "mae": None}
    y_pred = model.predict(X_test[vn_mask])
    yt = y_test[vn_mask]
    return {
        "n": n,
        "exact": accuracy_score(yt, y_pred),
        "within1": float(np.mean(np.abs(yt - y_pred) <= 1)),
        "mae": float(np.mean(np.abs(yt - y_pred))),
    }


def fit_variant(name, X_train, y_train, sample_weight):
    print(f"  → fitting {name}...", end=" ", flush=True)
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    print("done")
    return model


def main() -> None:
    df = load_female_upper()
    print(f"  female-upper rows with valid size + fit='fit': {len(df):,}")

    body_enc = LabelEncoder().fit(df["body_type_norm"])
    cat_enc = LabelEncoder().fit(df["category"])
    size_enc = LabelEncoder().fit(["XXS", "XS", "S", "M", "L", "XL", "XXL"])

    df["body_type_enc"] = body_enc.transform(df["body_type_norm"])
    df["category_enc"] = cat_enc.transform(df["category"])
    df["size_enc"] = size_enc.transform(df["size_label"])

    X = df[["height_cm", "weight_kg", "bmi", "age", "body_type_enc", "category_enc"]].values
    y = df["size_enc"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"  train n={len(X_train):,}  test n={len(X_test):,}")

    h_train, w_train = X_train[:, 0], X_train[:, 1]

    # 1. uniform
    n = len(X_train)
    w_uniform = np.ones(n)

    # 2. independence
    w_indep_raw, _ = compute_weights_independent(h_train, w_train)
    w_indep = w_indep_raw / w_indep_raw.mean()

    # 3. copula (no PSIS)
    w_cop_raw, diag_cop = compute_weights_copula(h_train, w_train, apply_psis=False)
    w_cop = w_cop_raw / w_cop_raw.mean()

    # 4. copula+PSIS
    w_psis_raw, diag_psis = compute_weights_copula(h_train, w_train, apply_psis=True)
    w_psis = w_psis_raw / w_psis_raw.mean()

    # 5. copula+PSIS α=0.75 (chosen)
    w_a075 = np.power(w_psis_raw, 0.75)
    w_a075 = w_a075 / w_a075.mean()

    # 6. copula+PSIS α=0.50
    w_a050 = np.power(w_psis_raw, 0.50)
    w_a050 = w_a050 / w_a050.mean()

    variants = [
        ("uniform",                w_uniform, "—"),
        ("independence",           w_indep,   "α=1.0"),
        ("copula",                 w_cop,     "α=1.0"),
        ("copula+PSIS",            w_psis,    "α=1.0"),
        ("copula+PSIS α=0.75",     w_a075,    "α=0.75"),
        ("copula+PSIS α=0.50",     w_a050,    "α=0.50"),
    ]

    label_to_idx = {l: i for i, l in enumerate(size_enc.classes_)}

    results = []
    for name, sw, alpha in variants:
        model = fit_variant(name, X_train, y_train, sw)
        full = evaluate(model, X_test, y_test, label_to_idx)
        vn = evaluate_vn_subset(model, X_test, y_test)
        row = {
            "variant": name,
            "alpha": alpha,
            "weights_mean": float(sw.mean()),
            "weights_std": float(sw.std()),
            "weights_max": float(sw.max()),
            "full_test": full,
            "vn_subset": vn,
        }
        results.append(row)

    # Pretty-print as a markdown table for the paper
    print("\n" + "=" * 78)
    print("  ABLATION TABLE (paper §4, female upper-body subset)")
    print("=" * 78)
    print()
    print("| Variant              | Tempering | Exact | Within-1 |  MAE  | VN-Exact | VN-W1 |  k̂  |")
    print("|----------------------|-----------|------:|---------:|------:|---------:|------:|----:|")
    for r in results:
        f, v = r["full_test"], r["vn_subset"]
        ve = f"{v['exact']:.3f}" if v['exact'] is not None else "—"
        vw = f"{v['within1']:.3f}" if v['within1'] is not None else "—"
        khat = "—"
        if "PSIS" in r["variant"]:
            khat = f"{diag_psis.get('k_hat', 0):.2f}"
        print(f"| {r['variant']:<20s} | {r['alpha']:^9s} | {f['exact']:.3f} | {f['within1']:.3f} | {f['mae']:.3f} | {ve:>8s} | {vw:>5s} | {khat:>3s} |")

    # Save JSON
    out = VARIANTS_DIR / "ablation_female_upper.json"
    out.write_text(json.dumps({
        "results": results,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "psis_k_hat": float(diag_psis.get("k_hat", 0)),
        "copula_rho": float(diag_psis.get("us_rho", 0)) if "us_rho" in diag_psis else None,
    }, indent=2, default=float))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
