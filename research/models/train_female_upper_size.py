"""Train a female-only, upper-body-specific size GBM with copula-tempered VN reweighting.

RTR is 100% female by platform (women's rental). We restrict the training set
to upper-body female garment categories so the size head is specialized for
upper-body female sizing — the central object of our paper.

Pipeline:
  1. Filter RTR to upper-body categories (top, blouse, shirt, jumpsuit, romper,
     jacket, blazer, cardigan, sweater, vest, tank, t-shirt)
  2. Drop rows missing height/weight or with non-fit verdicts
  3. 80/20 stratified split by size
  4. Compute copula importance weights against VN female anthropometric prior,
     temper with α=0.75 (best-performing variant from ablation study)
  5. Train GBM(7-class), evaluate exact + within-1 accuracy
  6. Save bundle to research/models/variants/gbm_female_upper_vn.pkl
"""

# intent: produce a female-upper-body-specialized size model for the paper
# status: done
# next: register in research_models.py and expose via /recommend-size/visual
# confidence: high

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from research.weighting.copula_psis import compute_weights_copula  # noqa: E402

DATA_DIR = ROOT / "research/datasets/processed"
VARIANTS_DIR = ROOT / "research/models/variants"
VARIANTS_DIR.mkdir(parents=True, exist_ok=True)

SIZE_MAP = {
    0: "XXS", 1: "XXS", 2: "XS", 3: "XS", 4: "S", 5: "S", 6: "S",
    7: "M", 8: "M", 9: "M", 10: "L", 11: "L", 12: "L", 13: "L",
    14: "XL", 15: "XL", 16: "XL", 17: "XXL", 18: "XXL", 19: "XXL", 20: "XXL",
}
SIZE_ORDER = {"XXS": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5, "XXL": 6}

# Female upper-body categories present in RTR
UPPER_CATEGORIES = {
    "top", "blouse", "shirt", "jumpsuit", "romper",
    "jacket", "blazer", "cardigan", "sweater",
    "vest", "tank", "t-shirt", "tunic", "sweatershirt",
    "buttondown", "crewneck",
}

ALPHA_TEMPER = 0.75


def load_female_upper_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "rtr_body_fit.parquet")
    print(f"  RTR total rows: {len(df):,}")

    # Female upper-body filter
    df = df[df["category"].str.lower().isin(UPPER_CATEGORIES)].copy()
    print(f"  female upper-body rows: {len(df):,}")

    df = df[
        df["height_cm"].notna()
        & df["weight_kg"].notna()
        & df["fit"].isin(["fit", "small", "large"])
        & df["body_type_norm"].notna()
    ].copy()

    df_fit = df[df["fit"] == "fit"].copy()
    df_fit["size_label"] = df_fit["size"].map(SIZE_MAP)
    df_fit = df_fit[df_fit["size_label"].notna()].copy()
    df_fit["age"] = df_fit["age"].fillna(df_fit["age"].median())

    print(f"  fit='fit' rows with valid size: {len(df_fit):,}")
    print(f"  size distribution:")
    for s, n in df_fit["size_label"].value_counts().sort_index(key=lambda x: x.map(SIZE_ORDER)).items():
        print(f"    {s:<5}  {n:>5,}")
    return df_fit


def main() -> None:
    df = load_female_upper_data()

    body_enc = LabelEncoder().fit(df["body_type_norm"])
    cat_enc = LabelEncoder().fit(df["category"])
    size_enc = LabelEncoder().fit(["XXS", "XS", "S", "M", "L", "XL", "XXL"])

    df["body_type_enc"] = body_enc.transform(df["body_type_norm"])
    df["category_enc"] = cat_enc.transform(df["category"])
    df["size_enc"] = size_enc.transform(df["size_label"])

    X = df[["height_cm", "weight_kg", "bmi", "age", "body_type_enc", "category_enc"]].values
    y = df["size_enc"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)), test_size=0.2, random_state=42, stratify=y,
    )
    print(f"\n  train: {len(X_train):,}  test: {len(X_test):,}")

    # VN female anthropometric prior is built into copula_psis.VN_FEMALE
    # (h=156.2±5.5, w=53.9±8.0 from Tran et al. 2024)
    h = X_train[:, 0]
    w = X_train[:, 1]
    weights, diag = compute_weights_copula(h, w, apply_psis=True)
    print(f"\n  copula weights via PSIS-smoothed density ratio:")
    print(f"    PSIS k_hat:    {diag.get('k_hat', '?'):.3f}")
    print(f"    mean before α: {weights.mean():.3f}")
    weights_tempered = np.power(weights, ALPHA_TEMPER)
    weights_tempered = weights_tempered / weights_tempered.mean()
    print(f"    after α={ALPHA_TEMPER}: mean={weights_tempered.mean():.3f}  "
          f"std={weights_tempered.std():.3f}  "
          f"range=[{weights_tempered.min():.3f}, {weights_tempered.max():.3f}]")

    # ── Train GBM ──
    print("\n  training GBM(n_estimators=200)...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=weights_tempered)

    # ── Evaluate ──
    y_pred = model.predict(X_test)
    exact = accuracy_score(y_test, y_pred)
    within1 = np.mean(np.abs(y_test - y_pred) <= 1)
    print(f"\n  ── test eval ──")
    print(f"    exact accuracy:     {exact:.4f}")
    print(f"    within-1 accuracy:  {within1:.4f}")

    # Restrict to VN-range subset
    vn_mask = (X_test[:, 0] < 160) & (X_test[:, 1] < 55)
    if vn_mask.sum() > 20:
        vn_exact = accuracy_score(y_test[vn_mask], y_pred[vn_mask])
        vn_within1 = np.mean(np.abs(y_test[vn_mask] - y_pred[vn_mask]) <= 1)
        print(f"\n  ── VN-range subset (h<160, w<55) — n={vn_mask.sum()} ──")
        print(f"    exact accuracy:     {vn_exact:.4f}")
        print(f"    within-1 accuracy:  {vn_within1:.4f}")
    else:
        vn_exact = vn_within1 = None

    # Confusion matrix
    print("\n  ── confusion matrix (test) ──")
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(size_enc.classes_, key=lambda x: SIZE_ORDER[x])
    label_idx = [size_enc.transform([l])[0] for l in labels]
    cm_ordered = cm[label_idx][:, label_idx]
    print("        " + "  ".join(f"{l:>4}" for l in labels))
    for i, l in enumerate(labels):
        print(f"    {l:<4}" + "  ".join(f"{n:>4}" for n in cm_ordered[i]))

    # ── Save bundle ──
    bundle = {
        "model": model,
        "body_enc": body_enc,
        "cat_enc": cat_enc,
        "size_enc": size_enc,
        "results": {
            "exact_acc": exact,
            "within1_acc": within1,
            "vn_exact_acc": vn_exact,
            "vn_within1_acc": vn_within1,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "alpha_temper": ALPHA_TEMPER,
            "scope": "female_upper_body",
            "vn_prior": {"h_mean": 156.2, "h_std": 5.5,
                         "w_mean": 53.9, "w_std": 8.0,
                         "source": "Tran et al. 2024 VN_FEMALE"},
            "psis_k_hat": float(diag.get("k_hat", 0)),
            "rtr_filter": "category in upper-body female set, fit='fit'",
        },
    }
    out = VARIANTS_DIR / "gbm_female_upper_vn.pkl"
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  wrote {out}  ({out.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
