"""
Train size recommendation models for EA-VTON.

Produces two models:
  1. baseline_size_model.pkl — trained on raw RTR (US population)
  2. eavton_size_model.pkl   — trained on VN-weighted RTR

Both models: given (height_cm, weight_kg, bmi, body_type, category) → predict size.

Note: pickle is used here intentionally for sklearn model serialization.
These are trusted, locally-generated model files — never load untrusted pickles.

Usage:
    cd ea-vton
    source .venv/bin/activate
    python models/train_size_rec.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent.parent / "datasets" / "processed"
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare():
    """Load RTR data and prepare features."""
    df = pd.read_parquet(DATA_DIR / "rtr_body_fit.parquet")

    # Filter to rows with body data and valid fit labels
    df = df[
        df["height_cm"].notna()
        & df["weight_kg"].notna()
        & df["fit"].isin(["fit", "small", "large"])
    ].copy()

    # Only train on fit="fit" rows (ground truth: this body got the right size)
    df_fit = df[df["fit"] == "fit"].copy()

    # Map RTR numeric sizes to standard labels
    SIZE_MAP = {
        0: "XXS", 1: "XXS", 2: "XS", 3: "XS", 4: "S",
        5: "S", 6: "S", 7: "M", 8: "M",
        9: "M", 10: "L", 11: "L", 12: "L",
        13: "L", 14: "XL", 15: "XL", 16: "XL",
        17: "XXL", 18: "XXL", 19: "XXL", 20: "XXL",
    }
    df_fit["size_label"] = df_fit["size"].map(SIZE_MAP)
    df_fit = df_fit[df_fit["size_label"].notna()].copy()

    print(f"Training rows (fit='fit', has body, valid size): {len(df_fit):,}")
    print(f"Size distribution:\n{df_fit['size_label'].value_counts().sort_index().to_string()}")

    return df_fit


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, LabelEncoder, LabelEncoder, LabelEncoder]:
    """Build feature matrix and labels."""
    body_enc = LabelEncoder()
    cat_enc = LabelEncoder()
    size_enc = LabelEncoder()

    df["body_type_clean"] = df["body_type_norm"].fillna("unknown")

    body_encoded = body_enc.fit_transform(df["body_type_clean"])
    cat_encoded = cat_enc.fit_transform(df["category"])
    size_labels = size_enc.fit_transform(df["size_label"])

    X = np.column_stack([
        df["height_cm"].values,
        df["weight_kg"].values,
        df["bmi"].fillna(22.0).values,
        df["age"].fillna(30).values,
        body_encoded,
        cat_encoded,
    ])

    return X, size_labels, body_enc, cat_enc, size_enc


def train_model(X_train, y_train, sample_weight=None):
    """Train a gradient boosting classifier."""
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def evaluate(model, X_test, y_test, size_enc, label=""):
    """Evaluate model accuracy."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    size_order = {"XXS": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5, "XXL": 6}
    y_test_labels = size_enc.inverse_transform(y_test)
    y_pred_labels = size_enc.inverse_transform(y_pred)
    within_1 = np.mean([
        abs(size_order.get(t, 3) - size_order.get(p, 3)) <= 1
        for t, p in zip(y_test_labels, y_pred_labels)
    ])

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Top-1 Accuracy:    {acc:.3f} ({acc*100:.1f}%)")
    print(f"  Within-1 Accuracy: {within_1:.3f} ({within_1*100:.1f}%)")
    print(f"\n  Size distribution (predicted vs true):")
    for s in ["XXS", "XS", "S", "M", "L", "XL", "XXL"]:
        pred_pct = np.mean(y_pred_labels == s) * 100
        true_pct = np.mean(y_test_labels == s) * 100
        print(f"    {s:>3s}: pred {pred_pct:5.1f}% | true {true_pct:5.1f}%")

    return acc, within_1


def main():
    print("=" * 60)
    print("  EA-VTON Size Recommendation Model Training")
    print("=" * 60)

    df = load_and_prepare()
    X, y, body_enc, cat_enc, size_enc = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Model 1: Baseline (no population weighting) ──────────────────
    print("\nTraining baseline model (US population)...")
    baseline = train_model(X_train, y_train)
    baseline_acc, baseline_w1 = evaluate(baseline, X_test, y_test, size_enc, "BASELINE (US Population)")

    # ── Model 2: EA-VTON (Vietnamese-weighted) ───────────────────────
    from scipy import stats

    vn_h_mean, vn_h_std = 156.2, 5.5
    vn_w_mean, vn_w_std = 53.9, 8.0

    train_heights = X_train[:, 0]
    train_weights = X_train[:, 1]
    train_bmis = X_train[:, 2]

    rtr_h_mean, rtr_h_std = train_heights.mean(), train_heights.std()
    rtr_w_mean, rtr_w_std = train_weights.mean(), train_weights.std()
    rtr_b_mean, rtr_b_std = train_bmis.mean(), train_bmis.std()

    w_h = np.clip(
        stats.norm.pdf(train_heights, vn_h_mean, vn_h_std) /
        (stats.norm.pdf(train_heights, rtr_h_mean, rtr_h_std) + 1e-10),
        0.01, 100
    )
    w_w = np.clip(
        stats.norm.pdf(train_weights, vn_w_mean, vn_w_std) /
        (stats.norm.pdf(train_weights, rtr_w_mean, rtr_w_std) + 1e-10),
        0.01, 100
    )
    w_b = np.clip(
        stats.norm.pdf(train_bmis, 21.4, 2.5) /
        (stats.norm.pdf(train_bmis, rtr_b_mean, rtr_b_std) + 1e-10),
        0.01, 100
    )
    vn_weights = w_h * w_w * w_b
    vn_weights = vn_weights / vn_weights.mean()

    eff_n = (vn_weights.sum() ** 2) / (vn_weights ** 2).sum()
    print(f"\nTraining EA-VTON model (Vietnamese-weighted)...")
    print(f"  Effective sample size: {eff_n:.0f} / {len(X_train):,}")

    eavton = train_model(X_train, y_train, sample_weight=vn_weights)
    eavton_acc, eavton_w1 = evaluate(eavton, X_test, y_test, size_enc, "EA-VTON (Vietnamese Population)")

    # ── Vietnamese body simulation test ──────────────────────────────
    print("\n" + "=" * 60)
    print("  VIETNAMESE BODY SIMULATION TEST")
    print("=" * 60)

    vn_test_profiles = [
        {"name": "VN Cluster 1 (short, thin)", "h": 151, "w": 46, "bmi": 20.2, "age": 25},
        {"name": "VN Cluster 3 (medium, dominant)", "h": 156, "w": 52, "bmi": 21.4, "age": 28},
        {"name": "VN Cluster 4 (short, heavy)", "h": 152, "w": 58, "bmi": 25.1, "age": 35},
        {"name": "US Average (RTR mean)", "h": 166, "w": 62, "bmi": 22.5, "age": 30},
    ]

    test_cat = "dress" if "dress" in cat_enc.classes_ else cat_enc.classes_[0]
    test_cat_idx = cat_enc.transform([test_cat])[0]
    test_body = body_enc.transform(["hourglass"])[0] if "hourglass" in body_enc.classes_ else 0

    for profile in vn_test_profiles:
        x = np.array([[profile["h"], profile["w"], profile["bmi"], profile["age"], test_body, test_cat_idx]])
        baseline_pred = size_enc.inverse_transform(baseline.predict(x))[0]
        eavton_pred = size_enc.inverse_transform(eavton.predict(x))[0]

        baseline_conf = baseline.predict_proba(x)[0].max()
        eavton_conf = eavton.predict_proba(x)[0].max()

        diff = "SAME" if baseline_pred == eavton_pred else "DIFFERENT"
        print(f"\n  {profile['name']}")
        print(f"    H={profile['h']}cm W={profile['w']}kg BMI={profile['bmi']}")
        print(f"    Baseline:  {baseline_pred:>3s} (conf {baseline_conf:.2f})")
        print(f"    EA-VTON:   {eavton_pred:>3s} (conf {eavton_conf:.2f})")
        print(f"    --> {diff}")

    # ── Save models (pickle for sklearn — trusted local files only) ──
    baseline_bundle = {
        "model": baseline,
        "body_enc": body_enc,
        "cat_enc": cat_enc,
        "size_enc": size_enc,
        "type": "baseline",
        "metrics": {"accuracy": baseline_acc, "within_1": baseline_w1},
        "population": "US (RTR)",
    }

    eavton_bundle = {
        "model": eavton,
        "body_enc": body_enc,
        "cat_enc": cat_enc,
        "size_enc": size_enc,
        "type": "eavton",
        "metrics": {"accuracy": eavton_acc, "within_1": eavton_w1},
        "population": "Vietnamese (importance-weighted)",
    }

    baseline_path = OUT_DIR / "baseline_size_model.pkl"
    eavton_path = OUT_DIR / "eavton_size_model.pkl"

    # NOTE: pickle used intentionally for sklearn model serialization.
    # These are trusted, locally-generated files — never deserialize untrusted pickles.
    with open(baseline_path, "wb") as f:
        pickle.dump(baseline_bundle, f)
    with open(eavton_path, "wb") as f:
        pickle.dump(eavton_bundle, f)

    config = {
        "baseline": {
            "path": str(baseline_path),
            "accuracy": round(baseline_acc, 4),
            "within_1": round(baseline_w1, 4),
            "population": "US (RTR)",
        },
        "eavton": {
            "path": str(eavton_path),
            "accuracy": round(eavton_acc, 4),
            "within_1": round(eavton_w1, 4),
            "population": "Vietnamese",
        },
        "size_order": ["XXS", "XS", "S", "M", "L", "XL", "XXL"],
        "body_types": list(body_enc.classes_),
        "categories": list(cat_enc.classes_),
    }
    config_path = OUT_DIR / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Models saved:")
    print(f"    {baseline_path.name} ({baseline_path.stat().st_size / 1e3:.1f} KB)")
    print(f"    {eavton_path.name}  ({eavton_path.stat().st_size / 1e3:.1f} KB)")
    print(f"    {config_path.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
