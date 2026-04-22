"""
Train body measurement regression models for EA-VTON.

Predicts detailed body measurements (chest, waist, hip, shoulder breadth,
arm length, leg length) from minimal inputs: height_cm, weight_kg, gender.

Trained on the BodyM dataset (3D-scanned ground truth, ~2.5K subjects).

Produces two models:
  1. linear_body_regression.pkl  -- Ridge regression baseline
  2. mlp_body_regression.pkl     -- sklearn MLPRegressor (2 hidden layers, 64 units)

Also exposes `predict_measurements(height_cm, weight_kg, gender)` for import.

Note: pickle is used intentionally for sklearn model serialization.
These are trusted, locally-generated model files -- never load untrusted pickles.

Usage:
    cd research
    source .venv/bin/activate
    python models/train_body_regression.py
"""

# intent: body measurement regression from height + weight + gender
# status: done
# next: integrate into recommendation service via predict_measurements()
# confidence: high

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "processed"
OUT_DIR = Path(__file__).resolve().parent / "body_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────
INPUT_FEATURES = ["height_cm", "weight_kg", "bmi", "gender_enc"]
TARGET_MEASUREMENTS = [
    "chest_girth_cm",
    "waist_girth_cm",
    "hip_girth_cm",
    "shoulder_breadth_cm",
    "arm_length_cm",
    "leg_length_cm",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load BodyM data and split into train/test using the existing split column."""
    df = pd.read_parquet(DATA_DIR / "bodym_measurements.parquet")

    # Encode gender as numeric (female=0, male=1)
    df["gender_enc"] = (df["gender"] == "male").astype(int)

    # Ensure BMI is present (should already be in the data)
    if "bmi" not in df.columns:
        df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

    # Drop rows with any NaN in features or targets
    cols_needed = INPUT_FEATURES + TARGET_MEASUREMENTS
    df = df.dropna(subset=cols_needed)

    train = df[df["split"] == "train"].copy()
    test = df[df["split"].isin(["testA", "testB"])].copy()

    return train, test


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray, label: str
) -> dict[str, dict[str, float]]:
    """Evaluate a multi-output regressor. Returns per-target MAE and R2."""
    y_pred = model.predict(X_test)

    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")
    print(f"  {'Measurement':<22s} {'MAE (cm)':>10s} {'R²':>8s}  {'±3cm?':>6s}")
    print(f"  {'-' * 52}")

    results = {}
    for i, name in enumerate(TARGET_MEASUREMENTS):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        within_3 = "YES" if mae < 3.0 else "no"
        print(f"  {name:<22s} {mae:>10.2f} {r2:>8.3f}  {within_3:>6s}")
        results[name] = {"mae": round(mae, 3), "r2": round(r2, 4)}

    avg_mae = np.mean([r["mae"] for r in results.values()])
    avg_r2 = np.mean([r["r2"] for r in results.values()])
    print(f"  {'-' * 52}")
    print(f"  {'AVERAGE':<22s} {avg_mae:>10.2f} {avg_r2:>8.3f}")

    return results


def train_linear(X_train, y_train) -> Pipeline:
    """Train a Ridge regression baseline with feature scaling."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_mlp(X_train, y_train) -> Pipeline:
    """Train a small MLP (2 hidden layers, 64 units each)."""
    # intent: small MLP that mirrors the SizeMLP architecture but for regression
    # status: done
    # confidence: high
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
            batch_size=64,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_measurements(
    height_cm: float,
    weight_kg: float,
    gender: str = "female",
) -> dict[str, float]:
    """
    Predict body measurements from height, weight, and gender.

    Returns a dict with keys: chest_girth_cm, waist_girth_cm, hip_girth_cm,
    shoulder_breadth_cm, arm_length_cm, leg_length_cm.

    Loads the best saved model on first call (cached thereafter).
    """
    # intent: inference function importable by the recommendation service
    # status: done
    # confidence: high

    global _cached_model, _cached_meta
    if "_cached_model" not in globals() or _cached_model is None:
        _cached_model, _cached_meta = _load_best_model()

    bmi = weight_kg / (height_cm / 100) ** 2
    gender_enc = 1 if gender.lower() == "male" else 0

    X = np.array([[height_cm, weight_kg, bmi, gender_enc]])
    y_pred = _cached_model.predict(X)[0]

    return {name: round(float(val), 1) for name, val in zip(TARGET_MEASUREMENTS, y_pred)}


# Module-level cache for the inference model
_cached_model = None
_cached_meta = None


def _load_best_model():
    """Load the best model from disk."""
    meta_path = OUT_DIR / "best_model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {meta_path}. Run train_body_regression.py first."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    model_path = OUT_DIR / meta["model_file"]
    # NOTE: pickle used intentionally for sklearn model serialization.
    # These are trusted, locally-generated files -- never deserialize untrusted pickles.
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, meta


def main():
    print("=" * 65)
    print("  EA-VTON Body Measurement Regression")
    print("=" * 65)

    # ── Load data ────────────────────────────────────────────────────
    train_df, test_df = load_data()
    print(f"\n  Train samples: {len(train_df):,}")
    print(f"  Test samples:  {len(test_df):,} (testA + testB)")
    print(f"  Gender split (train): {train_df['gender'].value_counts().to_dict()}")

    X_train = train_df[INPUT_FEATURES].values
    y_train = train_df[TARGET_MEASUREMENTS].values
    X_test = test_df[INPUT_FEATURES].values
    y_test = test_df[TARGET_MEASUREMENTS].values

    # ── Print target stats for context ───────────────────────────────
    print(f"\n  Target measurement ranges (train):")
    for name in TARGET_MEASUREMENTS:
        col = train_df[name]
        print(f"    {name:<22s}  mean={col.mean():.1f}  std={col.std():.1f}  "
              f"range=[{col.min():.1f}, {col.max():.1f}]")

    # ── Model 1: Ridge regression baseline ───────────────────────────
    print("\n  Training Ridge regression baseline...")
    linear_model = train_linear(X_train, y_train)
    linear_results = evaluate_model(linear_model, X_test, y_test, "LINEAR BASELINE (Ridge)")

    # ── Model 2: MLP regressor ───────────────────────────────────────
    print("\n  Training MLP regressor (64-64, relu, adam)...")
    mlp_model = train_mlp(X_train, y_train)
    mlp_results = evaluate_model(mlp_model, X_test, y_test, "MLP REGRESSOR (64-64)")

    # ── Determine best model ─────────────────────────────────────────
    linear_avg_mae = np.mean([v["mae"] for v in linear_results.values()])
    mlp_avg_mae = np.mean([v["mae"] for v in mlp_results.values()])

    if mlp_avg_mae < linear_avg_mae:
        best_name = "mlp"
        best_model = mlp_model
        best_results = mlp_results
        best_avg_mae = mlp_avg_mae
    else:
        best_name = "linear"
        best_model = linear_model
        best_results = linear_results
        best_avg_mae = linear_avg_mae

    print(f"\n  >>> Best model: {best_name} (avg MAE = {best_avg_mae:.2f} cm)")

    # ── Save both models ─────────────────────────────────────────────
    # NOTE: pickle used intentionally for sklearn model serialization.
    # These are trusted, locally-generated files -- never deserialize untrusted pickles.
    linear_path = OUT_DIR / "linear_body_regression.pkl"
    mlp_path = OUT_DIR / "mlp_body_regression.pkl"

    with open(linear_path, "wb") as f:
        pickle.dump(linear_model, f)
    with open(mlp_path, "wb") as f:
        pickle.dump(mlp_model, f)

    # ── Save metadata pointing to best model ─────────────────────────
    meta = {
        "best_model": best_name,
        "model_file": f"{best_name}_body_regression.pkl",
        "input_features": INPUT_FEATURES,
        "target_measurements": TARGET_MEASUREMENTS,
        "avg_mae_cm": round(best_avg_mae, 3),
        "per_target": best_results,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "models": {
            "linear": {
                "file": "linear_body_regression.pkl",
                "avg_mae": round(linear_avg_mae, 3),
                "per_target": linear_results,
            },
            "mlp": {
                "file": "mlp_body_regression.pkl",
                "avg_mae": round(mlp_avg_mae, 3),
                "per_target": mlp_results,
            },
        },
    }

    meta_path = OUT_DIR / "best_model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Models saved to {OUT_DIR}/")
    print(f"    {linear_path.name}  ({linear_path.stat().st_size / 1e3:.1f} KB)")
    print(f"    {mlp_path.name}     ({mlp_path.stat().st_size / 1e3:.1f} KB)")
    print(f"    {meta_path.name}")

    # ── Quick inference demo ─────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  INFERENCE DEMO")
    print(f"{'=' * 65}")

    demo_profiles = [
        ("Vietnamese female (petite)", 155.0, 48.0, "female"),
        ("Vietnamese female (average)", 158.0, 53.0, "female"),
        ("Vietnamese male (average)", 168.0, 62.0, "male"),
        ("US female (average RTR)", 165.0, 63.0, "female"),
        ("US male (tall)", 180.0, 82.0, "male"),
    ]

    # Use the best model directly for the demo (bypass file loading)
    for desc, h, w, g in demo_profiles:
        bmi = w / (h / 100) ** 2
        gender_enc = 1 if g == "male" else 0
        X = np.array([[h, w, bmi, gender_enc]])
        preds = best_model.predict(X)[0]

        print(f"\n  {desc}: H={h}cm, W={w}kg, BMI={bmi:.1f}")
        for name, val in zip(TARGET_MEASUREMENTS, preds):
            print(f"    {name:<22s} {val:>6.1f} cm")

    # ── Answer the key question ──────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  KEY QUESTION: Can we predict chest/waist/hip within +/-3cm?")
    print(f"{'=' * 65}")
    sizing_targets = ["chest_girth_cm", "waist_girth_cm", "hip_girth_cm"]
    for t in sizing_targets:
        mae = best_results[t]["mae"]
        verdict = "YES" if mae < 3.0 else "NO"
        print(f"  {t:<22s}  MAE = {mae:.2f} cm  -->  {verdict}")

    all_within = all(best_results[t]["mae"] < 3.0 for t in sizing_targets)
    print(f"\n  Overall: {'ALL within +/-3cm' if all_within else 'NOT all within +/-3cm'}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
