"""Retrain tuned GBM size models with recomputed density-ratio weights.

This script is intentionally GBM-only: it refreshes the model artifacts served by
the recommendation API without requiring PyTorch for the older MLP experiments.
"""

# intent: make Copula/PSIS density-ratio model refresh reproducible
# status: done
# next: promote artifacts through a model registry instead of ignored pickle files
# blockers: trained model pickles are intentionally gitignored as large artifacts
# confidence: high

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

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

DATA_DIR = ROOT / "research" / "datasets" / "processed"
VARIANTS_DIR = ROOT / "research" / "models" / "variants"
EVAL_DIR = ROOT / "research" / "evaluation"

SIZE_MAP = {
    0: "XXS",
    1: "XXS",
    2: "XS",
    3: "XS",
    4: "S",
    5: "S",
    6: "S",
    7: "M",
    8: "M",
    9: "M",
    10: "L",
    11: "L",
    12: "L",
    13: "L",
    14: "XL",
    15: "XL",
    16: "XL",
    17: "XXL",
    18: "XXL",
    19: "XXL",
    20: "XXL",
}
SIZE_ORDER = {"XXS": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5, "XXL": 6}

SUBSETS = {
    "full_test": lambda h, w: np.ones(len(h), dtype=bool),
    "vn_range": lambda h, w: (h < 160) & (w < 55),
}


def load_data() -> tuple[np.ndarray, np.ndarray, LabelEncoder, LabelEncoder, LabelEncoder]:
    df = pd.read_parquet(DATA_DIR / "rtr_body_fit.parquet")
    df = df[
        df["height_cm"].notna()
        & df["weight_kg"].notna()
        & df["fit"].isin(["fit", "small", "large"])
    ].copy()

    df_fit = df[df["fit"] == "fit"].copy()
    df_fit["size_label"] = df_fit["size"].map(SIZE_MAP)
    df_fit = df_fit[df_fit["size_label"].notna()].copy()
    df_fit["body_type_clean"] = df_fit["body_type_norm"].fillna("unknown")

    body_enc = LabelEncoder()
    cat_enc = LabelEncoder()
    size_enc = LabelEncoder()

    body_encoded = body_enc.fit_transform(df_fit["body_type_clean"])
    cat_encoded = cat_enc.fit_transform(df_fit["category"])
    y = size_enc.fit_transform(df_fit["size_label"])

    X = np.column_stack(
        [
            df_fit["height_cm"].values,
            df_fit["weight_kg"].values,
            df_fit["bmi"].fillna(22.0).values,
            df_fit["age"].fillna(30).values,
            body_encoded,
            cat_encoded,
        ]
    )

    return X, y, body_enc, cat_enc, size_enc


def temper_weights(weights: np.ndarray, alpha: float) -> np.ndarray:
    tempered = np.power(np.asarray(weights, dtype=np.float64), alpha)
    return tempered / tempered.mean()


def effective_sample_size(weights: np.ndarray) -> float:
    return float((weights.sum() ** 2) / np.square(weights).sum())


def with_weight_diagnostics(base: dict[str, Any], weights: np.ndarray, alpha: float | None) -> dict[str, Any]:
    diagnostics = dict(base)
    diagnostics["alpha"] = alpha
    diagnostics["ess"] = effective_sample_size(weights)
    diagnostics["ess_pct"] = diagnostics["ess"] / len(weights) * 100
    diagnostics["weight_stats"] = {
        "mean": float(weights.mean()),
        "std": float(weights.std()),
        "max": float(weights.max()),
        "min": float(weights.min()),
        "pct_99": float(np.percentile(weights, 99)),
    }
    return diagnostics


def train_gbm(X_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray | None):
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, size_enc: LabelEncoder) -> dict[str, Any]:
    y_true_labels = size_enc.inverse_transform(y_true)
    y_pred_labels = size_enc.inverse_transform(y_pred)
    diff = np.array(
        [SIZE_ORDER.get(pred, 3) - SIZE_ORDER.get(true, 3) for true, pred in zip(y_true_labels, y_pred_labels)]
    )
    return {
        "n": int(len(y_true)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "within_1": float(np.mean(np.abs(diff) <= 1)),
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "over_size_pct": float(np.mean(diff > 0) * 100),
        "under_size_pct": float(np.mean(diff < 0) * 100),
    }


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, size_enc: LabelEncoder) -> dict[str, Any]:
    heights = X_test[:, 0]
    weights_kg = X_test[:, 1]
    y_pred = model.predict(X_test)
    return {
        subset: compute_metrics(y_test[mask_fn(heights, weights_kg)], y_pred[mask_fn(heights, weights_kg)], size_enc)
        for subset, mask_fn in SUBSETS.items()
    }


def save_bundle(
    name: str,
    model,
    body_enc: LabelEncoder,
    cat_enc: LabelEncoder,
    size_enc: LabelEncoder,
    results: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(VARIANTS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "body_enc": body_enc,
                "cat_enc": cat_enc,
                "size_enc": size_enc,
                "results": results,
                "diagnostics": diagnostics,
            },
            f,
        )


def main() -> None:
    X, y, body_enc, cat_enc, size_enc = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_heights = X_train[:, 0]
    train_weights_kg = X_train[:, 1]

    w_indep, d_indep = compute_weights_independent(train_heights, train_weights_kg)
    w_copula, d_copula = compute_weights_copula(train_heights, train_weights_kg, apply_psis=True)

    variant_weights: dict[str, tuple[np.ndarray | None, dict[str, Any]]] = {
        "gbm_uniform_current": (
            None,
            {
                "method": "uniform",
                "alpha": None,
                "ess": float(len(X_train)),
                "ess_pct": 100.0,
                "weight_stats": {"mean": 1.0, "std": 0.0, "max": 1.0, "min": 1.0, "pct_99": 1.0},
            },
        ),
        "gbm_copula_current": (w_copula, with_weight_diagnostics(d_copula, w_copula, None)),
        "gbm_indep_tempered_a05": (
            temper_weights(w_indep, 0.5),
            with_weight_diagnostics(d_indep, temper_weights(w_indep, 0.5), 0.5),
        ),
        "gbm_copula_tempered_a075": (
            temper_weights(w_copula, 0.75),
            with_weight_diagnostics(d_copula, temper_weights(w_copula, 0.75), 0.75),
        ),
    }

    tuned_results: dict[str, Any] = {}

    print(f"Training rows: {len(X_train):,}; test rows: {len(X_test):,}")
    for name, (sample_weight, diagnostics) in variant_weights.items():
        print(f"Training {name}...")
        model = train_gbm(X_train, y_train, sample_weight)
        results = evaluate_model(model, X_test, y_test, size_enc)
        tuned_results[name] = {**results, "diagnostics": diagnostics}
        if name != "gbm_copula_current":
            save_bundle(name, model, body_enc, cat_enc, size_enc, results["full_test"], diagnostics)

        full = results["full_test"]
        vn = results["vn_range"]
        print(
            f"  full acc={full['acc']:.4f} w1={full['within_1']:.4f}; "
            f"vn acc={vn['acc']:.4f} w1={vn['within_1']:.4f}; "
            f"ess={diagnostics['ess']:.0f} ({diagnostics['ess_pct']:.1f}%)"
        )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / "tuned_results.json", "w") as f:
        json.dump(tuned_results, f, indent=2)
        f.write("\n")

    print(f"Saved tuned metrics to {EVAL_DIR / 'tuned_results.json'}")
    print(f"Saved served GBM artifacts to {VARIANTS_DIR}")


if __name__ == "__main__":
    main()
