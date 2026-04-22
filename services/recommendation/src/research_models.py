# intent: Load and serve trained research model variants (GBM + MLP) for size prediction
# status: done
# next: add more model variants as they are trained
# confidence: high

"""Research model loader — loads trained GBM and MLP variants from /research/models/variants/.

Provides a unified interface for:
  - GBM models (sklearn GradientBoosting, .pkl files)
  - MLP models (PyTorch, .pt files — optional, requires torch)

Each .pkl bundle contains: model, body_enc, cat_enc, size_enc, results.
MLP .pt bundles contain: model_state_dict, body_enc, cat_enc, size_enc, config, results.

NOTE: pickle is used intentionally for sklearn model serialization.
These are trusted, locally-generated model files — never load untrusted pickles.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default path to variant models (can be overridden via RESEARCH_MODELS_DIR env var)
_DEFAULT_VARIANTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "research" / "models" / "variants"

SIZE_MAP = {0: "XXS", 1: "XS", 2: "S", 3: "M", 4: "L", 5: "XL", 6: "XXL"}
SIZE_ORDER = {"XXS": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5, "XXL": 6}

# GBM variant names that have .pkl files
GBM_VARIANTS = [
    "gbm_uniform",
    "gbm_indep",
    "gbm_copula",
    "gbm_indep_tempered_a05",
    "gbm_copula_tempered_a075",
]

# MLP variant names that have .pt files (require torch)
MLP_VARIANTS = ["mlp_ce_uniform", "mlp_ce_copula", "mlp_corn_copula"]

ALL_VARIANTS = GBM_VARIANTS + MLP_VARIANTS


class ResearchModelRegistry:
    """Singleton registry that lazily loads research model variants."""

    _instance: ResearchModelRegistry | None = None
    _lock: Lock = Lock()

    def __init__(self, variants_dir: Path | None = None) -> None:
        import os
        self._variants_dir = Path(
            os.environ.get("RESEARCH_MODELS_DIR", str(variants_dir or _DEFAULT_VARIANTS_DIR))
        )
        self._loaded: dict[str, dict[str, Any]] = {}
        self._load_errors: dict[str, str] = {}
        self._torch_available: bool = False

        try:
            import torch  # noqa: F401
            self._torch_available = True
        except ImportError:
            logger.info("torch not available — MLP variants will be skipped")

    @classmethod
    def get_instance(cls) -> ResearchModelRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    # ── Loading ──

    def _load_gbm(self, name: str) -> dict[str, Any] | None:
        """Load a GBM variant from .pkl file."""
        pkl_path = self._variants_dir / f"{name}.pkl"
        if not pkl_path.exists():
            self._load_errors[name] = f"File not found: {pkl_path}"
            logger.warning("GBM variant '%s' not found at %s", name, pkl_path)
            return None

        # NOTE: pickle used intentionally — trusted local model files only
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)  # noqa: S301

        logger.info("Loaded GBM variant '%s' from %s", name, pkl_path)
        return bundle

    def _load_mlp(self, name: str) -> dict[str, Any] | None:
        """Load an MLP variant from .pt file. Requires torch."""
        if not self._torch_available:
            self._load_errors[name] = "torch not installed"
            return None

        import torch

        pt_path = self._variants_dir / f"{name}.pt"
        if not pt_path.exists():
            self._load_errors[name] = f"File not found: {pt_path}"
            logger.warning("MLP variant '%s' not found at %s", name, pt_path)
            return None

        try:
            bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
            logger.info("Loaded MLP variant '%s' from %s", name, pt_path)
            return bundle
        except Exception as e:
            self._load_errors[name] = str(e)
            logger.error("Failed to load MLP variant '%s': %s", name, e)
            return None

    def get_model(self, name: str) -> dict[str, Any] | None:
        """Get a loaded model bundle by variant name. Loads lazily on first access."""
        if name in self._loaded:
            return self._loaded[name]
        if name in self._load_errors:
            return None

        bundle = None
        if name in GBM_VARIANTS:
            bundle = self._load_gbm(name)
        elif name in MLP_VARIANTS:
            bundle = self._load_mlp(name)
        else:
            self._load_errors[name] = f"Unknown variant: {name}"

        if bundle is not None:
            self._loaded[name] = bundle
        return bundle

    def available_variants(self) -> list[str]:
        """Return list of variant names that can be loaded successfully."""
        available = []
        for name in ALL_VARIANTS:
            bundle = self.get_model(name)
            if bundle is not None:
                available.append(name)
        return available

    def get_load_errors(self) -> dict[str, str]:
        return dict(self._load_errors)

    # ── Prediction ──

    def predict_gbm(
        self,
        name: str,
        height_cm: float,
        weight_kg: float,
        age: float = 30.0,
        body_type: str = "unknown",
        category: str = "dress",
    ) -> dict[str, Any] | None:
        """Run prediction through a GBM variant.

        Returns dict with predicted_size, confidence, probabilities.
        """
        bundle = self.get_model(name)
        if bundle is None:
            return None

        model = bundle["model"]
        body_enc = bundle["body_enc"]
        cat_enc = bundle["cat_enc"]
        size_enc = bundle["size_enc"]

        # Encode inputs with fallback for unseen labels
        body_type_clean = body_type if body_type in body_enc.classes_ else "unknown"
        if body_type_clean not in body_enc.classes_:
            # If even "unknown" isn't in classes, use first class
            body_type_clean = body_enc.classes_[0]
        body_encoded = body_enc.transform([body_type_clean])[0]

        category_clean = category if category in cat_enc.classes_ else "dress"
        if category_clean not in cat_enc.classes_:
            category_clean = cat_enc.classes_[0]
        cat_encoded = cat_enc.transform([category_clean])[0]

        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m ** 2) if height_m > 0 else 22.0

        X = np.array([[height_cm, weight_kg, bmi, age, body_encoded, cat_encoded]])

        pred_idx = model.predict(X)[0]
        pred_label = size_enc.inverse_transform([pred_idx])[0]

        proba = model.predict_proba(X)[0]
        max_conf = float(proba.max())

        # Build probability map over all classes the model knows
        prob_map = {}
        for i, cls_idx in enumerate(model.classes_):
            label = size_enc.inverse_transform([cls_idx])[0]
            prob_map[label] = round(float(proba[i]), 4)

        confidence = "high" if max_conf > 0.5 else ("medium" if max_conf > 0.25 else "low")

        return {
            "predicted_size": pred_label,
            "confidence": confidence,
            "confidence_score": round(max_conf, 4),
            "probabilities": prob_map,
        }

    def predict_mlp(
        self,
        name: str,
        height_cm: float,
        weight_kg: float,
        age: float = 30.0,
        body_type: str = "unknown",
        category: str = "dress",
    ) -> dict[str, Any] | None:
        """Run prediction through an MLP variant.

        The .pt bundle contains: state_dict, X_mean, X_std, loss_type, results.
        We need to reconstruct the SizeMLP model from the state_dict.
        """
        if not self._torch_available:
            return None

        bundle = self.get_model(name)
        if bundle is None:
            return None

        import torch
        import torch.nn as nn

        # Load eval_data.pkl for encoders (shared across all variants)
        eval_data = self._get_eval_data()
        if eval_data is None:
            return None

        body_enc = eval_data["body_enc"]
        cat_enc = eval_data["cat_enc"]
        size_enc = eval_data["size_enc"]

        # Reconstruct model from state_dict
        loss_type = bundle.get("loss_type", "ce")
        input_dim = 6  # height, weight, bmi, age, body_type, category
        num_classes = 7
        out_dim = num_classes if loss_type == "ce" else num_classes - 1

        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim),
        )

        # Map state_dict keys: SizeMLP wraps in self.net
        state_dict = bundle["state_dict"]
        # Remove "net." prefix if present
        clean_dict = {}
        for k, v in state_dict.items():
            clean_k = k.replace("net.", "") if k.startswith("net.") else k
            clean_dict[clean_k] = v
        model.load_state_dict(clean_dict)
        model.eval()

        X_mean = bundle["X_mean"]
        X_std = bundle["X_std"]

        # Encode inputs
        body_type_clean = body_type if body_type in body_enc.classes_ else "unknown"
        if body_type_clean not in body_enc.classes_:
            body_type_clean = body_enc.classes_[0]
        body_encoded = body_enc.transform([body_type_clean])[0]

        category_clean = category if category in cat_enc.classes_ else "dress"
        if category_clean not in cat_enc.classes_:
            category_clean = cat_enc.classes_[0]
        cat_encoded = cat_enc.transform([category_clean])[0]

        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m ** 2) if height_m > 0 else 22.0

        X = np.array([[height_cm, weight_kg, bmi, age, float(body_encoded), float(cat_encoded)]])
        X_norm = (X - X_mean) / X_std
        features = torch.tensor(X_norm, dtype=torch.float32)

        with torch.inference_mode():
            logits = model(features)

            if loss_type == "corn":
                # CORN: cumulative product of sigmoid for class probabilities
                probs_gt = torch.sigmoid(logits).squeeze(0)
                cum_probs = torch.cumprod(probs_gt, dim=0)
                class_probs = torch.zeros(num_classes)
                class_probs[0] = 1.0 - cum_probs[0]
                for k in range(1, num_classes - 1):
                    class_probs[k] = cum_probs[k - 1] - cum_probs[k]
                class_probs[-1] = cum_probs[-1]
                class_probs = torch.clamp(class_probs, min=0.0)
                probs = class_probs
            else:
                probs = torch.softmax(logits, dim=-1).squeeze(0)

        pred_idx = int(probs.argmax().item())
        pred_label = size_enc.inverse_transform([pred_idx])[0]
        max_conf = float(probs[pred_idx].item())

        prob_map = {}
        for i in range(len(probs)):
            try:
                label = size_enc.inverse_transform([i])[0]
                prob_map[label] = round(float(probs[i].item()), 4)
            except ValueError:
                continue

        confidence = "high" if max_conf > 0.5 else ("medium" if max_conf > 0.25 else "low")

        return {
            "predicted_size": pred_label,
            "confidence": confidence,
            "confidence_score": round(max_conf, 4),
            "probabilities": prob_map,
        }

    def _get_eval_data(self) -> dict | None:
        """Load eval_data.pkl containing shared encoders."""
        if hasattr(self, "_eval_data_cache"):
            return self._eval_data_cache

        eval_path = self._variants_dir / "eval_data.pkl"
        if not eval_path.exists():
            logger.warning("eval_data.pkl not found at %s", eval_path)
            return None

        with open(eval_path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self._eval_data_cache = data
        return data

    def predict(
        self,
        name: str,
        height_cm: float,
        weight_kg: float,
        age: float = 30.0,
        body_type: str = "unknown",
        category: str = "dress",
    ) -> dict[str, Any] | None:
        """Unified predict — dispatches to GBM or MLP based on variant name."""
        if name in GBM_VARIANTS:
            return self.predict_gbm(name, height_cm, weight_kg, age, body_type, category)
        elif name in MLP_VARIANTS:
            return self.predict_mlp(name, height_cm, weight_kg, age, body_type, category)
        return None

    def predict_all(
        self,
        height_cm: float,
        weight_kg: float,
        age: float = 30.0,
        body_type: str = "unknown",
        category: str = "dress",
    ) -> dict[str, dict[str, Any]]:
        """Run prediction through all available variants. Returns {name: result}."""
        results = {}
        for name in ALL_VARIANTS:
            result = self.predict(name, height_cm, weight_kg, age, body_type, category)
            if result is not None:
                results[name] = result
        return results
