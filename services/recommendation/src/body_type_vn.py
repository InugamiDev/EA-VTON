"""Vietnamese female body type classifier (5-cluster, Mahalanobis kNN).

Based on Trieu et al. 2024 cluster definitions from N=480 VN women, age 18-50.
Input: height_cm + 5 ratios (shoulder/hip, waist/hip, shoulder/torso, torso/leg, arm/torso).
Output: cluster id (1..5), label, confidence ∈ [0, 1].

The classifier is parameter-free at training time — we use the published cluster
descriptions as centroids and a diagonal covariance estimated from VN STEPS
within-population variance. Confidence is a softmax over negative Mahalanobis
distances with temperature τ.
"""

# intent: 5-cluster VN body type classifier for style recommendation
# status: done
# next: validate against held-out simulated users + add boundary-case handling
# confidence: high

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

_DATA = Path(__file__).resolve().parent.parent / "data" / "vn_body_centroids_v1.json"


@dataclass
class BodyTypeResult:
    cluster_id: int
    label: str
    label_vi: str
    confidence: float
    distances: dict[int, float]  # per-cluster Mahalanobis distance
    feature_vector: list[float]


class VNBodyTypeClassifier:
    """Singleton classifier loaded from vn_body_centroids_v1.json."""

    _instance: "VNBodyTypeClassifier | None" = None
    _lock: Lock = Lock()

    def __init__(self, data_path: Path | None = None) -> None:
        path = data_path or _DATA
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.feature_order: list[str] = data["feature_order"]
        self.clusters: list[dict[str, Any]] = data["clusters"]
        self.feature_std: list[float] = data["feature_std"]
        self.n_features = len(self.feature_order)
        # Temperature for softmax over -d_M; tuned so confidence ≈ 0.5 when
        # nearest two clusters are roughly equidistant.
        self.temperature = 1.5

    @classmethod
    def get_instance(cls) -> "VNBodyTypeClassifier":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @staticmethod
    def _build_feature_vector(
        height_cm: float,
        shoulder_to_hip: float,
        waist_to_hip: float,
        shoulder_to_torso: float,
        torso_to_leg: float,
        arm_to_torso: float,
    ) -> list[float]:
        return [
            height_cm,
            shoulder_to_hip,
            waist_to_hip,
            shoulder_to_torso,
            torso_to_leg,
            arm_to_torso,
        ]

    def _mahalanobis(self, x: list[float], centroid: list[float]) -> float:
        """Diagonal-covariance Mahalanobis distance.

        d_M(x, μ) = sqrt( Σ ((x_j - μ_j) / σ_j)² )
        """
        s = 0.0
        for xi, mi, si in zip(x, centroid, self.feature_std):
            d = (xi - mi) / si
            s += d * d
        return math.sqrt(s)

    def predict(
        self,
        height_cm: float,
        shoulder_to_hip: float,
        waist_to_hip: float,
        shoulder_to_torso: float = 0.66,
        torso_to_leg: float = 0.45,
        arm_to_torso: float = 1.10,
    ) -> BodyTypeResult:
        """Classify a user's body type.

        Defaults for the optional ratios match the modal (Group 3) cluster so
        callers passing only the two reliable ratios still get sensible output.
        """
        x = self._build_feature_vector(
            height_cm,
            shoulder_to_hip,
            waist_to_hip,
            shoulder_to_torso,
            torso_to_leg,
            arm_to_torso,
        )

        distances: dict[int, float] = {}
        for c in self.clusters:
            distances[c["id"]] = self._mahalanobis(x, c["centroid"])

        # Softmax over negative distances with temperature
        neg_d = {cid: -d / self.temperature for cid, d in distances.items()}
        max_v = max(neg_d.values())
        exps = {cid: math.exp(v - max_v) for cid, v in neg_d.items()}
        denom = sum(exps.values())
        probs = {cid: e / denom for cid, e in exps.items()}

        best_id = min(distances, key=distances.get)
        best_cluster = next(c for c in self.clusters if c["id"] == best_id)
        confidence = probs[best_id]

        return BodyTypeResult(
            cluster_id=best_id,
            label=best_cluster["label"],
            label_vi=best_cluster["label_vi"],
            confidence=round(confidence, 4),
            distances={k: round(v, 4) for k, v in distances.items()},
            feature_vector=[round(v, 3) for v in x],
        )


def classify(
    height_cm: float,
    shoulder_to_hip: float,
    waist_to_hip: float,
    shoulder_to_torso: float = 0.66,
    torso_to_leg: float = 0.45,
    arm_to_torso: float = 1.10,
) -> BodyTypeResult:
    """Module-level convenience wrapper."""
    return VNBodyTypeClassifier.get_instance().predict(
        height_cm,
        shoulder_to_hip,
        waist_to_hip,
        shoulder_to_torso,
        torso_to_leg,
        arm_to_torso,
    )


if __name__ == "__main__":
    # Quick smoke test
    cases = [
        ("avg VN woman (Group 3 expected)", 155.0, 1.08, 0.78),
        ("petite/thin (Group 1 expected)",  149.0, 1.04, 0.74),
        ("tall slim (Group 2 expected)",    161.0, 1.13, 0.78),
        ("short fuller (Group 4 expected)", 151.0, 1.07, 0.86),
        ("broad heavy (Group 5 expected)",  155.5, 1.16, 0.92),
    ]
    for desc, h, sh, wh in cases:
        r = classify(h, sh, wh)
        print(f"{desc:38s} -> Group {r.cluster_id} ({r.label}) conf={r.confidence}")
