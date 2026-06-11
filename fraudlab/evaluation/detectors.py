"""Detector definitions used by the benchmark.

Three deliberately different detectors bracket the difficulty of the dataset:

* ``rule_baseline`` — a hand-written per-row heuristic (the kind of naive rule engine the
  dataset is meant to defeat). It has *no* training and *no* sequence memory.
* ``logistic`` — a linear model on standardized features (weak interactions).
* ``gradient_boosting`` — a strong row-wise tabular learner (the realistic upper bound for
  a model that still sees one row at a time).

All are row-wise: none consumes the ordered transaction sequence directly. The gap between
their easy-tier and hard-tier recall is the evidence that hard fraud needs sequence models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RuleBaseline:
    """Stateless per-row heuristic scorer (no fit). Returns a pseudo-probability in [0,1].

    Mirrors common naive fraud rules: unusually large vs the user's own history, risky
    device/merchant, foreign or night-time, and high short-window velocity.
    """

    name = "rule_baseline"
    trainable = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RuleBaseline":  # noqa: D401
        return self

    def predict_proba_fraud(self, X: pd.DataFrame) -> np.ndarray:
        score = np.zeros(len(X), dtype=float)
        if "amount_to_avg" in X:
            score += 0.30 * (np.clip(X["amount_to_avg"].to_numpy(), 0, 10) / 10.0)
        if "device_risk_score" in X:
            score += 0.20 * np.clip(X["device_risk_score"].to_numpy(), 0, 1)
        if "merchant_risk_score" in X:
            score += 0.15 * np.clip(X["merchant_risk_score"].to_numpy(), 0, 1)
        if "is_foreign" in X:
            score += 0.10 * (X["is_foreign"].to_numpy() > 0)
        if "hour" in X:
            hr = X["hour"].to_numpy()
            score += 0.10 * ((hr < 5) | (hr >= 23))
        if "velocity_1h" in X:
            score += 0.15 * (np.clip(X["velocity_1h"].to_numpy(), 0, 6) / 6.0)
        return np.clip(score, 0.0, 1.0)


class SklearnDetector:
    """Adapter wrapping an sklearn estimator with a uniform scoring interface."""

    trainable = True

    def __init__(self, name: str, estimator) -> None:
        self.name = name
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SklearnDetector":
        self.estimator.fit(X, y)
        return self

    def predict_proba_fraud(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.estimator.predict_proba(X)
        classes = list(self.estimator.classes_)
        idx = classes.index(1) if 1 in classes else 1
        return proba[:, idx]


def build_detectors(random_state: int = 42) -> list:
    """Instantiate the detector suite. Imports sklearn lazily so import stays cheap."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    logistic = SklearnDetector(
        "logistic",
        Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    )

    gbm = SklearnDetector(
        "gradient_boosting",
        GradientBoostingClassifier(random_state=random_state),
    )

    return [RuleBaseline(), logistic, gbm]
