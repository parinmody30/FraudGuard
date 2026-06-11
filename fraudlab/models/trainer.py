"""Detector registry for the benchmark: supervised row-level models + one unsupervised model.

Every detector is wrapped to expose a uniform interface:
* ``fit(X, y)``           - train (unsupervised models ignore ``y``).
* ``score(X) -> float[]`` - continuous fraud score, higher = riskier (for AUC / top-K).
* ``predict_label(X)``    - binary decision at the model's natural operating point.

Optional dependencies (XGBoost / LightGBM / CatBoost) are imported lazily and skipped
gracefully when unavailable, with the skip recorded so the report can state what ran.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd

RANDOM_STATE = 42


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


# --------------------------------------------------------------------------- wrappers

class SupervisedModel:
    """Wrap a probabilistic sklearn-style classifier (uses predict_proba)."""

    kind = "supervised"

    def __init__(self, name: str, estimator) -> None:
        self.name = name
        self.estimator = estimator

    def fit(self, X, y) -> "SupervisedModel":
        self.estimator.fit(X, y)
        return self

    def _proba(self, X) -> np.ndarray:
        proba = self.estimator.predict_proba(X)
        classes = list(self.estimator.classes_)
        idx = classes.index(1) if 1 in classes else 1
        return proba[:, idx]

    def score(self, X) -> np.ndarray:
        return self._proba(X)

    def predict_label(self, X) -> np.ndarray:
        return (self._proba(X) >= 0.5).astype(int)


class XGBModel(SupervisedModel):
    """XGBoost wrapper that sets ``scale_pos_weight`` from the training imbalance."""

    def fit(self, X, y) -> "XGBModel":
        y = np.asarray(y)
        pos = max(1, int((y == 1).sum()))
        neg = max(1, int((y == 0).sum()))
        self.estimator.set_params(scale_pos_weight=neg / pos)
        self.estimator.fit(X, y)
        return self


class IsolationForestModel:
    """Unsupervised anomaly detector. Higher score = more anomalous (more fraud-like)."""

    kind = "unsupervised"

    def __init__(self, name: str, estimator) -> None:
        self.name = name
        self.estimator = estimator

    def fit(self, X, y=None) -> "IsolationForestModel":
        self.estimator.fit(X)  # labels intentionally ignored
        return self

    def score(self, X) -> np.ndarray:
        # decision_function: lower = more anomalous; negate so higher = more fraud-like.
        return -self.estimator.decision_function(X)

    def predict_label(self, X) -> np.ndarray:
        return (self.estimator.predict(X) == -1).astype(int)


# --------------------------------------------------------------------------- registry

def build_models(*, random_state: int = RANDOM_STATE) -> tuple[dict[str, Any], list[str]]:
    """Instantiate all available detectors. Returns ``(models, skipped)``."""
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        IsolationForest,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression

    models: dict[str, Any] = {}
    skipped: list[str] = []

    models["logistic_regression"] = SupervisedModel(
        "logistic_regression",
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
    )
    models["random_forest"] = SupervisedModel(
        "random_forest",
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    )
    models["gradient_boosting"] = SupervisedModel(
        "gradient_boosting",
        GradientBoostingClassifier(random_state=random_state),
    )

    if _available("xgboost"):
        from xgboost import XGBClassifier

        models["xgboost"] = XGBModel(
            "xgboost",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            ),
        )
    else:
        skipped.append("xgboost")

    if _available("lightgbm"):
        from lightgbm import LGBMClassifier

        models["lightgbm"] = SupervisedModel(
            "lightgbm",
            LGBMClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
        )
    else:
        skipped.append("lightgbm")

    if _available("catboost"):
        from catboost import CatBoostClassifier

        models["catboost"] = SupervisedModel(
            "catboost",
            CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.1,
                auto_class_weights="Balanced",
                random_seed=random_state,
                verbose=False,
            ),
        )
    else:
        skipped.append("catboost")

    models["isolation_forest"] = IsolationForestModel(
        "isolation_forest",
        IsolationForest(n_estimators=200, random_state=random_state, n_jobs=-1),
    )

    return models, skipped


def run_model(model, X_train, y_train, X_test) -> tuple[np.ndarray, np.ndarray]:
    """Fit a wrapped model and return ``(test_scores, test_labels)``."""
    model.fit(X_train, y_train)
    return model.score(X_test), model.predict_label(X_test)
