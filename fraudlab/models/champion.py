"""Champion model (XGBoost) training + central decision/confidence policy (Phase 4).

Operationalizes the Phase 3 champion: it reuses the Phase 3 preprocessing
(``preprocessing.py``) and the *exact* XGBoost hyperparameters (``trainer.build_models``)
with no tuning, trains on the same user-grouped split, evaluates, and packages a
deployable bundle (fitted preprocessor + fitted model + feature names + metadata).

Decision thresholds and confidence bands live here as the single source of truth shared by
the predictor, the scoring CLI, and (later) the frontend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from fraudlab.models.metrics import compute_metrics
from fraudlab.models.preprocessing import build_design, grouped_split, make_preprocessor
from fraudlab.models.trainer import build_models

CHAMPION_MODEL = "xgboost"


# --------------------------------------------------------------- decision policy

@dataclass(frozen=True, slots=True)
class DecisionThresholds:
    """Probability cut-offs for Allow / Review / Block (configurable, defaults locked)."""

    review: float = 0.40  # >= review -> REVIEW
    block: float = 0.80   # >= block  -> BLOCK

    def decide(self, prob: float) -> str:
        if prob >= self.block:
            return "BLOCK"
        if prob >= self.review:
            return "REVIEW"
        return "ALLOW"


@dataclass(frozen=True, slots=True)
class ConfidenceBands:
    """Probability bands mapping to LOW / MEDIUM / HIGH confidence."""

    medium: float = 0.60  # >= medium -> MEDIUM
    high: float = 0.85    # > high    -> HIGH

    def level(self, prob: float) -> str:
        if prob > self.high:
            return "HIGH"
        if prob >= self.medium:
            return "MEDIUM"
        return "LOW"


DEFAULT_THRESHOLDS = DecisionThresholds()
DEFAULT_CONFIDENCE = ConfidenceBands()


# --------------------------------------------------------------------- bundle

@dataclass(slots=True)
class ChampionBundle:
    """Everything needed to score unseen transactions with the champion model."""

    preprocessor: Any
    model: Any  # fitted XGBClassifier
    feature_names: list[str]
    metadata: dict[str, Any]
    thresholds: DecisionThresholds = field(default_factory=DecisionThresholds)
    confidence: ConfidenceBands = field(default_factory=ConfidenceBands)


# --------------------------------------------------------------------- training

def train_champion(
    df: pd.DataFrame,
    *,
    simulator_config: dict[str, Any] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[ChampionBundle, dict[str, Any]]:
    """Train the XGBoost champion on a simulator dataset and return ``(bundle, metrics)``.

    Pipeline: grouped split by user -> fit preprocessor on train -> fit XGBoost (Phase 3
    hyperparameters, no tuning) -> evaluate on the held-out users.
    """
    design = build_design(df)
    train_idx, test_idx = grouped_split(design, test_size=test_size, random_state=random_state)

    preprocessor = make_preprocessor(design)
    preprocessor.fit(design.X.iloc[train_idx])  # train-only fit (no leakage)
    X_train = preprocessor.transform(design.X.iloc[train_idx])
    X_test = preprocessor.transform(design.X.iloc[test_idx])
    y_train, y_test = design.y[train_idx], design.y[test_idx]
    tiers_test = design.tiers[test_idx]

    # Reuse the exact Phase 3 XGBoost wrapper (sets scale_pos_weight from train imbalance).
    wrapper = build_models(random_state=random_state)[0][CHAMPION_MODEL]
    wrapper.fit(X_train, y_train)
    model = wrapper.estimator  # fitted XGBClassifier

    scores = wrapper.score(X_test)
    labels = wrapper.predict_label(X_test)
    metrics = compute_metrics(y_test, scores, labels, tiers_test)

    feature_names = [str(c) for c in preprocessor.get_feature_names_out()]
    metadata = {
        "model_type": "XGBoostClassifier",
        "champion": CHAMPION_MODEL,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "test_size": test_size,
        "simulator_config": simulator_config or {},
        "dataset_size": int(len(df)),
        "n_users": int(np.unique(design.groups).size),
        "fraud_rate": round(float(design.y.mean()), 6),
        "n_features": len(feature_names),
        "split": {
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "test_fraud": int(y_test.sum()),
            "users_disjoint": bool(
                set(np.unique(design.groups[train_idx])).isdisjoint(
                    set(np.unique(design.groups[test_idx]))
                )
            ),
        },
        "metrics": {
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "recall_hard": metrics["recall_hard"],
            "recall_easy": metrics["recall_easy"],
            "recall_medium": metrics["recall_medium"],
            "recall_at_1pct": metrics["recall_at_1pct"],
            "recall_at_2pct": metrics["recall_at_2pct"],
            "recall_at_5pct": metrics["recall_at_5pct"],
        },
        "thresholds": asdict(DEFAULT_THRESHOLDS),
        "confidence_bands": asdict(DEFAULT_CONFIDENCE),
    }

    bundle = ChampionBundle(
        preprocessor=preprocessor,
        model=model,
        feature_names=feature_names,
        metadata=metadata,
    )
    return bundle, metrics
