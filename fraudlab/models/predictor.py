"""ChampionPredictor: load the champion bundle and score unseen transactions (Phase 4).

Exposes the internal prediction API used by the scoring CLI and (in Phase 5) the frontend:
``load`` / ``predict_proba`` / ``predict`` / ``predict_batch`` / ``decision``. Batch scoring
appends the three mandatory frontend-contract fields: ``fraud_probability``, ``decision``,
``confidence``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudlab.models.champion import ChampionBundle
from fraudlab.models.persistence import DEFAULT_MODEL_PATH, load_bundle
from fraudlab.models.preprocessing import build_design

DECISION_LABEL = 0.5  # binary fraud cut-off for predict() (sklearn-style)


class ChampionPredictor:
    """Scores transactions with the persisted champion model and decision policy."""

    def __init__(self, bundle: ChampionBundle) -> None:
        self.bundle = bundle
        self._fraud_class_idx = self._resolve_fraud_index(bundle.model)

    # --- construction ----------------------------------------------------------

    @classmethod
    def load(cls, model_path: str = DEFAULT_MODEL_PATH) -> "ChampionPredictor":
        """Load a predictor from a persisted bundle artifact."""
        return cls(load_bundle(model_path=model_path))

    @staticmethod
    def _resolve_fraud_index(model) -> int:
        classes = list(getattr(model, "classes_", [0, 1]))
        return classes.index(1) if 1 in classes else 1

    # --- core scoring ----------------------------------------------------------

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted preprocessor to raw transactions (same design as training)."""
        design = build_design(df)
        return self.bundle.preprocessor.transform(design.X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return the fraud probability for each transaction in ``df``."""
        X = self._features(df)
        proba = self.bundle.model.predict_proba(X)
        return proba[:, self._fraud_class_idx]

    def predict(self, df: pd.DataFrame, *, threshold: float = DECISION_LABEL) -> np.ndarray:
        """Return a binary fraud label per transaction (1 = fraud) at ``threshold``."""
        return (self.predict_proba(df) >= threshold).astype(int)

    def decision(self, probability: float) -> str:
        """Map a single fraud probability to ALLOW / REVIEW / BLOCK."""
        return self.bundle.thresholds.decide(float(probability))

    def confidence(self, probability: float) -> str:
        """Map a single fraud probability to LOW / MEDIUM / HIGH confidence."""
        return self.bundle.confidence.level(float(probability))

    # --- batch -----------------------------------------------------------------

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ``df`` plus ``fraud_probability``, ``decision``, ``confidence`` columns."""
        probs = self.predict_proba(df)
        out = df.copy()
        out["fraud_probability"] = np.round(probs, 6)
        out["decision"] = [self.bundle.thresholds.decide(float(p)) for p in probs]
        out["confidence"] = [self.bundle.confidence.level(float(p)) for p in probs]
        return out

    def predict_response(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Return the mandatory frontend-contract records (one per transaction).

        Shape: ``{transaction_id, fraud_probability, decision, confidence}``.
        """
        probs = self.predict_proba(df)
        ids = (
            df["transaction_id"].astype(str).tolist()
            if "transaction_id" in df.columns
            else [str(i) for i in range(len(df))]
        )
        return [
            {
                "transaction_id": ids[i],
                "fraud_probability": round(float(probs[i]), 6),
                "decision": self.bundle.thresholds.decide(float(probs[i])),
                "confidence": self.bundle.confidence.level(float(probs[i])),
            }
            for i in range(len(probs))
        ]
