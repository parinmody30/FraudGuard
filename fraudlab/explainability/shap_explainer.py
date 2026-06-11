"""ChampionExplainer: SHAP TreeExplainer over the persisted champion XGBoost model.

Loads the Phase 4 bundle (preprocessor + model + decision policy) and explains predictions
without retraining. One-hot columns are aggregated back to their base feature so analysts see
per-attribute attribution. The TreeExplainer and the feature->base mapping are built once and
reused across all requests (single-init, vectorized batch).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudlab.explainability.formatter import friendly_label
from fraudlab.models.champion import ChampionBundle
from fraudlab.models.persistence import DEFAULT_MODEL_PATH, load_bundle
from fraudlab.models.preprocessing import (
    CATEGORICAL_FEATURES,
    build_design,
)

TOP_POSITIVE = 5
TOP_NEGATIVE = 3


def _base_feature(name: str) -> str:
    """Map an engineered column name back to its base schema feature.

    ``continuous__amount`` -> ``amount``; ``boolean__is_foreign`` -> ``is_foreign``;
    ``categorical__city_Mumbai`` -> ``city`` (longest known categorical prefix).
    """
    rest = name.split("__", 1)[1] if "__" in name else name
    if name.startswith("categorical__"):
        for col in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
            if rest == col or rest.startswith(col + "_"):
                return col
        return rest.split("_", 1)[0]
    return rest


class ChampionExplainer:
    """Explains champion-model predictions with SHAP TreeExplainer (single-init, reused)."""

    def __init__(self, bundle: ChampionBundle) -> None:
        import shap

        self.bundle = bundle
        self.model = bundle.model
        self.preprocessor = bundle.preprocessor
        self.feature_names: list[str] = list(bundle.feature_names)
        self._fraud_idx = self._resolve_fraud_index(self.model)

        # Build the explainer ONCE and reuse it for every request.
        self._explainer = shap.TreeExplainer(self.model)

        # Pre-compute base-feature grouping (feature engineering map) ONCE.
        bases = [_base_feature(n) for n in self.feature_names]
        self._base_of = bases
        self._groups: dict[str, list[int]] = {}
        for i, b in enumerate(bases):
            self._groups.setdefault(b, []).append(i)

    # --- construction ----------------------------------------------------------

    @classmethod
    def load(cls, model_path: str = DEFAULT_MODEL_PATH) -> "ChampionExplainer":
        """Load a champion bundle from disk and build the explainer."""
        return cls(load_bundle(model_path=model_path))

    @staticmethod
    def _resolve_fraud_index(model) -> int:
        classes = list(getattr(model, "classes_", [0, 1]))
        return classes.index(1) if 1 in classes else 1

    # --- internals -------------------------------------------------------------

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.preprocessor.transform(build_design(df).X), dtype=float)

    def _shap_matrix(self, X: np.ndarray) -> np.ndarray:
        """Return per-row, per-engineered-feature SHAP values for the fraud class."""
        sv = self._explainer.shap_values(X)
        if isinstance(sv, list):  # legacy multiclass list output
            sv = sv[self._fraud_idx] if len(sv) > self._fraud_idx else sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (rows, features, classes)
            sv = sv[:, :, self._fraud_idx]
        return sv

    def _aggregate(self, shap_row: np.ndarray) -> dict[str, float]:
        """Sum SHAP values of engineered columns into their base feature."""
        return {
            base: float(shap_row[idxs].sum()) for base, idxs in self._groups.items()
        }

    def top_contributors(self, shap_row: np.ndarray) -> list[dict[str, Any]]:
        """Top positive/negative base-feature contributors, sorted by |impact|."""
        agg = self._aggregate(shap_row)
        positives = sorted(
            ((b, v) for b, v in agg.items() if v > 0), key=lambda kv: kv[1], reverse=True
        )[:TOP_POSITIVE]
        negatives = sorted(
            ((b, v) for b, v in agg.items() if v < 0), key=lambda kv: abs(kv[1]), reverse=True
        )[:TOP_NEGATIVE]
        merged = positives + negatives
        merged.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return [
            {
                "feature": friendly_label(base),
                "impact": round(val, 4),
                "direction": "increase" if val > 0 else "decrease",
            }
            for base, val in merged
        ]

    def _probabilities(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(X)
        return proba[:, self._fraud_idx]

    def _response(
        self, df: pd.DataFrame, X: np.ndarray, shap_mat: np.ndarray, probs: np.ndarray
    ) -> list[dict[str, Any]]:
        ids = (
            df["transaction_id"].astype(str).tolist()
            if "transaction_id" in df.columns
            else [str(i) for i in range(len(df))]
        )
        th, cb = self.bundle.thresholds, self.bundle.confidence
        out = []
        for i in range(len(df)):
            p = float(probs[i])
            out.append(
                {
                    "transaction_id": ids[i],
                    "fraud_probability": round(p, 6),
                    "decision": th.decide(p),
                    "confidence": cb.level(p),
                    "contributors": self.top_contributors(shap_mat[i]),
                }
            )
        return out

    # --- public API ------------------------------------------------------------

    def explain(self, df: pd.DataFrame) -> dict[str, Any]:
        """Explain a single transaction (first row of ``df``)."""
        row = df.iloc[[0]]
        X = self._transform(row)
        shap_mat = self._shap_matrix(X)
        probs = self._probabilities(X)
        return self._response(row, X, shap_mat, probs)[0]

    def explain_batch(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Explain every transaction in ``df`` (single vectorized SHAP pass)."""
        if len(df) == 0:
            return []
        X = self._transform(df)
        shap_mat = self._shap_matrix(X)
        probs = self._probabilities(X)
        return self._response(df, X, shap_mat, probs)

    def frontend_response(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Alias for :meth:`explain_batch` - the Investigation Workspace contract."""
        return self.explain_batch(df)
