"""SHAP explainability for the FraudLab champion model (Phase 5).

Loads the persisted champion bundle (no retraining) and explains predictions with a
``shap.TreeExplainer``, returning analyst-friendly, frontend-ready contributor breakdowns
that answer "why was this transaction flagged?". Deterministic; no LLMs, no network.
"""

from __future__ import annotations

from fraudlab.explainability.formatter import (
    friendly_label,
    format_analyst_summary,
)
from fraudlab.explainability.shap_explainer import ChampionExplainer

__all__ = ["ChampionExplainer", "friendly_label", "format_analyst_summary"]
