"""Leaderboards, Simulator-vs-Hybrid improvement, and the final decision rule.

Ranks each dataset's models by PR-AUC, computes per-model ``Hybrid - Simulator`` deltas for
every metric, and applies the Phase-3 decision rules to recommend whether the hybrid dataset
is worth using for model training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Metrics surfaced in the leaderboard / CSV, in display order.
LEADERBOARD_METRICS: tuple[str, ...] = (
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "recall_easy",
    "recall_medium",
    "recall_hard",
    "recall_at_1pct",
    "recall_at_2pct",
    "recall_at_5pct",
)

# A per-model PR-AUC / recall_hard delta within +/- this band counts as "no change".
NEUTRAL_BAND = 0.01


def build_leaderboard(results: dict[str, dict], *, sort_by: str = "pr_auc") -> pd.DataFrame:
    """Rank models (row-level + sequential) by ``sort_by`` descending."""
    rows = []
    for name, m in results.items():
        if m.get("skipped"):
            continue
        row = {"model": name}
        for metric in LEADERBOARD_METRICS:
            row[metric] = m.get(metric)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(sort_by, ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _val(x: Any) -> float:
    return float(x) if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)) else float("nan")


def compute_improvement(
    simulator: dict[str, dict], hybrid: dict[str, dict]
) -> pd.DataFrame:
    """Per-model ``Hybrid - Simulator`` deltas for every leaderboard metric."""
    common = [m for m in simulator if m in hybrid]
    rows = []
    for name in common:
        s, h = simulator[name], hybrid[name]
        if s.get("skipped") or h.get("skipped"):
            continue
        row: dict[str, Any] = {"model": name}
        for metric in LEADERBOARD_METRICS:
            sv, hv = _val(s.get(metric)), _val(h.get(metric))
            row[f"sim_{metric}"] = s.get(metric)
            row[f"hyb_{metric}"] = h.get(metric)
            row[f"delta_{metric}"] = (
                round(hv - sv, 4) if not (np.isnan(sv) or np.isnan(hv)) else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _delta_stats(improvement: pd.DataFrame, metric: str) -> dict[str, Any]:
    col = f"delta_{metric}"
    deltas = [d for d in improvement[col].tolist() if d is not None] if not improvement.empty else []
    if not deltas:
        return {"n": 0, "improved": 0, "degraded": 0, "neutral": 0, "median_delta": None}
    improved = sum(1 for d in deltas if d > NEUTRAL_BAND)
    degraded = sum(1 for d in deltas if d < -NEUTRAL_BAND)
    neutral = len(deltas) - improved - degraded
    return {
        "n": len(deltas),
        "improved": improved,
        "degraded": degraded,
        "neutral": neutral,
        "median_delta": round(float(np.median(deltas)), 4),
    }


def decide(improvement: pd.DataFrame) -> dict[str, Any]:
    """Apply the decision rules to recommend HYBRID value (A / B / C)."""
    pr = _delta_stats(improvement, "pr_auc")
    hard = _delta_stats(improvement, "recall_hard")

    def _majority_improves(stats: dict[str, Any]) -> bool:
        return stats["n"] > 0 and stats["improved"] > stats["n"] / 2

    def _majority_degrades(stats: dict[str, Any]) -> bool:
        return stats["n"] > 0 and stats["degraded"] > stats["n"] / 2

    if _majority_improves(pr) or _majority_improves(hard):
        verdict = "A"
        recommendation = "HYBRID DATASET ADDS VALUE"
    elif _majority_degrades(pr) and (hard["n"] == 0 or _majority_degrades(hard)):
        verdict = "C"
        recommendation = "ABANDON CTGAN FOR MODEL TRAINING; USE SIMULATOR ONLY"
    else:
        verdict = "B"
        recommendation = "CTGAN PROVIDES DIVERSITY BUT LIMITED DETECTION BENEFIT"

    return {
        "verdict": verdict,
        "recommendation": recommendation,
        "pr_auc_delta_stats": pr,
        "recall_hard_delta_stats": hard,
        "neutral_band": NEUTRAL_BAND,
    }
