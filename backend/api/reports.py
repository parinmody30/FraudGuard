"""Reports + system-status routes.

``GET /api/reports/summary`` serves the offline benchmark artifacts already produced by the
training scripts (``reports/leaderboard.csv`` and ``reports/benchmark_results.json``) so the
frontend Reports page can display them. ``GET /api/system/status`` reports which optional
components (database, Groq key, model artifacts) are available, for the Settings page.
Both routes are read-only and never fail hard: missing artifacts surface as empty fields.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from fastapi import APIRouter

from backend.api.dependencies import API_VERSION

router = APIRouter(tags=["reports"])

REPORTS_DIR = "reports"
LEADERBOARD_PATH = os.path.join(REPORTS_DIR, "leaderboard.csv")
BENCHMARK_PATH = os.path.join(REPORTS_DIR, "benchmark_results.json")
MODEL_PATH = os.path.join("models", "champion_model.joblib")
METADATA_PATH = os.path.join("models", "champion_metadata.json")

_NUMERIC_FIELDS = {
    "rank", "pr_auc", "roc_auc", "precision", "recall", "f1", "recall_easy", "recall_medium",
    "recall_hard", "recall_at_1pct", "recall_at_2pct", "recall_at_5pct",
}


def _to_number(value: str) -> float | int | None:
    if value in ("", None):
        return None
    try:
        num = float(value)
    except ValueError:
        return None
    return int(num) if num.is_integer() and "." not in value else num


def _read_leaderboard() -> list[dict[str, Any]]:
    if not os.path.exists(LEADERBOARD_PATH):
        return []
    with open(LEADERBOARD_PATH, newline="", encoding="utf-8") as fh:
        rows = []
        for row in csv.DictReader(fh):
            rows.append({k: (_to_number(v) if k in _NUMERIC_FIELDS else v) for k, v in row.items()})
        return rows


def _read_benchmark() -> dict[str, Any]:
    if not os.path.exists(BENCHMARK_PATH):
        return {}
    with open(BENCHMARK_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/api/reports/summary")
def reports_summary() -> dict[str, Any]:
    """Model leaderboard + CTGAN (hybrid data) experiment outcome from the offline benchmark."""
    benchmark = _read_benchmark()
    datasets = benchmark.get("datasets", {}) if isinstance(benchmark, dict) else {}
    return {
        "leaderboard": _read_leaderboard(),
        "benchmark_config": datasets.get("config"),
        "hybrid_composition": datasets.get("hybrid_composition"),
        "ctgan_decision": benchmark.get("decision"),
        "improvement": benchmark.get("improvement", []),
    }


@router.get("/api/system/status")
def system_status() -> dict[str, Any]:
    """Availability of the optional components the UI depends on."""
    from backend import db

    return {
        "api": "ok",
        "version": API_VERSION,
        "database": bool(db.DB_AVAILABLE),
        "groq_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "model_available": os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH),
        "reports_available": os.path.exists(LEADERBOARD_PATH),
    }
