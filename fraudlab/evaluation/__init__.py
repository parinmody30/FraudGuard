"""FraudLab dataset evaluation framework (Phase 1.1).

This subpackage *evaluates* datasets produced by the FraudLab generator. It is strictly
read-only with respect to the simulator: it never imports or mutates generator internals
beyond the public config/schema, and it does not generate or synthesize data.

Public entry point: :func:`evaluate_dataset`, which runs descriptive metrics, a detector
benchmark (with per-difficulty-tier recall), label-leakage detection, a sequential-
complexity analysis, and a CTGAN-readiness assessment, returning a single report dict.
"""

from __future__ import annotations

from fraudlab.evaluation.report import evaluate_dataset, format_report

__all__ = ["evaluate_dataset", "format_report"]
