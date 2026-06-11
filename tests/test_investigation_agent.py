"""Tests for the Phase 6 Investigation Agent (grounding, schema, fallback, batch).

These run fully offline (no Groq calls): fallback mode plus a fake client to exercise the
Groq parse path and the graceful degradation path.
"""

from __future__ import annotations

import json

import pytest

from fraudlab.agent.investigator import InvestigationAgent
from fraudlab.agent.schemas import (
    FRONTEND_FIELDS,
    REPORT_FIELDS,
    ExplanationValidationError,
    is_valid_report,
    validate_explanation,
)


def _explanation(decision="REVIEW", confidence="HIGH", prob=0.94):
    return {
        "transaction_id": "T0000020057",
        "fraud_probability": prob,
        "decision": decision,
        "confidence": confidence,
        "contributors": [
            {"feature": "Device Risk", "impact": 6.87, "direction": "increase"},
            {"feature": "Velocity Spike (1 Hour)", "impact": 1.16, "direction": "increase"},
            {"feature": "Transaction City", "impact": -0.22, "direction": "decrease"},
        ],
    }


# --------------------------------------------------------------------- validation

def test_validate_requires_mandatory_keys():
    with pytest.raises(ExplanationValidationError):
        validate_explanation({"decision": "BLOCK"})


def test_validate_rejects_bad_enums():
    bad = _explanation(decision="MAYBE")
    with pytest.raises(ExplanationValidationError):
        validate_explanation(bad)


def test_validate_normalizes():
    ev = validate_explanation(_explanation())
    assert ev["transaction_id"] == "T0000020057"
    assert ev["decision"] == "REVIEW"
    assert len(ev["contributors"]) == 3


# ------------------------------------------------------------------- fallback mode

def test_fallback_agent_offline():
    agent = InvestigationAgent(use_groq=False)  # no client -> deterministic
    assert agent.groq_enabled is False
    report = agent.investigate(_explanation())
    assert agent.last_mode == "fallback"
    assert set(report) == set(FRONTEND_FIELDS)
    assert is_valid_report(report)


def test_fallback_is_deterministic():
    agent = InvestigationAgent(use_groq=False)
    r1 = agent.investigate(_explanation())
    r2 = agent.investigate(_explanation())
    assert r1 == r2


def test_recommendation_mapping():
    agent = InvestigationAgent(use_groq=False)
    assert agent.investigate(_explanation("ALLOW", "LOW", 0.05))["recommendation"] == \
        "Proceed without intervention."
    assert agent.investigate(_explanation("REVIEW", "MEDIUM", 0.5))["recommendation"] == \
        "Escalate for analyst verification."
    assert agent.investigate(_explanation("BLOCK", "HIGH", 0.97))["recommendation"] == \
        "Prevent execution and initiate fraud protocols."


def test_next_steps_mapping():
    agent = InvestigationAgent(use_groq=False)
    assert agent.investigate(_explanation("ALLOW", "LOW", 0.05))["next_steps"] == \
        ["Monitor for recurrence."]
    block_steps = agent.investigate(_explanation("BLOCK", "HIGH", 0.97))["next_steps"]
    assert any("Suspend" in s for s in block_steps)


def test_confidence_explanation_mentions_level():
    agent = InvestigationAgent(use_groq=False)
    for level in ("LOW", "MEDIUM", "HIGH"):
        r = agent.investigate(_explanation("REVIEW", level, 0.5))
        assert level in r["confidence_explanation"]


def test_signals_only_from_contributors():
    agent = InvestigationAgent(use_groq=False)
    r = agent.investigate(_explanation())
    joined = " ".join(r["signals_detected"])
    assert "Device Risk" in joined and "Velocity Spike" in joined
    # No invented signals / raw names.
    assert "__" not in joined


def test_no_contributors_states_absence():
    exp = _explanation()
    exp["contributors"] = []
    agent = InvestigationAgent(use_groq=False)
    r = agent.investigate(exp)
    assert "No individual contributing signals" in r["signals_detected"][0]


# ------------------------------------------------------------------- groq + batch

class _FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _FakeClient:
    """Minimal stand-in for the Groq client (no network)."""

    def __init__(self, content, raise_error=False):
        self._content = content
        self._raise = raise_error
        self.calls = 0

        class _Completions:
            def create(_self, **kwargs):
                self.calls += 1
                if self._raise:
                    raise RuntimeError("groq down")
                return type("R", (), {"choices": [_FakeChoice(self._content)]})

        self.chat = type("Chat", (), {"completions": _Completions()})


def test_groq_path_overlays_narrative():
    content = json.dumps({
        "summary": "LLM summary.",
        "signals_detected": ["Device Risk elevated."],
        "confidence_explanation": "LLM confidence note.",
    })
    agent = InvestigationAgent(client=_FakeClient(content), use_groq=True)
    r = agent.investigate(_explanation("BLOCK", "HIGH", 0.97))
    assert agent.last_mode == "groq"
    assert r["summary"] == "LLM summary."
    assert r["signals_detected"] == ["Device Risk elevated."]
    # recommendation/next_steps stay policy-derived, never from the LLM.
    assert r["recommendation"] == "Prevent execution and initiate fraud protocols."


def test_groq_failure_falls_back():
    agent = InvestigationAgent(client=_FakeClient("", raise_error=True), use_groq=True)
    r = agent.investigate(_explanation())
    assert agent.last_mode == "fallback"
    assert is_valid_report(r)
    assert set(r) == set(FRONTEND_FIELDS)


def test_batch_reuses_client():
    content = json.dumps({"summary": "s", "signals_detected": ["x"], "confidence_explanation": "c"})
    client = _FakeClient(content)
    agent = InvestigationAgent(client=client, use_groq=True)
    reports = agent.investigate_batch([_explanation(), _explanation("BLOCK", "HIGH", 0.99)])
    assert len(reports) == 2
    assert client.calls == 2  # one call per item, same client reused
    for r in reports:
        assert set(r) == set(FRONTEND_FIELDS)


def test_frontend_response_single_and_list():
    agent = InvestigationAgent(use_groq=False)
    single = agent.frontend_response(_explanation())
    assert isinstance(single, dict) and set(single) == set(FRONTEND_FIELDS)
    many = agent.frontend_response([_explanation(), _explanation()])
    assert isinstance(many, list) and len(many) == 2


def test_report_fields_exact():
    agent = InvestigationAgent(use_groq=False)
    r = agent.investigate(_explanation())
    assert all(k in r for k in REPORT_FIELDS)
    assert r["transaction_id"] == "T0000020057"
