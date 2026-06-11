"""InvestigationAgent: grounded fraud-investigation narratives (Phase 6).

Turns a Phase 5 explanation into a five-section investigation report. Groq
(``llama-3.3-70b-versatile``, deterministic) generates the prose for ``summary``,
``signals_detected``, and ``confidence_explanation``; ``recommendation`` and ``next_steps``
are *always* derived from the decision policy (never from the LLM) so the actionable output
is guaranteed correct and grounded.

If Groq is unavailable (no key, no SDK, or any API error) the agent transparently falls back
to deterministic local summaries built from :mod:`fraudlab.explainability.formatter` evidence.
The frontend contract is byte-identical in both modes; the system keeps working offline.
"""

from __future__ import annotations

from typing import Any

from fraudlab.agent import prompts
from fraudlab.agent.schemas import (
    NEXT_STEPS_BY_DECISION,
    RECOMMENDATION_BY_DECISION,
    is_valid_report,
    validate_explanation,
)


class InvestigationAgent:
    """Produces grounded investigation reports, Groq-backed with deterministic fallback."""

    def __init__(self, *, client: Any | None = None, use_groq: bool = True,
                 model: str = prompts.GROQ_MODEL) -> None:
        self.model = model
        self._client = client
        self._use_groq = bool(use_groq and client is not None)
        self.last_mode: str = "fallback"

    # --- construction ----------------------------------------------------------

    @classmethod
    def load(cls, *, use_groq: bool = True) -> "InvestigationAgent":
        """Build an agent, initializing a reusable Groq client when possible."""
        client = None
        if use_groq and prompts.api_key():
            try:
                from groq import Groq

                client = Groq(api_key=prompts.api_key())
            except Exception:
                client = None
        return cls(client=client, use_groq=use_groq and client is not None)

    @property
    def groq_enabled(self) -> bool:
        return self._use_groq and self._client is not None

    # --- public API ------------------------------------------------------------

    def investigate(self, explanation: dict[str, Any]) -> dict[str, Any]:
        """Return the frontend investigation report for a single explanation."""
        evidence = validate_explanation(explanation)
        report = self._deterministic_report(evidence)  # always valid + grounded baseline

        if self.groq_enabled:
            try:
                llm = self._groq_sections(evidence)
                for key in ("summary", "signals_detected", "confidence_explanation"):
                    if llm.get(key):
                        report[key] = llm[key]
                self.last_mode = "groq"
            except Exception:
                self.last_mode = "fallback"
        else:
            self.last_mode = "fallback"

        return {"transaction_id": evidence["transaction_id"], **report}

    def investigate_batch(self, explanations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Investigate many explanations, reusing the same client (and prompt scaffolding)."""
        return [self.investigate(e) for e in explanations]

    def frontend_response(self, explanations) -> Any:
        """Frontend contract: single dict for one explanation, list for many."""
        if isinstance(explanations, dict):
            return self.investigate(explanations)
        return self.investigate_batch(list(explanations))

    # --- Groq path -------------------------------------------------------------

    def _groq_sections(self, evidence: dict[str, Any]) -> dict[str, Any]:
        """Call Groq and return the narrative sections (raises on any failure)."""
        import json

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": prompts.build_user_prompt(evidence)},
            ],
            temperature=prompts.TEMPERATURE,
            top_p=prompts.TOP_P,
            max_tokens=prompts.MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        signals = data.get("signals_detected")
        if isinstance(signals, str):
            signals = [signals]
        return {
            "summary": (data.get("summary") or "").strip() or None,
            "signals_detected": [str(s) for s in signals] if isinstance(signals, list) else None,
            "confidence_explanation": (data.get("confidence_explanation") or "").strip() or None,
        }

    # --- deterministic fallback ------------------------------------------------

    def _deterministic_report(self, evidence: dict[str, Any]) -> dict[str, Any]:
        """Five-section report derived purely from the deterministic evidence."""
        decision = evidence["decision"]
        confidence = evidence["confidence"]
        prob = round(evidence["fraud_probability"], 4)
        contributors = evidence["contributors"]

        return {
            "summary": self._summary(decision, prob, confidence),
            "signals_detected": self._signals(contributors),
            "recommendation": RECOMMENDATION_BY_DECISION[decision],
            "confidence_explanation": self._confidence_explanation(confidence, prob, contributors),
            "next_steps": list(NEXT_STEPS_BY_DECISION[decision]),
        }

    @staticmethod
    def _summary(decision: str, prob: float, confidence: str) -> str:
        if decision == "BLOCK":
            return (
                f"A high-risk transaction (fraud probability {prob}) was blocked and routed "
                f"to fraud protocols with {confidence} confidence."
            )
        if decision == "REVIEW":
            return (
                f"A transaction with elevated fraud risk (probability {prob}) was escalated "
                f"for analyst review with {confidence} confidence."
            )
        return (
            f"A transaction with low fraud risk (probability {prob}) was allowed with "
            f"{confidence} confidence."
        )

    @staticmethod
    def _signals(contributors: list[dict[str, Any]]) -> list[str]:
        if not contributors:
            return ["No individual contributing signals were available for this transaction."]
        signals: list[str] = []
        for c in contributors:
            verb = "increased" if c["direction"] == "increase" else "reduced"
            signals.append(f"{c['feature']} {verb} fraud likelihood (impact {c['impact']:+.2f}).")
        return signals

    @staticmethod
    def _confidence_explanation(
        confidence: str, prob: float, contributors: list[dict[str, Any]]
    ) -> str:
        strong = sum(1 for c in contributors if c["direction"] == "increase" and abs(c["impact"]) >= 1.0)
        if confidence == "HIGH":
            extra = (
                f" Multiple independent indicators ({strong}) contributed strongly."
                if strong >= 2 else ""
            )
            return (
                f"The system assigned HIGH confidence because the fraud probability ({prob}) "
                f"is decisive.{extra}"
            )
        if confidence == "MEDIUM":
            return (
                f"The system assigned MEDIUM confidence because the fraud probability ({prob}) "
                "is moderate and not fully decisive."
            )
        return (
            f"The system assigned LOW confidence because the fraud probability ({prob}) sits "
            "far from the decision boundaries."
        )
