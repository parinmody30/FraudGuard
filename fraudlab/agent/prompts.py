"""Central Groq configuration and prompt construction for the Investigation Agent.

Deterministic by design: temperature 0.0 / top_p 1.0 so the same evidence yields the same
narrative. The system prompt hard-constrains the model to the supplied evidence and forbids
inventing facts. Only grounded evidence (transaction context, decision outputs, SHAP
contributors) is ever placed in the prompt - never keys, artifacts, or datasets.
"""

from __future__ import annotations

import json
import os
from typing import Any

# Central model configuration.
GROQ_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 700
API_KEY_ENV = "GROQ_API_KEY"


def api_key() -> str | None:
    return os.environ.get(API_KEY_ENV)


SYSTEM_PROMPT = """You are FraudLab's Investigation Assistant, an enterprise fraud-analysis \
explainer. You do NOT detect fraud; a champion XGBoost model already produced the decision. \
Your only job is to explain that decision to a human analyst.

Hard rules:
- Ground every statement ONLY in the evidence provided: the transaction context, the \
fraud probability, the decision, the confidence level, and the SHAP contributors.
- If evidence is not present in the SHAP contributors, do not mention it. Never invent \
devices, locations, velocities, merchants, or any signal that is not in the contributors.
- Do not speculate, do not infer unseen facts, and do not expose raw feature-engineering \
names. Use the analyst-friendly contributor labels exactly as given.
- Use concise, factual, enterprise language aimed at a fraud analyst.
- If there are no contributors, state that no individual signals were available.

Return ONLY a JSON object with exactly these keys:
  "summary": one or two sentences describing what happened and the action taken.
  "signals_detected": an array of short strings, each derived from one SHAP contributor.
  "confidence_explanation": one sentence explaining the confidence level from the \
probability and the strength of contributors.
You will be given the decision and confidence; reflect them faithfully."""


def build_user_prompt(evidence: dict[str, Any]) -> str:
    """Construct the grounded user prompt from validated evidence."""
    payload = {
        "transaction_context": evidence.get("transaction", {}),
        "fraud_probability": evidence["fraud_probability"],
        "decision": evidence["decision"],
        "confidence": evidence["confidence"],
        "shap_contributors": evidence["contributors"],
    }
    return (
        "Explain the following fraud decision for an analyst. Use ONLY this evidence.\n\n"
        + json.dumps(payload, indent=2, default=str)
        + "\n\nRespond with the JSON object described in the system instructions."
    )
