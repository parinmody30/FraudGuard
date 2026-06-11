"""FraudLab Investigation Agent (Phase 6).

Converts deterministic champion-model + SHAP evidence into grounded, analyst-facing
investigation narratives. Groq (``llama-3.3-70b-versatile``) generates the prose;
recommendation/next-steps come from the decision policy; a deterministic offline fallback
preserves the identical frontend contract when Groq is unavailable.
"""

from __future__ import annotations

from fraudlab.agent.investigator import InvestigationAgent
from fraudlab.agent.schemas import validate_explanation

__all__ = ["InvestigationAgent", "validate_explanation"]
