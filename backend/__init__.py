"""FraudLab backend API layer (Phase 7.0).

Thin FastAPI wiring over the already-implemented FraudLab modules (champion predictor, SHAP
explainer, investigation agent, dataset generator). This package contains *no* new business
logic - handlers delegate to the existing services and exist to define the stable
backend/frontend integration contract documented in ``docs/API_CONTRACT.md``.
"""
