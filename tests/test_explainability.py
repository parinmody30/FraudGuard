"""Tests for the Phase 5 SHAP explainability integration."""

from __future__ import annotations

import numpy as np
import pytest

from fraudlab.config.config import BusinessType, FraudLabConfig, RiskAppetite
from fraudlab.exporters.export import transactions_to_dataframe
from fraudlab.explainability.formatter import (
    FEATURE_LABELS,
    format_analyst_summary,
    friendly_label,
)
from fraudlab.explainability.shap_explainer import (
    TOP_NEGATIVE,
    TOP_POSITIVE,
    ChampionExplainer,
    _base_feature,
)
from fraudlab.models.champion import train_champion
from fraudlab.pipeline import build_dataset


@pytest.fixture(scope="module")
def explainer():
    config = FraudLabConfig(
        num_users=120,
        transactions_per_user=40,
        fraud_rate=0.06,
        business_type=BusinessType.BANK,
        risk_appetite=RiskAppetite.MEDIUM,
        random_seed=42,
    )
    transactions, _ = build_dataset(config)
    df = transactions_to_dataframe(transactions)
    bundle, _ = train_champion(df, simulator_config={"seed": 42}, random_state=42)
    return ChampionExplainer(bundle), df


# ------------------------------------------------------------------ labels / mapping

def test_friendly_labels_hide_raw_names():
    assert friendly_label("velocity_1h") == "Velocity Spike (1 Hour)"
    assert friendly_label("merchant_risk_score") == "Merchant Risk"
    # Unknown base feature still gets a clean title-cased label, never a raw __ name.
    assert friendly_label("some_new_feature") == "Some New Feature"
    assert "__" not in friendly_label("merchant_risk_score")


def test_base_feature_parsing():
    assert _base_feature("continuous__amount") == "amount"
    assert _base_feature("boolean__is_foreign") == "is_foreign"
    # One-hot category with underscores in the value maps to the base column.
    assert _base_feature("categorical__merchant_category_money_transfer") == "merchant_category"
    assert _base_feature("categorical__city_Mumbai") == "city"


# ------------------------------------------------------------------------ explainer

def test_explainer_builds_once(explainer):
    exp, _ = explainer
    # A single TreeExplainer and one precomputed feature->base map are reused.
    assert exp._explainer is not None
    assert len(exp._groups) >= 1
    assert all("__" not in b for b in exp._groups)


def test_single_explanation_schema(explainer):
    exp, df = explainer
    out = exp.explain(df.head(1))
    assert set(out) >= {"transaction_id", "fraud_probability", "decision", "confidence", "contributors"}
    assert out["decision"] in {"ALLOW", "REVIEW", "BLOCK"}
    assert out["confidence"] in {"LOW", "MEDIUM", "HIGH"}
    assert 0.0 <= out["fraud_probability"] <= 1.0


def test_contributor_ordering_and_counts(explainer):
    exp, df = explainer
    out = exp.explain_batch(df.head(8))
    for e in out:
        contribs = e["contributors"]
        pos = [c for c in contribs if c["direction"] == "increase"]
        neg = [c for c in contribs if c["direction"] == "decrease"]
        assert len(pos) <= TOP_POSITIVE
        assert len(neg) <= TOP_NEGATIVE
        # Sorted by absolute impact (descending).
        mags = [abs(c["impact"]) for c in contribs]
        assert mags == sorted(mags, reverse=True)


def test_direction_matches_sign(explainer):
    exp, df = explainer
    for e in exp.explain_batch(df.head(10)):
        for c in e["contributors"]:
            if c["impact"] > 0:
                assert c["direction"] == "increase"
            elif c["impact"] < 0:
                assert c["direction"] == "decrease"


def test_batch_matches_single(explainer):
    exp, df = explainer
    sample = df.head(4)
    batch = exp.explain_batch(sample)
    assert len(batch) == 4
    single = exp.explain(sample.head(1))
    assert batch[0]["transaction_id"] == single["transaction_id"]
    assert batch[0]["fraud_probability"] == pytest.approx(single["fraud_probability"], abs=1e-6)


def test_frontend_response_contract(explainer):
    exp, df = explainer
    records = exp.frontend_response(df.head(3))
    assert len(records) == 3
    for rec in records:
        assert set(rec) == {"transaction_id", "fraud_probability", "decision", "confidence", "contributors"}
        for c in rec["contributors"]:
            assert set(c) == {"feature", "impact", "direction"}
            assert c["direction"] in {"increase", "decrease"}
            assert isinstance(c["impact"], float)


def test_formatter_output(explainer):
    exp, df = explainer
    summary = format_analyst_summary(exp.explain(df.head(1)))
    assert "Positive Contributors:" in summary
    assert "Negative Contributors:" in summary
    # Deterministic: same input -> same output.
    assert summary == format_analyst_summary(exp.explain(df.head(1)))


def test_empty_batch(explainer):
    exp, df = explainer
    assert exp.explain_batch(df.head(0)) == []
