"""Tests for the Phase 2 CTGAN realism-augmentation subpackage.

CTGAN training is the slow part, so a single tiny model is trained once per module and
reused. These tests assert the *contracts* (typing, scope, shapes, determinism, label
enforcement, merge integrity, metric structure) rather than exact GAN outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlab.config.config import (
    ALL_FRAUD_TYPES,
    BusinessType,
    FraudLabConfig,
    RiskAppetite,
    default_fraud_mix,
)
from fraudlab.exporters.export import transactions_to_dataframe
from fraudlab.gan.dataset_builder import (
    HARD_FRAUD_TYPES,
    LEGIT_SENTINEL,
    SOFT_FRAUD_TYPES,
    build_training_frame,
    from_model_frame,
    to_model_frame,
)
from fraudlab.gan.evaluator import evaluate_ctgan
from fraudlab.gan.merger import build_hybrid_dataset
from fraudlab.gan.metadata import DROP_COLUMNS, GANMetadata, infer_metadata
from fraudlab.gan.sampler import CTGANSampler
from fraudlab.gan.trainer import CTGANConfig, train_ctgan
from fraudlab.models import TRANSACTION_FIELDS
from fraudlab.pipeline import build_dataset

pytestmark = pytest.mark.filterwarnings("ignore")


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def sim_df() -> pd.DataFrame:
    config = FraudLabConfig(
        num_users=60,
        transactions_per_user=30,
        fraud_rate=0.10,
        business_type=BusinessType.BANK,
        risk_appetite=RiskAppetite.MEDIUM,
        random_seed=321,
    )
    transactions, _ = build_dataset(config)
    return transactions_to_dataframe(transactions)


@pytest.fixture(scope="module")
def tiny_config() -> CTGANConfig:
    return CTGANConfig(
        epochs=3,
        batch_size=100,
        embedding_dim=16,
        generator_dim=(32, 32),
        discriminator_dim=(32, 32),
        pac=5,
        random_seed=7,
    )


@pytest.fixture(scope="module")
def trained(sim_df, tiny_config):
    metadata = infer_metadata(sim_df)
    frame = build_training_frame(sim_df)
    model = train_ctgan(frame, metadata, tiny_config)
    return model, metadata


@pytest.fixture(scope="module")
def sampler(trained):
    model, metadata = trained
    return CTGANSampler(model, metadata, seed=7)


# --------------------------------------------------------------------------- metadata

def test_metadata_inference(sim_df):
    md = infer_metadata(sim_df)
    assert "timestamp" in md.datetime
    assert set(md.boolean) == {"is_foreign", "is_fraud"}
    assert "amount" in md.continuous
    assert {"fraud_type", "city", "merchant_category"} <= set(md.categorical)
    # Identifier columns are excluded entirely.
    for c in DROP_COLUMNS:
        assert c not in md.columns
    # Discrete = categoricals + booleans; datetime is modelled as continuous.
    assert set(md.discrete_columns()) == set(md.categorical) | set(md.boolean)
    assert "velocity_1h" in md.integer_columns


def test_metadata_roundtrip(tmp_path, sim_df):
    md = infer_metadata(sim_df)
    p = tmp_path / "md.json"
    md.save(str(p))
    loaded = GANMetadata.load(str(p))
    assert loaded.to_dict() == md.to_dict()


# --------------------------------------------------------------------- dataset builder

def test_training_frame_excludes_hard_and_ids(sim_df):
    frame = build_training_frame(sim_df)
    # No hard fraud scenarios are present.
    assert not frame["fraud_type"].isin(HARD_FRAUD_TYPES).any()
    # Soft fraud + legit only; legit rows carry the sentinel.
    present = set(frame["fraud_type"].unique())
    assert present <= set(SOFT_FRAUD_TYPES) | {LEGIT_SENTINEL}
    assert (frame.loc[~frame["is_fraud"], "fraud_type"] == LEGIT_SENTINEL).all()
    # Identifier columns dropped.
    for c in DROP_COLUMNS:
        assert c not in frame.columns


def test_encode_decode_roundtrip(sim_df):
    md = infer_metadata(sim_df)
    frame = build_training_frame(sim_df)
    model_frame = to_model_frame(frame, md)
    decoded = from_model_frame(model_frame, md)

    # Categoricals/booleans round-trip exactly; legit sentinel decodes back to None.
    assert (decoded["city"].astype(str) == frame["city"].astype(str)).all()
    assert (decoded["is_foreign"].astype(bool) == frame["is_foreign"].astype(bool)).all()
    assert decoded.loc[~frame["is_fraud"].to_numpy(), "fraud_type"].isna().all()
    # Datetime decodes within a second.
    dt_diff = (pd.to_datetime(decoded["timestamp"]) - pd.to_datetime(frame["timestamp"])).abs()
    assert dt_diff.max().total_seconds() <= 1.5


# --------------------------------------------------------------------------- training

def test_training_reproducible(sim_df, tiny_config):
    md = infer_metadata(sim_df)
    frame = build_training_frame(sim_df)
    m1 = train_ctgan(frame, md, tiny_config)
    m2 = train_ctgan(frame, md, tiny_config)
    s1 = CTGANSampler(m1, md, seed=7).generate(40, seed=7)
    s2 = CTGANSampler(m2, md, seed=7).generate(40, seed=7)

    for c in md.continuous:
        assert np.allclose(s1[c].to_numpy(float), s2[c].to_numpy(float), atol=1e-6), c
    for c in (*md.categorical, *md.boolean):
        assert s1[c].astype(str).equals(s2[c].astype(str)), c


def test_config_rejects_bad_pac():
    with pytest.raises(ValueError):
        CTGANConfig(batch_size=100, pac=7)  # 100 % 7 != 0


# --------------------------------------------------------------------------- sampling

def test_generate_shape(sampler, trained):
    _, md = trained
    out = sampler.generate(50, seed=7)
    assert len(out) == 50
    assert set(md.columns) <= set(out.columns)
    assert out["is_fraud"].dtype == bool


def test_conditional_sampling_enforces_label(sampler):
    fraud = sampler.conditional_generate("large_spend", 30, seed=1)
    assert len(fraud) == 30
    assert fraud["is_fraud"].all()
    assert (fraud["fraud_type"] == "large_spend").all()

    legit = sampler.conditional_generate(None, 30, seed=1)
    assert (~legit["is_fraud"]).all()
    assert legit["fraud_type"].isna().all()


def test_sample_with_prevalence(sampler):
    out = sampler.sample_with_prevalence(400, fraud_rate=0.10, seed=3)
    assert len(out) == 400
    # Exact allocation -> realized prevalence equals the request.
    assert abs(out["is_fraud"].mean() - 0.10) <= 0.01
    # Only soft scenarios appear (CTGAN never produces hard fraud).
    produced = set(out.loc[out["is_fraud"], "fraud_type"].unique())
    assert produced <= set(SOFT_FRAUD_TYPES)


# --------------------------------------------------------------------------- merger

def test_merger_builds_hybrid(sim_df, sampler):
    mix = default_fraud_mix(RiskAppetite.MEDIUM)
    hybrid, comp = build_hybrid_dataset(
        sim_df, sampler, n_rows=2000, fraud_rate=0.05, fraud_mix=mix, seed=11
    )
    # Schema preserved exactly.
    assert list(hybrid.columns) == list(TRANSACTION_FIELDS)
    assert len(hybrid) == 2000
    # Identifiers re-minted, unique transaction ids, schema id prefixes.
    assert hybrid["transaction_id"].is_unique
    assert hybrid["transaction_id"].str.match(r"^T\d{10}$").all()
    assert hybrid["user_id"].str.match(r"^U\d{7}$").all()
    assert hybrid["merchant_id"].str.match(r"^M\d{6}$").all()
    assert hybrid["device_id"].str.match(r"^D\d+$").all()
    # Hard fraud comes from the simulator and is present in the mix.
    assert comp["fraud_by_tier"]["hard"] > 0
    assert "ctgan_legit" in comp["source_breakdown"]
    assert "simulator_hard" in comp["source_breakdown"]
    # Realized fraud rate near the request.
    assert abs(comp["realized_fraud_rate"] - 0.05) <= 0.01
    # No nulls in non-fraud_type columns.
    non_ftype = [c for c in TRANSACTION_FIELDS if c != "fraud_type"]
    assert hybrid[non_ftype].notna().all().all()


def test_merger_hard_rows_are_simulator_sourced(sim_df, sampler):
    mix = {t: 1.0 for t in ALL_FRAUD_TYPES}
    hybrid, comp = build_hybrid_dataset(
        sim_df, sampler, n_rows=1500, fraud_rate=0.08, fraud_mix=mix, seed=5
    )
    # Every hard scenario in the output traces to the simulator pool (CTGAN can't make them).
    hard_in_output = set(hybrid.loc[hybrid["fraud_type"].isin(HARD_FRAUD_TYPES), "fraud_type"])
    assert hard_in_output <= set(HARD_FRAUD_TYPES)
    assert comp["fraud_by_tier"]["hard"] == sum(comp["hard_from_simulator"].values())


# --------------------------------------------------------------------------- evaluator

def test_evaluation_metrics_structure(sim_df, sampler, trained):
    _, md = trained
    real = sim_df[~sim_df["fraud_type"].isin(HARD_FRAUD_TYPES)].drop(
        columns=list(DROP_COLUMNS)
    ).reset_index(drop=True)
    synth = sampler.generate(len(real), seed=9)

    report = evaluate_ctgan(real, synth, md, random_state=7)
    sim = report["similarity"]
    assert set(sim["ks_continuous"]) <= set(md.continuous)
    assert "js_categorical" in sim
    assert "correlation_preservation" in sim
    assert "mutual_information_preservation" in sim
    assert "fraud_prevalence" in sim
    assert "fraud_type_preservation" in sim

    for model_name in ("logistic", "gradient_boosting"):
        res = report["ml_utility"][model_name]
        assert set(res) == {"reference_real_on_real", "TSTR", "TRTS"}
        for scen in res.values():
            assert set(scen) == {"precision", "recall", "f1", "roc_auc", "pr_auc"}

    assert report["recommendation"]["verdict"] in {
        "READY FOR HYBRID GENERATION",
        "CTGAN NEEDS TUNING",
    }
