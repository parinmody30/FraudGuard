"""Export integrity across CSV / JSON / Parquet."""

from __future__ import annotations

import pandas as pd

from fraudlab.exporters.export import export_dataset, transactions_to_dataframe
from fraudlab.models import TRANSACTION_FIELDS


def test_dataframe_schema(small_dataset):
    transactions, _, _ = small_dataset
    df = transactions_to_dataframe(transactions)
    assert list(df.columns) == list(TRANSACTION_FIELDS)
    assert len(df) == len(transactions)


def test_csv_roundtrip(small_dataset, tmp_path):
    transactions, _, _ = small_dataset
    path = tmp_path / "data.csv"
    export_dataset(transactions, str(path), "csv")
    df = pd.read_csv(path)
    assert list(df.columns) == list(TRANSACTION_FIELDS)
    assert len(df) == len(transactions)
    assert int(df["is_fraud"].sum()) == sum(t.is_fraud for t in transactions)


def test_json_roundtrip(small_dataset, tmp_path):
    transactions, _, _ = small_dataset
    path = tmp_path / "data.json"
    export_dataset(transactions, str(path), "json")
    df = pd.read_json(path)
    assert list(df.columns) == list(TRANSACTION_FIELDS)
    assert len(df) == len(transactions)


def test_parquet_roundtrip(small_dataset, tmp_path):
    transactions, _, _ = small_dataset
    path = tmp_path / "data.parquet"
    export_dataset(transactions, str(path), "parquet")
    df = pd.read_parquet(path)
    assert list(df.columns) == list(TRANSACTION_FIELDS)
    assert len(df) == len(transactions)
    # Amounts preserved.
    assert abs(df["amount"].sum() - sum(t.amount for t in transactions)) < 1e-3
