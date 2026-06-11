"""Export transactions to CSV / JSON / Parquet while preserving schema integrity."""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd

from fraudlab.config.config import OutputFormat
from fraudlab.models import TRANSACTION_FIELDS, Transaction


def transactions_to_dataframe(transactions: Iterable[Transaction]) -> pd.DataFrame:
    """Build a DataFrame with columns in the canonical schema order."""
    rows = [t.to_row() for t in transactions]
    df = pd.DataFrame(rows, columns=list(TRANSACTION_FIELDS))
    return _coerce_dtypes(df)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply stable dtypes so exports are consistent and round-trip safe."""
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    int_cols = ["account_age_days", "velocity_1h", "velocity_24h"]
    for c in int_cols:
        df[c] = df[c].astype("int64")
    float_cols = [
        "amount",
        "avg_user_spend",
        "median_user_spend",
        "merchant_risk_score",
        "device_risk_score",
    ]
    for c in float_cols:
        df[c] = df[c].astype("float64")
    df["is_foreign"] = df["is_foreign"].astype("bool")
    df["is_fraud"] = df["is_fraud"].astype("bool")
    # fraud_type stays object/string with None for legitimate rows.
    return df


def export_dataset(
    transactions: Iterable[Transaction],
    path: str,
    output_format: OutputFormat | str,
) -> str:
    """Write transactions to ``path`` in the requested format. Returns the path."""
    fmt = OutputFormat(output_format) if not isinstance(output_format, OutputFormat) else output_format
    df = transactions_to_dataframe(transactions)

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    if fmt is OutputFormat.CSV:
        df.to_csv(path, index=False)
    elif fmt is OutputFormat.JSON:
        # ISO-8601 timestamps, one record per row.
        df.to_json(path, orient="records", date_format="iso", indent=2)
    elif fmt is OutputFormat.PARQUET:
        df.to_parquet(path, engine="pyarrow", index=False)
    else:  # pragma: no cover - guarded by enum
        raise ValueError(f"Unsupported output format: {fmt}")
    return path
