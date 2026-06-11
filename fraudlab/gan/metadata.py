"""Metadata inference and persistence for CTGAN training.

The metadata classifies each *modelled* column (i.e. after identifier columns are dropped)
into one of four kinds:

* ``continuous``  - real-valued numeric columns (CTGAN mode-specific normalization).
* ``categorical`` - discrete string columns (CTGAN one-hot / conditional vector).
* ``boolean``     - two-valued flags, modelled as discrete.
* ``datetime``    - the timestamp, encoded to epoch-seconds (a continuous column) for the
                    model and decoded back on output.

Persisting the metadata makes training reproducible: the same column typing and ordering
is reused at sample time. Inference is deterministic and dtype/name driven.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import pandas as pd

# Identifier columns that are dropped before training and re-minted after sampling.
DROP_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "user_id",
    "merchant_id",
    "merchant_name",
    "device_id",
)

# Known schema roles. Inference falls back to dtype when a column is not listed here.
KNOWN_DATETIME: tuple[str, ...] = ("timestamp",)
KNOWN_BOOLEAN: tuple[str, ...] = ("is_foreign", "is_fraud")
KNOWN_CATEGORICAL: tuple[str, ...] = (
    "currency",
    "merchant_category",
    "payment_method",
    "channel",
    "device_type",
    "city",
    "state",
    "country",
    "fraud_type",
)
# Numeric columns that are conceptually integers (rounded/clipped on decode).
KNOWN_INTEGER: tuple[str, ...] = ("account_age_days", "velocity_1h", "velocity_24h")
# Numeric columns bounded to [0, 1].
KNOWN_UNIT_INTERVAL: tuple[str, ...] = ("merchant_risk_score", "device_risk_score")


@dataclass(slots=True)
class GANMetadata:
    """Column typing + ordering used to train and sample a CTGAN reproducibly."""

    columns: list[str]  # modelled column order
    continuous: list[str]
    categorical: list[str]
    boolean: list[str]
    datetime: list[str]
    integer_columns: list[str] = field(default_factory=list)
    unit_interval_columns: list[str] = field(default_factory=list)

    def discrete_columns(self) -> list[str]:
        """Columns CTGAN must treat as discrete (categoricals + booleans).

        Datetime is *not* discrete: it is encoded to a continuous epoch column.
        """
        return [*self.categorical, *self.boolean]

    # --- persistence -----------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GANMetadata":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "GANMetadata":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def infer_metadata(
    df: pd.DataFrame, *, drop_columns: tuple[str, ...] = DROP_COLUMNS
) -> GANMetadata:
    """Infer a :class:`GANMetadata` from a (typically simulator-exported) DataFrame.

    Columns named in ``drop_columns`` are excluded. Remaining columns are typed by the
    known-schema role first, then by pandas dtype as a fallback.
    """
    modelled = [c for c in df.columns if c not in set(drop_columns)]

    continuous: list[str] = []
    categorical: list[str] = []
    boolean: list[str] = []
    datetimes: list[str] = []
    integer_cols: list[str] = []
    unit_cols: list[str] = []

    for c in modelled:
        s = df[c]
        if c in KNOWN_DATETIME or pd.api.types.is_datetime64_any_dtype(s):
            datetimes.append(c)
        elif c in KNOWN_BOOLEAN or pd.api.types.is_bool_dtype(s):
            boolean.append(c)
        elif c in KNOWN_CATEGORICAL or (
            not pd.api.types.is_numeric_dtype(s)
        ):
            categorical.append(c)
        else:
            continuous.append(c)
            if c in KNOWN_INTEGER or pd.api.types.is_integer_dtype(s):
                integer_cols.append(c)
            if c in KNOWN_UNIT_INTERVAL:
                unit_cols.append(c)

    return GANMetadata(
        columns=modelled,
        continuous=continuous,
        categorical=categorical,
        boolean=boolean,
        datetime=datetimes,
        integer_columns=integer_cols,
        unit_interval_columns=unit_cols,
    )
