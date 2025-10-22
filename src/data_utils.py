from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "Price",
    "Location",
    "Expensas",
    "surface_total",
    "rooms",
    "bedrooms",
    "garage",
    "type_building",
    "type_operation",
]

NUMERIC_COLUMNS = [
    "Price",
    "Expensas",
    "surface_total",
    "rooms",
    "bedrooms",
    "garage",
]

CATEGORICAL_COLUMNS = ["Location", "type_building", "type_operation"]


class SchemaError(ValueError):
    """Raised when the dataset does not comply with the expected schema."""


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise SchemaError(f"Missing required columns: {', '.join(missing_cols)}")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str).fillna("desconocido")

    if df[NUMERIC_COLUMNS].isnull().any().any():
        raise SchemaError("Numeric columns contain non-convertible values after coercion.")


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=["Price"])
    target = df["Price"]
    return features, target


def hash_dataframe(df: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    if columns is not None:
        df = df[list(columns)]
    data_bytes = pd.util.hash_pandas_object(df, index=True).values
    m = hashlib.sha256()
    m.update(np.ascontiguousarray(data_bytes).view(np.uint8))
    return m.hexdigest()


__all__ = [
    "CATEGORICAL_COLUMNS",
    "NUMERIC_COLUMNS",
    "REQUIRED_COLUMNS",
    "SchemaError",
    "hash_dataframe",
    "load_dataset",
    "split_features_target",
    "validate_schema",
]
