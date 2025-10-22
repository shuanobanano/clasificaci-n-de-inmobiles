from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import AppConfig
from .data_utils import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip values based on quantiles computed during fit."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: D401
        X = self._validate_data(X, reset=True)
        self.lower_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X = self._validate_data(X, reset=False)
        return np.clip(X, self.lower_, self.upper_)


def build_preprocessor(config: AppConfig) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "winsor",
                Winsorizer(
                    lower_quantile=config.winsor.low,
                    upper_quantile=config.winsor.high,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS[1:]),  # exclude target
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> Iterable[str]:
    feature_names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            feature_names.extend(transformer.get_feature_names_out(columns))
        else:
            feature_names.extend(columns)
    return feature_names


def get_pipeline_config(config: AppConfig) -> Dict[str, Tuple[str, object]]:
    """Return default parameter search space for the RandomForest pipeline."""
    return {
        "model__n_estimators": [200, 400, config.model.n_estimators],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
        "model__bootstrap": [True, False],
    }


__all__ = [
    "Winsorizer",
    "build_preprocessor",
    "get_feature_names",
    "get_pipeline_config",
]
