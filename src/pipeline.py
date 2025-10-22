from __future__ import annotations

from typing import Dict, Iterable, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import AppConfig
from .data_utils import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.05):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        # Validación básica de datos y conversión a DataFrame si es necesario
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Guardar nombres de columnas numéricas para usar en transform
        self.columns_ = list(X.columns)
        self.quantiles_ = {}
        for col in self.columns_:
            lower = X[col].quantile(self.lower_quantile)
            upper = X[col].quantile(1 - self.upper_quantile)
            self.quantiles_[col] = (lower, upper)
        return self

    def transform(self, X):
        # Operar sobre una copia para no modificar el original
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.columns_)
        else:
            X_df = X.copy()
        # Aplicar recorte (winsorización) por columna
        for col, (lower, upper) in self.quantiles_.items():
            X_df[col] = X_df[col].clip(lower=lower, upper=upper)
        # Devolver el mismo tipo que recibimos
        return X_df if isinstance(X, pd.DataFrame) else X_df.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return input_features




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
