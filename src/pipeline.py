from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import AppConfig
from .data_utils import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


FEATURE_WEIGHTS: Dict[str, float] = {
    "surface_total": 3.0,
    "rooms": 2.0,
    "garage": 1.0,
    "Location": 2.0,
    "type_building": 1.0,
}


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


class ColumnWeighter(BaseEstimator, TransformerMixin):
    """Multiplica columnas por pesos predefinidos."""

    def __init__(self, weights: Dict[str, float], feature_names: Iterable[str] | None = None):
        self.weights = dict(weights)
        self.feature_names = list(feature_names) if feature_names is not None else None

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        elif self.feature_names is not None:
            self.feature_names_in_ = list(self.feature_names)
        else:
            raise ValueError("Feature names are required when input data has no column metadata.")

        self.weights_ = np.array(
            [float(self.weights.get(name, 1.0)) for name in self.feature_names_in_],
            dtype=float,
        )
        return self

    def transform(self, X):
        X_array = np.asarray(X, dtype=float)
        return X_array * self.weights_

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        return getattr(self, "feature_names_in_", None)


class WeightedOneHotEncoder(OneHotEncoder):
    """OneHotEncoder que pondera las variables categóricas por columna original."""

    def __init__(self, *args, weights: Dict[str, float] | None = None, **kwargs):
        kwargs.setdefault("sparse_output", False)
        super().__init__(*args, **kwargs)
        self.weights = dict(weights or {})

    def fit(self, X, y=None):
        fitted = super().fit(X, y)
        feature_weights = []
        for column, categories in zip(self.feature_names_in_, self.categories_):
            weight = float(self.weights.get(column, 1.0))
            feature_weights.append(np.full(len(categories), weight, dtype=float))
        self.feature_weights_ = (
            np.concatenate(feature_weights) if feature_weights else np.array([], dtype=float)
        )
        return fitted

    def transform(self, X):
        transformed = super().transform(X)
        if transformed.size == 0 or getattr(self, "feature_weights_", None) is None:
            return transformed
        return transformed * self.feature_weights_


def build_preprocessor(config: AppConfig) -> ColumnTransformer:
    numeric_features = NUMERIC_COLUMNS[1:]
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
            ("weight", ColumnWeighter(FEATURE_WEIGHTS, feature_names=numeric_features)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                WeightedOneHotEncoder(handle_unknown="ignore", weights=FEATURE_WEIGHTS),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),  # exclude target
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
    "FEATURE_WEIGHTS",
    "Winsorizer",
    "ColumnWeighter",
    "WeightedOneHotEncoder",
    "build_preprocessor",
    "get_feature_names",
    "get_pipeline_config",
]
