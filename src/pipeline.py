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
    "Price": 4.0,
    "Location": 2.5,
    "Expensas": 2.0,
    "surface_total": 4.0,
    "rooms": 2.0,
    "bedrooms": 2.0,
    "garage": 3.0,
    "type_building": 0.1,
    "type_operation": 0.1,
    
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

    def __init__(self, weights: Dict[str, float] | None = None, feature_names: list[str] | None = None):
        self.weights = weights or {}
        self.feature_names = feature_names or []

    def fit(self, X, y=None):
        # Si X es un DataFrame, usar sus columnas
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            # Si no, usar los feature_names proporcionados
            self.feature_names_in_ = self.feature_names.copy() if self.feature_names else []
        
        # Crear array de pesos en el orden correcto de las features
        self.weights_ = np.array([
            float(self.weights.get(feature_name, 1.0)) 
            for feature_name in self.feature_names_in_
        ], dtype=float)
        
        return self

    def transform(self, X):
        X_array = np.asarray(X, dtype=float)
        return X_array * self.weights_

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        return self.feature_names_in_

class WeightedOneHotEncoder(OneHotEncoder):
    """OneHotEncoder que pondera las variables categóricas por columna original."""

    def __init__(
        self,
        weights=None,
        categories='auto',
        drop=None,
        sparse_output=False,
        dtype=np.float64,
        handle_unknown='error',
        min_frequency=None,
        max_categories=None,
        feature_name_combiner='concat'
    ):
        super().__init__(
            categories=categories,
            drop=drop,
            sparse_output=sparse_output,
            dtype=dtype,
            handle_unknown=handle_unknown,
            min_frequency=min_frequency,
            max_categories=max_categories,
            feature_name_combiner=feature_name_combiner
        )
        self.weights = weights
        self.original_feature_names_ = None

    def fit(self, X, y=None):
        fitted = super().fit(X, y)
        
        # Si no tenemos feature_names_in_, crear nombres basados en índices
        if not hasattr(self, 'feature_names_in_'):
            n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
            self.feature_names_in_ = [f'feature_{i}' for i in range(n_features)]
        
        # Convertir weights a dict si es necesario
        weights_dict = {}
        if self.weights is not None:
            if hasattr(self.weights, 'items'):
                weights_dict = dict(self.weights)
            elif isinstance(self.weights, (list, tuple)):
                # Si weights es una lista/tupla, asumir que coincide con feature_names_in_
                weights_dict = {name: weight for name, weight in zip(self.feature_names_in_, self.weights)}
        
        feature_weights = []
        for column, categories in zip(self.feature_names_in_, self.categories_):
            weight = float(weights_dict.get(column, 1.0))
            feature_weights.append(np.full(len(categories), weight, dtype=float))
        
        self.feature_weights_ = (
            np.concatenate(feature_weights) if feature_weights else np.array([], dtype=float)
        )
        return fitted

    def transform(self, X):
        transformed = super().transform(X)
        if (transformed.size == 0 or 
            not hasattr(self, 'feature_weights_') or 
            self.feature_weights_ is None):
            return transformed
        return transformed * self.feature_weights_

    def get_feature_names_out(self, input_features=None):
        return super().get_feature_names_out(input_features)
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


def _ensure_iterable(features) -> list[str]:
    if features is None:
        return []
    if isinstance(features, str):
        return [features]
    try:
        return list(features)
    except TypeError:
        return [features]


def _extract_feature_names(transformer, input_features) -> list[str]:
    if isinstance(transformer, Pipeline):
        current_features = _ensure_iterable(input_features)
        for _, step in transformer.steps:
            current_features = _extract_feature_names(step, current_features)
        return _ensure_iterable(current_features)

    if hasattr(transformer, "transformers_"):
        names: list[str] = []
        for name, sub_transformer, columns in transformer.transformers_:
            if name == "remainder":
                continue
            names.extend(_extract_feature_names(sub_transformer, columns))
        return names

    if hasattr(transformer, "get_feature_names_out"):
        try:
            return _ensure_iterable(transformer.get_feature_names_out())
        except TypeError:
            input_names = getattr(transformer, "feature_names_in_", input_features)
            return _ensure_iterable(transformer.get_feature_names_out(input_names))

    input_names = getattr(transformer, "feature_names_in_", input_features)
    return _ensure_iterable(input_names)


def get_feature_names(preprocessor: ColumnTransformer) -> Iterable[str]:
    feature_names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        feature_names.extend(_extract_feature_names(transformer, columns))
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
