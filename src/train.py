from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from .config import AppConfig, ensure_config_file, load_config
from .data_utils import REQUIRED_COLUMNS, hash_dataframe, load_dataset, split_features_target
from .pipeline import build_preprocessor, get_feature_names, get_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RandomForest regressor for fair price estimation.")
    parser.add_argument("--data", required=True, help="Path to the training CSV file.")
    parser.add_argument("--band_pct", type=float, default=None, help="Override band percentage for labeling.")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration YAML file.")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory to store trained artifacts and reports.",
    )
    parser.add_argument("--no-search", action="store_true", help="Disable hyper-parameter search regardless of config.")
    return parser.parse_args()


def build_pipeline(config: AppConfig) -> Pipeline:
    preprocessor = build_preprocessor(config)
    model = RandomForestRegressor(
        n_estimators=config.model.n_estimators,
        max_depth=config.model.max_depth,
        min_samples_split=config.model.min_samples_split,
        min_samples_leaf=config.model.min_samples_leaf,
        max_features=config.model.max_features,
        bootstrap=config.model.bootstrap,
        random_state=config.random_state,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train() -> Dict[str, Any]:
    args = parse_args()
    data_path = Path(args.data)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config) if args.config else ensure_config_file(artifacts_dir / "config.yaml")
    config = load_config(config_path)

    if args.band_pct is not None:
        config.band_pct = args.band_pct
    if args.no_search:
        config.search.enabled = False

    df = load_dataset(data_path)
    X, y = split_features_target(df)

    pipeline = build_pipeline(config)

    if config.search.enabled:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=get_pipeline_config(config),
            n_iter=config.search.n_iter,
            scoring=config.search.scoring,
            cv=config.cv_folds,
            random_state=config.random_state,
            n_jobs=config.search.n_jobs,
            refit=config.search.scoring,
        )
        search.fit(X, y)
        best_pipeline: Pipeline = search.best_estimator_
    else:
        pipeline.fit(X, y)
        best_pipeline = pipeline

    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
    }
    scores = cross_validate(best_pipeline, X, y, cv=config.cv_folds, scoring=scoring)
    metrics = {
        "r2_cv": float(np.mean(scores["test_r2"])),
        "mae_cv": float(-np.mean(scores["test_neg_mae"])),
        "rmse_cv": float(-np.mean(scores["test_neg_rmse"])),
    }

    median_price = float(np.median(y))
    if metrics["r2_cv"] < 0.60:
        raise RuntimeError(f"Cross-validated R^2 {metrics['r2_cv']:.3f} below threshold 0.60.")
    if metrics["mae_cv"] > 0.25 * median_price:
        raise RuntimeError(
            f"Cross-validated MAE {metrics['mae_cv']:.2f} exceeds 25% of median price {0.25 * median_price:.2f}."
        )

    best_pipeline.fit(X, y)

    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    joblib.dump(preprocessor, artifacts_dir / "preprocessor.pkl")
    joblib.dump(model, artifacts_dir / "rf_regressor.pkl")

    feature_names = list(get_feature_names(preprocessor))
    importances = model.feature_importances_
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(artifacts_dir / "feature_importance.csv", index=False)

    report = {
        "metrics": metrics,
        "median_price": median_price,
        "band_pct": config.band_pct,
        "cv_folds": config.cv_folds,
        "random_state": config.random_state,
        "data_path": str(data_path),
        "data_hash": hash_dataframe(df, REQUIRED_COLUMNS),
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
    }

    with (artifacts_dir / "train_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    train()
