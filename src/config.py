from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class WinsorConfig:
    low: float = 0.01
    high: float = 0.99


@dataclass
class ModelConfig:
    n_estimators: int = 600
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    bootstrap: bool = True


@dataclass
class SearchConfig:
    enabled: bool = True
    n_iter: int = 40
    scoring: str = "neg_mean_absolute_error"
    n_jobs: int = -1


@dataclass
class AppConfig:
    random_state: int = 42
    band_pct: float = 0.10
    cv_folds: int = 5
    winsor: WinsorConfig = field(default_factory=WinsorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        winsor = data.get("winsor", {})
        model = data.get("model", {})
        search = data.get("search", {})
        return cls(
            random_state=data.get("random_state", 42),
            band_pct=data.get("band_pct", 0.10),
            cv_folds=data.get("cv_folds", 5),
            winsor=WinsorConfig(**winsor) if not isinstance(winsor, WinsorConfig) else winsor,
            model=ModelConfig(**model) if not isinstance(model, ModelConfig) else model,
            search=SearchConfig(**search) if not isinstance(search, SearchConfig) else search,
        )


DEFAULT_CONFIG = AppConfig()


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load application configuration from YAML if available."""
    if path is None:
        path = Path("artifacts/config.yaml")

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}
        return AppConfig.from_dict(raw)

    return DEFAULT_CONFIG


def ensure_config_file(path: Optional[Path] = None) -> Path:
    """Create the config file with default values if it doesn't exist."""
    if path is None:
        path = Path("artifacts/config.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(_config_to_dict(DEFAULT_CONFIG), f, sort_keys=False)
    return path


def _config_to_dict(config: AppConfig) -> Dict[str, Any]:
    return {
        "random_state": config.random_state,
        "band_pct": config.band_pct,
        "cv_folds": config.cv_folds,
        "winsor": {
            "low": config.winsor.low,
            "high": config.winsor.high,
        },
        "model": {
            "n_estimators": config.model.n_estimators,
            "max_depth": config.model.max_depth,
            "min_samples_split": config.model.min_samples_split,
            "min_samples_leaf": config.model.min_samples_leaf,
            "max_features": config.model.max_features,
            "bootstrap": config.model.bootstrap,
        },
        "search": {
            "enabled": config.search.enabled,
            "n_iter": config.search.n_iter,
            "scoring": config.search.scoring,
            "n_jobs": config.search.n_jobs,
        },
    }


__all__ = [
    "AppConfig",
    "DEFAULT_CONFIG",
    "ModelConfig",
    "SearchConfig",
    "WinsorConfig",
    "ensure_config_file",
    "load_config",
]
