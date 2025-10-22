from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.data_utils import REQUIRED_COLUMNS, load_dataset

DATASET_PATH = Path("data/raw/dataset_alquileres.csv")


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    return load_dataset(DATASET_PATH)


def test_required_columns(dataset: pd.DataFrame) -> None:
    assert set(REQUIRED_COLUMNS).issubset(dataset.columns)


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("artifacts")
    cmd = [
        "python",
        "-m",
        "src.train",
        "--data",
        str(DATASET_PATH),
        "--artifacts-dir",
        str(tmp_dir),
        "--no-search",
    ]
    subprocess.run(cmd, check=True)
    return tmp_dir


def test_predict_jsonl_and_csv(trained_artifacts):
    jsonl_cmd = [
        "python",
        "-m",
        "src.predict",
        "--input",
        str(DATASET_PATH),
        "--artifacts-dir",
        str(trained_artifacts),
        "--jsonl",
    ]
    result = subprocess.run(jsonl_cmd, check=True, capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, "Prediction JSONL output is empty"
    first_record = json.loads(lines[0])
    pred = first_record["pred"]
    assert pytest.approx(abs(pred["deviation_pct"]), rel=1e-6) == pred["confidence"]

    csv_path = Path(trained_artifacts) / "predictions.csv"
    csv_cmd = [
        "python",
        "-m",
        "src.predict",
        "--input",
        str(DATASET_PATH),
        "--artifacts-dir",
        str(trained_artifacts),
        "--output",
        str(csv_path),
    ]
    subprocess.run(csv_cmd, check=True)
    df = pd.read_csv(csv_path)
    assert {"pred_fair_price", "class_label", "confidence", "summary"}.issubset(df.columns)
