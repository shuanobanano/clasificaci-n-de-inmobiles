from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import ensure_config_file, load_config
from .data_utils import REQUIRED_COLUMNS, load_dataset, split_features_target

CLASS_UI = {
    "Infravalorada": {"badge": {"text": "Infravalorada", "variant": "success"}, "color": "#16a34a", "icon": "trending-down"},
    "Regular": {"badge": {"text": "Regular", "variant": "neutral"}, "color": "#64748b", "icon": "activity"},
    "Sobrevalorada": {
        "badge": {"text": "Sobrevalorada", "variant": "danger"},
        "color": "#ef4444",
        "icon": "trending-up",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions for property pricing dataset.")
    parser.add_argument("--input", required=True, help="Path to the CSV file with property data.")
    parser.add_argument("--output", help="Optional output file (CSV or JSONL).")
    parser.add_argument("--jsonl", action="store_true", help="Emit predictions as JSONL (stdout if no output path).")
    parser.add_argument("--band_pct", type=float, default=None, help="Override band percentage for classification.")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory containing preprocessor/model artifacts.",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional config path (defaults to artifacts/config.yaml).")
    return parser.parse_args()


def load_pipeline(artifacts_dir: Path) -> Pipeline:
    preprocessor = joblib.load(artifacts_dir / "preprocessor.pkl")
    model = joblib.load(artifacts_dir / "rf_regressor.pkl")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def classify_deviation(deviation: float, band_pct: float) -> str:
    if deviation <= -band_pct:
        return "Infravalorada"
    if abs(deviation) < band_pct:
        return "Regular"
    return "Sobrevalorada"


def format_summary(price: float, fair_price: float, deviation: float) -> str:
    sign = "+" if deviation >= 0 else "-"
    pct = abs(deviation) * 100
    return f"Precio publicado ${price:,.0f} vs justo ${fair_price:,.0f} ({sign}{pct:.1f}%)".replace(",", ".")


def build_ui(label: str, summary: str) -> Dict[str, Any]:
    base = CLASS_UI[label]
    return {
        "badge": base["badge"],
        "color": base["color"],
        "icon": base["icon"],
        "summary": summary,
        "actions": [
            {"type": "openLink", "label": "Ver comps", "href": "#"},
            {"type": "details", "label": "Ver detalle", "payloadId": None},
        ],
    }


def build_prediction_records(
    df: pd.DataFrame,
    predictions: np.ndarray,
    band_pct: float,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    for idx, (row, fair_price) in enumerate(zip(df.to_dict(orient="records"), predictions), start=1):
        price = row["Price"]
        deviation = (price - fair_price) / fair_price if fair_price else 0.0
        label = classify_deviation(deviation, band_pct)
        summary = format_summary(price, fair_price, deviation)
        ui = build_ui(label, summary)
        ui["actions"][1]["payloadId"] = f"pred-{idx}"

        record = {
            "id": str(uuid4()),
            "input": row,
            "pred": {
                "pred_fair_price": float(fair_price),
                "deviation_pct": float(deviation),
                "class_label": label,
                "confidence": float(abs(deviation)),
                "band_pct": float(band_pct),
                "currency": "ARS",
                "locale": "es-AR",
            },
            "explain": {
                "top_features": [],
                "notes": "Importancias relativas disponibles en artifacts/feature_importance.csv",
            },
            "ui": ui,
            "inference_timestamp": timestamp,
        }
        records.append(record)
    return records


def predictions_to_frame(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        base = record["input"].copy()
        pred = record["pred"]
        base.update(
            {
                "pred_fair_price": pred["pred_fair_price"],
                "deviation_pct": pred["deviation_pct"],
                "class_label": pred["class_label"],
                "confidence": pred["confidence"],
                "summary": record["ui"]["summary"],
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def run_inference(args: argparse.Namespace) -> List[Dict[str, Any]]:
    artifacts_dir = Path(args.artifacts_dir)
    config_path = Path(args.config) if args.config else ensure_config_file(artifacts_dir / "config.yaml")
    config = load_config(config_path)

    if args.band_pct is not None:
        config.band_pct = args.band_pct

    pipeline = load_pipeline(artifacts_dir)

    df = load_dataset(Path(args.input))
    records_df = df[REQUIRED_COLUMNS]

    features, _ = split_features_target(records_df)
    predictions = pipeline.predict(features)

    return build_prediction_records(records_df, predictions, config.band_pct)


def main() -> None:
    args = parse_args()
    records = run_inference(args)

    if args.jsonl:
        if args.output:
            output_path = Path(args.output)
            with output_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            for record in records:
                print(json.dumps(record, ensure_ascii=False))
    else:
        frame = predictions_to_frame(records)
        if args.output:
            frame.to_csv(args.output, index=False)
        else:
            print(frame.to_csv(index=False))


if __name__ == "__main__":
    main()
