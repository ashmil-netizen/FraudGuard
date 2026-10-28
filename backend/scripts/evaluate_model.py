"""Evaluate the saved PaySim classifier once on the held-out chronological test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml_pipeline import (
    DATASET_IDENTIFIER,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    DataValidationError,
    classification_metrics,
    load_json,
    load_model_bundle,
)


def evaluate(
    processed_dir: Path,
    model_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    schema = load_json(processed_dir / "feature_schema.json")
    metadata = load_json(model_dir / "model_metadata.json")
    bundle = load_model_bundle(model_dir / "fraud_classifier.joblib")
    if schema.get("dataset_identifier") != DATASET_IDENTIFIER:
        raise DataValidationError("Preprocessing schema is not for PaySim.")
    if metadata.get("dataset_identifier") != DATASET_IDENTIFIER:
        raise DataValidationError("Model metadata is not for PaySim.")
    if metadata.get("feature_names") != list(FEATURE_COLUMNS):
        raise DataValidationError("Model metadata feature schema does not match.")
    if bundle.get("feature_names") != list(FEATURE_COLUMNS):
        raise DataValidationError("Saved model feature schema does not match.")
    if bundle.get("classification_threshold") != metadata.get("threshold", {}).get("value"):
        raise DataValidationError("Model threshold and metadata threshold do not match.")

    feature_path = processed_dir / "feature_matrix.csv"
    records_path = processed_dir / "processed_transactions.csv"
    labels_path = processed_dir / "labels.csv"
    dtype = {column: "float32" for column in FEATURE_COLUMNS if column != "type"}
    dtype["type"] = "string"
    features = pd.read_csv(feature_path, dtype=dtype)
    labels = pd.read_csv(labels_path, dtype={TARGET_COLUMN: "int8"})[TARGET_COLUMN]
    record_columns = [
        "transaction_id",
        "source_row_index",
        "step",
        "nameOrig",
        "nameDest",
        "type",
        "amount",
        TARGET_COLUMN,
        "split",
    ]
    records = pd.read_csv(
        records_path,
        usecols=record_columns,
        dtype={
            "transaction_id": "string",
            "nameOrig": "string",
            "nameDest": "string",
            "type": "string",
            TARGET_COLUMN: "int8",
            "split": "category",
        },
    )
    if list(features.columns) != list(FEATURE_COLUMNS):
        raise DataValidationError("Feature matrix columns/order do not match the model.")
    if not (len(features) == len(labels) == len(records)):
        raise DataValidationError("Feature, label, and transaction row counts differ.")
    if not records[TARGET_COLUMN].equals(labels):
        raise DataValidationError(
            "Offline target labels do not align with transaction records."
        )
    if not np.array_equal(
        features["step"].to_numpy(dtype=np.float32),
        records["step"].to_numpy(dtype=np.float32),
    ):
        raise DataValidationError(
            "Feature rows do not align with transaction time steps."
        )
    test_indices = np.flatnonzero(records["split"].eq("test").to_numpy())
    if test_indices.size == 0:
        raise DataValidationError("No held-out chronological test rows were found.")

    test_features = features.iloc[test_indices]
    test_labels = labels.iloc[test_indices].to_numpy(dtype=np.int8)
    test_scores = bundle["model"].predict_proba(test_features)[:, 1]
    threshold = float(metadata["threshold"]["value"])
    metrics = classification_metrics(test_labels, test_scores, threshold)
    predictions = test_scores >= threshold
    test_records = records.iloc[test_indices].copy()
    test_records["fraud_score"] = test_scores
    test_records["predicted_isFraud"] = predictions.astype("int8")
    test_records["classification_threshold"] = threshold
    graph_records = test_records.loc[
        :,
        [
            "transaction_id",
            "source_row_index",
            "step",
            "nameOrig",
            "nameDest",
            "type",
            "amount",
            "fraud_score",
            "predicted_isFraud",
            TARGET_COLUMN,
        ],
    ].copy()
    graph_records["classification_threshold"] = threshold
    graph_records["ground_truth_scope"] = "offline_simulation_evaluation_only"

    feature_description = test_features.describe(percentiles=[0.25, 0.5, 0.75])
    feature_summaries = {
        name: {
            key: (None if pd.isna(value) else float(value))
            for key, value in row.items()
        }
        for name, row in feature_description.transpose().iterrows()
    }
    diagnostics = {
        "dataset_identifier": DATASET_IDENTIFIER,
        "model_type": metadata["model_type"],
        "records_processed": int(len(features)),
        "train_rows": int(metadata["split_strategy"]["training_rows_available"]),
        "training_rows_used": int(metadata["split_strategy"]["training_rows_used"]),
        "validation_rows": int(metadata["split_strategy"]["validation_rows"]),
        "test_rows": int(len(test_indices)),
        "test_metrics": metrics,
        "feature_names": list(FEATURE_COLUMNS),
        "feature_statistics_on_test_split": feature_summaries,
        "missing_values_by_feature": {
            name: int(count)
            for name, count in test_features.isna().sum().items()
        },
        "infinite_numeric_values": int(
            np.isinf(
                test_features.loc[
                    :, [column for column in FEATURE_COLUMNS if column != "type"]
                ].to_numpy(dtype=np.float64)
            ).sum()
        ),
        "threshold_source": metadata["threshold"]["selection"],
        "model_reloaded_from_disk": True,
        "raw_score_reproducibility": bool(
            np.array_equal(
                test_scores[: min(20, len(test_scores))],
                bundle["model"].predict_proba(test_features.iloc[:20])[:, 1],
            )
        ),
        "graph_records": int(len(graph_records)),
        "graph_transaction_rows_match_test_split": bool(
            len(graph_records) == len(test_indices)
            and graph_records["transaction_id"].is_unique
        ),
        "interpretation": (
            "Metrics measure classification of simulated PaySim labels on a later "
            "chronological test period. Fraud scores are uncalibrated model scores, "
            "not calibrated probabilities or evidence of real-world fraud."
        ),
    }
    if not diagnostics["raw_score_reproducibility"]:
        raise DataValidationError("Repeated inference did not reproduce saved-model scores.")

    report_dir.mkdir(parents=True, exist_ok=True)
    test_records.to_csv(processed_dir / "scored_test_transactions.csv", index=False)
    graph_records.to_csv(processed_dir / "graph_test_transactions.csv", index=False)
    (report_dir / "model_report.md").write_text(
        _markdown_report(diagnostics), encoding="utf-8"
    )
    (report_dir / "model_evaluation.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return diagnostics


def _markdown_report(diagnostics: dict[str, Any]) -> str:
    metrics = diagnostics["test_metrics"]
    matrix = metrics["confusion_matrix"]
    lines = [
        "# PaySim supervised fraud-classification report",
        "",
        "> PaySim is simulated mobile-money data. Results below are predictions against "
        "the dataset's simulated labels, not validated real-world fraud detection.",
        "",
        "## Evaluation design",
        "",
        f"- **Model:** `{diagnostics['model_type']}`",
        f"- **Dataset rows processed:** {diagnostics['records_processed']:,}",
        f"- **Training rows available / used:** {diagnostics['train_rows']:,} / {diagnostics['training_rows_used']:,}",
        f"- **Validation rows:** {diagnostics['validation_rows']:,}",
        f"- **Held-out chronological test rows:** {diagnostics['test_rows']:,}",
        "- **Threshold selection:** validation split only; the test set was not used to select the threshold.",
        f"- **Selected threshold:** {metrics['threshold']:.8g}",
        f"- **Test fraud / non-fraud support:** {metrics['fraud_support']:,} / {metrics['non_fraud_support']:,}",
        "",
        "Accuracy is omitted because a model can appear highly accurate by predicting "
        "the majority non-fraud class while missing rare fraud cases.",
        "",
        "## Held-out test metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Precision | {metrics['precision']:.8f} |",
        f"| Recall | {metrics['recall']:.8f} |",
        f"| F1-score | {metrics['f1']:.8f} |",
        f"| Average precision (PR-AUC summary) | {metrics['average_precision'] if metrics['average_precision'] is not None else 'Not defined'} |",
        f"| ROC-AUC | {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'Not defined: test split has one class'} |",
        f"| Flagged transactions | {metrics['flagged_records']:,} ({metrics['flagged_percentage']:.5f}%) |",
        "",
        "### Confusion matrix",
        "",
        "| | Predicted non-fraud | Predicted fraud |",
        "|---|---:|---:|",
        f"| Actual non-fraud | {matrix['true_negative']:,} | {matrix['false_positive']:,} |",
        f"| Actual fraud | {matrix['false_negative']:,} | {matrix['true_positive']:,} |",
        "",
        "## Data quality and inference checks",
        "",
        f"- Missing values by feature: `{diagnostics['missing_values_by_feature']}`.",
        f"- Infinite numeric feature values: {diagnostics['infinite_numeric_values']:,}.",
        f"- Saved-model repeated inference matches: {diagnostics['raw_score_reproducibility']}.",
        f"- Graph-preparation records in held-out output: {diagnostics['graph_records']:,}.",
        f"- One unique graph transaction record is preserved for every held-out row: {diagnostics['graph_transaction_rows_match_test_split']}.",
        "",
        "## Score and scope interpretation",
        "",
        metrics["score_description"],
        "The decision threshold is selected to maximize F1 on validation data. "
        "Threshold choice is a trade-off and should reflect investigation capacity "
        "and the cost of missed cases versus false alerts.",
        "",
        "PaySim is simulated and its labels are generated within the simulation. "
        "These metrics do not establish effectiveness on genuine bank customers, "
        "production transactions, or other fraud types. The model uses post-transaction "
        "balances, so its score is intended only for review after posting.",
        "",
        "## Test feature summaries",
        "",
        "| Feature | Mean | Std. dev. | Min | 25% | Median | 75% | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in diagnostics["feature_statistics_on_test_split"].items():
        lines.append(
            f"| `{name}` | {values.get('mean')} | {values.get('std')} | "
            f"{values.get('min')} | {values.get('25%')} | {values.get('50%')} | "
            f"{values.get('75%')} | {values.get('max')} |"
        )
    lines += ["", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the saved PaySim classifier on its held-out test period."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "backend" / "data" / "processed",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "backend" / "models",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "backend" / "reports",
    )
    args = parser.parse_args()
    try:
        diagnostics = evaluate(
            args.processed_dir, args.model_dir, args.report_dir
        )
    except (DataValidationError, FileNotFoundError, OSError, ValueError) as exc:
        parser.exit(1, f"Could not evaluate PaySim model: {exc}\n")
    metrics = diagnostics["test_metrics"]
    print(
        f"Held-out test: {metrics['records']:,} rows, "
        f"fraud support={metrics['fraud_support']:,}, "
        f"precision={metrics['precision']:.6f}, recall={metrics['recall']:.6f}, "
        f"F1={metrics['f1']:.6f}, AP={metrics['average_precision']}."
    )
    print(f"Threshold (validation-selected): {metrics['threshold']:.8g}")
    print(f"Report: {args.report_dir.resolve() / 'model_report.md'}")
    print(
        f"Scored test records: "
        f"{args.processed_dir.resolve() / 'scored_test_transactions.csv'}"
    )
    print(
        f"Graph preparation: "
        f"{args.processed_dir.resolve() / 'graph_test_transactions.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
