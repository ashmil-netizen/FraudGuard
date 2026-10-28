"""Fit the PaySim Random Forest and select its threshold on validation data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml_pipeline import (
    DATASET_IDENTIFIER,
    FEATURE_COLUMNS,
    MODEL_TYPE,
    MODEL_VERSION,
    RANDOM_SEED,
    TARGET_COLUMN,
    DataValidationError,
    build_training_pipeline,
    classification_metrics,
    load_json,
    load_model_bundle,
    sample_training_rows,
    save_model_bundle,
    select_f1_threshold,
    sklearn_version,
    write_json,
)


def _read_training_data(
    processed_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    feature_path = processed_dir / "feature_matrix.csv"
    records_path = processed_dir / "processed_transactions.csv"
    labels_path = processed_dir / "labels.csv"
    schema_path = processed_dir / "feature_schema.json"
    required = (feature_path, records_path, labels_path, schema_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Preprocessing outputs are incomplete; run "
            "backend\\scripts\\preprocess.py first. Missing: "
            + ", ".join(missing)
        )

    schema = load_json(schema_path)
    if schema.get("dataset_identifier") != DATASET_IDENTIFIER:
        raise DataValidationError("Feature schema is not for the active PaySim dataset.")
    if schema.get("feature_names") != list(FEATURE_COLUMNS):
        raise DataValidationError("Feature schema does not match the model input schema.")
    dtype = {column: "float32" for column in FEATURE_COLUMNS if column != "type"}
    dtype["type"] = "string"
    features = pd.read_csv(feature_path, dtype=dtype)
    labels = pd.read_csv(labels_path, dtype={TARGET_COLUMN: "int8"})[TARGET_COLUMN]
    split = pd.read_csv(
        records_path,
        usecols=["step", "split"],
        dtype={"step": "int32", "split": "category"},
    )
    if list(features.columns) != list(FEATURE_COLUMNS):
        raise DataValidationError("Feature matrix columns/order do not match the schema.")
    if not (len(features) == len(labels) == len(split)):
        raise DataValidationError("Feature, label, and transaction row counts differ.")
    if not np.array_equal(
        features["step"].to_numpy(dtype=np.int32),
        split["step"].to_numpy(dtype=np.int32),
    ):
        raise DataValidationError("Feature rows do not align with transaction records.")
    if not set(labels.unique()).issubset({0, 1}):
        raise DataValidationError("Target values must be binary 0/1.")
    numeric = features.loc[:, list(FEATURE_COLUMNS[:-1])].to_numpy(dtype=np.float64)
    if np.isinf(numeric).any():
        raise DataValidationError("Feature matrix contains infinite numeric values.")
    if features["type"].isna().any():
        features["type"] = features["type"].fillna("__missing__")
    return features, labels.to_numpy(dtype=np.int8), split


def train(
    processed_dir: Path,
    model_dir: Path,
    report_dir: Path,
    *,
    max_training_rows: int = 500_000,
    n_estimators: int = 100,
    random_state: int = RANDOM_SEED,
) -> dict:
    if max_training_rows < 2:
        raise DataValidationError("max_training_rows must be at least two.")
    if n_estimators < 1:
        raise DataValidationError("n_estimators must be positive.")
    processed_dir = Path(processed_dir)
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    features, labels, split = _read_training_data(processed_dir)

    train_indices = np.flatnonzero(split["split"].eq("train").to_numpy())
    validation_indices = np.flatnonzero(split["split"].eq("validation").to_numpy())
    test_indices = np.flatnonzero(split["split"].eq("test").to_numpy())
    if train_indices.size == 0 or validation_indices.size == 0 or test_indices.size == 0:
        raise DataValidationError("Chronological train/validation/test splits must be non-empty.")
    if np.unique(labels[train_indices]).size != 2:
        raise DataValidationError("Training split must contain both fraud classes.")
    if np.unique(labels[validation_indices]).size != 2:
        raise DataValidationError(
            "Validation split must contain both classes to select a threshold."
        )

    sampled_indices = sample_training_rows(
        labels,
        train_indices,
        max_rows=max_training_rows,
        random_state=random_state,
    )
    model = build_training_pipeline(
        len(sampled_indices),
        random_state=random_state,
        n_estimators=n_estimators,
    )
    model.fit(features.iloc[sampled_indices], labels[sampled_indices])

    validation_scores = model.predict_proba(
        features.iloc[validation_indices]
    )[:, 1]
    threshold = select_f1_threshold(labels[validation_indices], validation_scores)
    validation_metrics = classification_metrics(
        labels[validation_indices], validation_scores, threshold
    )
    transformed_names = model.named_steps["preprocessor"].get_feature_names_out().tolist()
    schema = load_json(processed_dir / "feature_schema.json")
    bundle = {
        "dataset_identifier": DATASET_IDENTIFIER,
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "model": model,
        "feature_names": list(FEATURE_COLUMNS),
        "transformed_feature_names": transformed_names,
        "target_column": TARGET_COLUMN,
        "classification_threshold": float(threshold),
        "random_state": int(random_state),
        "preprocessing_schema_version": schema["schema_version"],
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "fraud_classifier.joblib"
    metadata_path = model_dir / "model_metadata.json"
    save_model_bundle(bundle, model_path)
    loaded_bundle = load_model_bundle(model_path)
    verification_rows = min(5, len(sampled_indices))
    before = model.predict_proba(features.iloc[sampled_indices[:verification_rows]])[:, 1]
    after = loaded_bundle["model"].predict_proba(
        features.iloc[sampled_indices[:verification_rows]]
    )[:, 1]
    if not np.allclose(before, after, rtol=0.0, atol=1e-12):
        raise DataValidationError("Reloaded model scores differ from in-memory scores.")

    metadata = {
        "dataset_identifier": DATASET_IDENTIFIER,
        "dataset_kind": "simulated mobile-money transactions",
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "parameters": model.named_steps["classifier"].get_params(deep=False),
        "library_versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn_version(),
        },
        "feature_names": list(FEATURE_COLUMNS),
        "transformed_feature_names": transformed_names,
        "target_label": {
            "column": TARGET_COLUMN,
            "positive_value": 1,
            "negative_value": 0,
            "meaning": "PaySim simulation label; not independently adjudicated real-world fraud.",
        },
        "excluded_columns": schema["excluded_columns"],
        "preprocessing_configuration": schema["preprocessing_configuration"],
        "prediction_point": "Post-transaction review after transaction/balance posting.",
        "split_strategy": {
            "name": "chronological by whole step values",
            "fractions_of_rows_at_boundary_selection": schema["split_configuration"][
                "fractions_of_rows_at_boundary_selection"
            ],
            "step_ranges": schema["split_configuration"]["step_partitions"],
            "training_rows_available": int(train_indices.size),
            "training_rows_used": int(sampled_indices.size),
            "validation_rows": int(validation_indices.size),
            "test_rows": int(test_indices.size),
            "random_state": int(random_state),
            "sampling": (
                f"Deterministic class-stratified sample capped at {max_training_rows:,} "
                "rows from the chronological training partition."
            ),
        },
        "class_distribution": {
            "train_full": pd.Series(labels[train_indices]).value_counts().sort_index().to_dict(),
            "train_sample": pd.Series(labels[sampled_indices]).value_counts().sort_index().to_dict(),
            "validation": pd.Series(labels[validation_indices]).value_counts().sort_index().to_dict(),
            "test_support_only": pd.Series(labels[test_indices]).value_counts().sort_index().to_dict(),
        },
        "threshold": {
            "value": float(threshold),
            "selection": "F1-maximizing threshold selected on validation scores only; final test set not used.",
            "validation_metrics_at_selected_threshold": validation_metrics,
        },
        "score_convention": (
            "model.predict_proba(X)[:, 1] positive-class score. It is an uncalibrated "
            "Random Forest score and must not be interpreted as a calibrated fraud probability."
        ),
        "dataset_limitations": [
            "PaySim is simulated data, not genuine bank transaction records.",
            "The fraud label represents the simulation's label, not external adjudication.",
            "The selected balances are post-transaction fields; scores apply only to post-transaction review.",
            "A chronological simulation split does not establish generalization to real institutions or future production data.",
        ],
        "artifact": model_path.name,
    }
    write_json(metadata_path, metadata)

    summary = {
        "model_type": MODEL_TYPE,
        "dataset_rows": int(len(features)),
        "training_rows_available": int(train_indices.size),
        "training_rows_used": int(sampled_indices.size),
        "validation_rows": int(validation_indices.size),
        "test_rows": int(test_indices.size),
        "feature_count": len(FEATURE_COLUMNS),
        "transformed_feature_count": len(transformed_names),
        "classification_threshold": float(threshold),
        "validation_metrics": validation_metrics,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "round_trip_prediction_match": True,
    }
    lines = [
        "# PaySim Random Forest training summary",
        "",
        f"- **Model:** `{MODEL_TYPE}`",
        f"- **Dataset rows:** {len(features):,}",
        f"- **Training rows available / used:** {len(train_indices):,} / {len(sampled_indices):,}",
        f"- **Validation rows:** {len(validation_indices):,}",
        f"- **Held-out test rows:** {len(test_indices):,} (not evaluated or used for threshold selection here)",
        f"- **Raw / encoded feature count:** {len(FEATURE_COLUMNS)} / {len(transformed_names)}",
        f"- **Parameters:** `{model.named_steps['classifier'].get_params(deep=False)}`",
        f"- **Validation-selected threshold:** {threshold:.8g}",
        f"- **Saved model reload check:** {summary['round_trip_prediction_match']}",
        "",
        "The threshold maximizes F1 on the chronological validation split only. The "
        "Random Forest positive-class score is uncalibrated, not a calibrated fraud "
        "probability. Final metrics must be read from the separate held-out evaluation.",
        "",
        "```json",
        pd.Series(summary).to_json(indent=2),
        "```",
        "",
    ]
    (report_dir / "model_training_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and save the PaySim supervised Random Forest baseline."
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
    parser.add_argument("--max-training-rows", type=int, default=500_000)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    try:
        summary = train(
            args.processed_dir,
            args.model_dir,
            args.report_dir,
            max_training_rows=args.max_training_rows,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
        )
    except (DataValidationError, FileNotFoundError, OSError, ValueError) as exc:
        parser.exit(1, f"Could not train PaySim model: {exc}\n")
    metrics = summary["validation_metrics"]
    print(
        f"Trained Random Forest on {summary['training_rows_used']:,} of "
        f"{summary['training_rows_available']:,} chronological training rows; "
        f"validation F1={metrics['f1']:.6f}, AP={metrics['average_precision']:.6f}; "
        f"threshold={summary['classification_threshold']:.8g}."
    )
    print(f"Saved model: {summary['model_path']}")
    print(f"Saved metadata: {summary['metadata_path']}")
    print("Final held-out test metrics are calculated by evaluate_model.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
