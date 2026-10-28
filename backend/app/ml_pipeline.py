"""PaySim preprocessing, supervised model, and evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASET_IDENTIFIER = "ealaxi/paysim1"
SCHEMA_VERSION = "2.0"
MODEL_VERSION = "2.0"
RANDOM_SEED = 42
TARGET_COLUMN = "isFraud"
ID_COLUMNS = ("nameOrig", "nameDest")
NUMERIC_FEATURE_COLUMNS = (
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)
CATEGORICAL_FEATURE_COLUMNS = ("type",)
FEATURE_COLUMNS = (*NUMERIC_FEATURE_COLUMNS, *CATEGORICAL_FEATURE_COLUMNS)
REQUIRED_COLUMNS = set((*FEATURE_COLUMNS, *ID_COLUMNS, TARGET_COLUMN))
RAW_NUMERIC_COLUMNS = (
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    TARGET_COLUMN,
    "isFlaggedFraud",
)
MODEL_TYPE = "sklearn.ensemble.RandomForestClassifier"

FEATURE_DEFINITIONS: dict[str, dict[str, str]] = {
    "step": {
        "meaning": "Integer simulation time step supplied by PaySim.",
        "calculation": "Parsed from the source column without transformation.",
        "sources": "step",
        "rationale": "Captures chronological changes in the simulated transaction stream.",
        "limitations": "The dataset uses simulated time steps, not real timestamps.",
    },
    "amount": {
        "meaning": "Transaction amount.",
        "calculation": "Parsed as a finite, non-negative number.",
        "sources": "amount",
        "rationale": "Transaction size can distinguish unusual transaction patterns.",
        "limitations": "Currency units and real-world representativeness are not established.",
    },
    "oldbalanceOrg": {
        "meaning": "Sender balance before the transaction.",
        "calculation": "Parsed as a non-negative number; invalid values are imputed by a training-fitted median.",
        "sources": "oldbalanceOrg",
        "rationale": "Provides sender-account context for a transaction.",
        "limitations": "The model's stated prediction point is after transaction posting; not valid for pre-transaction prediction.",
    },
    "newbalanceOrig": {
        "meaning": "Sender balance after the transaction.",
        "calculation": "Parsed as a non-negative number; invalid values are imputed by a training-fitted median.",
        "sources": "newbalanceOrig",
        "rationale": "Can expose unusual balance outcomes in the simulated data.",
        "limitations": "Post-transaction information; excluded for any pre-transaction decision use case.",
    },
    "oldbalanceDest": {
        "meaning": "Recipient balance before the transaction.",
        "calculation": "Parsed as a non-negative number; invalid values are imputed by a training-fitted median.",
        "sources": "oldbalanceDest",
        "rationale": "Provides recipient-account context for a transaction.",
        "limitations": "The model's stated prediction point is after transaction posting; not valid for pre-transaction prediction.",
    },
    "newbalanceDest": {
        "meaning": "Recipient balance after the transaction.",
        "calculation": "Parsed as a non-negative number; invalid values are imputed by a training-fitted median.",
        "sources": "newbalanceDest",
        "rationale": "Can expose unusual balance outcomes in the simulated data.",
        "limitations": "Post-transaction information; excluded for any pre-transaction decision use case.",
    },
    "type": {
        "meaning": "PaySim transaction type.",
        "calculation": "Trimmed string category; missing values use an explicit missing category.",
        "sources": "type",
        "rationale": "Different transaction types have different normal patterns.",
        "limitations": "Categories describe the simulation and may encode simulated behavior.",
    },
}


class DataValidationError(ValueError):
    """Raised when a PaySim input or saved artifact is invalid."""


def validate_paysim_columns(columns: Sequence[str]) -> None:
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        raise DataValidationError(
            "PaySim CSV is missing required column(s): "
            + ", ".join(sorted(missing))
        )


def discover_paysim_csv(data_dir: Path) -> Path:
    """Find the unique CSV whose header matches the verified PaySim schema."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    candidates: list[Path] = []
    for path in sorted(data_dir.rglob("*.csv")):
        try:
            columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError):
            continue
        if REQUIRED_COLUMNS.issubset(columns):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No CSV with the required PaySim columns was found under {data_dir}. "
            "Run backend\\scripts\\download_dataset.py and inspect the downloaded schema."
        )
    if len(candidates) > 1:
        raise DataValidationError(
            "Multiple CSV files match the PaySim schema; specify the dataset "
            "directory containing exactly one source file: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def _step_partitions(step_counts: dict[int, int]) -> dict[str, Any]:
    ordered = np.asarray(sorted(step_counts), dtype=np.int64)
    if ordered.size < 3:
        raise DataValidationError(
            "At least three distinct valid time steps are required for "
            "chronological train/validation/test splits."
        )
    row_counts = np.asarray([step_counts[int(step)] for step in ordered], dtype=np.int64)
    cumulative_rows = np.cumsum(row_counts)
    total_rows = int(cumulative_rows[-1])
    train_end = max(
        1,
        int(np.searchsorted(cumulative_rows, total_rows * 0.70, side="left")) + 1,
    )
    train_end = min(train_end, ordered.size - 2)
    validation_end = max(
        train_end + 1,
        int(np.searchsorted(cumulative_rows, total_rows * 0.85, side="left")) + 1,
    )
    validation_end = min(validation_end, ordered.size - 1)
    train_rows = int(row_counts[:train_end].sum())
    validation_rows = int(row_counts[train_end:validation_end].sum())
    test_rows = int(row_counts[validation_end:].sum())
    return {
        "train_step_max": int(ordered[train_end - 1]),
        "validation_step_max": int(ordered[validation_end - 1]),
        "test_step_min": int(ordered[validation_end]),
        "unique_steps": int(ordered.size),
        "rows_by_split_at_boundary_selection": {
            "train": train_rows,
            "validation": validation_rows,
            "test": test_rows,
            "total": total_rows,
        },
        "step_partitions": {
            "train": [int(ordered[0]), int(ordered[train_end - 1])],
            "validation": [
                int(ordered[train_end]),
                int(ordered[validation_end - 1]),
            ],
            "test": [int(ordered[validation_end]), int(ordered[-1])],
        },
        "fractions_of_rows_at_boundary_selection": {
            "train": float(train_rows / total_rows),
            "validation": float(validation_rows / total_rows),
            "test": float(test_rows / total_rows),
        },
    }


def _write_frame(frame: pd.DataFrame, path: Path, first: bool) -> None:
    frame.to_csv(path, mode="w" if first else "a", header=first, index=False)


def preprocess_dataset(
    csv_path: Path,
    processed_dir: Path,
    report_dir: Path,
    *,
    chunk_size: int = 250_000,
) -> dict[str, Any]:
    """Stream PaySim rows into model, lookup, label, and graph-preparation files."""
    csv_path = Path(csv_path)
    processed_dir = Path(processed_dir)
    report_dir = Path(report_dir)
    if chunk_size < 1:
        raise DataValidationError("chunk_size must be positive.")
    if not csv_path.is_file():
        raise FileNotFoundError(f"PaySim CSV does not exist: {csv_path}")

    header = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns
    validate_paysim_columns(header)

    step_counts: dict[int, int] = {}
    for step_chunk in pd.read_csv(
        csv_path, usecols=["step"], chunksize=chunk_size, low_memory=False
    ):
        step_values = pd.to_numeric(step_chunk["step"], errors="coerce")
        valid = step_values[
            np.isfinite(step_values)
            & step_values.gt(0)
            & step_values.eq(np.floor(step_values))
        ].astype("int64")
        for step, count in valid.value_counts().items():
            step_counts[int(step)] = step_counts.get(int(step), 0) + int(count)
    split_config = _step_partitions(step_counts)

    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "records": processed_dir / "processed_transactions.csv",
        "features": processed_dir / "feature_matrix.csv",
        "labels": processed_dir / "labels.csv",
        "graph": processed_dir / "graph_transactions.csv",
    }
    audit: dict[str, Any] = {
        "input_rows": 0,
        "retained_rows": 0,
        "dropped_invalid_core_rows": 0,
        "missing_or_invalid_balance_values": {
            column: 0 for column in NUMERIC_FEATURE_COLUMNS[2:]
        },
        "missing_transaction_type_rows": 0,
        "class_distribution": {"0": 0, "1": 0},
        "split_distribution": {},
        "split_class_distribution": {},
    }
    first_write = {key: True for key in output_paths}
    row_offset = 0
    read_dtypes = {"type": "string", "nameOrig": "string", "nameDest": "string"}
    for chunk in pd.read_csv(
        csv_path, chunksize=chunk_size, low_memory=False, dtype=read_dtypes
    ):
        source_row_indices = np.arange(
            row_offset, row_offset + len(chunk), dtype=np.int64
        )
        row_offset += len(chunk)
        audit["input_rows"] += int(len(chunk))

        for column in RAW_NUMERIC_COLUMNS:
            if column in chunk.columns:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

        step = chunk["step"]
        amount = chunk["amount"]
        label = chunk[TARGET_COLUMN]
        identifiers_valid = (
            chunk["nameOrig"].notna()
            & chunk["nameOrig"].str.strip().ne("")
            & chunk["nameDest"].notna()
            & chunk["nameDest"].str.strip().ne("")
        )
        valid_core = (
            identifiers_valid
            & np.isfinite(step)
            & step.gt(0)
            & step.eq(np.floor(step))
            & np.isfinite(amount)
            & amount.ge(0)
            & label.isin([0, 1])
        )
        audit["dropped_invalid_core_rows"] += int((~valid_core).sum())
        chunk = chunk.loc[valid_core].copy()
        source_row_indices = source_row_indices[valid_core.to_numpy()]
        if chunk.empty:
            continue

        chunk["step"] = chunk["step"].astype("int32")
        chunk["amount"] = chunk["amount"].astype("float32")
        chunk[TARGET_COLUMN] = chunk[TARGET_COLUMN].astype("int8")
        chunk["type"] = chunk["type"].str.strip()
        missing_type = chunk["type"].isna() | chunk["type"].eq("")
        audit["missing_transaction_type_rows"] += int(missing_type.sum())
        chunk.loc[missing_type, "type"] = "__missing__"

        for column in NUMERIC_FEATURE_COLUMNS[2:]:
            invalid = ~np.isfinite(chunk[column]) | chunk[column].lt(0)
            audit["missing_or_invalid_balance_values"][column] += int(invalid.sum())
            chunk.loc[invalid, column] = np.nan
            chunk[column] = chunk[column].astype("float32")

        split_values = np.select(
            [
                chunk["step"].le(split_config["train_step_max"]),
                chunk["step"].le(split_config["validation_step_max"]),
            ],
            ["train", "validation"],
            default="test",
        )
        transaction_ids = [f"paysim-{index:012d}" for index in source_row_indices]
        feature_frame = chunk.loc[:, list(FEATURE_COLUMNS)].copy()
        records = pd.DataFrame(
            {
                "transaction_id": transaction_ids,
                "source_row_index": source_row_indices,
                "step": chunk["step"].to_numpy(),
                "type": chunk["type"].to_numpy(),
                "amount": chunk["amount"].to_numpy(),
                "nameOrig": chunk["nameOrig"].str.strip().to_numpy(),
                "oldbalanceOrg": chunk["oldbalanceOrg"].to_numpy(),
                "newbalanceOrig": chunk["newbalanceOrig"].to_numpy(),
                "nameDest": chunk["nameDest"].str.strip().to_numpy(),
                "oldbalanceDest": chunk["oldbalanceDest"].to_numpy(),
                "newbalanceDest": chunk["newbalanceDest"].to_numpy(),
                TARGET_COLUMN: chunk[TARGET_COLUMN].to_numpy(),
                "split": split_values,
            }
        )
        graph_records = records.loc[
            :,
            [
                "transaction_id",
                "source_row_index",
                "step",
                "nameOrig",
                "nameDest",
                "type",
                "amount",
                TARGET_COLUMN,
                "split",
            ],
        ]
        for key, frame in (
            ("records", records),
            ("features", feature_frame),
            ("labels", chunk.loc[:, [TARGET_COLUMN]]),
            ("graph", graph_records),
        ):
            _write_frame(frame, output_paths[key], first_write[key])
            first_write[key] = False

        audit["retained_rows"] += int(len(chunk))
        class_counts = chunk[TARGET_COLUMN].value_counts()
        for label_value, count in class_counts.items():
            audit["class_distribution"][str(int(label_value))] += int(count)
        split_series = pd.Series(split_values, index=chunk.index)
        for split_name, split_indices in split_series.groupby(split_series).groups.items():
            audit["split_distribution"][split_name] = (
                audit["split_distribution"].get(split_name, 0) + len(split_indices)
            )
            split_counts = chunk.loc[split_indices, TARGET_COLUMN].value_counts()
            target_split = audit["split_class_distribution"].setdefault(
                split_name, {"0": 0, "1": 0}
            )
            for label_value, count in split_counts.items():
                target_split[str(int(label_value))] += int(count)

    if not audit["retained_rows"]:
        raise DataValidationError(
            "No valid PaySim transactions remain after input validation."
        )
    if any(audit["class_distribution"][label] == 0 for label in ("0", "1")):
        raise DataValidationError(
            "The retained data must contain both isFraud classes for supervised training."
        )

    schema = {
        "schema_version": SCHEMA_VERSION,
        "dataset_identifier": DATASET_IDENTIFIER,
        "source_filename": csv_path.name,
        "prediction_unit": "one PaySim transaction per row",
        "target_column": TARGET_COLUMN,
        "feature_names": list(FEATURE_COLUMNS),
        "numeric_feature_names": list(NUMERIC_FEATURE_COLUMNS),
        "categorical_feature_names": list(CATEGORICAL_FEATURE_COLUMNS),
        "feature_definitions": FEATURE_DEFINITIONS,
        "excluded_columns": {
            "identifiers": list(ID_COLUMNS),
            "target": [TARGET_COLUMN],
            "existing_rule_flag": [
                "isFlaggedFraud (excluded because it is an existing detection flag)"
            ],
        },
        "preprocessing_configuration": {
            "invalid_core_rows": "Drop rows with invalid/non-positive/non-integer step, invalid/negative amount, missing account identifiers, or a target outside {0, 1}.",
            "balance_values": "Negative, nonnumeric, or non-finite balance values are set to missing; training-fitted median imputation is applied by the saved pipeline.",
            "transaction_type": "Trim values; missing/blank categories become __missing__; the saved encoder ignores unseen categories.",
            "identifiers": "Preserve source identifiers and a stable source-row-derived transaction_id for record lookup and graph preparation; never use account IDs as model features.",
            "target": "isFraud is stored separately in labels.csv and offline lookup/graph data; it is never in the feature matrix.",
            "existing_flag": "isFlaggedFraud is excluded from all model inputs to avoid encoding an existing detection rule.",
            "split": "Chronological split by whole sorted step values using cumulative transaction-row counts to target 70% train, 15% validation, and 15% test; complete time steps are never divided between splits.",
            "prediction_point": "Post-transaction review after the transaction and resulting balance fields are available. This pipeline is not valid for pre-transaction authorization.",
            "raw_data": "Input CSV is read only and remains unchanged.",
        },
        "split_configuration": split_config,
        "audit": audit,
    }
    schema_path = processed_dir / "feature_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    _write_preprocessing_report(schema, report_dir / "preprocessing_report.md")
    return schema


def _write_preprocessing_report(schema: dict[str, Any], path: Path) -> None:
    audit = schema["audit"]
    lines = [
        "# PaySim preprocessing report",
        "",
        f"- **Dataset:** `{schema['dataset_identifier']}`",
        f"- **Source file:** `{schema['source_filename']}`",
        f"- **Prediction unit:** {schema['prediction_unit']}",
        f"- **Input rows:** {audit['input_rows']:,}",
        f"- **Retained rows:** {audit['retained_rows']:,}",
        f"- **Dropped invalid rows:** {audit['dropped_invalid_core_rows']:,}",
        f"- **Fraud-label distribution:** `{audit['class_distribution']}`",
        "",
        "## Features",
        "",
        "| Feature | Meaning | Calculation | Sources | Limitations |",
        "|---|---|---|---|---|",
    ]
    for name in schema["feature_names"]:
        definition = schema["feature_definitions"][name]
        lines.append(
            f"| `{name}` | {definition['meaning']} | {definition['calculation']} | "
            f"`{definition['sources']}` | {definition['limitations']} |"
        )
    lines += [
        "",
        "## Data handling",
        "",
        f"- Balance-field invalid counts: `{audit['missing_or_invalid_balance_values']}`.",
        f"- Missing/blank transaction-type rows: {audit['missing_transaction_type_rows']:,}.",
        "- `nameOrig` and `nameDest` are preserved as identifiers but are not model features.",
        "- `isFraud` is stored separately as the target; `isFlaggedFraud` is excluded as an existing rule flag.",
        "- Account IDs are not converted into arbitrary numeric features.",
        "- No raw source rows are modified.",
        "",
        "## Chronological partitions",
        "",
        "| Partition | Rows | Non-fraud | Simulated fraud | Step range |",
        "|---|---:|---:|---:|---|",
    ]
    for split_name in ("train", "validation", "test"):
        counts = audit["split_class_distribution"].get(
            split_name, {"0": 0, "1": 0}
        )
        step_range = schema["split_configuration"]["step_partitions"][split_name]
        lines.append(
            f"| {split_name} | {audit['split_distribution'].get(split_name, 0):,} | "
            f"{counts['0']:,} | {counts['1']:,} | {step_range[0]}-{step_range[1]} |"
        )
    lines += [
        "",
        "## Split and prediction point",
        "",
        "The split is chronological by whole simulation steps and uses cumulative row "
        "counts to target 70/15/15. The feature set includes "
        "post-transaction balances, so the intended decision point is after the "
        "transaction has posted. It must not be interpreted as a pre-transaction model.",
        "",
        "PaySim is simulated mobile-money data; this report describes performance on "
        "that synthetic dataset, not on genuine bank customers or deployed fraud.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_model(
    max_samples: int, random_state: int, n_estimators: int
) -> Pipeline:
    numeric = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            )
        ]
    )
    categorical = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__missing__",
                    keep_empty_features=True,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURE_COLUMNS)),
            ("categorical", categorical, list(CATEGORICAL_FEATURE_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=16,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
        bootstrap=True,
        max_samples=max_samples,
    )
    return Pipeline([("preprocessor", preprocessing), ("classifier", classifier)])


def select_f1_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.size == 0 or labels_array.shape != scores_array.shape:
        raise DataValidationError(
            "Threshold selection requires matching non-empty inputs."
        )
    if not np.isfinite(scores_array).all():
        raise DataValidationError("Threshold selection scores must be finite.")
    if np.unique(labels_array).size != 2:
        raise DataValidationError(
            "Threshold selection requires both classes in validation data."
        )
    precision, recall, thresholds = precision_recall_curve(labels_array, scores_array)
    if thresholds.size == 0:
        return 0.5
    f1_values = (
        2.0
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    )
    best = np.flatnonzero(f1_values == f1_values.max())
    return float(thresholds[best[-1]])


def classification_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, Any]:
    actual = np.asarray(labels, dtype=np.int8)
    score_values = np.asarray(scores, dtype=np.float64)
    if actual.size == 0 or actual.shape != score_values.shape:
        raise DataValidationError(
            "Evaluation requires matching non-empty labels and scores."
        )
    if not np.isfinite(score_values).all():
        raise DataValidationError("Evaluation scores must be finite.")
    predicted = score_values >= threshold
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    both_classes = np.unique(actual).size == 2
    return {
        "records": int(actual.size),
        "fraud_support": int(actual.sum()),
        "non_fraud_support": int((actual == 0).sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "average_precision": (
            float(average_precision_score(actual, score_values))
            if actual.sum() > 0
            else None
        ),
        "roc_auc": float(roc_auc_score(actual, score_values)) if both_classes else None,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "flagged_records": int(predicted.sum()),
        "flagged_percentage": float(predicted.mean() * 100.0),
        "score_description": (
            "Uncalibrated Random Forest positive-class score; not a calibrated "
            "probability of fraud."
        ),
    }


def sample_training_rows(
    labels: Sequence[int],
    indices: np.ndarray,
    *,
    max_rows: int,
    random_state: int,
) -> np.ndarray:
    if max_rows < 2:
        raise DataValidationError("max_training_rows must be at least two.")
    if indices.size <= max_rows:
        return indices
    selected_labels = np.asarray(labels)[indices]
    classes, class_counts = np.unique(selected_labels, return_counts=True)
    if len(classes) != 2:
        raise DataValidationError("Training split must contain both fraud classes.")
    proportions = class_counts / class_counts.sum()
    allocation = np.floor(proportions * max_rows).astype(int)
    allocation = np.maximum(allocation, 1)
    while allocation.sum() > max_rows:
        largest = int(np.argmax(allocation))
        if allocation[largest] <= 1:
            raise DataValidationError(
                "Cannot create a class-stratified training sample."
            )
        allocation[largest] -= 1
    rng = np.random.default_rng(random_state)
    sampled: list[np.ndarray] = []
    for class_value, class_allocation in zip(classes, allocation):
        class_indices = indices[selected_labels == class_value]
        count = min(int(class_allocation), len(class_indices))
        sampled.append(rng.choice(class_indices, size=count, replace=False))
    result = np.concatenate(sampled)
    rng.shuffle(result)
    return result


def save_model_bundle(bundle: dict[str, Any], model_path: Path) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved PaySim model was not found: {model_path}")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise DataValidationError("Saved model artifact is not a PaySim model bundle.")
    if bundle.get("dataset_identifier") != DATASET_IDENTIFIER:
        raise DataValidationError(
            "Saved model dataset identifier does not match the active PaySim dataset."
        )
    if bundle.get("model_type") != MODEL_TYPE:
        raise DataValidationError(
            "Saved model type does not match the active PaySim classifier."
        )
    if bundle.get("feature_names") != list(FEATURE_COLUMNS):
        raise DataValidationError("Saved model feature schema is incompatible.")
    return bundle


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"Could not read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"Expected a JSON object in {path}.")
    return payload


def build_training_pipeline(
    training_rows: int,
    random_state: int = RANDOM_SEED,
    n_estimators: int = 100,
) -> Pipeline:
    if training_rows < 2:
        raise DataValidationError("At least two training rows are required.")
    if n_estimators < 1:
        raise DataValidationError("n_estimators must be positive.")
    return _build_model(
        max_samples=min(100_000, training_rows),
        random_state=random_state,
        n_estimators=n_estimators,
    )


def sklearn_version() -> str:
    return sklearn.__version__
