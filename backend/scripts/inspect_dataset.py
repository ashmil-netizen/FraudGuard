"""Inspect the actual PaySim CSV using chunked, non-destructive profiling."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from backend.app.ml_pipeline import (
    DATASET_IDENTIFIER,
    ID_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    DataValidationError,
    discover_paysim_csv,
)

CHUNK_SIZE = 100_000
QUANTILE_SAMPLE_SIZE = 10_000


def _update_numeric_stats(
    stats: dict[str, dict[str, Any]],
    column: str,
    values: pd.Series,
    rng: np.random.Generator,
) -> None:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return
    entry = stats.setdefault(
        column,
        {
            "count": 0,
            "sum": 0.0,
            "sum_squares": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "sample": np.empty(0, dtype=np.float64),
        },
    )
    entry["count"] += int(numeric.size)
    entry["sum"] += float(numeric.sum(dtype=np.float64))
    entry["sum_squares"] += float(np.square(numeric).sum(dtype=np.float64))
    entry["min"] = min(entry["min"], float(numeric.min()))
    entry["max"] = max(entry["max"], float(numeric.max()))
    sample = np.concatenate((entry["sample"], numeric))
    if sample.size > QUANTILE_SAMPLE_SIZE:
        entry["sample"] = rng.choice(sample, size=QUANTILE_SAMPLE_SIZE, replace=False)
    else:
        entry["sample"] = sample


def _scan_csv(csv_path: Path) -> dict[str, Any]:
    columns = list(pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns)
    missing_columns = REQUIRED_COLUMNS - set(columns)
    if missing_columns:
        raise DataValidationError(
            "PaySim source CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    missing_counts = Counter()
    dtype_names: dict[str, set[str]] = {column: set() for column in columns}
    numeric_stats: dict[str, dict[str, Any]] = {}
    type_counts: Counter[str] = Counter()
    fraud_counts: Counter[str] = Counter()
    flagged_counts: Counter[str] = Counter()
    rows = 0
    self_transfer_rows = 0
    invalid_label_rows = 0
    rng = np.random.default_rng(2026)
    temp = tempfile.TemporaryDirectory(prefix="fraudguard-paysim-inspection-")
    connection = sqlite3.connect(Path(temp.name) / "row_hashes.sqlite")
    connection.execute("CREATE TABLE fingerprints (value INTEGER PRIMARY KEY) WITHOUT ROWID")
    duplicate_rows = 0
    try:
        for chunk in pd.read_csv(
            csv_path,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            encoding="utf-8-sig",
            dtype={"type": "string", "nameOrig": "string", "nameDest": "string"},
        ):
            rows += len(chunk)
            for column in columns:
                missing_counts[column] += int(chunk[column].isna().sum())
                dtype_names[column].add(str(chunk[column].dtype))
                if pd.api.types.is_numeric_dtype(chunk[column].dtype):
                    _update_numeric_stats(numeric_stats, column, chunk[column], rng)

            type_counts.update(
                {str(value): int(count) for value, count in chunk["type"].value_counts(dropna=False).items()}
            )
            fraud_counts.update(
                {str(value): int(count) for value, count in chunk[TARGET_COLUMN].value_counts(dropna=False).items()}
            )
            if "isFlaggedFraud" in chunk.columns:
                flagged_counts.update(
                    {
                        str(value): int(count)
                        for value, count in chunk["isFlaggedFraud"].value_counts(dropna=False).items()
                    }
                )
            labels = pd.to_numeric(chunk[TARGET_COLUMN], errors="coerce")
            invalid_label_rows += int((~labels.isin([0, 1])).sum())
            self_transfer_rows += int(
                chunk["nameOrig"].astype("string").eq(
                    chunk["nameDest"].astype("string")
                ).sum()
            )

            row_hashes = (
                pd.util.hash_pandas_object(chunk, index=False)
                .to_numpy(dtype=np.uint64)
                .view(np.int64)
            )
            before = connection.total_changes
            connection.executemany(
                "INSERT OR IGNORE INTO fingerprints(value) VALUES (?)",
                ((int(value),) for value in row_hashes),
            )
            unique_added = connection.total_changes - before
            duplicate_rows += len(chunk) - unique_added
            connection.commit()
    finally:
        connection.close()
        temp.cleanup()

    column_profiles = []
    for column in columns:
        column_profiles.append(
            {
                "name": column,
                "data_types": sorted(dtype_names[column]),
                "missing_values": int(missing_counts[column]),
                "missing_percentage": (
                    float(missing_counts[column] * 100.0 / rows) if rows else 0.0
                ),
                "unique_values": (
                    type_counts.most_common()
                    if column == "type"
                    else (
                        fraud_counts.most_common()
                        if column == TARGET_COLUMN
                        else (
                            flagged_counts.most_common()
                            if column == "isFlaggedFraud"
                            else None
                        )
                    )
                ),
                "identifier": column in ID_COLUMNS,
            }
        )

    finalized_numeric = {}
    for column, entry in numeric_stats.items():
        mean = entry["sum"] / entry["count"]
        variance = max(0.0, entry["sum_squares"] / entry["count"] - mean * mean)
        quantiles = np.quantile(entry["sample"], [0.25, 0.5, 0.75])
        finalized_numeric[column] = {
            "count": entry["count"],
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "min": entry["min"],
            "25%": float(quantiles[0]),
            "50%": float(quantiles[1]),
            "75%": float(quantiles[2]),
            "max": entry["max"],
            "quantiles_are_sampled": entry["count"] > QUANTILE_SAMPLE_SIZE,
            "quantile_sample_size": int(len(entry["sample"])),
        }

    return {
        "filename": csv_path.name,
        "path": str(csv_path.resolve()),
        "rows": int(rows),
        "columns_count": len(columns),
        "columns": column_profiles,
        "column_names": columns,
        "duplicate_rows": int(duplicate_rows),
        "numeric_statistics": finalized_numeric,
        "transaction_type_counts": dict(sorted(type_counts.items())),
        "fraud_label_counts": dict(sorted(fraud_counts.items())),
        "existing_flag_counts": dict(sorted(flagged_counts.items())),
        "invalid_label_rows": invalid_label_rows,
        "self_transfer_rows": self_transfer_rows,
        "account_identifier_columns": [
            column for column in ID_COLUMNS if column in columns
        ],
        "timestamp_columns": [
            column
            for column in columns
            if any(token in column.lower() for token in ("date", "time", "timestamp"))
        ],
        "numeric_columns": sorted(finalized_numeric),
        "duplicate_method": (
            "Streaming pandas stable 64-bit row fingerprints deduplicated in a temporary SQLite table."
        ),
    }


def _analysis_report(report: dict[str, Any]) -> str:
    profile = report["file"]
    labels = profile["fraud_label_counts"]
    fraud_count = int(labels.get("1", 0))
    total = int(profile["rows"])
    fraud_percent = fraud_count * 100.0 / total if total else 0.0
    lines = [
        "# PaySim dataset analysis",
        "",
        "## Verified dataset facts",
        "",
        f"- Dataset identifier: `{DATASET_IDENTIFIER}`.",
        f"- Actual CSV: `{profile['filename']}` ({profile['rows']:,} rows × {profile['columns_count']} columns).",
        f"- Columns: {', '.join(f'`{name}`' for name in profile['column_names'])}.",
        f"- Duplicate rows found by the recorded row-fingerprint method: {profile['duplicate_rows']:,}.",
        f"- Missing values by column: `{ {item['name']: item['missing_values'] for item in profile['columns']} }`.",
        f"- Observed transaction types: `{profile['transaction_type_counts']}`.",
        f"- Observed `isFraud` counts: `{labels}`; fraud rows are {fraud_percent:.6f}% of the scanned CSV.",
        f"- Observed existing `isFlaggedFraud` counts: `{profile['existing_flag_counts']}`.",
        f"- Sender/recipient identifier columns: `{profile['account_identifier_columns']}`.",
        f"- Timestamp columns: `{profile['timestamp_columns'] or 'none'}`; `step` is the available simulation-time field.",
        "",
        "## Dataset meaning and relationships",
        "",
        "PaySim is a simulated mobile-money transaction dataset, not genuine bank "
        "customer records. Each row describes one transaction and contains a sender "
        "identifier, recipient identifier, transaction type, amount, balance snapshots, "
        "and a simulated fraud label.",
        "",
        "- `nameOrig` and `nameDest` define directed sender-to-recipient relationships within each transaction row.",
        "- No separate account table or source transaction ID column was observed. The preprocessing pipeline creates a stable ID from the source row position for lookup.",
        "- Repeated sender/recipient pairs are separate transaction events; they must not be collapsed into a single edge.",
        "- `isFraud` is the observed target label. `isFlaggedFraud` is an existing rule flag and is excluded from model features.",
        "",
        "## Methodology recommendations",
        "",
        "- Use one transaction per prediction record and preserve identifiers separately from model features.",
        "- A chronological split by whole `step` values is preferable to a random split because it evaluates later simulated transactions after earlier ones.",
        "- The baseline's decision point is defined as post-transaction review because the selected feature set includes resulting balances (`newbalanceOrig`, `newbalanceDest`). These are leakage for pre-transaction authorization.",
        "- Report precision, recall, F1, confusion matrix, average precision, ROC-AUC, and class support; accuracy alone obscures the effect of rare fraud labels.",
        "- Choose the classification threshold using validation data only and report Random Forest scores as uncalibrated model scores, not calibrated fraud probabilities.",
        "",
        "## Proposed graph representation for the next phase",
        "",
        "Represent sender and recipient accounts as account nodes and each transaction "
        "as its own transaction node, connected by directed `sender -> transaction -> "
        "recipient` edges. This preserves parallel transactions, transaction IDs, "
        "step, amount, type, and offline score/label attributes without overwriting "
        "repeated account pairs. A flagged transaction does not prove either account "
        "is fraudulent. Keep the ground-truth label in offline evaluation data only.",
        "",
        "## Limitations",
        "",
        "- All records are simulated; performance does not establish performance on real financial institutions.",
        "- The target is a simulation label, not independently adjudicated real-world fraud.",
        "- Identifiers are synthetic and are not meaningful numeric risk measurements.",
        "- Simulation time steps are not calendar timestamps.",
        "- Threshold and metrics depend on the simulated data distribution and chronological split.",
        "",
    ]
    return "\n".join(lines)


def _inspection_report(report: dict[str, Any]) -> str:
    profile = report["file"]
    lines = [
        "# PaySim dataset inspection report",
        "",
        f"- **Dataset identifier:** `{DATASET_IDENTIFIER}`",
        f"- **Dataset directory:** `{report['data_directory']}`",
        f"- **Actual CSV:** `{profile['filename']}`",
        f"- **Rows × columns:** {profile['rows']:,} × {profile['columns_count']}",
        f"- **Duplicate rows:** {profile['duplicate_rows']:,}",
        f"- **Duplicate method:** {profile['duplicate_method']}",
        f"- **Timestamp columns:** {', '.join(profile['timestamp_columns']) or 'None'}",
        "",
        "| Column | Inferred data type(s) | Missing | Missing % | Unique values / counts |",
        "|---|---|---:|---:|---|",
    ]
    for column in profile["columns"]:
        values = column["unique_values"]
        lines.append(
            f"| `{column['name']}` | {', '.join(column['data_types'])} | "
            f"{column['missing_values']:,} | {column['missing_percentage']:.5f}% | "
            f"`{values}` |"
        )
    lines += [
        "",
        "## Numeric descriptive statistics",
        "",
        "| Column | Count | Mean | Std. dev. | Min | 25% | Median | 75% | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in profile["numeric_statistics"].items():
        lines.append(
            f"| `{name}` | {stats['count']:,} | {stats['mean']:.6g} | "
            f"{stats['std']:.6g} | {stats['min']:.6g} | {stats['25%']:.6g} | "
            f"{stats['50%']:.6g} | {stats['75%']:.6g} | {stats['max']:.6g} |"
        )
    lines += [
        "",
        "Quartiles are calculated from a deterministic sample when the column has "
        f"more than {QUANTILE_SAMPLE_SIZE:,} valid values; count, mean, standard "
        "deviation, minimum, and maximum use all valid values.",
        "",
        "## Verified transaction relationships",
        "",
        "- Each row contains `nameOrig` (sender) and `nameDest` (recipient), defining a directed transaction relationship.",
        "- There is no separate account table or relational foreign key in the inspected CSV.",
        f"- Rows whose sender and recipient identifiers match: {profile['self_transfer_rows']:,}.",
        "- A source transaction ID column is not present; preprocessing derives `transaction_id` from the stable source row index.",
        "- Parallel sender/recipient transactions must be preserved as distinct transaction records.",
        "",
        "## Identifier/key assessment",
        "",
        "- A source transaction ID or separate account table was not present in the inspected CSV schema.",
        "- `nameOrig` and `nameDest` are account identifiers used as graph endpoints; neither column is a foreign key to a separate table in this one-file dataset.",
        "- Preprocessing derives `transaction_id`/`source_row_index` from the row position to preserve transaction lookup. It is a dataset-snapshot surrogate, not a source-provided business key.",
        "- No conventional relational primary-key/foreign-key pair can be verified from this CSV alone.",
        "",
    ]
    return "\n".join(lines)


def inspect_dataset(data_dir: Path, report_dir: Path) -> dict[str, Any]:
    data_dir = Path(data_dir).resolve()
    report_dir = Path(report_dir).resolve()
    csv_path = discover_paysim_csv(data_dir)
    profile = _scan_csv(csv_path)
    report = {
        "dataset_identifier": DATASET_IDENTIFIER,
        "data_directory": str(csv_path.parent.resolve()),
        "csv_files_found": len(list(csv_path.parent.glob("*.csv"))),
        "csv_files_inspected": 1,
        "file": profile,
        "key_assessment": {
            "source_transaction_id_column": None,
            "derived_lookup_key": "source_row_index / transaction_id",
            "derived_key_scope": "unique within this downloaded file snapshot; not source-provided",
            "account_columns": list(ID_COLUMNS),
            "separate_account_table": False,
            "verified_relational_foreign_keys": [],
        },
        "relationships": [
            {
                "source_column": "nameOrig",
                "target_column": "nameDest",
                "relationship": "One directed sender-to-recipient relationship per transaction row",
                "key_exists_in_both_columns": True,
                "cardinality": "many-to-many across transaction rows; repeated pairs are preserved",
                "separate_account_table": False,
                "self_transfer_rows": profile["self_transfer_rows"],
            }
        ],
        "potential_label_columns": [TARGET_COLUMN]
        if TARGET_COLUMN in profile["column_names"]
        else [],
        "limitations": [
            "PaySim is simulated mobile-money transaction data, not genuine customer bank records.",
            "isFlaggedFraud is an existing rule flag and is not a model feature.",
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dataset_inspection.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (report_dir / "dataset_inspection.md").write_text(
        _inspection_report(report), encoding="utf-8"
    )
    (report_dir / "dataset_analysis.md").write_text(
        _analysis_report(report), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the discovered PaySim CSV and write verified reports."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "backend" / "data" / "raw",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "backend" / "reports",
    )
    args = parser.parse_args()
    try:
        report = inspect_dataset(args.data_dir, args.report_dir)
    except (DataValidationError, FileNotFoundError, OSError, pd.errors.ParserError) as exc:
        parser.exit(1, f"Could not inspect PaySim data: {exc}\n")
    profile = report["file"]
    print(
        f"Inspected {profile['filename']}: {profile['rows']:,} rows × "
        f"{profile['columns_count']} columns; {profile['duplicate_rows']:,} "
        "duplicate rows."
    )
    for report_name in (
        "dataset_inspection.md",
        "dataset_inspection.json",
        "dataset_analysis.md",
    ):
        print(f"Report: {args.report_dir.resolve() / report_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
