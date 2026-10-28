"""Validate PaySim data and stream transaction-level ML/graph outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml_pipeline import (
    DataValidationError,
    discover_paysim_csv,
    preprocess_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess the verified PaySim CSV into transaction-level data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "backend" / "data" / "raw",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "backend" / "data" / "processed",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Explicit PaySim CSV path; otherwise discover it under --data-dir.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "backend" / "reports",
    )
    args = parser.parse_args()
    try:
        csv_path = args.csv or discover_paysim_csv(args.data_dir)
        result = preprocess_dataset(
            csv_path, args.processed_dir, args.report_dir
        )
    except (DataValidationError, FileNotFoundError, OSError) as exc:
        parser.exit(1, f"Could not preprocess PaySim data: {exc}\n")
    print(
        f"Preprocessed {result['audit']['retained_rows']:,} transactions "
        f"with {len(result['feature_names'])} raw model features."
    )
    print(f"Records: {args.processed_dir.resolve() / 'processed_transactions.csv'}")
    print(f"Feature matrix: {args.processed_dir.resolve() / 'feature_matrix.csv'}")
    print(f"Labels: {args.processed_dir.resolve() / 'labels.csv'}")
    print(f"Graph preparation: {args.processed_dir.resolve() / 'graph_transactions.csv'}")
    print(f"Feature schema: {args.processed_dir.resolve() / 'feature_schema.json'}")
    print(f"Preprocessing report: {args.report_dir.resolve() / 'preprocessing_report.md'}")
    print(f"Class distribution: {result['audit']['class_distribution']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
