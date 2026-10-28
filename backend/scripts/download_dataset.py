"""Download and list the PaySim dataset using KaggleHub."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "backend" / "data" / "raw"
DATASET_HANDLE = "ealaxi/paysim1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the PaySim transaction dataset with KaggleHub."
    )
    parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Keep KaggleHub's versioned cache inside the ignored raw-data folder unless
    # the user has explicitly selected a different cache location.
    os.environ.setdefault("KAGGLEHUB_CACHE", str(RAW_DATA_DIR.resolve()))

    try:
        import kagglehub
    except ImportError:
        print(
            "KaggleHub is not installed. From the project root, run:\n"
            "  python -m pip install -r backend\\requirements.txt",
            file=sys.stderr,
        )
        return 1

    try:
        dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE)).resolve()
    except Exception as exc:
        print(
            f"Could not download {DATASET_HANDLE}: {exc}\n"
            "Check your internet connection and Kaggle access. If Kaggle requests "
            "authentication, create an API token in your Kaggle account and run "
            "`python -c \"import kagglehub; kagglehub.login()\"` in an interactive "
            "terminal, or configure KaggleHub credentials using its official "
            "instructions. Do not put credentials in this project.",
            file=sys.stderr,
        )
        return 1

    if not dataset_path.is_dir():
        print(
            f"KaggleHub returned a path that is not a directory: {dataset_path}",
            file=sys.stderr,
        )
        return 1

    csv_files = sorted(
        (
            path
            for path in dataset_path.rglob("*")
            if path.is_file() and path.suffix.lower() == ".csv"
        ),
        key=lambda path: str(path).lower(),
    )
    print(f"Dataset directory: {dataset_path}")
    if not csv_files:
        print("No CSV files were found in the downloaded dataset directory.", file=sys.stderr)
        return 2

    print(f"CSV files found: {len(csv_files)}")
    for csv_file in csv_files:
        print(f"  - {csv_file.relative_to(dataset_path)}")
    print("KaggleHub reuses its local cache on subsequent runs when the dataset is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
