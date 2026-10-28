"""Paths and bounded API configuration."""

from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
MODEL_PATH = BACKEND_ROOT / "models" / "fraud_classifier.joblib"
MODEL_METADATA_PATH = BACKEND_ROOT / "models" / "model_metadata.json"
INSPECTION_REPORT_PATH = BACKEND_ROOT / "reports" / "dataset_inspection.json"
EVALUATION_REPORT_PATH = BACKEND_ROOT / "reports" / "model_evaluation.json"
SCORED_TRANSACTIONS_PATH = (
    BACKEND_ROOT / "data" / "processed" / "graph_test_transactions.csv"
)
API_INDEX_PATH = BACKEND_ROOT / "data" / "processed" / "api_transactions.sqlite3"

MAX_PAGE_SIZE = 200
MAX_GRAPH_TRANSACTIONS = 5_000
INDEX_CHUNK_SIZE = 50_000
CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
