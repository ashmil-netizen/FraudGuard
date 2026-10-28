"""FastAPI tests use a tiny synthetic scored transaction CSV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.api import services
from backend.app.api.services import FraudGuardService
from backend.app.ml_pipeline import DATASET_IDENTIFIER, FEATURE_COLUMNS
from backend.app.api.main import create_app


class SmallClassifier:
    classes_ = [0, 1]

    def predict_proba(self, frame: pd.DataFrame) -> list[list[float]]:
        return [[0.2, 0.8] if frame.iloc[0]["amount"] > 100 else [0.8, 0.2]]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
    threshold = 0.5
    records = pd.DataFrame(
        [
            ("t1", 1, 379, "A", "B", "TRANSFER", 10.0, 0.1, 0, threshold),
            ("t2", 2, 380, "A", "B", "CASH_OUT", 200.0, 0.8, 1, threshold),
            ("t3", 3, 381, "B", "C", "PAYMENT", 30.0, 0.2, 0, threshold),
            ("t4", 4, 382, "C", "A", "TRANSFER", 40.0, 0.7, 1, threshold),
            ("t5", 5, 383, "D", "E", "DEBIT", 50.0, 0.1, 0, threshold),
        ],
        columns=services.TRANSACTION_COLUMNS,
    )
    scored_path = tmp_path / "scored.csv"
    records.to_csv(scored_path, index=False)
    metadata_path = tmp_path / "metadata.json"
    inspection_path = tmp_path / "inspection.json"
    evaluation_path = tmp_path / "evaluation.json"
    metadata = {
        "dataset_identifier": DATASET_IDENTIFIER,
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "feature_names": list(FEATURE_COLUMNS),
        "threshold": {"value": threshold},
        "parameters": {"n_estimators": 1},
        "split_strategy": {"name": "chronological"},
        "prediction_point": "post-transaction review",
        "score_convention": "uncalibrated score",
        "dataset_limitations": ["simulated data"],
        "class_distribution": {"test_support_only": {"1": 2}},
    }
    _write_json(metadata_path, metadata)
    _write_json(
        inspection_path,
        {
            "dataset_identifier": DATASET_IDENTIFIER,
            "file": {
                "rows": 10,
                "columns_count": 11,
                "columns": [
                    {"name": "type", "unique_values": [["TRANSFER", 4], ["CASH_OUT", 3]]},
                    {"name": "isFraud", "unique_values": [["0", 8], ["1", 2]]},
                ],
            },
        },
    )
    _write_json(
        evaluation_path,
        {
            "dataset_identifier": DATASET_IDENTIFIER,
            "test_metrics": {"precision": 0.9},
        },
    )
    def fake_load_model_bundle(path: Path) -> dict[str, Any]:
        assert path == tmp_path / "model.joblib"
        return {
            "dataset_identifier": DATASET_IDENTIFIER,
            "model_type": metadata["model_type"],
            "feature_names": list(FEATURE_COLUMNS),
            "classification_threshold": threshold,
            "model": SmallClassifier(),
        }

    monkeypatch.setattr(services, "load_model_bundle", fake_load_model_bundle)
    service = FraudGuardService(
        scored_path=scored_path,
        index_path=tmp_path / "index.sqlite3",
        model_path=tmp_path / "model.joblib",
        metadata_path=metadata_path,
        inspection_path=inspection_path,
        evaluation_path=evaluation_path,
    )
    app = create_app(service)
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_reports_are_ready(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["data"]["components"]["artifacts_loaded"] is True
    assert client.get("/api/dataset/summary").json()["data"]["dataset_identifier"] == DATASET_IDENTIFIER
    dashboard = client.get("/api/dashboard/metrics").json()["data"]
    assert dashboard["transactions_available"] == 5
    assert dashboard["model_predicted_candidates"] == 2
    assert dashboard["ground_truth_is_not_prediction_count"] is True
    assert sum(item["count"] for item in dashboard["transaction_type_distribution"]) == 5
    assert sum(item["count"] for item in dashboard["amount_distribution"]) == 5


def test_transaction_search_detail_and_pagination(client: TestClient) -> None:
    response = client.get("/api/transactions", params={"page_size": 2, "type": "TRANSFER"})
    assert response.status_code == 200
    assert response.json()["data"]["total"] == 2
    detail = client.get("/api/transactions/t2")
    assert detail.status_code == 200
    assert detail.json()["data"]["prediction"]["predicted_class"] == 1
    assert "ground_truth" not in detail.json()["data"]
    assert client.get("/api/transactions/unknown").status_code == 404
    assert client.get("/api/transactions?page_size=201").status_code == 422


def test_hypothetical_prediction_and_input_validation(client: TestClient) -> None:
    payload = {
        "step": 400,
        "amount": 200,
        "oldbalanceOrg": 250,
        "newbalanceOrig": 50,
        "oldbalanceDest": 10,
        "newbalanceDest": 210,
        "type": "TRANSFER",
    }
    result = client.post("/api/predict", json=payload)
    assert result.status_code == 200
    assert result.json()["data"]["prediction_type"] == "hypothetical"
    assert result.json()["data"]["predicted_class"] == 1
    assert result.json()["data"]["ground_truth_used"] is False
    assert client.post("/api/predict", json={**payload, "isFraud": 1}).status_code == 422
    assert client.post("/api/predict", json={**payload, "amount": -1}).status_code == 422
    infinite_amount = json.dumps(payload).replace('"amount": 200', '"amount": 1e999')
    assert (
        client.post(
            "/api/predict",
            content=infinite_amount,
            headers={"Content-Type": "application/json"},
        ).status_code
        == 422
    )


def test_graph_filter_parallel_edges_and_account_lookup(client: TestClient) -> None:
    graph = client.get("/api/graph", params={"limit": 4}).json()["data"]
    assert graph["counts"]["edges"] == 4
    assert len({edge["id"] for edge in graph["edges"]}) == 4
    assert graph["truncated"] is True
    account = client.get("/api/graph/node/A").json()["data"]
    assert account["counts"]["edges"] == 3
    assert client.get("/api/graph/node/missing").status_code == 404


@pytest.mark.parametrize("algorithm", ["bfs", "dfs"])
def test_graph_traversal_and_invalid_account(
    client: TestClient, algorithm: str
) -> None:
    response = client.post(
        "/api/graph/traverse",
        json={
            "start_node": "A",
            "algorithm": algorithm,
            "maximum_depth": 2,
            "direction": "out",
        },
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["traversal_order"] == ["A", "B", "C"]
    assert data["statistics"]["maximum_depth_reached"] == 2
    assert data["ground_truth_labels_included"] is False
    assert client.post(
        "/api/graph/traverse",
        json={"start_node": "unknown", "algorithm": "bfs", "maximum_depth": 1},
    ).status_code == 404


def test_model_summary_and_error_shapes(client: TestClient) -> None:
    assert client.get("/api/model/summary").json()["data"]["feature_schema"] == list(FEATURE_COLUMNS)
    validation = client.post("/api/graph/traverse", json={"algorithm": "bad"})
    assert validation.status_code == 422
    assert validation.json()["error"]["code"] == "validation_error"


def test_missing_artifacts_are_reported_as_unavailable(tmp_path: Path) -> None:
    service = FraudGuardService(
        scored_path=tmp_path / "missing-scored.csv",
        index_path=tmp_path / "missing-index.sqlite3",
        model_path=tmp_path / "missing-model.joblib",
        metadata_path=tmp_path / "missing-metadata.json",
        inspection_path=tmp_path / "missing-inspection.json",
        evaluation_path=tmp_path / "missing-evaluation.json",
    )
    with TestClient(create_app(service)) as test_client:
        health = test_client.get("/api/health")
        prediction = test_client.post(
            "/api/predict",
            json={
                "step": 400,
                "amount": 100,
                "oldbalanceOrg": 100,
                "newbalanceOrig": 0,
                "oldbalanceDest": 0,
                "newbalanceDest": 100,
                "type": "TRANSFER",
            },
        )
    assert health.status_code == 503
    assert health.json()["data"]["components"]["model"] is False
    assert "C:\\" not in json.dumps(health.json())
    assert prediction.status_code == 503
    assert prediction.json()["error"]["code"] == "model_unavailable"
