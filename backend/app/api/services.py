"""Artifact loading, transaction index, model inference, and graph services."""

from __future__ import annotations

import json
import logging
import pickle
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Generator

import networkx as nx
import pandas as pd

from backend.app.api import config
from backend.app.ml_pipeline import (
    DATASET_IDENTIFIER,
    FEATURE_COLUMNS,
    DataValidationError,
    load_json,
    load_model_bundle,
)
from backend.app.transaction_graph import build_transaction_graph, traverse

LOGGER = logging.getLogger(__name__)
TRANSACTION_COLUMNS = (
    "transaction_id",
    "source_row_index",
    "step",
    "nameOrig",
    "nameDest",
    "type",
    "amount",
    "fraud_score",
    "predicted_isFraud",
    "classification_threshold",
)
SQL_COLUMNS = ", ".join(f'"{column}"' for column in TRANSACTION_COLUMNS)


class ApiServiceError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class FraudGuardService:
    """Loads trusted artifacts once and builds a reusable local SQLite index."""

    def __init__(
        self,
        *,
        scored_path: Path = config.SCORED_TRANSACTIONS_PATH,
        index_path: Path = config.API_INDEX_PATH,
        model_path: Path = config.MODEL_PATH,
        metadata_path: Path = config.MODEL_METADATA_PATH,
        inspection_path: Path = config.INSPECTION_REPORT_PATH,
        evaluation_path: Path = config.EVALUATION_REPORT_PATH,
    ) -> None:
        self.scored_path = scored_path
        self.index_path = index_path
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.inspection_path = inspection_path
        self.evaluation_path = evaluation_path
        self.model_bundle: dict[str, Any] | None = None
        self.metadata: dict[str, Any] | None = None
        self.inspection: dict[str, Any] | None = None
        self.evaluation: dict[str, Any] | None = None
        self.failures: dict[str, str] = {}

    def initialize(self) -> None:
        self._load_model()
        self._load_reports()
        try:
            self._ensure_index()
        except (OSError, sqlite3.Error, ValueError, KeyError, pd.errors.ParserError) as exc:
            self.failures["dataset"] = str(exc)
            LOGGER.exception("Transaction index initialization failed")

    def _load_model(self) -> None:
        try:
            self.metadata = load_json(self.metadata_path)
            if self.metadata.get("dataset_identifier") != DATASET_IDENTIFIER:
                raise DataValidationError("Model metadata does not identify PaySim.")
            if self.metadata.get("feature_names") != list(FEATURE_COLUMNS):
                raise DataValidationError("Model metadata feature schema is incompatible.")
            self.model_bundle = load_model_bundle(self.model_path)
            loaded_model = self.model_bundle.get("model")
            if not callable(getattr(loaded_model, "predict_proba", None)):
                raise DataValidationError("Saved model does not support score inference.")
            if list(getattr(loaded_model, "classes_", [])) != [0, 1]:
                raise DataValidationError("Saved model does not expose both PaySim classes.")
            if self.model_bundle.get("classification_threshold") != self.metadata.get(
                "threshold", {}
            ).get("value"):
                raise DataValidationError("Saved model threshold does not match metadata.")
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            EOFError,
            ImportError,
            pickle.UnpicklingError,
        ) as exc:
            self.model_bundle = None
            self.failures["model"] = str(exc)
            LOGGER.exception("Model initialization failed")

    def _load_reports(self) -> None:
        for component, path, attribute in (
            ("dataset_report", self.inspection_path, "inspection"),
            ("evaluation_report", self.evaluation_path, "evaluation"),
        ):
            try:
                payload = load_json(path)
                if payload.get("dataset_identifier") != DATASET_IDENTIFIER:
                    raise DataValidationError(f"{component} does not identify PaySim.")
                setattr(self, attribute, payload)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.failures[component] = str(exc)
                LOGGER.exception("%s initialization failed", component)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        if not self.index_path.is_file():
            raise ApiServiceError(503, "dataset_unavailable", "Transaction index is unavailable.")
        connection = sqlite3.connect(self.index_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _ensure_index(self) -> None:
        if not self.scored_path.is_file():
            raise FileNotFoundError("Scored held-out PaySim transaction file is missing.")
        stat = self.scored_path.stat()
        source_signature = f"{stat.st_size}:{stat.st_mtime_ns}"
        if self.index_path.is_file():
            try:
                with closing(sqlite3.connect(self.index_path)) as connection:
                    connection.row_factory = sqlite3.Row
                    row = connection.execute(
                        "SELECT value FROM metadata WHERE key='source_signature'"
                    ).fetchone()
                    if row and row[0] == source_signature:
                        connection.execute("SELECT COUNT(*) FROM transactions").fetchone()
                        connection.execute("SELECT COUNT(*) FROM accounts").fetchone()
                        self._ensure_dashboard_summary(connection)
                        connection.commit()
                        return
            except sqlite3.Error:
                LOGGER.warning("Existing transaction index is invalid; rebuilding it.")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.index_path.with_name(self.index_path.name + ".building")
        temporary_path.unlink(missing_ok=True)
        try:
            self._build_index(temporary_path, source_signature)
            temporary_path.replace(self.index_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _build_index(self, destination: Path, source_signature: str) -> None:
        required = set(TRANSACTION_COLUMNS)
        headers = set(pd.read_csv(self.scored_path, nrows=0, encoding="utf-8-sig").columns)
        if not required.issubset(headers):
            raise DataValidationError(
                "Scored transaction file is missing required prediction or graph columns."
            )
        dtypes = {
            "transaction_id": "string",
            "nameOrig": "string",
            "nameDest": "string",
            "type": "string",
        }
        with closing(sqlite3.connect(destination)) as connection, connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                """CREATE TABLE transactions (
                    transaction_id TEXT PRIMARY KEY,
                    source_row_index INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    nameOrig TEXT NOT NULL,
                    nameDest TEXT NOT NULL,
                    type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    fraud_score REAL NOT NULL,
                    predicted_isFraud INTEGER NOT NULL,
                    classification_threshold REAL NOT NULL
                )"""
            )
            connection.execute("CREATE TABLE accounts (account_id TEXT PRIMARY KEY)")
            insert_transaction = (
                "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            for chunk in pd.read_csv(
                self.scored_path,
                usecols=list(TRANSACTION_COLUMNS),
                dtype=dtypes,
                chunksize=config.INDEX_CHUNK_SIZE,
                low_memory=False,
                encoding="utf-8-sig",
            ):
                chunk = chunk.loc[:, list(TRANSACTION_COLUMNS)]
                if chunk["transaction_id"].isna().any() or chunk["transaction_id"].duplicated().any():
                    raise DataValidationError("Transaction IDs must be present and unique.")
                if not chunk["predicted_isFraud"].isin((0, 1)).all():
                    raise DataValidationError("Predicted classes must be binary.")
                if not chunk["fraud_score"].between(0, 1).all():
                    raise DataValidationError("Prediction scores must be between 0 and 1.")
                rows = list(chunk.itertuples(index=False, name=None))
                connection.executemany(insert_transaction, rows)
                accounts = set(chunk["nameOrig"].dropna()).union(
                    chunk["nameDest"].dropna()
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO accounts VALUES (?)",
                    ((str(account),) for account in accounts),
                )
            connection.execute("CREATE INDEX transaction_sender_idx ON transactions(nameOrig)")
            connection.execute("CREATE INDEX transaction_recipient_idx ON transactions(nameDest)")
            connection.execute("CREATE INDEX transaction_type_idx ON transactions(type)")
            connection.execute("CREATE INDEX transaction_prediction_idx ON transactions(predicted_isFraud)")
            connection.execute("CREATE INDEX transaction_amount_idx ON transactions(amount)")
            connection.execute("CREATE INDEX transaction_source_row_idx ON transactions(source_row_index)")
            connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO metadata VALUES ('source_signature', ?)",
                (source_signature,),
            )
            connection.execute("ANALYZE")
            self._ensure_dashboard_summary(connection)

    @staticmethod
    def _ensure_dashboard_summary(connection: sqlite3.Connection) -> None:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS dashboard_summary (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                payload TEXT NOT NULL
            )"""
        )
        cached = connection.execute(
            "SELECT 1 FROM dashboard_summary WHERE id = 1"
        ).fetchone()
        if cached:
            return

        stats = connection.execute(
            """SELECT COUNT(*) AS transactions,
                      SUM(predicted_isFraud) AS predicted_candidates
               FROM transactions"""
        ).fetchone()
        account_count = connection.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
        transaction_types = connection.execute(
            """SELECT type, COUNT(*) AS count
               FROM transactions GROUP BY type ORDER BY count DESC, type"""
        ).fetchall()
        amount_ranges = connection.execute(
            """SELECT CASE
                      WHEN amount < 1000 THEN '< 1K'
                      WHEN amount < 10000 THEN '1K-10K'
                      WHEN amount < 100000 THEN '10K-100K'
                      WHEN amount < 1000000 THEN '100K-1M'
                      ELSE '1M+'
                    END AS range,
                    COUNT(*) AS count
               FROM transactions
               GROUP BY range
               ORDER BY MIN(amount)"""
        ).fetchall()
        payload = {
            "transactions": stats["transactions"],
            "accounts_represented": account_count,
            "model_predicted_candidates": stats["predicted_candidates"] or 0,
            "transaction_type_distribution": [
                {"type": row["type"], "count": row["count"]}
                for row in transaction_types
            ],
            "amount_distribution": [
                {"range": row["range"], "count": row["count"]}
                for row in amount_ranges
            ],
        }
        connection.execute(
            "INSERT INTO dashboard_summary (id, payload) VALUES (1, ?)",
            (json.dumps(payload),),
        )

    def health(self) -> dict[str, Any]:
        dataset_ready = self._index_ready()
        model_ready = self.model_bundle is not None and self.metadata is not None
        report_ready = self.inspection is not None and self.evaluation is not None
        components = {
            "dataset": dataset_ready,
            "model": model_ready,
            "graph": dataset_ready,
            "reports": report_ready,
            "artifacts_loaded": dataset_ready and model_ready and report_ready,
        }
        return {
            "status": "ready" if all(components.values()) else "degraded",
            "components": components,
            "errors": {
                component: "Unavailable or failed validation."
                for component in self.failures
            },
        }

    def _index_ready(self) -> bool:
        try:
            return self.index_path.is_file() and self.index_path.stat().st_size > 0
        except OSError:
            return False

    def require(self, component: str) -> None:
        if component == "model" and self.model_bundle is None:
            raise ApiServiceError(503, "model_unavailable", "Saved model is unavailable.")
        if component == "dataset" and not self._index_ready():
            raise ApiServiceError(503, "dataset_unavailable", "Transaction index is unavailable.")

    @staticmethod
    def _record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "transaction_id": row["transaction_id"],
            "source_row_index": row["source_row_index"],
            "step": row["step"],
            "sender": row["nameOrig"],
            "recipient": row["nameDest"],
            "type": row["type"],
            "amount": row["amount"],
            "prediction": {
                "predicted_class": int(row["predicted_isFraud"]),
                "score": row["fraud_score"],
                "threshold": row["classification_threshold"],
                "score_interpretation": "uncalibrated positive-class model score",
            },
        }

    def list_transactions(
        self,
        *,
        page: int,
        page_size: int,
        transaction_id: str | None,
        transaction_type: str | None,
        predicted: bool | None,
        minimum_amount: float | None,
        maximum_amount: float | None,
    ) -> dict[str, Any]:
        self.require("dataset")
        conditions: list[str] = []
        parameters: list[Any] = []
        if transaction_id:
            conditions.append("transaction_id = ?")
            parameters.append(transaction_id)
        if transaction_type:
            conditions.append("type = ?")
            parameters.append(transaction_type)
        if predicted is not None:
            conditions.append("predicted_isFraud = ?")
            parameters.append(int(predicted))
        if minimum_amount is not None:
            conditions.append("amount >= ?")
            parameters.append(minimum_amount)
        if maximum_amount is not None:
            conditions.append("amount <= ?")
            parameters.append(maximum_amount)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        offset = (page - 1) * page_size
        with self._connection() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM transactions{where}", parameters
            ).fetchone()[0]
            rows = connection.execute(
                f"SELECT {SQL_COLUMNS} FROM transactions{where} "
                "ORDER BY source_row_index DESC LIMIT ? OFFSET ?",
                (*parameters, page_size, offset),
            ).fetchall()
        return {
            "items": [self._record(row) for row in rows],
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_next": offset + len(rows) < total,
        }

    def transaction(self, transaction_id: str) -> dict[str, Any]:
        self.require("dataset")
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT {SQL_COLUMNS} FROM transactions WHERE transaction_id = ?",
                (transaction_id,),
            ).fetchone()
        if row is None:
            raise ApiServiceError(404, "transaction_not_found", "Transaction was not found.")
        return self._record(row)

    def dataset_summary(self) -> dict[str, Any]:
        self.require("dataset")
        report = self.inspection
        if report is None:
            raise ApiServiceError(503, "report_unavailable", "Dataset inspection report is unavailable.")
        file_report = report.get("file", {})
        return {
            "dataset_identifier": DATASET_IDENTIFIER,
            "dataset_kind": "simulated mobile-money transactions",
            "rows": file_report.get("rows"),
            "columns_count": file_report.get("columns_count"),
            "columns": file_report.get("columns"),
            "transaction_types": [
                value
                for value in file_report.get("columns", [])
                if value.get("name") == "type"
                for value, _ in value.get("unique_values", [])
            ],
            "ground_truth_labels": {
                "column": "isFraud",
                "counts": next(
                    (
                        column.get("unique_values")
                        for column in file_report.get("columns", [])
                        if column.get("name") == "isFraud"
                    ),
                    None,
                ),
                "meaning": "PaySim simulation labels; not independently adjudicated real-world fraud.",
            },
            "limitations": (self.metadata or {}).get("dataset_limitations", []),
            "api_transaction_scope": "Scored chronological held-out test split only.",
        }

    def dashboard_metrics(self) -> dict[str, Any]:
        self.require("dataset")
        with self._connection() as connection:
            cached = connection.execute(
                "SELECT payload FROM dashboard_summary WHERE id = 1"
            ).fetchone()
        if cached is None:
            raise ApiServiceError(503, "dataset_summary_unavailable", "Dataset metrics are not prepared.")
        stats = json.loads(cached["payload"])
        evaluation = self.evaluation
        return {
            "scope": "scored chronological held-out test split",
            "transactions_available": stats["transactions"],
            "accounts_represented": stats["accounts_represented"],
            "model_predicted_candidates": stats["model_predicted_candidates"],
            "evaluation_metrics": evaluation.get("test_metrics") if evaluation else None,
            "transaction_type_distribution": stats["transaction_type_distribution"],
            "amount_distribution": stats["amount_distribution"],
            "ground_truth_positive_count": (self.metadata or {}).get(
                "class_distribution", {}
            ).get("test_support_only", {}).get("1"),
            "ground_truth_is_not_prediction_count": True,
        }

    def model_summary(self) -> dict[str, Any]:
        self.require("model")
        assert self.metadata is not None
        assert self.model_bundle is not None
        return {
            "model_type": self.metadata["model_type"],
            "dataset_identifier": DATASET_IDENTIFIER,
            "feature_schema": self.metadata["feature_names"],
            "training_configuration": self.metadata.get("parameters"),
            "split_strategy": self.metadata.get("split_strategy"),
            "evaluation_metrics": (
                self.evaluation.get("test_metrics") if self.evaluation else None
            ),
            "classification_threshold": self.model_bundle["classification_threshold"],
            "threshold_selection": self.metadata.get("threshold", {}).get("selection"),
            "score_interpretation": self.metadata.get("score_convention"),
            "limitations": self.metadata.get("dataset_limitations"),
        }

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        self.require("model")
        assert self.model_bundle is not None
        model = self.model_bundle["model"]
        normalized = dict(features)
        for column in (
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ):
            if normalized[column] is not None and normalized[column] < 0:
                normalized[column] = None
        row = pd.DataFrame([{name: normalized[name] for name in FEATURE_COLUMNS}])
        try:
            scores = model.predict_proba(row)[0]
            classes = list(model.classes_)
            positive_index = classes.index(1)
            score = float(scores[positive_index])
        except (AttributeError, IndexError, ValueError, TypeError) as exc:
            raise ApiServiceError(503, "model_inference_failed", "The saved model could not score this request.") from exc
        threshold = float(self.model_bundle["classification_threshold"])
        return {
            "prediction_type": "hypothetical",
            "predicted_class": int(score >= threshold),
            "score": score,
            "threshold": threshold,
            "score_interpretation": "uncalibrated positive-class model score, not a calibrated probability",
            "ground_truth_used": False,
            "graph_relationship_created": False,
            "prediction_point": (self.metadata or {}).get("prediction_point"),
        }

    def _select_neighborhood(
        self,
        start_node: str,
        maximum_depth: int,
        direction: str,
        maximum_transactions: int,
    ) -> pd.DataFrame:
        self.require("dataset")
        if maximum_depth > 6:
            raise ApiServiceError(422, "depth_limit_exceeded", "Maximum graph depth is 6.")
        selected: dict[str, tuple[Any, ...]] = {}
        discovered = {start_node}
        frontier = {start_node}
        with self._connection() as connection:
            account = connection.execute(
                "SELECT 1 FROM accounts WHERE account_id = ?", (start_node,)
            ).fetchone()
            if account is None:
                raise ApiServiceError(404, "account_not_found", "Account was not found in the indexed test split.")
            for _ in range(maximum_depth):
                if not frontier:
                    break
                new_frontier: set[str] = set()
                frontier_list = list(frontier)
                for start in range(0, len(frontier_list), 300):
                    batch = frontier_list[start : start + 300]
                    placeholders = ",".join("?" for _ in batch)
                    clauses: list[str] = []
                    params: list[str] = []
                    if direction in ("out", "both"):
                        clauses.append(f"nameOrig IN ({placeholders})")
                        params.extend(batch)
                    if direction in ("in", "both"):
                        clauses.append(f"nameDest IN ({placeholders})")
                        params.extend(batch)
                    query = (
                        f"SELECT {SQL_COLUMNS} FROM transactions WHERE "
                        + " OR ".join(clauses)
                    )
                    for row in connection.execute(query, params):
                        record = self._record_tuple(row)
                        transaction_id = str(record[0])
                        if transaction_id not in selected:
                            if len(selected) >= maximum_transactions:
                                raise ApiServiceError(
                                    413,
                                    "graph_limit_exceeded",
                                    f"Neighborhood exceeds {maximum_transactions} transactions; no partial graph was returned.",
                                )
                            selected[transaction_id] = record
                        sender = str(record[3])
                        recipient = str(record[4])
                        neighbors: set[str] = set()
                        if direction in ("out", "both") and sender in frontier:
                            neighbors.add(recipient)
                        if direction in ("in", "both") and recipient in frontier:
                            neighbors.add(sender)
                        for neighbor in neighbors - discovered:
                            discovered.add(neighbor)
                            new_frontier.add(neighbor)
                frontier = new_frontier
        return pd.DataFrame.from_records(
            list(selected.values()), columns=TRANSACTION_COLUMNS
        )

    @staticmethod
    def _record_tuple(row: sqlite3.Row) -> tuple[Any, ...]:
        return tuple(row[column] for column in TRANSACTION_COLUMNS)

    def graph_data(
        self,
        *,
        limit: int,
        transaction_type: str | None,
        predicted: bool | None,
        minimum_amount: float | None,
        maximum_amount: float | None,
    ) -> dict[str, Any]:
        self.require("dataset")
        conditions: list[str] = []
        params: list[Any] = []
        if transaction_type:
            conditions.append("type = ?")
            params.append(transaction_type)
        if predicted is not None:
            conditions.append("predicted_isFraud = ?")
            params.append(int(predicted))
        if minimum_amount is not None:
            conditions.append("amount >= ?")
            params.append(minimum_amount)
        if maximum_amount is not None:
            conditions.append("amount <= ?")
            params.append(maximum_amount)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT {SQL_COLUMNS} FROM transactions{where} "
                "ORDER BY source_row_index DESC LIMIT ?",
                (*params, limit + 1),
            ).fetchall()
        truncated = len(rows) > limit
        graph = self._build_graph(rows[:limit])
        return self._serialize_graph(graph, truncated=truncated)

    def account_graph(
        self,
        node_id: str,
        *,
        direction: str,
        maximum_transactions: int,
    ) -> dict[str, Any]:
        transactions = self._select_neighborhood(
            node_id, 1, direction, maximum_transactions
        )
        if transactions.empty:
            return {"account_id": node_id, "nodes": [{"id": node_id}], "edges": [], "counts": {"nodes": 1, "edges": 0}}
        graph = build_transaction_graph(
            transactions, max_transactions=maximum_transactions
        )
        return {
            "account_id": node_id,
            **self._serialize_graph(graph, truncated=False),
        }

    def graph_traversal(
        self,
        *,
        start_node: str,
        algorithm: str,
        maximum_depth: int,
        direction: str,
        maximum_nodes: int,
        maximum_edges: int,
    ) -> dict[str, Any]:
        transaction_limit = min(maximum_edges, config.MAX_GRAPH_TRANSACTIONS)
        transactions = self._select_neighborhood(
            start_node, maximum_depth, direction, transaction_limit
        )
        graph = build_transaction_graph(
            transactions,
            max_transactions=transaction_limit,
        ) if not transactions.empty else nx.MultiDiGraph()
        if not graph.has_node(start_node):
            graph.add_node(start_node, node_type="account", account_id=start_node)
        result = traverse(
            graph,
            start_node,
            maximum_depth,
            algorithm=algorithm,  # validated Literal request field
            direction=direction,  # validated Literal request field
            maximum_nodes=maximum_nodes,
            maximum_edges=maximum_edges,
        )
        return {
            **result.to_dict(),
            "interpretation": "Traversal reveals connectivity; it does not classify fraud.",
            "ground_truth_labels_included": False,
        }

    @staticmethod
    def _build_graph(rows: list[sqlite3.Row]) -> nx.MultiDiGraph:
        frame = pd.DataFrame.from_records(
            [FraudGuardService._record_tuple(row) for row in rows],
            columns=TRANSACTION_COLUMNS,
        )
        if frame.empty:
            return nx.MultiDiGraph()
        return build_transaction_graph(
            frame,
            max_transactions=config.MAX_GRAPH_TRANSACTIONS,
        )

    @staticmethod
    def _serialize_graph(graph: nx.MultiDiGraph, *, truncated: bool) -> dict[str, Any]:
        nodes = [
            {"id": node_id, "node_type": "account"}
            for node_id in graph.nodes
        ]
        edges = [
            {
                "id": str(key),
                "source": source,
                "target": target,
                "transaction_id": attributes["transaction_id"],
                "step": attributes["step"],
                "amount": attributes["amount"],
                "type": attributes["type"],
                "prediction": {
                    "predicted_class": int(attributes["predicted_fraud"]),
                    "score": attributes["predicted_fraud_score"],
                    "threshold": attributes["classification_threshold"],
                },
            }
            for source, target, key, attributes in graph.edges(keys=True, data=True)
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "counts": {"nodes": len(nodes), "edges": len(edges)},
            "truncated": truncated,
        }
