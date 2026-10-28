"""PaySim transaction multigraph construction and bounded account traversal."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import networkx as nx
import pandas as pd

from backend.app.ml_pipeline import DataValidationError


TraversalDirection = Literal["out", "in", "both"]
TraversalAlgorithm = Literal["bfs", "dfs"]

REQUIRED_TRANSACTION_COLUMNS = {
    "transaction_id",
    "step",
    "nameOrig",
    "nameDest",
    "type",
    "amount",
}
PREDICTION_COLUMNS = {
    "fraud_score",
    "predicted_isFraud",
    "classification_threshold",
}


@dataclass(frozen=True)
class TraversalResult:
    """Serializable account neighborhood and traversal statistics."""

    algorithm: TraversalAlgorithm
    start_node: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    traversal_order: list[str]
    depths: dict[str, int]
    statistics: dict[str, int | bool | str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "start_node": self.start_node,
            "nodes": self.nodes,
            "edges": self.edges,
            "traversal_order": self.traversal_order,
            "depths": self.depths,
            "statistics": self.statistics,
        }


def _clean_text(value: Any, name: str) -> str:
    if pd.isna(value):
        raise DataValidationError(f"{name} must not be missing.")
    result = str(value).strip()
    if not result:
        raise DataValidationError(f"{name} must not be blank.")
    return result


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be a finite number.") from exc
    if not isfinite(number) or (minimum is not None and number < minimum):
        raise DataValidationError(f"{name} must be finite and at least {minimum}.")
    return number


def _binary_value(value: Any, name: str) -> int:
    number = _finite_number(value, name)
    if number not in (0.0, 1.0):
        raise DataValidationError(f"{name} must be binary 0/1.")
    return int(number)


def _prediction_attributes(
    row: Mapping[str, Any], *, include_ground_truth: bool
) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    has_prediction = PREDICTION_COLUMNS.issubset(row.keys())
    if has_prediction:
        score = _finite_number(row["fraud_score"], "fraud_score", minimum=0.0)
        if score > 1.0:
            raise DataValidationError("fraud_score must be between 0 and 1.")
        threshold = _finite_number(
            row["classification_threshold"], "classification_threshold", minimum=0.0
        )
        if threshold > 1.0:
            raise DataValidationError(
                "classification_threshold must be between 0 and 1."
            )
        predicted = _binary_value(row["predicted_isFraud"], "predicted_isFraud")
        if predicted != int(score >= threshold):
            raise DataValidationError(
                "predicted_isFraud does not match fraud_score and classification_threshold."
            )
        attributes.update(
            {
                "predicted_fraud_score": score,
                "predicted_fraud": bool(predicted),
                "classification_threshold": threshold,
            }
        )
    elif any(column in row for column in PREDICTION_COLUMNS):
        raise DataValidationError(
            "Prediction fields must include fraud_score, predicted_isFraud, "
            "and classification_threshold together."
        )

    if include_ground_truth and "isFraud" in row:
        label = _binary_value(row["isFraud"], "Offline isFraud label")
        attributes["ground_truth_is_fraud"] = bool(label)
        attributes["ground_truth_scope"] = "offline_simulation_evaluation_only"
    return attributes


def build_transaction_graph(
    transactions: pd.DataFrame,
    *,
    include_ground_truth: bool = False,
    max_transactions: int = 100_000,
) -> nx.MultiDiGraph:
    """Build account nodes and one keyed directed edge per transaction.

    A source account points to its recipient. The transaction ID is the multiedge
    key, so repeated account pairs remain distinct events rather than overwriting.
    """
    missing = REQUIRED_TRANSACTION_COLUMNS - set(transactions.columns)
    if missing:
        raise DataValidationError(
            "Transaction graph input is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if max_transactions < 1:
        raise DataValidationError("max_transactions must be positive.")
    if len(transactions) > max_transactions:
        raise DataValidationError(
            f"Graph input has {len(transactions):,} transactions, exceeding "
            f"the explicit limit of {max_transactions:,}."
        )
    if transactions.empty:
        raise DataValidationError("Cannot build a transaction graph from empty input.")
    if transactions["transaction_id"].duplicated().any():
        raise DataValidationError("transaction_id values must be unique.")

    graph = nx.MultiDiGraph(
        graph_type="PaySim transaction multigraph",
        node_type="account",
        edge_type="transaction",
        direction="sender_to_recipient",
        transaction_count=0,
    )
    columns = list(transactions.columns)
    seen_transaction_ids: set[str] = set()
    for values in transactions.itertuples(index=False, name=None):
        row = dict(zip(columns, values))
        transaction_id = _clean_text(row["transaction_id"], "transaction_id")
        if transaction_id in seen_transaction_ids:
            raise DataValidationError(
                f"Duplicate normalized transaction_id {transaction_id!r}."
            )
        seen_transaction_ids.add(transaction_id)
        sender = _clean_text(row["nameOrig"], "nameOrig")
        recipient = _clean_text(row["nameDest"], "nameDest")
        transaction_type = _clean_text(row["type"], "type")
        step_value = _finite_number(row["step"], "step", minimum=0.0)
        amount = _finite_number(row["amount"], "amount", minimum=0.0)
        if not step_value.is_integer():
            raise DataValidationError("step must be an integer simulation time.")
        attributes: dict[str, Any] = {
            "transaction_id": transaction_id,
            "step": int(step_value),
            "amount": amount,
            "type": transaction_type,
        }
        if "source_row_index" in row and not pd.isna(row["source_row_index"]):
            attributes["source_row_index"] = int(row["source_row_index"])
        attributes.update(_prediction_attributes(row, include_ground_truth=include_ground_truth))
        graph.add_node(sender, node_type="account", account_id=sender)
        graph.add_node(recipient, node_type="account", account_id=recipient)
        graph.add_edge(sender, recipient, key=transaction_id, **attributes)
    graph.graph["transaction_count"] = graph.number_of_edges()
    return graph


def load_account_neighborhood_csv(
    csv_path: Path,
    start_account: str,
    maximum_depth: int,
    *,
    direction: TraversalDirection = "out",
    maximum_transactions: int = 50_000,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Scan a scored transaction CSV and retain only a bounded account neighborhood.

    The CSV is rescanned once per requested hop, avoiding loading the full dataset
    or its complete graph into memory. If the neighborhood exceeds the cap, this
    fails explicitly instead of returning a silently incomplete graph.
    """
    path = Path(csv_path)
    start = _clean_text(start_account, "start_account")
    if not path.is_file():
        raise FileNotFoundError(f"Scored transaction CSV was not found: {path}")
    if not isinstance(maximum_depth, int) or maximum_depth < 1:
        raise DataValidationError("maximum_depth must be at least one for file loading.")
    if direction not in ("out", "in", "both"):
        raise DataValidationError("direction must be 'out', 'in', or 'both'.")
    if maximum_transactions < 1 or chunk_size < 1:
        raise DataValidationError("maximum_transactions and chunk_size must be positive.")

    columns = list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    required = REQUIRED_TRANSACTION_COLUMNS | PREDICTION_COLUMNS
    missing = required - set(columns)
    if missing:
        raise DataValidationError(
            "Scored transaction CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )
    optional_columns = {"source_row_index", "isFraud"}
    read_columns = [
        column for column in columns if column in required | optional_columns
    ]
    dtype = {
        column: "string"
        for column in ("transaction_id", "nameOrig", "nameDest", "type")
    }

    discovered = {start}
    frontier = {start}
    seen_ids: set[str] = set()
    selected_rows: list[dict[str, Any]] = []
    for _ in range(maximum_depth):
        if not frontier:
            break
        next_frontier: set[str] = set()
        for chunk in pd.read_csv(
            path,
            usecols=read_columns,
            dtype=dtype,
            chunksize=chunk_size,
            low_memory=False,
            encoding="utf-8-sig",
        ):
            sender_match = (
                chunk["nameOrig"].astype("string").isin(frontier)
                if direction in ("out", "both")
                else pd.Series(False, index=chunk.index)
            )
            recipient_match = (
                chunk["nameDest"].astype("string").isin(frontier)
                if direction in ("in", "both")
                else pd.Series(False, index=chunk.index)
            )
            matched = chunk.loc[sender_match | recipient_match]
            for values in matched.itertuples(index=False, name=None):
                row = dict(zip(read_columns, values))
                transaction_id = _clean_text(
                    row["transaction_id"], "transaction_id"
                )
                if transaction_id in seen_ids:
                    continue
                if len(selected_rows) >= maximum_transactions:
                    raise DataValidationError(
                        f"Account neighborhood exceeds the {maximum_transactions:,} "
                        "transaction limit. Increase the explicit limit or narrow "
                        "the scored input period; no partial graph was returned."
                    )
                seen_ids.add(transaction_id)
                selected_rows.append(row)
                sender = _clean_text(row["nameOrig"], "nameOrig")
                recipient = _clean_text(row["nameDest"], "nameDest")
                neighbor = (
                    recipient
                    if sender in frontier and direction in ("out", "both")
                    else sender
                )
                if neighbor not in discovered:
                    next_frontier.add(neighbor)
                    discovered.add(neighbor)
        frontier = next_frontier
    if not selected_rows:
        raise DataValidationError(
            f"No scored transactions were found within depth {maximum_depth} "
            f"from account {start!r} in {path}."
        )
    return pd.DataFrame.from_records(selected_rows, columns=read_columns)


def filter_transaction_edges(
    graph: nx.MultiDiGraph,
    *,
    minimum_score: float | None = None,
    predicted_class: int | None = None,
) -> list[dict[str, Any]]:
    """Return transaction edge records matching prediction-based filters."""
    if not graph.is_directed() or not graph.is_multigraph():
        raise DataValidationError("Expected a directed multigraph.")
    if minimum_score is not None:
        minimum_score = _finite_number(minimum_score, "minimum_score", minimum=0.0)
        if minimum_score > 1.0:
            raise DataValidationError("minimum_score must be between 0 and 1.")
    if predicted_class not in (None, 0, 1):
        raise DataValidationError("predicted_class must be 0, 1, or None.")

    matches = []
    for source, target, key, attributes in graph.edges(keys=True, data=True):
        score = attributes.get("predicted_fraud_score")
        predicted = attributes.get("predicted_fraud")
        if minimum_score is not None and (
            score is None or float(score) < minimum_score
        ):
            continue
        if predicted_class is not None and (
            predicted is None or int(bool(predicted)) != predicted_class
        ):
            continue
        matches.append(
            {
                "source": source,
                "target": target,
                "transaction_id": str(key),
                "attributes": dict(attributes),
            }
        )
    return matches


def _adjacent_edges(
    graph: nx.MultiDiGraph,
    node: str,
    direction: TraversalDirection,
) -> list[tuple[str, str, str, dict[str, Any], str]]:
    edges: list[tuple[str, str, str, dict[str, Any], str]] = []
    if direction in ("out", "both"):
        edges.extend(
            (source, target, str(key), data, target)
            for source, target, key, data in graph.out_edges(
                node, keys=True, data=True
            )
        )
    if direction in ("in", "both"):
        edges.extend(
            (source, target, str(key), data, source)
            for source, target, key, data in graph.in_edges(
                node, keys=True, data=True
            )
        )
    edges.sort(
        key=lambda edge: (
            int(edge[3].get("step", 0)),
            edge[2],
            edge[1],
            edge[0],
        )
    )
    return edges


def _make_result(
    graph: nx.MultiDiGraph,
    algorithm: TraversalAlgorithm,
    start: str,
    traversal_order: list[str],
    depths: dict[str, int],
    edge_records: list[dict[str, Any]],
    *,
    maximum_depth: int,
    maximum_nodes: int,
    maximum_edges: int,
    edges_examined: int,
    truncated: bool,
) -> TraversalResult:
    nodes = [
        {
            "id": node,
            "node_type": "account",
            "attributes": dict(graph.nodes[node]),
            "depth": depths[node],
        }
        for node in traversal_order
    ]
    return TraversalResult(
        algorithm=algorithm,
        start_node=start,
        nodes=nodes,
        edges=edge_records,
        traversal_order=traversal_order,
        depths=depths,
        statistics={
            "visited_accounts": len(traversal_order),
            "transactions_returned": len(edge_records),
            "transactions_examined": edges_examined,
            "maximum_depth": maximum_depth,
            "maximum_depth_reached": max(depths.values(), default=0),
            "maximum_nodes": maximum_nodes,
            "maximum_edges": maximum_edges,
            "truncated": truncated,
        },
    )


def _validate_traversal(
    graph: nx.MultiDiGraph,
    start_node: str,
    maximum_depth: int,
    direction: TraversalDirection,
    maximum_nodes: int,
    maximum_edges: int,
) -> str:
    if not graph.is_directed() or not graph.is_multigraph():
        raise DataValidationError("Traversal requires a directed transaction multigraph.")
    start = _clean_text(start_node, "start_node")
    if start not in graph:
        raise DataValidationError(f"Starting account {start!r} is not in this graph.")
    if not isinstance(maximum_depth, int) or maximum_depth < 0:
        raise DataValidationError("maximum_depth must be a non-negative integer.")
    if direction not in ("out", "in", "both"):
        raise DataValidationError("direction must be 'out', 'in', or 'both'.")
    if maximum_nodes < 1 or maximum_edges < 1:
        raise DataValidationError("maximum_nodes and maximum_edges must be positive.")
    return start


def breadth_first_search(
    graph: nx.MultiDiGraph,
    start_node: str,
    maximum_depth: int,
    *,
    direction: TraversalDirection = "out",
    maximum_nodes: int = 10_000,
    maximum_edges: int = 50_000,
) -> TraversalResult:
    """Explore account neighbors level by level and retain connecting transactions."""
    start = _validate_traversal(
        graph,
        start_node,
        maximum_depth,
        direction,
        maximum_nodes,
        maximum_edges,
    )
    order = [start]
    depths = {start: 0}
    queue = deque([start])
    edges: list[dict[str, Any]] = []
    seen_edge_ids: set[str] = set()
    examined = 0
    truncated = False

    while queue and not truncated:
        current = queue.popleft()
        depth = depths[current]
        if depth >= maximum_depth:
            continue
        for source, target, transaction_id, attributes, neighbor in _adjacent_edges(
            graph, current, direction
        ):
            examined += 1
            if neighbor not in depths and len(order) >= maximum_nodes:
                truncated = True
                break
            if transaction_id not in seen_edge_ids:
                if len(edges) >= maximum_edges:
                    truncated = True
                    break
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "transaction_id": transaction_id,
                        "attributes": dict(attributes),
                    }
                )
                seen_edge_ids.add(transaction_id)
            if neighbor not in depths:
                depths[neighbor] = depth + 1
                order.append(neighbor)
                queue.append(neighbor)

    return _make_result(
        graph,
        "bfs",
        start,
        order,
        depths,
        edges,
        maximum_depth=maximum_depth,
        maximum_nodes=maximum_nodes,
        maximum_edges=maximum_edges,
        edges_examined=examined,
        truncated=truncated,
    )


def depth_first_search(
    graph: nx.MultiDiGraph,
    start_node: str,
    maximum_depth: int,
    *,
    direction: TraversalDirection = "out",
    maximum_nodes: int = 10_000,
    maximum_edges: int = 50_000,
) -> TraversalResult:
    """Explore one account path deeply before backtracking, without recursion."""
    start = _validate_traversal(
        graph,
        start_node,
        maximum_depth,
        direction,
        maximum_nodes,
        maximum_edges,
    )
    order = [start]
    depths = {start: 0}
    edges: list[dict[str, Any]] = []
    seen_edge_ids: set[str] = set()
    examined = 0
    truncated = False
    stack: list[
        tuple[str, int, Sequence[tuple[str, str, str, dict[str, Any], str]], int]
    ] = []

    def push(node: str, depth: int) -> None:
        stack.append((node, depth, _adjacent_edges(graph, node, direction), 0))

    if maximum_depth > 0:
        push(start, 0)
    while stack and not truncated:
        current, depth, adjacent, position = stack[-1]
        if position >= len(adjacent):
            stack.pop()
            continue
        stack[-1] = (current, depth, adjacent, position + 1)
        source, target, transaction_id, attributes, neighbor = adjacent[position]
        examined += 1
        if neighbor not in depths and len(order) >= maximum_nodes:
            truncated = True
            break
        if transaction_id not in seen_edge_ids:
            if len(edges) >= maximum_edges:
                truncated = True
                break
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "transaction_id": transaction_id,
                    "attributes": dict(attributes),
                }
            )
            seen_edge_ids.add(transaction_id)
        if neighbor not in depths:
            neighbor_depth = depth + 1
            depths[neighbor] = neighbor_depth
            order.append(neighbor)
            if neighbor_depth < maximum_depth:
                push(neighbor, neighbor_depth)

    return _make_result(
        graph,
        "dfs",
        start,
        order,
        depths,
        edges,
        maximum_depth=maximum_depth,
        maximum_nodes=maximum_nodes,
        maximum_edges=maximum_edges,
        edges_examined=examined,
        truncated=truncated,
    )


def traverse(
    graph: nx.MultiDiGraph,
    start_node: str,
    maximum_depth: int,
    *,
    algorithm: TraversalAlgorithm = "bfs",
    direction: TraversalDirection = "out",
    maximum_nodes: int = 10_000,
    maximum_edges: int = 50_000,
) -> TraversalResult:
    """Select BFS or DFS without coupling traversal to classifier decisions."""
    if algorithm == "bfs":
        return breadth_first_search(
            graph,
            start_node,
            maximum_depth,
            direction=direction,
            maximum_nodes=maximum_nodes,
            maximum_edges=maximum_edges,
        )
    if algorithm == "dfs":
        return depth_first_search(
            graph,
            start_node,
            maximum_depth,
            direction=direction,
            maximum_nodes=maximum_nodes,
            maximum_edges=maximum_edges,
        )
    raise DataValidationError("algorithm must be 'bfs' or 'dfs'.")


def verify_prediction_schema(metadata: Mapping[str, Any]) -> float:
    """Validate PaySim model metadata and return its classification threshold."""
    if metadata.get("dataset_identifier") != "ealaxi/paysim1":
        raise DataValidationError("Prediction metadata is not for the PaySim dataset.")
    if metadata.get("model_type") != "sklearn.ensemble.RandomForestClassifier":
        raise DataValidationError("Prediction metadata does not describe the PaySim classifier.")
    if metadata.get("feature_names") != [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "type",
    ]:
        raise DataValidationError("Prediction metadata feature schema is incompatible.")
    threshold = metadata.get("threshold", {}).get("value")
    return _finite_number(threshold, "metadata threshold", minimum=0.0)
