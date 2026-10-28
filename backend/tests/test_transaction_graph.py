"""Graph tests use tiny artificial PaySim-like transaction rows."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import networkx as nx
import pandas as pd

from backend.app.ml_pipeline import DataValidationError
from backend.app.transaction_graph import (
    breadth_first_search,
    build_transaction_graph,
    depth_first_search,
    filter_transaction_edges,
    load_account_neighborhood_csv,
)


def graph_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transaction_id": "t1",
                "step": 1,
                "nameOrig": "A",
                "nameDest": "B",
                "type": "TRANSFER",
                "amount": 10.0,
                "fraud_score": 0.1,
                "predicted_isFraud": 0,
                "classification_threshold": 0.5,
                "isFraud": 0,
            },
            {
                "transaction_id": "t2",
                "step": 2,
                "nameOrig": "A",
                "nameDest": "B",
                "type": "CASH_OUT",
                "amount": 20.0,
                "fraud_score": 0.8,
                "predicted_isFraud": 1,
                "classification_threshold": 0.5,
                "isFraud": 1,
            },
            {
                "transaction_id": "t3",
                "step": 3,
                "nameOrig": "B",
                "nameDest": "C",
                "type": "TRANSFER",
                "amount": 30.0,
                "fraud_score": 0.7,
                "predicted_isFraud": 1,
                "classification_threshold": 0.5,
                "isFraud": 0,
            },
            {
                "transaction_id": "t4",
                "step": 4,
                "nameOrig": "A",
                "nameDest": "D",
                "type": "PAYMENT",
                "amount": 40.0,
                "fraud_score": 0.2,
                "predicted_isFraud": 0,
                "classification_threshold": 0.5,
                "isFraud": 0,
            },
            {
                "transaction_id": "t5",
                "step": 5,
                "nameOrig": "C",
                "nameDest": "A",
                "type": "TRANSFER",
                "amount": 50.0,
                "fraud_score": 0.9,
                "predicted_isFraud": 1,
                "classification_threshold": 0.5,
                "isFraud": 1,
            },
            {
                "transaction_id": "t6",
                "step": 6,
                "nameOrig": "D",
                "nameDest": "E",
                "type": "CASH_OUT",
                "amount": 60.0,
                "fraud_score": 0.6,
                "predicted_isFraud": 1,
                "classification_threshold": 0.5,
                "isFraud": 0,
            },
            {
                "transaction_id": "t7",
                "step": 7,
                "nameOrig": "F",
                "nameDest": "G",
                "type": "PAYMENT",
                "amount": 70.0,
                "fraud_score": 0.1,
                "predicted_isFraud": 0,
                "classification_threshold": 0.5,
                "isFraud": 0,
            },
        ]
    )


class GraphConstructionTests(unittest.TestCase):
    def test_builds_directed_multigraph_and_preserves_parallel_transactions(self) -> None:
        graph = build_transaction_graph(graph_fixture())
        self.assertIsInstance(graph, nx.MultiDiGraph)
        self.assertTrue(graph.is_directed())
        self.assertEqual(graph.number_of_edges(), 7)
        self.assertEqual(graph.number_of_edges("A", "B"), 2)
        self.assertEqual(graph["A"]["B"]["t2"]["amount"], 20.0)
        self.assertTrue(graph["A"]["B"]["t2"]["predicted_fraud"])
        self.assertNotIn("ground_truth_is_fraud", graph["A"]["B"]["t2"])
        self.assertNotIn("isFraud", graph["A"]["B"]["t2"])

    def test_ground_truth_is_opt_in_and_explicitly_offline(self) -> None:
        graph = build_transaction_graph(
            graph_fixture(), include_ground_truth=True
        )
        attributes = graph["A"]["B"]["t2"]
        self.assertTrue(attributes["ground_truth_is_fraud"])
        self.assertEqual(
            attributes["ground_truth_scope"],
            "offline_simulation_evaluation_only",
        )
        self.assertNotIn("fraudulent", graph.nodes["A"])

    def test_prediction_filter_uses_score_and_predicted_class_only(self) -> None:
        graph = build_transaction_graph(graph_fixture())
        selected = filter_transaction_edges(
            graph, minimum_score=0.7, predicted_class=1
        )
        self.assertEqual(
            [edge["transaction_id"] for edge in selected], ["t2", "t3", "t5"]
        )
        self.assertTrue(
            all(
                "ground_truth_is_fraud" not in edge["attributes"]
                for edge in selected
            )
        )

    def test_rejects_missing_columns_duplicate_ids_and_invalid_scores(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "missing required"):
            build_transaction_graph(pd.DataFrame([{"transaction_id": "x"}]))
        duplicate = pd.concat(
            [graph_fixture().iloc[[0]], graph_fixture().iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaisesRegex(DataValidationError, "unique"):
            build_transaction_graph(duplicate)
        invalid = graph_fixture().iloc[[0]].copy()
        invalid.loc[:, "predicted_isFraud"] = 1
        with self.assertRaisesRegex(DataValidationError, "does not match"):
            build_transaction_graph(invalid)
        fractional = graph_fixture().iloc[[0]].copy()
        fractional["predicted_isFraud"] = fractional[
            "predicted_isFraud"
        ].astype(float)
        fractional.loc[:, "predicted_isFraud"] = 0.5
        with self.assertRaisesRegex(DataValidationError, "binary"):
            build_transaction_graph(fractional)

    def test_hard_transaction_limit_prevents_unbounded_graph_creation(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "exceeding"):
            build_transaction_graph(graph_fixture(), max_transactions=2)


class TraversalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_transaction_graph(graph_fixture())

    def test_bfs_traverses_level_by_level_and_keeps_parallel_edges(self) -> None:
        result = breadth_first_search(self.graph, "A", maximum_depth=2)
        self.assertEqual(result.traversal_order, ["A", "B", "D", "C", "E"])
        self.assertEqual(result.depths, {"A": 0, "B": 1, "D": 1, "C": 2, "E": 2})
        self.assertEqual(
            {edge["transaction_id"] for edge in result.edges},
            {"t1", "t2", "t3", "t4", "t6"},
        )
        self.assertEqual(result.statistics["visited_accounts"], 5)
        self.assertEqual(result.statistics["transactions_returned"], 5)

    def test_dfs_follows_a_path_before_backtracking(self) -> None:
        result = depth_first_search(self.graph, "A", maximum_depth=3)
        self.assertEqual(result.traversal_order, ["A", "B", "C", "D", "E"])
        self.assertIn("t5", {edge["transaction_id"] for edge in result.edges})

    def test_depth_zero_and_one_are_respected(self) -> None:
        zero = breadth_first_search(self.graph, "A", maximum_depth=0)
        one = breadth_first_search(self.graph, "A", maximum_depth=1)
        self.assertEqual(zero.traversal_order, ["A"])
        self.assertEqual(zero.edges, [])
        self.assertEqual(set(one.traversal_order), {"A", "B", "D"})
        self.assertEqual(
            {edge["transaction_id"] for edge in one.edges}, {"t1", "t2", "t4"}
        )

    def test_cycles_and_disconnected_components_are_safe(self) -> None:
        result = breadth_first_search(self.graph, "A", maximum_depth=20)
        self.assertEqual(len(result.traversal_order), 5)
        self.assertEqual(len(set(result.traversal_order)), 5)
        self.assertNotIn("F", result.traversal_order)
        self.assertNotIn("G", result.traversal_order)
        self.assertIn("t5", {edge["transaction_id"] for edge in result.edges})

    def test_incoming_and_bidirectional_traversal(self) -> None:
        incoming = breadth_first_search(
            self.graph, "C", maximum_depth=1, direction="in"
        )
        both = breadth_first_search(
            self.graph, "C", maximum_depth=1, direction="both"
        )
        self.assertEqual(incoming.traversal_order, ["C", "B"])
        self.assertIn("t3", {edge["transaction_id"] for edge in incoming.edges})
        self.assertEqual(set(both.traversal_order), {"C", "B", "A"})
        both_from_sender = breadth_first_search(
            self.graph, "A", maximum_depth=2, direction="both"
        )
        edge_ids = [edge["transaction_id"] for edge in both_from_sender.edges]
        self.assertEqual(len(edge_ids), len(set(edge_ids)))

    def test_invalid_start_depth_and_caps_are_explicit(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "not in this graph"):
            breadth_first_search(self.graph, "missing", maximum_depth=1)
        with self.assertRaisesRegex(DataValidationError, "non-negative"):
            depth_first_search(self.graph, "A", maximum_depth=-1)
        limited = breadth_first_search(
            self.graph, "A", maximum_depth=3, maximum_nodes=2
        )
        self.assertTrue(limited.statistics["truncated"])
        self.assertLessEqual(len(limited.nodes), 2)


class NeighborhoodLoadingTests(unittest.TestCase):
    def test_chunked_loader_returns_bounded_scored_neighborhood(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scored.csv"
            graph_fixture().to_csv(path, index=False)
            rows = load_account_neighborhood_csv(
                path, "A", maximum_depth=2, chunk_size=2
            )
            self.assertEqual(
                set(rows["transaction_id"]), {"t1", "t2", "t3", "t4", "t6"}
            )
            with self.assertRaisesRegex(DataValidationError, "exceeds the"):
                load_account_neighborhood_csv(
                    path,
                    "A",
                    maximum_depth=2,
                    maximum_transactions=2,
                    chunk_size=2,
                )

    def test_missing_start_neighborhood_fails_without_fabricated_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scored.csv"
            graph_fixture().to_csv(path, index=False)
            with self.assertRaisesRegex(DataValidationError, "No scored transactions"):
                load_account_neighborhood_csv(
                    path, "unknown", maximum_depth=2, chunk_size=2
                )


if __name__ == "__main__":
    unittest.main()
