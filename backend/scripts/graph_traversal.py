"""Run bounded BFS/DFS over scored PaySim test transactions for one account."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml_pipeline import (
    DataValidationError,
    load_json,
    load_model_bundle,
)
from backend.app.transaction_graph import (
    TraversalAlgorithm,
    TraversalDirection,
    build_transaction_graph,
    load_account_neighborhood_csv,
    traverse,
    verify_prediction_schema,
)


def investigate_account(
    start_account: str,
    *,
    algorithm: TraversalAlgorithm,
    maximum_depth: int,
    direction: TraversalDirection,
    minimum_score: float | None,
    maximum_transactions: int,
    maximum_nodes: int,
    maximum_edges: int,
    include_ground_truth: bool,
    scored_path: Path,
    model_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    metadata = load_json(metadata_path)
    threshold = verify_prediction_schema(metadata)
    bundle = load_model_bundle(model_path)
    if bundle.get("classification_threshold") != threshold:
        raise DataValidationError(
            "Saved model threshold does not match model metadata."
        )

    transactions = load_account_neighborhood_csv(
        scored_path,
        start_account,
        maximum_depth,
        direction=direction,
        maximum_transactions=maximum_transactions,
    )
    if not transactions["classification_threshold"].astype(float).eq(threshold).all():
        raise DataValidationError(
            "Scored graph rows use a different classification threshold from the saved model."
        )
    graph = build_transaction_graph(
        transactions,
        include_ground_truth=include_ground_truth,
        max_transactions=maximum_transactions,
    )
    result = traverse(
        graph,
        start_account,
        maximum_depth,
        algorithm=algorithm,
        direction=direction,
        maximum_nodes=maximum_nodes,
        maximum_edges=maximum_edges,
    )
    filtered_edges = [
        edge
        for edge in result.edges
        if (
            minimum_score is None
            or edge["attributes"].get("predicted_fraud_score", -1.0)
            >= minimum_score
        )
    ]
    return {
        "dataset_identifier": metadata["dataset_identifier"],
        "model_type": metadata["model_type"],
        "score_interpretation": metadata["score_convention"],
        "classification_threshold": threshold,
        "graph_scope": {
            "source": str(scored_path),
            "population": "held-out chronological test-period transactions with saved-model predictions",
            "neighborhood_transactions_loaded": int(len(transactions)),
            "graph_nodes": int(graph.number_of_nodes()),
            "graph_transactions": int(graph.number_of_edges()),
            "max_transactions": maximum_transactions,
            "ground_truth_included": include_ground_truth,
            "ground_truth_warning": (
                "Ground-truth labels are included only because this is offline "
                "evaluation data; do not expose them in inference-time investigation."
                if include_ground_truth
                else "Ground-truth labels are omitted from this investigation result."
            ),
        },
        "traversal": result.to_dict(),
        "prediction_filter": {
            "minimum_score": minimum_score,
            "transactions_matching_filter": len(filtered_edges),
            "transactions": filtered_edges,
        },
        "account_classification": (
            "No account is labeled fraudulent solely because it is connected to a "
            "transaction predicted as fraud."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Investigate a bounded account neighborhood with BFS or DFS."
    )
    parser.add_argument("--start-account", required=True)
    parser.add_argument("--algorithm", choices=("bfs", "dfs"), default="bfs")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--direction", choices=("out", "in", "both"), default="out")
    parser.add_argument("--minimum-score", type=float)
    parser.add_argument("--max-transactions", type=int, default=50_000)
    parser.add_argument("--max-nodes", type=int, default=10_000)
    parser.add_argument("--max-edges", type=int, default=50_000)
    parser.add_argument(
        "--include-ground-truth",
        action="store_true",
        help="Include the offline simulation label for evaluation only; omitted by default.",
    )
    parser.add_argument(
        "--scored-csv",
        type=Path,
        default=PROJECT_ROOT
        / "backend"
        / "data"
        / "processed"
        / "graph_test_transactions.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "backend" / "models" / "fraud_classifier.joblib",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "backend" / "models" / "model_metadata.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "backend" / "reports" / "graph_investigation.json",
    )
    args = parser.parse_args()
    if args.max_depth < 1:
        parser.error("--max-depth must be at least one.")
    if args.minimum_score is not None and not 0.0 <= args.minimum_score <= 1.0:
        parser.error("--minimum-score must be between 0 and 1.")
    try:
        result = investigate_account(
            args.start_account,
            algorithm=args.algorithm,
            maximum_depth=args.max_depth,
            direction=args.direction,
            minimum_score=args.minimum_score,
            maximum_transactions=args.max_transactions,
            maximum_nodes=args.max_nodes,
            maximum_edges=args.max_edges,
            include_ground_truth=args.include_ground_truth,
            scored_path=args.scored_csv,
            model_path=args.model,
            metadata_path=args.metadata,
        )
    except (DataValidationError, FileNotFoundError, OSError, ValueError) as exc:
        parser.exit(1, f"Could not run graph investigation: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    stats = result["traversal"]["statistics"]
    print(
        f"{args.algorithm.upper()} from {args.start_account}: "
        f"{stats['visited_accounts']} accounts, "
        f"{stats['transactions_returned']} transactions, "
        f"{result['prediction_filter']['transactions_matching_filter']} above filter."
    )
    print(f"Investigation saved: {args.output.resolve()}")
    print(result["account_classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
