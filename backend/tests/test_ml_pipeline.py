"""PaySim pipeline tests use small, explicitly synthetic transactions only."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml_pipeline import (
    DATASET_IDENTIFIER,
    FEATURE_COLUMNS,
    MODEL_TYPE,
    TARGET_COLUMN,
    DataValidationError,
    build_training_pipeline,
    classification_metrics,
    discover_paysim_csv,
    load_model_bundle,
    preprocess_dataset,
    sample_training_rows,
    save_model_bundle,
    select_f1_threshold,
    validate_paysim_columns,
)


def synthetic_transactions(step_count: int = 30) -> pd.DataFrame:
    rows = []
    for step in range(1, step_count + 1):
        for sequence in range(2):
            fraud = (step + sequence) % 2
            rows.append(
                {
                    "step": step,
                    "type": "TRANSFER" if fraud else "CASH_OUT",
                    "amount": 1000.0 + 500.0 * fraud + sequence,
                    "nameOrig": f"sender-{step % 4}",
                    "oldbalanceOrg": 5000.0,
                    "newbalanceOrig": 4000.0 - 500.0 * fraud,
                    "nameDest": f"recipient-{step % 3}",
                    "oldbalanceDest": 100.0,
                    "newbalanceDest": 100.0 + 500.0 * fraud,
                    "isFraud": fraud,
                    "isFlaggedFraud": 0,
                }
            )
    return pd.DataFrame(rows)


def write_fixture(path: Path, *, step_count: int = 30) -> pd.DataFrame:
    frame = synthetic_transactions(step_count)
    frame.to_csv(path, index=False)
    return frame


class DatasetDiscoveryTests(unittest.TestCase):
    def test_discovers_pay_sim_csv_by_verified_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "olist.csv").write_text("order_id\n1\n", encoding="utf-8")
            source = root / "paysim.csv"
            write_fixture(source)

            self.assertEqual(discover_paysim_csv(root), source)

    def test_missing_and_ambiguous_pay_sim_files_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                discover_paysim_csv(root)
            write_fixture(root / "one.csv")
            write_fixture(root / "two.csv")
            with self.assertRaisesRegex(DataValidationError, "Multiple CSV"):
                discover_paysim_csv(root)

    def test_schema_validation_lists_missing_required_columns(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "isFraud"):
            validate_paysim_columns(["step", "type"])


class PreprocessingTests(unittest.TestCase):
    def test_one_row_per_transaction_and_label_identifier_separation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "paysim.csv"
            original = write_fixture(source)
            before = source.read_bytes()
            processed = root / "processed"
            reports = root / "reports"

            schema = preprocess_dataset(source, processed, reports, chunk_size=7)

            features = pd.read_csv(processed / "feature_matrix.csv")
            labels = pd.read_csv(processed / "labels.csv")
            records = pd.read_csv(processed / "processed_transactions.csv")
            graph = pd.read_csv(processed / "graph_transactions.csv")
            self.assertEqual(len(features), len(original))
            self.assertEqual(len(labels), len(original))
            self.assertEqual(len(records), len(original))
            self.assertEqual(len(graph), len(original))
            self.assertEqual(features.columns.tolist(), list(FEATURE_COLUMNS))
            self.assertNotIn(TARGET_COLUMN, features.columns)
            self.assertNotIn("nameOrig", features.columns)
            self.assertNotIn("nameDest", features.columns)
            self.assertNotIn("isFlaggedFraud", features.columns)
            self.assertIn("isFraud", labels.columns)
            self.assertEqual(graph["transaction_id"].nunique(), len(original))
            repeated_pair = graph.groupby(["nameOrig", "nameDest"]).size().max()
            self.assertGreater(repeated_pair, 1)
            self.assertEqual(schema["dataset_identifier"], DATASET_IDENTIFIER)
            self.assertEqual(schema["audit"]["retained_rows"], len(original))
            self.assertEqual(before, source.read_bytes())
            self.assertTrue((reports / "preprocessing_report.md").is_file())

    def test_invalid_core_rows_are_removed_and_invalid_balances_are_left_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "paysim.csv"
            frame = synthetic_transactions()
            frame.loc[0, "amount"] = -1
            frame.loc[1, "isFraud"] = 2
            frame.loc[2, "nameOrig"] = None
            frame.loc[3, "oldbalanceOrg"] = -10
            frame.loc[4, "type"] = None
            frame.to_csv(source, index=False)

            schema = preprocess_dataset(
                source, root / "processed", root / "reports", chunk_size=11
            )
            features = pd.read_csv(root / "processed" / "feature_matrix.csv")

            self.assertEqual(schema["audit"]["dropped_invalid_core_rows"], 3)
            self.assertEqual(
                schema["audit"]["missing_or_invalid_balance_values"]["oldbalanceOrg"],
                1,
            )
            self.assertEqual(schema["audit"]["missing_transaction_type_rows"], 1)
            self.assertTrue(features["oldbalanceOrg"].isna().any())
            self.assertIn("__missing__", set(features["type"]))

    def test_same_step_is_never_split_across_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "paysim.csv"
            write_fixture(source, step_count=20)
            preprocess_dataset(source, root / "processed", root / "reports")
            records = pd.read_csv(root / "processed" / "processed_transactions.csv")
            self.assertTrue(
                records.groupby("step")["split"].nunique().eq(1).all()
            )

    def test_empty_or_invalid_training_data_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "missing.csv"
            with self.assertRaises(FileNotFoundError):
                preprocess_dataset(source, root / "processed", root / "reports")

            frame = synthetic_transactions()
            frame["amount"] = -1
            invalid_source = root / "invalid.csv"
            frame.to_csv(invalid_source, index=False)
            with self.assertRaisesRegex(DataValidationError, "No valid"):
                preprocess_dataset(
                    invalid_source, root / "invalid_processed", root / "invalid_reports"
                )


class ModelTests(unittest.TestCase):
    def test_train_fitted_pipeline_scores_and_uses_feature_schema(self) -> None:
        frame = synthetic_transactions()
        features = frame.loc[:, list(FEATURE_COLUMNS)]
        labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
        model = build_training_pipeline(len(features), n_estimators=8)
        model.fit(features, labels)
        scores = model.predict_proba(features)[:, 1]

        self.assertEqual(len(scores), len(labels))
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[-1], 0.0)
        self.assertTrue(
            set(model.named_steps["classifier"].classes_) == {0, 1}
        )

    def test_validation_threshold_and_metrics_are_consistent(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 1])
        scores = np.asarray([0.05, 0.2, 0.75, 0.9, 0.4, 0.6])
        threshold = select_f1_threshold(labels, scores)
        result = classification_metrics(labels, scores, threshold)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertEqual(
            result["flagged_records"], int((scores >= threshold).sum())
        )
        self.assertEqual(result["fraud_support"], 3)
        self.assertIn("true_positive", result["confusion_matrix"])
        self.assertIsNotNone(result["average_precision"])
        self.assertIsNotNone(result["roc_auc"])

    def test_class_stratified_sampling_is_reproducible_and_keeps_both_classes(self) -> None:
        labels = np.asarray([0] * 90 + [1] * 10)
        indices = np.arange(len(labels))
        first = sample_training_rows(
            labels, indices, max_rows=30, random_state=42
        )
        second = sample_training_rows(
            labels, indices, max_rows=30, random_state=42
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 30)
        self.assertEqual(set(labels[first]), {0, 1})

    def test_invalid_threshold_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "both classes"):
            select_f1_threshold([0, 0], [0.2, 0.3])
        with self.assertRaisesRegex(DataValidationError, "finite"):
            classification_metrics([0, 1], [0.1, np.inf], 0.5)

    def test_model_bundle_round_trip_and_dataset_guard(self) -> None:
        frame = synthetic_transactions()
        features = frame.loc[:, list(FEATURE_COLUMNS)]
        labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
        model = build_training_pipeline(len(features), n_estimators=4)
        model.fit(features, labels)
        bundle = {
            "dataset_identifier": DATASET_IDENTIFIER,
            "model_type": MODEL_TYPE,
            "feature_names": list(FEATURE_COLUMNS),
            "model": model,
            "classification_threshold": 0.5,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fraud_classifier.joblib"
            save_model_bundle(bundle, path)
            loaded = load_model_bundle(path)
            np.testing.assert_array_equal(
                model.predict_proba(features)[:, 1],
                loaded["model"].predict_proba(features)[:, 1],
            )
            wrong_dataset = dict(bundle, dataset_identifier="olistbr/brazilian-ecommerce")
            save_model_bundle(wrong_dataset, path)
            with self.assertRaisesRegex(DataValidationError, "dataset identifier"):
                load_model_bundle(path)


class InspectionTests(unittest.TestCase):
    def test_inspection_writes_actual_fixture_statistics_without_modifying_input(self) -> None:
        from backend.scripts.inspect_dataset import inspect_dataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "paysim.csv"
            frame = synthetic_transactions(step_count=6)
            frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
            frame.to_csv(source, index=False)
            before = source.read_bytes()

            report = inspect_dataset(root, root / "reports")

            self.assertEqual(report["file"]["rows"], len(frame))
            self.assertEqual(report["file"]["duplicate_rows"], 1)
            self.assertEqual(report["file"]["fraud_label_counts"]["0"], 6)
            self.assertTrue((root / "reports" / "dataset_inspection.md").is_file())
            self.assertTrue((root / "reports" / "dataset_analysis.md").is_file())
            self.assertEqual(source.read_bytes(), before)


class EndToEndScriptTests(unittest.TestCase):
    def test_train_reload_and_held_out_evaluation(self) -> None:
        from backend.scripts.evaluate_model import evaluate
        from backend.scripts.train_model import train

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "paysim.csv"
            write_fixture(source, step_count=30)
            processed = root / "processed"
            models = root / "models"
            reports = root / "reports"
            preprocess_dataset(source, processed, reports, chunk_size=13)

            training = train(
                processed,
                models,
                reports,
                max_training_rows=200,
                n_estimators=4,
            )
            evaluation = evaluate(processed, models, reports)

            self.assertEqual(training["dataset_rows"], 60)
            self.assertTrue(training["round_trip_prediction_match"])
            self.assertEqual(evaluation["test_metrics"]["records"], 8)
            self.assertTrue(evaluation["raw_score_reproducibility"])
            self.assertEqual(evaluation["graph_records"], 8)
            self.assertTrue(
                (processed / "scored_test_transactions.csv").is_file()
            )
            self.assertTrue(
                (processed / "graph_test_transactions.csv").is_file()
            )
            self.assertTrue((reports / "model_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
