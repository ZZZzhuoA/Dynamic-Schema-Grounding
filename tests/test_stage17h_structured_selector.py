import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SELECTOR = load_module(
    "stage17h_run_structured_selector",
    "src/evaluation/stage17h_run_structured_selector.py",
)


def graph_record():
    nodes = [
        {"id": 10, "type": "table", "name": "orders"},
        {
            "id": 11,
            "type": "column",
            "name": "orders.total",
            "table": "orders",
            "column": "total",
        },
        {
            "id": 12,
            "type": "column",
            "name": "orders.date",
            "table": "orders",
            "column": "date",
        },
        {"id": 20, "type": "table", "name": "customers"},
        {
            "id": 21,
            "type": "column",
            "name": "customers.name",
            "table": "customers",
            "column": "name",
        },
        {
            "id": 22,
            "type": "column",
            "name": "customers.city",
            "table": "customers",
            "column": "city",
        },
    ]
    return {
        "inference_inputs": {
            "db_id": "shop",
            "question": "What are the order totals?",
            "schema_nodes": nodes,
            "schema_edges": [
                {"src": 10, "dst": 10, "type": "self_loop"},
                {"src": 10, "dst": 11, "type": "table_to_column"},
                {"src": 11, "dst": 10, "type": "column_to_table"},
                {"src": 10, "dst": 12, "type": "table_to_column"},
                {"src": 12, "dst": 10, "type": "column_to_table"},
                {"src": 20, "dst": 21, "type": "table_to_column"},
                {"src": 21, "dst": 20, "type": "column_to_table"},
                {"src": 20, "dst": 22, "type": "table_to_column"},
                {"src": 22, "dst": 20, "type": "column_to_table"},
                {"src": 11, "dst": 21, "type": "synthetic_path"},
            ],
        },
        "metadata": {"record_index": 0, "question_id": 7},
    }


def label_record():
    graph = graph_record()
    return {
        "db_id": "shop",
        "question_id": 7,
        "question": "What are the order totals?",
        "schema_items": graph["inference_inputs"]["schema_nodes"],
        "whole_sql_labels": [10, 11],
    }


def prediction_row():
    scores = {10: 0.1, 11: 4.0, 12: 3.0, 20: 0.0, 21: 3.5, 22: 2.5}
    nodes = graph_record()["inference_inputs"]["schema_nodes"]
    ordered = sorted(nodes, key=lambda node: (-scores[node["id"]], node["id"]))
    return {
        "record_index": 0,
        "db_id": "shop",
        "question_id": 7,
        "schema_node_count": len(nodes),
        "ranked_schema": [
            {
                "schema_item_id": node["id"],
                "name": node["name"],
                "type": node["type"],
                "logit": scores[node["id"]],
                "probability": 0.5,
                "rank": rank,
            }
            for rank, node in enumerate(ordered, start=1)
        ],
    }


def args(top_k=2):
    return SimpleNamespace(
        top_k=top_k,
        max_tables=8,
        min_tables=-1,
        connectivity_weight=0.1,
        baseline_retention_weight=0.05,
        direct_edge_types=SELECTOR.DEFAULT_DIRECT_EDGE_TYPES,
    )


class Stage17HStructuredSelectorTest(unittest.TestCase):
    def aligned_example(self):
        examples, _ = SELECTOR.align_graphs_and_labels(
            [graph_record()], [label_record()], "dev"
        )
        return examples[0]

    def test_adapter_maps_owners_and_excludes_non_direct_edges(self):
        example = self.aligned_example()
        adapted = SELECTOR.stage17_selector_example(
            example, [11, 21], SELECTOR.DEFAULT_DIRECT_EDGE_TYPES
        )
        self.assertEqual(adapted["candidate_nodes"][1]["owner_table_id"], 10)
        self.assertEqual(adapted["candidate_nodes"][1]["owner_local_id"], 0)
        self.assertEqual(len(adapted["schema_edges"]), 8)
        self.assertNotIn(
            "synthetic_path", {edge["type"] for edge in adapted["schema_edges"]}
        )
        self.assertNotIn("self_loop", {edge["type"] for edge in adapted["schema_edges"]})

    def test_selector_is_deterministic_owner_closed_and_exact_budget(self):
        example = self.aligned_example()
        first = SELECTOR.select_one(example, prediction_row(), args(top_k=2))
        second = SELECTOR.select_one(example, prediction_row(), args(top_k=2))
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[2], [11, 10])
        self.assertEqual(first[3]["selected_count"], 2)
        self.assertTrue(first[3]["owner_closed"])

    def test_selector_recovers_owner_complete_gold_set(self):
        example = self.aligned_example()
        output, raw_ids, structured_ids, _ = SELECTOR.select_one(
            example, prediction_row(), args(top_k=2)
        )
        self.assertEqual(raw_ids, [11, 21])
        self.assertEqual(set(structured_ids), {10, 11})
        raw_metrics = SELECTOR.selection_metrics([example], {0: raw_ids}, 2)
        structured_metrics = SELECTOR.selection_metrics(
            [example], {0: structured_ids}, 2
        )
        self.assertEqual(raw_metrics["complete_coverage@2"], 0.0)
        self.assertEqual(structured_metrics["complete_coverage@2"], 1.0)
        self.assertEqual(output["top_10_ids"][:2], structured_ids)

    def test_prediction_validation_rejects_leakage_and_identity_mismatch(self):
        example = self.aligned_example()
        leaked = {**prediction_row(), "gold_ids": [10, 11]}
        with self.assertRaisesRegex(ValueError, "Gold/label fields"):
            SELECTOR.prediction_rows_by_record([leaked], "leaked")
        mismatched = prediction_row()
        mismatched["ranked_schema"][0]["name"] = "wrong.name"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            SELECTOR.select_one(example, mismatched, args())

    def test_run_outputs_leakage_free_predictions_and_separate_eval_cases(self):
        example = self.aligned_example()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_file = root / "dev_predictions.jsonl"
            prediction_file.write_text(
                json.dumps(prediction_row()) + "\n", encoding="utf-8"
            )
            summary = SELECTOR.run_selector(
                "42", prediction_file, [example], args(top_k=2), root / "out"
            )
            output_file = Path(summary["structured_prediction_file"])
            output = json.loads(output_file.read_text(encoding="utf-8").strip())
            SELECTOR.assert_leakage_free(output)
            self.assertNotIn("gold_ids", output)
            self.assertEqual(
                summary["paired_outcomes"]["recovered_complete_count"], 1
            )
            evaluation_file = Path(summary["evaluation_cases_file"])
            evaluation = json.loads(
                evaluation_file.read_text(encoding="utf-8").strip()
            )
            self.assertEqual(evaluation["outcome"], "recovered")
            self.assertIn("raw_missing_gold_ids", evaluation)

    def test_aggregate_requires_positive_complete_and_column_deltas(self):
        run = {
            "raw_metrics": {
                "schema_recall@2": 0.5,
                "schema_precision@2": 0.5,
                "complete_coverage@2": 0.0,
                "table_recall@2": 0.0,
                "column_recall@2": 1.0,
                "mrr": 1.0,
            },
            "structured_metrics": {
                "schema_recall@2": 1.0,
                "schema_precision@2": 1.0,
                "complete_coverage@2": 1.0,
                "table_recall@2": 1.0,
                "column_recall@2": 1.0,
                "mrr": 1.0,
            },
            "paired_outcomes": {
                "recovered_complete_count": 1,
                "regressed_complete_count": 0,
            },
            "prediction_leakage_free": True,
            "selector_statistics": {
                "all_owner_closed": True,
                "all_budgets_exact": True,
            },
        }
        aggregate = SELECTOR.aggregate_runs({"42": run}, 2)
        self.assertTrue(aggregate["decision_passed"])

    def test_cli_writes_root_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_file = root / "dev_graph.jsonl"
            label_file = root / "dev_label.jsonl"
            prediction_file = root / "dev_predictions.jsonl"
            output_dir = root / "output"
            graph_file.write_text(
                json.dumps(graph_record()) + "\n", encoding="utf-8"
            )
            label_file.write_text(
                json.dumps(label_record()) + "\n", encoding="utf-8"
            )
            prediction_file.write_text(
                json.dumps(prediction_row()) + "\n", encoding="utf-8"
            )
            argv = [
                "stage17h_run_structured_selector.py",
                "--prediction-file",
                f"42={prediction_file}",
                "--dev-graph-file",
                str(graph_file),
                "--dev-label-file",
                str(label_file),
                "--output-dir",
                str(output_dir),
                "--top-k",
                "2",
            ]
            with mock.patch("sys.argv", argv), redirect_stdout(StringIO()):
                SELECTOR.main()
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["stage"], "17-H")
            self.assertEqual(
                summary["aggregate"]["net_recovered_complete_count"], 1
            )
            self.assertIn("dev_graph_sha256", summary["config"])


if __name__ == "__main__":
    unittest.main()
