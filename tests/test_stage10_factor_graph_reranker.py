import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_module(
    "stage10_build_factor_graph_data",
    "src/data/stage10_build_factor_graph_data.py",
)
SELECTOR = load_module(
    "stage10_constrained_selector",
    "src/grounding/stage10_constrained_selector.py",
)
TRAINING = load_module(
    "stage10_train_factor_graph_reranker",
    "src/training/stage10_train_factor_graph_reranker.py",
)


def synthetic_graph():
    return {
        "example_id": "demo:0",
        "inference_inputs": {
            "db_id": "demo",
            "question": "schools in Alameda",
            "evidence": "Alameda is a county value",
            "schema_nodes": [
                {"id": 0, "type": "table", "name": "schools"},
                {"id": 1, "type": "column", "name": "schools.name", "table": "schools", "column": "name"},
                {"id": 2, "type": "column", "name": "schools.county", "table": "schools", "column": "county"},
                {"id": 3, "type": "table", "name": "districts"},
                {"id": 4, "type": "column", "name": "districts.hidden_gold", "table": "districts", "column": "hidden_gold"},
            ],
            "schema_edges": [
                {"src": 0, "dst": 1, "type": "table_to_column"},
                {"src": 1, "dst": 0, "type": "column_to_table"},
                {"src": 0, "dst": 2, "type": "table_to_column"},
                {"src": 2, "dst": 0, "type": "column_to_table"},
                {"src": 3, "dst": 4, "type": "table_to_column"},
            ],
        },
    }


class Stage10FactorGraphDataTest(unittest.TestCase):
    def test_builder_creates_evidence_factors_without_gold_injection(self):
        graph = synthetic_graph()
        aligned_item = {
            "record_index": 0,
            "graph_example": graph,
            "clause_record": {
                "db_id": "demo",
                "question_id": 0,
                "whole_sql_labels": [0, 2, 4],
                "relation_labels": {"PREDICATE_COLUMN": [2]},
            },
        }
        relation_rows = {
            "PREDICATE_COLUMN": {
                "record_index": 0,
                "clause": "PREDICATE_COLUMN",
                "top_30": [
                    {"id": 2, "score": 2.0},
                    {"id": 1, "score": 1.0},
                    {"id": 0, "score": 0.5},
                ],
            }
        }
        baseline = {
            "record_index": 0,
            "top_30": [{"id": 2, "score": 2.0}, {"id": 0, "score": 0.5}],
        }
        evidence = {
            "record_index": 0,
            "value_matches": [
                {
                    "schema_item_id": 2,
                    "value_anchor": "alameda",
                    "value_confidence": 0.8,
                    "raw_value_score": 1.0,
                    "value_ambiguity_score": 1.0,
                    "value_margin_score": 0.5,
                    "value_semantic_support": 0.7,
                    "eligible_for_injection": True,
                    "eligible_for_terminal": True,
                }
            ],
            "join_path_candidates": [],
            "join_path": {"paths": []},
        }
        args = SimpleNamespace(
            relation_top_m=20,
            max_candidates=4,
            value_priority_weight=1.0,
            join_priority_weight=0.8,
        )
        result = BUILDER.build_one_factor_graph(
            aligned_item,
            relation_rows,
            baseline,
            evidence,
            ["PREDICATE_COLUMN"],
            args,
        )
        candidate_ids = {node["schema_item_id"] for node in result["candidate_nodes"]}
        factor_kinds = {factor["kind"] for factor in result["factors"]}
        self.assertIn("relation", factor_kinds)
        self.assertIn("value", factor_kinds)
        self.assertNotIn(4, candidate_ids)
        self.assertEqual(result["missing_gold_ids"], [4])
        self.assertLess(result["candidate_oracle_recall"], 1.0)


class Stage10OOFFilteringTest(unittest.TestCase):
    def test_record_index_manifest_filters_without_reindexing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indices.json"
            path.write_text(json.dumps({"record_indices": [2, 5]}), encoding="utf-8")
            allowed = TRAINING.read_record_index_set(path)
            examples = [{"record_index": value} for value in [1, 2, 5, 8]]
            selected = TRAINING.filter_record_indices(examples, allowed, "heldout")
            self.assertEqual([row["record_index"] for row in selected], [2, 5])

    def test_record_index_manifest_rejects_missing_rows(self):
        with self.assertRaisesRegex(ValueError, "missing rows"):
            TRAINING.filter_record_indices(
                [{"record_index": 1}], {1, 9}, "heldout"
            )

    def test_validation_skips_only_empty_unlabeled_candidate_graphs(self):
        empty = {
            "record_index": 7,
            "db_id": "demo",
            "candidate_nodes": [],
            "gold_ids": [],
        }
        valid = {
            "record_index": 8,
            "candidate_nodes": [
                {"numeric_features": [1.0, 0.0]}
            ],
            "gold_ids": [1],
            "role_labels": [[1.0]],
        }
        usable, report = TRAINING.validate_and_filter_examples(
            [empty, valid], "train"
        )
        self.assertEqual([row["record_index"] for row in usable], [8])
        self.assertEqual(report["skipped_empty_unlabeled_count"], 1)

    def test_validation_rejects_empty_gold_bearing_candidate_graph(self):
        with self.assertRaises(ValueError):
            TRAINING.validate_and_filter_examples(
                [{"record_index": 9, "candidate_nodes": [], "gold_ids": [2]}],
                "train",
            )

    def test_constrained_selector_never_returns_orphan_column(self):
        example = {
            "candidate_nodes": [
                {"local_id": 0, "schema_item_id": 0, "type": "table", "owner_table_id": 0, "owner_local_id": 0},
                {"local_id": 1, "schema_item_id": 1, "type": "column", "owner_table_id": 0, "owner_local_id": 0},
                {"local_id": 2, "schema_item_id": 2, "type": "table", "owner_table_id": 2, "owner_local_id": 2},
                {"local_id": 3, "schema_item_id": 3, "type": "column", "owner_table_id": 2, "owner_local_id": 2},
            ],
            "schema_edges": [
                {"src": 0, "dst": 1, "type": "table_to_column"},
                {"src": 2, "dst": 3, "type": "table_to_column"},
            ],
            "baseline_selected_ids": [0, 1],
        }
        selected, _ = SELECTOR.constrained_topk(
            example,
            scores=[0.1, 0.2, 0.0, 5.0],
            top_k=3,
            max_tables=2,
            min_tables=1,
        )
        selected_set = set(selected)
        for local_id in selected:
            node = example["candidate_nodes"][local_id]
            if node["type"] == "column":
                self.assertIn(node["owner_local_id"], selected_set)


class Stage10ModelSmokeTest(unittest.TestCase):
    @staticmethod
    def structured_example():
        return {
            "candidate_nodes": [
                {"local_id": 0, "schema_item_id": 0, "type": "table", "owner_table_id": 0, "owner_local_id": 0},
                {"local_id": 1, "schema_item_id": 1, "type": "column", "owner_table_id": 0, "owner_local_id": 0},
                {"local_id": 2, "schema_item_id": 2, "type": "table", "owner_table_id": 2, "owner_local_id": 2},
                {"local_id": 3, "schema_item_id": 3, "type": "column", "owner_table_id": 2, "owner_local_id": 2},
            ],
            "schema_edges": [
                {"src": 0, "dst": 1, "type": "table_to_column"},
                {"src": 2, "dst": 3, "type": "table_to_column"},
            ],
            "baseline_selected_ids": [2, 3],
            "candidate_oracle_recall": 1.0,
        }

    def test_required_selector_builds_owner_closed_gold_feasible_set(self):
        example = self.structured_example()
        selected, detail = SELECTOR.constrained_topk(
            example,
            scores=[0.0, 0.0, 2.0, 2.0],
            top_k=2,
            max_tables=1,
            min_tables=1,
            required_local_ids={1},
        )
        self.assertEqual(set(selected), {0, 1})
        self.assertTrue(detail["required_feasible"])
        self.assertEqual(detail["forced_owner_additions"], 1)

    def test_structured_coverage_loss_swaps_intruder_package_for_gold_package(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable in the current interpreter")
        logits = torch.tensor([0.0, 0.0, 2.0, 2.0], requires_grad=True)
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        args = SimpleNamespace(
            output_top_k=2,
            max_tables=1,
            min_tables=1,
            connectivity_weight=0.0,
            baseline_retention_weight=0.0,
            structured_coverage_margin=0.1,
        )
        loss, detail = TRAINING.constrained_structured_coverage_loss(
            logits, labels, self.structured_example(), args
        )
        loss.backward()
        self.assertEqual(detail["structured_active"], 1.0)
        self.assertEqual(detail["structured_missing_gold"], 2.0)
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertLess(float(logits.grad[0]), 0.0)
        self.assertLess(float(logits.grad[1]), 0.0)
        self.assertGreater(float(logits.grad[2]), 0.0)
        self.assertGreater(float(logits.grad[3]), 0.0)

    def test_structured_coverage_loss_skips_infeasible_gold_table_budget(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable in the current interpreter")
        example = self.structured_example()
        logits = torch.tensor([0.0, 0.0, 2.0, 2.0], requires_grad=True)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        args = SimpleNamespace(
            output_top_k=2,
            max_tables=1,
            min_tables=1,
            connectivity_weight=0.0,
            baseline_retention_weight=0.0,
            structured_coverage_margin=0.1,
        )
        loss, detail = TRAINING.constrained_structured_coverage_loss(
            logits, labels, example, args
        )
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(detail["structured_active"], 0.0)
        self.assertEqual(detail["structured_feasible"], 0.0)

    def test_budget_aware_coverage_loss_targets_weakest_positive_and_boundary(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable in the current interpreter")
        logits = torch.tensor([2.0, 0.0, 3.0, 1.0, -1.0], requires_grad=True)
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
        loss, detail = TRAINING.topk_coverage_loss(
            logits,
            labels,
            top_k=3,
            margin=0.1,
            temperature=0.2,
            eligible=True,
        )
        loss.backward()

        # P=2 and K=3 means one negative may outrank the weakest positive; the
        # second-highest negative (score 1.0) is the first violating boundary.
        self.assertEqual(detail["coverage_boundary_rank"], 2.0)
        self.assertEqual(detail["coverage_active"], 1.0)
        self.assertAlmostEqual(detail["coverage_boundary_negative"], 1.0, places=5)
        self.assertLess(float(logits.grad[1]), 0.0)
        self.assertGreater(float(logits.grad[3]), 0.0)

    def test_coverage_loss_is_inactive_for_incomplete_or_trivial_candidates(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable in the current interpreter")
        logits = torch.tensor([1.0, 0.0, -1.0], requires_grad=True)
        labels = torch.tensor([1.0, 0.0, 0.0])
        incomplete, incomplete_detail = TRAINING.topk_coverage_loss(
            logits, labels, top_k=2, margin=0.1, temperature=0.2, eligible=False
        )
        trivial, trivial_detail = TRAINING.topk_coverage_loss(
            logits, labels, top_k=3, margin=0.1, temperature=0.2, eligible=True
        )
        self.assertEqual(float(incomplete.detach()), 0.0)
        self.assertEqual(float(trivial.detach()), 0.0)
        self.assertEqual(incomplete_detail["coverage_active"], 0.0)
        self.assertEqual(trivial_detail["coverage_active"], 0.0)

    def test_factor_graph_forward_and_backward(self):
        try:
            import torch
            from src.modeling.factor_graph_reranker import HeterogeneousFactorGraphReranker
        except ImportError:
            self.skipTest("PyTorch is unavailable in the current interpreter")
        model = HeterogeneousFactorGraphReranker(
            dense_dim=8,
            numeric_dim=6,
            factor_numeric_dim=3,
            relation_count=2,
            schema_relation_count=2,
            hidden_dim=16,
            num_layers=2,
            dropout=0.0,
        )
        output = model(
            dense_nodes=torch.randn(4, 8),
            numeric_features=torch.randn(4, 6),
            query_embedding=torch.randn(1, 8),
            factor_kind=torch.tensor([0, 1]),
            factor_relation=torch.tensor([0, -1]),
            factor_numeric=torch.randn(2, 3),
            schema_edge_index=torch.tensor([[0, 1, 2], [1, 0, 3]]),
            schema_edge_type=torch.tensor([0, 1, 0]),
            factor_edge_index=torch.tensor([[1, 2, 3], [0, 0, 1]]),
            factor_edge_type=torch.tensor([0, 0, 1]),
            factor_edge_weight=torch.tensor([1.0, 0.5, 0.8]),
        )
        self.assertEqual(tuple(output["logits"].shape), (4,))
        self.assertEqual(tuple(output["role_logits"].shape), (4, 2))
        output["logits"].sum().backward()
        self.assertIsNotNone(model.schema_dense.weight.grad)

    def test_one_training_step_and_constrained_evaluation(self):
        try:
            import numpy as np
            import torch
            import torch.nn as nn
            from src.modeling.factor_graph_reranker import HeterogeneousFactorGraphReranker
        except ImportError:
            self.skipTest("PyTorch/numpy is unavailable in the current interpreter")
        example = {
            "record_index": 0,
            "db_id": "demo",
            "question_id": 0,
            "question": "q",
            "candidate_nodes": [
                {"local_id": 0, "schema_item_id": 0, "schema_position": 0, "type": "table", "owner_table_id": 0, "owner_local_id": 0, "numeric_features": [1, 0, 1, 0, 1, 0]},
                {"local_id": 1, "schema_item_id": 1, "schema_position": 1, "type": "column", "owner_table_id": 0, "owner_local_id": 0, "numeric_features": [0, 1, 1, 0, 1, 1]},
                {"local_id": 2, "schema_item_id": 2, "schema_position": 2, "type": "column", "owner_table_id": 0, "owner_local_id": 0, "numeric_features": [0, 1, 0, 0, 1, 0]},
            ],
            "schema_edges": [
                {"src": 0, "dst": 1, "type": "table_to_column"},
                {"src": 0, "dst": 2, "type": "table_to_column"},
            ],
            "factors": [
                {"id": 0, "kind_id": 0, "relation_id": 0, "numeric_features": [0.67, 1.0, 0.75]},
            ],
            "factor_edges": [
                {"schema": 1, "factor": 0, "type_id": 0, "weight": 1.0},
                {"schema": 2, "factor": 0, "type_id": 0, "weight": 0.5},
            ],
            "baseline_selected_ids": [0, 2],
            "whole_labels": [1.0, 1.0, 0.0],
            "role_labels": [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            "gold_ids": [0, 1],
            "gold_table_ids": [0],
            "gold_column_ids": [1],
            "candidate_oracle_recall": 1.0,
        }
        runtime = {
            "np": np,
            "torch": torch,
            "nn": nn,
            "model": HeterogeneousFactorGraphReranker,
        }
        cache = {
            "query": np.random.randn(1, 8).astype("float32"),
            "node": np.random.randn(3, 8).astype("float32"),
            "index": {0: {"query_embedding_index": 0, "node_embedding_start": 0, "node_count": 3}},
            "dense_dim": 8,
        }
        maps = {"schema_relation_to_id": {"table_to_column": 0}, "factor_numeric_dim": 3}
        args = SimpleNamespace(
            seed=42,
            pos_weight=2.0,
            role_loss_weight=0.3,
            pairwise_loss_weight=0.5,
            pairwise_margin=0.5,
            hard_negative_k=2,
            coverage_loss_weight=0.3,
            coverage_margin=0.1,
            coverage_temperature=0.2,
            structured_coverage_loss_weight=0.2,
            structured_coverage_margin=0.1,
            max_grad_norm=1.0,
            output_top_k=2,
            max_tables=1,
            min_tables=-1,
            connectivity_weight=0.1,
            baseline_retention_weight=0.05,
            gradient_accumulation_steps=2,
            eval_every_examples=2,
        )
        model = HeterogeneousFactorGraphReranker(
            dense_dim=8,
            numeric_dim=6,
            factor_numeric_dim=3,
            relation_count=2,
            schema_relation_count=1,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        progress = []
        train_metrics = TRAINING.train_epoch(
            model,
            [example] * 5,
            cache,
            maps,
            args,
            runtime,
            torch.device("cpu"),
            optimizer,
            1,
            progress_callback=lambda metrics: progress.append(dict(metrics)),
        )
        metrics, predictions = TRAINING.evaluate(
            model, [example], cache, maps, args, runtime, torch.device("cpu"), "dev"
        )
        self.assertGreater(train_metrics["loss"], 0.0)
        self.assertEqual(train_metrics["optimizer_steps"], 3)
        self.assertEqual(train_metrics["effective_batch_size"], 2)
        self.assertEqual([row["example_count"] for row in progress], [2, 4])
        self.assertIn("constrained_complete_coverage@2", metrics)
        selected = predictions[0]["top_2"]
        selected_ids = {item["schema_item_id"] for item in selected}
        for item in selected:
            if item["type"] == "column":
                self.assertIn(item["owner_table_id"], selected_ids)


if __name__ == "__main__":
    unittest.main()
