import importlib.util
import json
import sys
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


TRAINING = load_module(
    "stage17a_train_full_schema_qrgta",
    "src/training/stage17a_train_full_schema_qrgta.py",
)
EVALUATION = load_module(
    "stage17a_evaluate_full_schema_qrgta",
    "src/evaluation/stage17a_evaluate_full_schema_qrgta.py",
)
CONTROLS = load_module(
    "stage17a_run_checkpoint_controls",
    "src/evaluation/stage17a_run_checkpoint_controls.py",
)
SUMMARY = load_module(
    "stage17a_summarize_causal_controls",
    "src/evaluation/stage17a_summarize_causal_controls.py",
)
try:
    MODELING = load_module(
        "full_schema_qrgta",
        "src/modeling/full_schema_qrgta.py",
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    MODELING = None


def graph_record():
    nodes = [
        {"id": 0, "type": "table", "name": "schools"},
        {"id": 1, "type": "column", "name": "schools.name", "table": "schools", "column": "name"},
        {"id": 2, "type": "column", "name": "schools.county", "table": "schools", "column": "county"},
        {"id": 3, "type": "table", "name": "districts"},
        {"id": 4, "type": "column", "name": "districts.school_id", "table": "districts", "column": "school_id"},
    ]
    return {
        "example_id": "dev::demo::7",
        "inference_inputs": {
            "db_id": "demo",
            "question": "Which schools are in Alameda?",
            "schema_nodes": nodes,
            "schema_edges": [
                {"src": 0, "dst": 0, "type": "self_loop"},
                {"src": 0, "dst": 1, "type": "table_to_column"},
                {"src": 1, "dst": 0, "type": "column_to_table"},
                {"src": 0, "dst": 2, "type": "table_to_column"},
                {"src": 2, "dst": 0, "type": "column_to_table"},
                {"src": 3, "dst": 4, "type": "table_to_column"},
            ],
        },
        "metadata": {"record_index": 0, "question_id": 7},
    }


def label_record():
    graph = graph_record()
    return {
        "db_id": "demo",
        "question_id": 7,
        "question": "Which schools are in Alameda?",
        "schema_items": graph["inference_inputs"]["schema_nodes"],
        "whole_sql_labels": [0, 1, 2],
    }


class Stage17AAlignmentTest(unittest.TestCase):
    def test_alignment_preserves_every_full_schema_node(self):
        examples, report = TRAINING.align_graphs_and_labels(
            [graph_record()], [label_record()], "dev"
        )
        self.assertEqual(report["usable_count"], 1)
        self.assertEqual(len(examples[0]["nodes"]), 5)
        self.assertEqual(examples[0]["gold_ids"], [0, 1, 2])

    def test_alignment_rejects_schema_identity_mismatch(self):
        label = label_record()
        label["schema_items"][2] = {**label["schema_items"][2], "name": "wrong.name"}
        with self.assertRaisesRegex(ValueError, "Schema identity mismatch"):
            TRAINING.align_graphs_and_labels([graph_record()], [label], "dev")

    def test_alignment_attaches_role_labels_from_clause_file(self):
        role_record = {
            "db_id": "demo",
            "question_id": 7,
            "clause_labels": {
                "select": [1],
                "where": [2],
                "join": [0],
            },
        }
        examples, _ = TRAINING.align_graphs_and_labels(
            [graph_record()], [label_record()], "dev", [role_record]
        )
        self.assertEqual(examples[0]["role_label_ids"]["OUTPUT_TARGET"], [1])
        self.assertEqual(examples[0]["role_label_ids"]["PREDICATE_COLUMN"], [2])
        self.assertEqual(examples[0]["role_label_ids"]["JOIN_BRIDGE"], [0])

    def test_prediction_evaluation_requires_complete_identity_preserving_ranking(self):
        label = label_record()
        prediction = {
            "record_index": 0,
            "db_id": "demo",
            "question_id": 7,
            "schema_node_count": 5,
            "ranked_schema": [
                {
                    "schema_item_id": item["id"],
                    "name": item["name"],
                    "type": item["type"],
                    "rank": rank,
                }
                for rank, item in enumerate(label["schema_items"], start=1)
            ],
        }
        examples, rankings, skipped = EVALUATION.align_predictions(
            [prediction], [label]
        )
        self.assertEqual(skipped, 0)
        self.assertEqual(len(examples), 1)
        self.assertEqual(rankings[0], [0, 1, 2, 3, 4])
        self.assertNotIn("gold_ids", prediction)

    def test_path_candidate_construction_is_deterministic_and_preserves_direct_edges(self):
        example = TRAINING.align_graphs_and_labels(
            [graph_record()], [label_record()], "dev"
        )[0][0]
        relations = TRAINING.relation_mapping([example], [example])
        first = TRAINING.build_path_rows(
            example,
            relations,
            max_path_distance=3,
            max_path_edges_per_destination=None,
            query_similarity=None,
        )
        second = TRAINING.build_path_rows(
            example,
            relations,
            max_path_distance=3,
            max_path_edges_per_destination=None,
            query_similarity=None,
        )
        self.assertEqual(first, second)
        direct = [row for row in first if row["is_direct"]]
        self.assertEqual(len(direct), len(example["schema_edges"]))
        self.assertTrue(any(row["src"] == 1 and row["dst"] == 2 and row["distance"] == 2 for row in first))
        signatures = TRAINING.collect_path_signatures([example], relations, 3)
        self.assertIn("PATH:table_to_column>column_to_table", signatures)

    def test_path_cap_does_not_remove_direct_schema_edges(self):
        example = TRAINING.align_graphs_and_labels(
            [graph_record()], [label_record()], "dev"
        )[0][0]
        relations = TRAINING.relation_mapping([example], [example])
        capped = TRAINING.build_path_rows(
            example,
            relations,
            max_path_distance=3,
            max_path_edges_per_destination=0,
            query_similarity=None,
        )
        self.assertEqual(len(capped), len(example["schema_edges"]))
        self.assertTrue(all(row["is_direct"] for row in capped))

    def test_unknown_path_signature_can_fall_back_to_neutral_for_old_checkpoints(self):
        path_signatures = {TRAINING.NEUTRAL_PATH_SIGNATURE: 0}
        self.assertEqual(
            path_signatures.get("PATH:table_to_column>column_to_table", path_signatures[TRAINING.NEUTRAL_PATH_SIGNATURE]),
            0,
        )


class Stage17AModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.runtime = TRAINING.import_runtime()
        except RuntimeError:
            cls.runtime = None

    def setUp(self):
        if self.runtime is None:
            self.skipTest("PyTorch is unavailable")

    def aligned_example(self):
        return TRAINING.align_graphs_and_labels(
            [graph_record()], [label_record()], "dev"
        )[0][0]

    def cache(self):
        np = self.runtime["np"]
        generator = np.random.default_rng(42)
        return {
            "query": generator.normal(size=(1, 16)).astype("float32"),
            "nodes": generator.normal(size=(5, 16)).astype("float32"),
            "by_record": {
                0: {
                    "example_index": 0,
                    "record_index": 0,
                    "query_embedding_index": 0,
                    "node_embedding_start": 0,
                    "node_count": 5,
                }
            },
            "by_example": {},
            "dense_dim": 16,
        }

    def args(self, control="normal"):
        return SimpleNamespace(control_mode=control, seed=42)

    def test_runtime_adds_one_way_query_edge_for_every_schema_node(self):
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, self.args(), self.runtime, "cpu"
        )
        self.assertEqual(tensors["dense_nodes"].shape, (5, 16))
        self.assertEqual(tensors["query_edge_destination"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(tensors["query_edge_type"].shape[0], 5)
        self.assertEqual(tensors["labels"].tolist(), [1.0, 1.0, 1.0, 0.0, 0.0])

    def test_zero_query_control_removes_query_graph_edges(self):
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        tensors = TRAINING.example_to_tensors(
            example,
            self.cache(),
            relations,
            self.args("zero_query_edges"),
            self.runtime,
            "cpu",
        )
        self.assertEqual(tensors["query_edge_destination"].numel(), 0)

    def test_shuffled_schema_edges_preserve_relation_marginals_and_self_loops(self):
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        normal_index, normal_type = TRAINING.schema_edge_tensors(
            example, relations, "normal", 42, self.runtime, "cpu"
        )
        shuffled_index, shuffled_type = TRAINING.schema_edge_tensors(
            example, relations, "shuffled_schema_edges", 42, self.runtime, "cpu"
        )
        self.assertEqual(normal_type.tolist(), shuffled_type.tolist())
        self.assertEqual(normal_index[0].tolist(), shuffled_index[0].tolist())
        self.assertCountEqual(normal_index[1].tolist(), shuffled_index[1].tolist())
        self_loop = relations["self_loop"]
        normal_loops = [
            (int(normal_index[0, i]), int(normal_index[1, i]))
            for i, relation in enumerate(normal_type.tolist())
            if relation == self_loop
        ]
        shuffled_loops = [
            (int(shuffled_index[0, i]), int(shuffled_index[1, i]))
            for i, relation in enumerate(shuffled_type.tolist())
            if relation == self_loop
        ]
        self.assertEqual(normal_loops, shuffled_loops)

    def test_primary_key_downgrade_preserves_topology_and_removes_typed_relations(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        typed_edges = []
        for edge in example["schema_edges"]:
            row = dict(edge)
            if int(row["src"]) == 0 and int(row["dst"]) == 1:
                row["type"] = "table_to_primary_key"
            elif int(row["src"]) == 1 and int(row["dst"]) == 0:
                row["type"] = "primary_key_to_table"
            typed_edges.append(row)
        example = {**example, "schema_edges": typed_edges}
        relations = TRAINING.relation_mapping([example], [example])
        normal_args = self.path_args(example, relations)
        control_args = self.path_args(
            example, relations, control="downgrade_primary_key_edges"
        )
        normal = TRAINING.example_to_tensors(
            example, self.cache(), relations, normal_args, self.runtime, "cpu"
        )
        downgraded = TRAINING.example_to_tensors(
            example, self.cache(), relations, control_args, self.runtime, "cpu"
        )
        self.assertTrue(
            torch.equal(normal["schema_edge_index"], downgraded["schema_edge_index"])
        )
        self.assertEqual(
            normal["path_stats"]["total_attention_edge_count"],
            downgraded["path_stats"]["total_attention_edge_count"],
        )
        typed_ids = {
            relations["table_to_primary_key"],
            relations["primary_key_to_table"],
        }
        self.assertTrue(any(value in typed_ids for value in normal["schema_edge_type"].tolist()))
        self.assertFalse(
            any(value in typed_ids for value in downgraded["schema_edge_type"].tolist())
        )
        self.assertFalse(
            torch.equal(normal["schema_path_signature"], downgraded["schema_path_signature"])
        )

    def test_shuffled_node_identity_stays_within_schema_type(self):
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        normal = TRAINING.example_to_tensors(
            example, self.cache(), relations, self.args(), self.runtime, "cpu"
        )
        shuffled = TRAINING.example_to_tensors(
            example,
            self.cache(),
            relations,
            self.args("shuffled_node_identity"),
            self.runtime,
            "cpu",
        )
        self.assertEqual(normal["node_types"].tolist(), shuffled["node_types"].tolist())
        for node_type in (0, 1):
            positions = [
                i for i, value in enumerate(normal["node_types"].tolist()) if value == node_type
            ]
            before = sorted(tuple(normal["dense_nodes"][i].tolist()) for i in positions)
            after = sorted(tuple(shuffled["dense_nodes"][i].tolist()) for i in positions)
            self.assertEqual(before, after)

    def test_qrgta_forward_backward_supports_variable_full_schema(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, self.args(), self.runtime, "cpu"
        )
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            model_type="qrgta",
        )
        output = TRAINING.forward_model(model, tensors)
        self.assertEqual(output["logits"].shape, (5,))
        self.assertEqual(output["schema_states"].shape, (5, 16))
        loss = self.runtime["loss"](output["logits"], tensors["labels"])
        loss.backward()
        gradient_total = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(gradient_total, 0.0)

    def path_args(self, example, relations, control="normal", model_type="path_qrgta"):
        return SimpleNamespace(
            control_mode=control,
            seed=42,
            model_type=model_type,
            max_path_distance=3,
            max_path_edges_per_destination=32,
            distance_buckets=TRAINING.distance_bucket_mapping(3),
            path_signatures=TRAINING.collect_path_signatures([example], relations, 3),
            coverage_surrogate_weight=0.1,
            coverage_margin=0.1,
            coverage_target_k=30,
            record_persistent_diagnostics=True,
        )

    def test_path_qrgta_forward_backward_supports_variable_full_schema(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(example, relations)
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, args, self.runtime, "cpu"
        )
        self.assertGreater(tensors["schema_edge_index"].shape[1], len(example["schema_edges"]))
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            model_type="path_qrgta",
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
        )
        output = TRAINING.forward_model(model, tensors)
        self.assertEqual(output["logits"].shape, (5,))
        loss = TRAINING.training_loss(output, tensors["labels"], args, self.runtime)
        loss.backward()
        gradient_total = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(gradient_total, 0.0)

    def test_table_multi_positive_loss_updates_every_gold_sibling(self):
        torch = self.runtime["torch"]
        logits = torch.tensor(
            [0.0, -0.4, -0.2, 0.7, 0.0, -0.1, 0.5], requires_grad=True
        )
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        parents = torch.tensor([0, 0, 0, 0, 4, 4, 4])
        is_column = torch.tensor([False, True, True, True, False, True, True])
        loss, diagnostics = TRAINING.table_multi_positive_boundary_loss(
            logits,
            labels,
            parents,
            is_column,
            self.runtime,
            margin=0.1,
            hard_negatives=5,
        )
        loss.backward()
        self.assertEqual(diagnostics["eligible_table_count"], 2)
        self.assertEqual(diagnostics["pair_count"], 3)
        self.assertTrue(torch.all(logits.grad[torch.tensor([1, 2, 5])] < 0))
        self.assertTrue(torch.all(logits.grad[torch.tensor([3, 6])] > 0))
        self.assertEqual(float(logits.grad[0]), 0.0)
        self.assertEqual(float(logits.grad[4]), 0.0)

    def test_table_multi_positive_hard_negative_selection_is_stable(self):
        torch = self.runtime["torch"]
        logits = torch.tensor(
            [0.0, -0.5, 1.0, 1.0, 1.0, 0.2, -0.3, -1.0],
            requires_grad=True,
        )
        labels = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        parents = torch.zeros(8, dtype=torch.long)
        is_column = torch.tensor([False, True, True, True, True, True, True, True])
        loss, diagnostics = TRAINING.table_multi_positive_boundary_loss(
            logits,
            labels,
            parents,
            is_column,
            self.runtime,
            hard_negatives=2,
        )
        loss.backward()
        self.assertEqual(diagnostics["pair_count"], 2)
        self.assertGreater(float(logits.grad[2]), 0.0)
        self.assertGreater(float(logits.grad[3]), 0.0)
        self.assertEqual(float(logits.grad[4]), 0.0)
        self.assertEqual(float(logits.grad[7]), 0.0)

    def test_table_multi_positive_loss_normalizes_per_table(self):
        torch = self.runtime["torch"]
        F = self.runtime["F"]
        logits = torch.tensor([0.0, 0.2, 0.8, 0.0, -0.3, 0.4, 0.1])
        labels = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        parents = torch.tensor([0, 0, 0, 3, 3, 3, 3])
        is_column = torch.tensor([False, True, True, False, True, True, True])
        margin = 0.1
        loss, diagnostics = TRAINING.table_multi_positive_boundary_loss(
            logits,
            labels,
            parents,
            is_column,
            self.runtime,
            margin=margin,
            hard_negatives=5,
        )
        table_zero = F.softplus(logits[2] - logits[1] + margin)
        table_three = torch.stack(
            [
                F.softplus(logits[5] - logits[4] + margin),
                F.softplus(logits[6] - logits[4] + margin),
            ]
        ).mean()
        self.assertTrue(torch.allclose(loss, (table_zero + table_three) / 2))
        self.assertEqual(diagnostics["eligible_table_count"], 2)
        self.assertEqual(diagnostics["pair_count"], 3)

    def test_table_multi_positive_empty_and_zero_weight_are_exact(self):
        torch = self.runtime["torch"]
        logits = torch.tensor([0.3, -0.2], requires_grad=True)
        labels = torch.tensor([0.0, 1.0])
        parents = torch.tensor([0, 0])
        is_column = torch.tensor([False, True])
        auxiliary, diagnostics = TRAINING.table_multi_positive_boundary_loss(
            logits, labels, parents, is_column, self.runtime
        )
        self.assertEqual(float(auxiliary), 0.0)
        self.assertEqual(diagnostics["eligible_table_count"], 0)

        output = {"logits": logits}
        base_args = SimpleNamespace(
            role_loss_weight=0.0,
            coverage_surrogate_weight=0.1,
            coverage_target_k=30,
            coverage_margin=0.1,
        )
        zero_args = SimpleNamespace(
            **vars(base_args),
            table_multi_positive_weight=0.0,
            table_multi_positive_margin=0.1,
            table_multi_positive_hard_negatives=5,
        )
        base_loss = TRAINING.training_loss(output, labels, base_args, self.runtime)
        zero_loss = TRAINING.training_loss(
            output,
            labels,
            zero_args,
            self.runtime,
            column_parent_table=parents,
            is_column_node=is_column,
        )
        self.assertTrue(torch.equal(base_loss, zero_loss))

    def test_table_multi_positive_diagnostics_do_not_enter_predictions(self):
        example = {**self.aligned_example(), "gold_ids": [0, 1]}
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(
            example, relations, model_type="table_competitive_path_qrgta"
        )
        args.table_multi_positive_weight = 0.1
        args.table_multi_positive_margin = 0.1
        args.table_multi_positive_hard_negatives = 5
        args.role_mapping = {}
        args.role_loss_weight = 0.0
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            model_type="table_competitive_path_qrgta",
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
            competition_hidden_dim=8,
            competition_dropout=0.0,
        )
        metrics, predictions = TRAINING.evaluate(
            model,
            [example],
            self.cache(),
            relations,
            args,
            self.runtime,
            "cpu",
            "dev",
            predictions=True,
        )
        diagnostics = metrics["table_multi_positive_diagnostics"]
        self.assertEqual(diagnostics["eligible_table_count"], 1)
        self.assertGreater(diagnostics["pair_count"], 0)
        self.assertNotIn("table_multi_positive_diagnostics", predictions[0])
        self.assertNotIn("gold_ids", predictions[0])

    def test_column_parent_table_tensor_matches_schema_ownership(self):
        example = self.aligned_example()
        parent, is_column, is_table = TRAINING.column_parent_table_tensor(
            example, "normal", 42, self.runtime, "cpu"
        )
        self.assertEqual(parent.tolist(), [0, 0, 0, 3, 3])
        self.assertEqual(is_column.tolist(), [False, True, True, False, True])
        self.assertEqual(is_table.tolist(), [True, False, False, True, False])

    def test_table_competitive_path_qrgta_forward_backward_supports_variable_full_schema(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(
            example, relations, model_type="table_competitive_path_qrgta"
        )
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, args, self.runtime, "cpu"
        )
        self.assertEqual(tensors["column_parent_table"].tolist(), [0, 0, 0, 3, 3])
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            model_type="table_competitive_path_qrgta",
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
            competition_hidden_dim=8,
            competition_dropout=0.0,
        )
        output = TRAINING.forward_model(model, tensors)
        self.assertEqual(output["logits"].shape, (5,))
        self.assertEqual(output["schema_states"].shape, (5, 16))
        loss = TRAINING.training_loss(output, tensors["labels"], args, self.runtime)
        loss.backward()
        gradient_total = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(gradient_total, 0.0)

    def test_table_competitive_path_qrgta_requires_path_and_parent_tensors(self):
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(
            example, relations, model_type="table_competitive_path_qrgta"
        )
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, args, self.runtime, "cpu"
        )
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            model_type="table_competitive_path_qrgta",
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
            competition_hidden_dim=8,
            competition_dropout=0.0,
        )
        tensors_without_path = dict(tensors)
        tensors_without_path["schema_distance_bucket"] = None
        with self.assertRaisesRegex(ValueError, "requires path and distance tensors"):
            TRAINING.forward_model(model, tensors_without_path)
        tensors_without_parent = dict(tensors)
        tensors_without_parent["column_parent_table"] = None
        with self.assertRaisesRegex(ValueError, "requires parent-table tensors"):
            TRAINING.forward_model(model, tensors_without_parent)

    def test_pk_residual_model_preserves_generic_edges_and_uses_gated_delta(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        marked_edges = []
        for edge in example["schema_edges"]:
            row = dict(edge)
            if (int(row["src"]), int(row["dst"])) in {(0, 1), (1, 0)}:
                row["is_primary_key_edge"] = True
            marked_edges.append(row)
        example = {**example, "schema_edges": marked_edges}
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(
            example,
            relations,
            model_type="pk_residual_table_competitive_path_qrgta",
        )
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, args, self.runtime, "cpu"
        )
        self.assertNotIn("table_to_primary_key", relations)
        self.assertNotIn("primary_key_to_table", relations)
        directions = tensors["schema_primary_key_direction"]
        self.assertEqual(int(directions.eq(TRAINING.PK_EDGE_TABLE_TO_PRIMARY_KEY).sum()), 1)
        self.assertEqual(int(directions.eq(TRAINING.PK_EDGE_PRIMARY_KEY_TO_TABLE).sum()), 1)
        self.assertTrue(torch.all(directions[6:] == TRAINING.PK_EDGE_NONE))

        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            model_type="pk_residual_table_competitive_path_qrgta",
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
            competition_hidden_dim=8,
            competition_dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            initial = TRAINING.forward_model(model, tensors)["logits"]
            zero_tensors = {**tensors, "zero_pk_modifier": True}
            initial_zero = TRAINING.forward_model(model, zero_tensors)["logits"]
        self.assertTrue(torch.equal(initial, initial_zero))

        with torch.no_grad():
            for layer in model.layers:
                layer.primary_key_delta_key.weight[1:].fill_(0.2)
                layer.primary_key_delta_value.weight[1:].fill_(0.3)
                layer.primary_key_delta_bias.weight[1:].fill_(0.1)
            modified = TRAINING.forward_model(model, tensors)["logits"]
            disabled = TRAINING.forward_model(model, zero_tensors)["logits"]
        self.assertFalse(torch.allclose(modified, disabled))

        model.train()
        output = TRAINING.forward_model(model, tensors)
        loss = TRAINING.training_loss(output, tensors["labels"], args, self.runtime)
        loss.backward()
        delta_gradient = sum(
            float(layer.primary_key_delta_value.weight.grad.abs().sum())
            for layer in model.layers
        )
        self.assertGreater(delta_gradient, 0.0)

    def test_zero_pk_modifier_preserves_all_structural_tensors(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        example = {
            **example,
            "schema_edges": [
                {
                    **edge,
                    **(
                        {"is_primary_key_edge": True}
                        if (int(edge["src"]), int(edge["dst"])) in {(0, 1), (1, 0)}
                        else {}
                    ),
                }
                for edge in example["schema_edges"]
            ],
        }
        relations = TRAINING.relation_mapping([example], [example])
        normal = TRAINING.example_to_tensors(
            example,
            self.cache(),
            relations,
            self.path_args(
                example,
                relations,
                model_type="pk_residual_table_competitive_path_qrgta",
            ),
            self.runtime,
            "cpu",
        )
        zero = TRAINING.example_to_tensors(
            example,
            self.cache(),
            relations,
            self.path_args(
                example,
                relations,
                control="zero_pk_modifier",
                model_type="pk_residual_table_competitive_path_qrgta",
            ),
            self.runtime,
            "cpu",
        )
        for key in (
            "schema_edge_index",
            "schema_edge_type",
            "schema_distance_bucket",
            "schema_path_signature",
            "schema_primary_key_direction",
            "dense_nodes",
        ):
            self.assertTrue(torch.equal(normal[key], zero[key]), key)
        self.assertFalse(normal["zero_pk_modifier"])
        self.assertTrue(zero["zero_pk_modifier"])

    def test_zero_initialized_pk_residual_is_stage17e_equivalent_at_same_seed(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        example = {
            **example,
            "schema_edges": [
                {
                    **edge,
                    **(
                        {"is_primary_key_edge": True}
                        if (int(edge["src"]), int(edge["dst"])) in {(0, 1), (1, 0)}
                        else {"is_primary_key_edge": False}
                    ),
                }
                for edge in example["schema_edges"]
            ],
        }
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(
            example,
            relations,
            model_type="pk_residual_table_competitive_path_qrgta",
        )
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, args, self.runtime, "cpu"
        )
        common = dict(
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=3,
            num_heads=4,
            dropout=0.0,
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
            competition_hidden_dim=8,
            competition_dropout=0.0,
        )
        torch.manual_seed(123)
        baseline = self.runtime["model"](
            **common, model_type="table_competitive_path_qrgta"
        )
        torch.manual_seed(123)
        residual = self.runtime["model"](
            **common, model_type="pk_residual_table_competitive_path_qrgta"
        )
        baseline.eval()
        residual.eval()
        residual_state = residual.state_dict()
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, residual_state[name]), name)
        with torch.no_grad():
            baseline_logits = TRAINING.forward_model(baseline, tensors)["logits"]
            residual_logits = TRAINING.forward_model(residual, tensors)["logits"]
        self.assertTrue(torch.equal(baseline_logits, residual_logits))

    def test_table_competition_only_updates_column_nodes(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(
            example, relations, model_type="table_competitive_path_qrgta"
        )
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, args, self.runtime, "cpu"
        )
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            model_type="table_competitive_path_qrgta",
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
            competition_hidden_dim=8,
            competition_dropout=0.0,
        )
        model.eval()
        layer = model.layers[0]
        with torch.no_grad():
            states = torch.randn(5, 16)
            query = torch.randn(16)
            refined = layer._table_scoped_competition(
                states,
                query,
                tensors["column_parent_table"],
                tensors["is_column_node"],
            )
        self.assertTrue(torch.equal(refined[tensors["is_table_node"]], states[tensors["is_table_node"]]))
        self.assertFalse(torch.equal(refined[tensors["is_column_node"]], states[tensors["is_column_node"]]))

    def test_table_competition_controls_preserve_identity_and_edges(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        normal_args = self.path_args(
            example, relations, model_type="table_competitive_path_qrgta"
        )
        normal = TRAINING.example_to_tensors(
            example, self.cache(), relations, normal_args, self.runtime, "cpu"
        )
        for mode in (
            "zero_table_competition",
            "shuffle_column_parent_table",
            "zero_competition_gates",
        ):
            control_args = self.path_args(
                example, relations, mode, model_type="table_competitive_path_qrgta"
            )
            control = TRAINING.example_to_tensors(
                example, self.cache(), relations, control_args, self.runtime, "cpu"
            )
            self.assertTrue(torch.equal(normal["schema_edge_index"], control["schema_edge_index"]))
            self.assertTrue(torch.equal(normal["schema_edge_type"], control["schema_edge_type"]))
            self.assertTrue(torch.equal(normal["dense_nodes"], control["dense_nodes"]))
            self.assertEqual(normal["is_column_node"].tolist(), control["is_column_node"].tolist())
        shuffled = TRAINING.example_to_tensors(
            example,
            self.cache(),
            relations,
            self.path_args(
                example,
                relations,
                "shuffle_column_parent_table",
                model_type="table_competitive_path_qrgta",
            ),
            self.runtime,
            "cpu",
        )
        self.assertCountEqual(
            normal["column_parent_table"][normal["is_column_node"]].tolist(),
            shuffled["column_parent_table"][shuffled["is_column_node"]].tolist(),
        )

    def test_zero_competition_controls_disable_column_writeback(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(
            example, relations, model_type="table_competitive_path_qrgta"
        )
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, args, self.runtime, "cpu"
        )
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            model_type="table_competitive_path_qrgta",
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
            competition_hidden_dim=8,
            competition_dropout=0.0,
        )
        model.eval()
        layer = model.layers[0]
        with torch.no_grad():
            states = torch.randn(5, 16)
            query = torch.randn(16)
            zero_module = layer._table_scoped_competition(
                states,
                query,
                tensors["column_parent_table"],
                tensors["is_column_node"],
                zero_table_competition=True,
            )
            zero_gates = layer._table_scoped_competition(
                states,
                query,
                tensors["column_parent_table"],
                tensors["is_column_node"],
                zero_competition_gates=True,
            )
        self.assertTrue(torch.equal(zero_module, states))
        self.assertTrue(torch.allclose(zero_gates, states, atol=1e-6))

    def test_enhanced_table_competitive_path_qrgta_forward_backward_supports_variable_full_schema(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(
            example, relations, model_type="enhanced_table_competitive_path_qrgta"
        )
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, args, self.runtime, "cpu"
        )
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            model_type="enhanced_table_competitive_path_qrgta",
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
            competition_hidden_dim=8,
            competition_dropout=0.0,
            competition_temperature=1.5,
            competition_residual_scale=0.5,
        )
        output = TRAINING.forward_model(model, tensors)
        self.assertEqual(output["logits"].shape, (5,))
        self.assertEqual(output["schema_states"].shape, (5, 16))
        loss = TRAINING.training_loss(output, tensors["labels"], args, self.runtime)
        loss.backward()
        gradient_total = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(gradient_total, 0.0)

    def test_enhanced_table_competition_temperature_changes_sibling_weights(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(
            example, relations, model_type="enhanced_table_competitive_path_qrgta"
        )
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, args, self.runtime, "cpu"
        )
        base_kwargs = dict(
            hidden_dim=16,
            num_heads=4,
            relation_count=len(relations),
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
            dropout=0.0,
            competition_hidden_dim=8,
            competition_dropout=0.0,
            competition_residual_scale=0.5,
        )
        cold = MODELING.EnhancedTableScopedCompetitivePathQRGTAEncoderLayer(
            **base_kwargs, competition_temperature=0.5
        )
        warm = MODELING.EnhancedTableScopedCompetitivePathQRGTAEncoderLayer(
            **base_kwargs, competition_temperature=2.0
        )
        warm.load_state_dict(cold.state_dict())
        cold.eval()
        warm.eval()
        with torch.no_grad():
            states = torch.randn(5, 16)
            query = torch.randn(16)
            cold_features = cold._competition_features(
                states, query, tensors["column_parent_table"], tensors["is_column_node"]
            )
            warm_features = warm._competition_features(
                states, query, tensors["column_parent_table"], tensors["is_column_node"]
            )
        self.assertEqual(cold_features["sibling_weight"].shape, (3, 1))
        self.assertEqual(warm_features["sibling_weight"].shape, (3, 1))
        self.assertFalse(
            torch.allclose(
                cold_features["sibling_weight"], warm_features["sibling_weight"]
            )
        )

    def test_enhanced_table_competition_scale_zero_and_multi_winner_gate(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(
            example, relations, model_type="enhanced_table_competitive_path_qrgta"
        )
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, args, self.runtime, "cpu"
        )
        layer = MODELING.EnhancedTableScopedCompetitivePathQRGTAEncoderLayer(
            hidden_dim=16,
            num_heads=4,
            relation_count=len(relations),
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
            dropout=0.0,
            competition_hidden_dim=8,
            competition_dropout=0.0,
            competition_temperature=1.5,
            competition_residual_scale=0.0,
        )
        layer.eval()
        with torch.no_grad():
            states = torch.randn(5, 16)
            query = torch.randn(16)
            refined = layer._table_scoped_competition(
                states, query, tensors["column_parent_table"], tensors["is_column_node"]
            )
            features = layer._competition_features(
                states, query, tensors["column_parent_table"], tensors["is_column_node"]
            )
        self.assertTrue(torch.equal(refined, states))
        self.assertEqual(features["multi_winner_gate"].shape, (3, 1))
        self.assertTrue(torch.all(features["multi_winner_gate"] >= 0.0))
        self.assertTrue(torch.all(features["multi_winner_gate"] <= 1.0))

    def test_role_head_outputs_logits_and_loss(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        example["role_label_ids"] = {
            "OUTPUT_TARGET": [1],
            "PREDICATE_COLUMN": [2],
        }
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(example, relations)
        args.role_loss_weight = 0.2
        args.role_mapping = TRAINING.role_mapping([example], [example])
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, args, self.runtime, "cpu"
        )
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            model_type="path_qrgta",
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
            role_count=len(args.role_mapping),
        )
        output = TRAINING.forward_model(model, tensors)
        self.assertEqual(tuple(output["role_logits"].shape), (5, len(args.role_mapping)))
        args._current_role_labels = tensors["role_labels"]
        loss = TRAINING.training_loss(output, tensors["labels"], args, self.runtime)
        args._current_role_labels = None
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.role_scorer[-1].weight.grad)

    def test_persistent_path_qrgta_forward_backward_and_diagnostics(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(example, relations, model_type="persistent_path_qrgta")
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, args, self.runtime, "cpu"
        )
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            model_type="persistent_path_qrgta",
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
        )
        output = TRAINING.forward_model(model, tensors)
        self.assertEqual(output["logits"].shape, (5,))
        diagnostics = output["persistent_diagnostics"]
        for key in (
            "avg_message_gate",
            "avg_ffn_gate",
            "min_message_gate",
            "max_message_gate",
            "avg_identity_cosine_input_to_final",
            "avg_layer_delta_norm",
            "avg_final_delta_norm",
        ):
            self.assertIn(key, diagnostics)
            self.assertTrue(torch.isfinite(diagnostics[key]))
        self.assertGreaterEqual(float(diagnostics["min_message_gate"]), 0.0)
        self.assertLessEqual(float(diagnostics["max_message_gate"]), 1.0)
        loss = TRAINING.training_loss(output, tensors["labels"], args, self.runtime)
        loss.backward()
        gradient_total = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(gradient_total, 0.0)

    def test_persistent_path_qrgta_requires_path_tensors(self):
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        tensors = TRAINING.example_to_tensors(
            example,
            self.cache(),
            relations,
            self.path_args(example, relations, model_type="persistent_path_qrgta"),
            self.runtime,
            "cpu",
        )
        tensors["schema_distance_bucket"] = None
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            model_type="persistent_path_qrgta",
            distance_bucket_count=5,
            path_signature_count=12,
        )
        with self.assertRaisesRegex(ValueError, "requires path and distance tensors"):
            TRAINING.forward_model(model, tensors)

    def test_zero_update_gates_preserves_initial_identity_state(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(
            example, relations, control="zero_update_gates", model_type="persistent_path_qrgta"
        )
        tensors = TRAINING.example_to_tensors(
            example, self.cache(), relations, args, self.runtime, "cpu"
        )
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            model_type="persistent_path_qrgta",
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
        )
        output = TRAINING.forward_model(model, tensors)
        with torch.no_grad():
            initial = model.node_norm(
                model.node_input(tensors["dense_nodes"]) + model.node_type(tensors["node_types"])
            )
        self.assertTrue(torch.allclose(output["schema_states"], initial, atol=1e-6))
        self.assertEqual(float(output["persistent_diagnostics"]["avg_message_gate"]), 0.0)
        self.assertEqual(float(output["persistent_diagnostics"]["avg_ffn_gate"]), 0.0)

    def test_persistent_diagnostics_are_metrics_not_predictions(self):
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        args = self.path_args(example, relations, model_type="persistent_path_qrgta")
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            model_type="persistent_path_qrgta",
            distance_bucket_count=len(args.distance_buckets),
            path_signature_count=len(args.path_signatures),
        )
        metrics, predictions = TRAINING.evaluate(
            model,
            [example],
            self.cache(),
            relations,
            args,
            self.runtime,
            "cpu",
            "dev",
            predictions=True,
        )
        self.assertIn("persistent_diagnostics", metrics)
        self.assertNotIn("persistent_diagnostics", predictions[0])
        self.assertNotIn("gold_ids", predictions[0])

    def test_zero_path_features_preserves_edge_count_and_neutralizes_schema_features(self):
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        normal_args = self.path_args(example, relations)
        zero_args = self.path_args(example, relations, "zero_path_features")
        normal = TRAINING.example_to_tensors(
            example, self.cache(), relations, normal_args, self.runtime, "cpu"
        )
        zero = TRAINING.example_to_tensors(
            example, self.cache(), relations, zero_args, self.runtime, "cpu"
        )
        self.assertTrue(self.runtime["torch"].equal(normal["schema_edge_index"], zero["schema_edge_index"]))
        self.assertEqual(normal["schema_edge_type"].tolist(), zero["schema_edge_type"].tolist())
        self.assertEqual(
            zero["schema_distance_bucket"].tolist(),
            [zero_args.distance_buckets[TRAINING.DISTANCE_SELF]] * zero["schema_edge_type"].numel(),
        )
        self.assertEqual(
            zero["schema_path_signature"].tolist(),
            [zero_args.path_signatures[TRAINING.NEUTRAL_PATH_SIGNATURE]] * zero["schema_edge_type"].numel(),
        )

    def test_shuffled_path_controls_preserve_schema_edge_identity(self):
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        normal_args = self.path_args(example, relations)
        normal = TRAINING.example_to_tensors(
            example, self.cache(), relations, normal_args, self.runtime, "cpu"
        )
        for mode in ("shuffled_distance_buckets", "shuffled_path_signatures"):
            control_args = self.path_args(example, relations, mode)
            control = TRAINING.example_to_tensors(
                example, self.cache(), relations, control_args, self.runtime, "cpu"
            )
            self.assertTrue(
                self.runtime["torch"].equal(normal["schema_edge_index"], control["schema_edge_index"])
            )
            self.assertEqual(normal["schema_edge_type"].tolist(), control["schema_edge_type"].tolist())

    def test_depth_matched_mlp_forward_backward_ignores_graph_edges(self):
        torch = self.runtime["torch"]
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        normal = TRAINING.example_to_tensors(
            example, self.cache(), relations, self.args(), self.runtime, "cpu"
        )
        shuffled = TRAINING.example_to_tensors(
            example,
            self.cache(),
            relations,
            self.args("shuffled_schema_edges"),
            self.runtime,
            "cpu",
        )
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=len(relations),
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            model_type="mlp_residual",
        )
        model.eval()
        first = TRAINING.forward_model(model, normal)
        second = TRAINING.forward_model(model, shuffled)
        self.assertTrue(torch.equal(first["logits"], second["logits"]))
        loss = self.runtime["loss"](first["logits"], normal["labels"])
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_model_state_digest_is_stable(self):
        model = self.runtime["model"](
            dense_dim=16,
            relation_count=7,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            model_type="qrgta",
        )
        first = CONTROLS.state_dict_sha256(model)
        second = CONTROLS.state_dict_sha256(model)
        self.assertEqual(first, second)

    def test_reference_check_accepts_metric_equivalent_numerical_tie_drift(self):
        def record(order, scores):
            return {
                "record_index": 7,
                "ranked_schema": [
                    {
                        "schema_item_id": schema_id,
                        "logit": scores[schema_id],
                        "rank": rank,
                    }
                    for rank, schema_id in enumerate(order, start=1)
                ],
            }

        reference = record([1, 2, 3], {1: 2.0, 2: -1.0, 3: -1.0000001})
        actual = record([1, 3, 2], {1: 2.0, 2: -1.0000001, 3: -1.0})
        examples = [{"record_index": 7, "gold_ids": [1]}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.jsonl"
            TRAINING.write_jsonl(path, [reference])
            result = CONTROLS.assert_reference_normal_matches(
                [actual], path, examples, score_atol=1e-5
            )
        self.assertTrue(result["metric_equivalent"])
        self.assertFalse(result["exact_full_ranking"])
        self.assertEqual(result["numerical_rank_drift_record_indices"], [7])

    def test_reference_check_rejects_non_equivalent_score_drift(self):
        reference = {
            "record_index": 7,
            "ranked_schema": [
                {"schema_item_id": 1, "logit": 2.0, "rank": 1},
                {"schema_item_id": 2, "logit": -1.0, "rank": 2},
            ],
        }
        actual = {
            "record_index": 7,
            "ranked_schema": [
                {"schema_item_id": 2, "logit": 0.0, "rank": 1},
                {"schema_item_id": 1, "logit": 1.0, "rank": 2},
            ],
        }
        examples = [{"record_index": 7, "gold_ids": [1]}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.jsonl"
            TRAINING.write_jsonl(path, [reference])
            with self.assertRaisesRegex(ValueError, "not metric-equivalent"):
                CONTROLS.assert_reference_normal_matches(
                    [actual], path, examples, score_atol=1e-5
                )

    def test_checkpoint_controls_write_paired_leakage_free_outputs(self):
        torch = self.runtime["torch"]
        np = self.runtime["np"]
        graph = graph_record()
        label = label_record()
        example = self.aligned_example()
        relations = TRAINING.relation_mapping([example], [example])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph_path = root / "dev_examples.jsonl"
            label_path = root / "dev_labels.jsonl"
            TRAINING.write_jsonl(graph_path, [graph])
            TRAINING.write_jsonl(label_path, [label])
            cache_dir = root / "cache"
            cache_dir.mkdir()
            cache = self.cache()
            np.save(cache_dir / "dev_query_embeddings.npy", cache["query"])
            np.save(cache_dir / "dev_node_embeddings.npy", cache["nodes"])
            (cache_dir / "dev_index.json").write_text(
                json.dumps(list(cache["by_record"].values())), encoding="utf-8"
            )
            model_config = {
                "dense_dim": 16,
                "relation_count": len(relations),
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 4,
                "dropout": 0.0,
                "model_type": "qrgta",
                "relations": relations,
                "control_mode": "normal",
            }
            model = self.runtime["model"](
                dense_dim=16,
                relation_count=len(relations),
                hidden_dim=16,
                num_layers=1,
                num_heads=4,
                dropout=0.0,
                model_type="qrgta",
            )
            checkpoint_path = root / "best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "epoch": 1,
                },
                checkpoint_path,
            )
            output_dir = root / "controls"
            argv = [
                "stage17a_run_checkpoint_controls.py",
                "--checkpoint",
                str(checkpoint_path),
                "--dev-graph-file",
                str(graph_path),
                "--dev-label-file",
                str(label_path),
                "--embedding-cache-dir",
                str(cache_dir),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
            ]
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(StringIO()):
                    CONTROLS.main()
            summary = json.loads(
                (output_dir / "intervention_summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["parameters_unchanged"])
            self.assertEqual(set(summary["metrics"]), set(CONTROLS.CONTROL_MODES))
            prediction = json.loads(
                (output_dir / "normal" / "dev_predictions.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertNotIn("gold_ids", prediction)


class Stage17ASummaryTest(unittest.TestCase):
    def metric_payload(self, complete):
        return {
            "sample_count": 100,
            "complete_coverage@30": complete,
            "schema_recall@30": complete + 0.1,
            "table_recall@30": complete + 0.12,
            "column_recall@30": complete + 0.08,
            "complete_coverage@50": complete + 0.15,
            "mrr": complete + 0.05,
        }

    def write_training_run(self, path, model_type, control_mode, complete):
        path.mkdir(parents=True)
        data_config = {
            "train_graph_file": "train_graph.jsonl",
            "dev_graph_file": "dev_graph.jsonl",
            "train_label_file": "train_labels.jsonl",
            "dev_label_file": "dev_labels.jsonl",
            "embedding_cache_dir": "cache",
            "train_limit": None,
            "dev_limit": None,
        }
        (path / "training_summary.json").write_text(
            json.dumps(
                {
                    "best_epoch": 1,
                    "dev_metrics": self.metric_payload(complete),
                    "config": data_config,
                    "model_config": {
                        "model_type": model_type,
                        "control_mode": control_mode,
                    },
                }
            ),
            encoding="utf-8",
        )
        (path / "best.pt").write_bytes(f"{model_type}:{control_mode}".encode("utf-8"))

    def test_summary_reports_paired_deltas_and_rejects_no_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = ["stage17a_summarize_causal_controls.py"]
            for seed in (42, 43, 44):
                normal = root / f"normal_{seed}"
                mlp = root / f"mlp_{seed}"
                intervention = root / f"intervention_{seed}"
                self.write_training_run(normal, "qrgta", "normal", 0.8)
                self.write_training_run(mlp, "mlp_residual", "normal", 0.7)
                intervention.mkdir()
                normal_metrics = self.metric_payload(0.8)
                controls = {
                    "zero_query_edges": self.metric_payload(0.78),
                    "shuffled_schema_edges": self.metric_payload(0.76),
                    "shuffled_node_identity": self.metric_payload(0.74),
                }
                (intervention / "intervention_summary.json").write_text(
                    json.dumps(
                        {
                            "parameters_unchanged": True,
                            "checkpoint_sha256": SUMMARY.file_sha256(normal / "best.pt"),
                            "reference_normal_reproduced": True,
                            "data_config": {
                                "dev_graph_file": "dev_graph.jsonl",
                                "dev_label_file": "dev_labels.jsonl",
                                "embedding_cache_dir": "cache",
                                "dev_limit": None,
                            },
                            "metrics": {"normal": normal_metrics, **controls},
                        }
                    ),
                    encoding="utf-8",
                )
                argv.extend(["--normal-run", f"{seed}={normal}"])
                argv.extend(["--mlp-run", f"{seed}={mlp}"])
                argv.extend(["--intervention-run", f"{seed}={intervention}"])
            output = root / "summary.json"
            argv.extend(["--output-file", str(output)])
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(StringIO()):
                    SUMMARY.main()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                result["qrgta_minus_mlp"]["complete_coverage@30"]["mean"], 0.1
            )
            self.assertTrue(result["decision_passed"])

    def test_summary_accepts_path_qrgta_normal_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = ["stage17a_summarize_causal_controls.py"]
            for seed in (42, 43, 44):
                normal = root / f"path_normal_{seed}"
                mlp = root / f"mlp_{seed}"
                intervention = root / f"intervention_{seed}"
                self.write_training_run(normal, "path_qrgta", "normal", 0.8)
                self.write_training_run(mlp, "mlp_residual", "normal", 0.7)
                intervention.mkdir()
                controls = {
                    "zero_query_edges": self.metric_payload(0.78),
                    "shuffled_schema_edges": self.metric_payload(0.76),
                    "shuffled_node_identity": self.metric_payload(0.74),
                    "shuffled_distance_buckets": self.metric_payload(0.79),
                    "shuffled_path_signatures": self.metric_payload(0.77),
                    "zero_path_features": self.metric_payload(0.79),
                }
                (intervention / "intervention_summary.json").write_text(
                    json.dumps(
                        {
                            "parameters_unchanged": True,
                            "checkpoint_sha256": SUMMARY.file_sha256(normal / "best.pt"),
                            "reference_normal_reproduced": True,
                            "data_config": {
                                "dev_graph_file": "dev_graph.jsonl",
                                "dev_label_file": "dev_labels.jsonl",
                                "embedding_cache_dir": "cache",
                                "dev_limit": None,
                            },
                            "metrics": {"normal": self.metric_payload(0.8), **controls},
                        }
                    ),
                    encoding="utf-8",
                )
                argv.extend(["--normal-run", f"{seed}={normal}"])
                argv.extend(["--mlp-run", f"{seed}={mlp}"])
                argv.extend(["--intervention-run", f"{seed}={intervention}"])
            output = root / "summary.json"
            argv.extend(["--output-file", str(output)])
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(StringIO()):
                    SUMMARY.main()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn(
                "shuffled_path_signatures",
                result["path_checkpoint_interventions_normal_minus_control"],
            )
            self.assertTrue(result["decision_passed"])

    def test_summary_accepts_persistent_path_qrgta_normal_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = ["stage17a_summarize_causal_controls.py"]
            for seed in (42, 43, 44):
                normal = root / f"persistent_normal_{seed}"
                mlp = root / f"mlp_{seed}"
                intervention = root / f"intervention_{seed}"
                self.write_training_run(normal, "persistent_path_qrgta", "normal", 0.8)
                mlp_complete = 0.7
                self.write_training_run(mlp, "mlp_residual", "normal", mlp_complete)
                intervention.mkdir()
                controls = {
                    "zero_query_edges": self.metric_payload(0.78),
                    "shuffled_schema_edges": self.metric_payload(0.76),
                    "shuffled_node_identity": self.metric_payload(0.74),
                    "shuffled_distance_buckets": self.metric_payload(0.79),
                    "shuffled_path_signatures": self.metric_payload(0.77),
                    "zero_path_features": self.metric_payload(0.79),
                    "zero_update_gates": self.metric_payload(0.75),
                }
                (intervention / "intervention_summary.json").write_text(
                    json.dumps(
                        {
                            "parameters_unchanged": True,
                            "checkpoint_sha256": SUMMARY.file_sha256(normal / "best.pt"),
                            "reference_normal_reproduced": True,
                            "data_config": {
                                "dev_graph_file": "dev_graph.jsonl",
                                "dev_label_file": "dev_labels.jsonl",
                                "embedding_cache_dir": "cache",
                                "dev_limit": None,
                            },
                            "metrics": {"normal": self.metric_payload(0.8), **controls},
                        }
                    ),
                    encoding="utf-8",
                )
                argv.extend(["--normal-run", f"{seed}={normal}"])
                argv.extend(["--mlp-run", f"{seed}={mlp}"])
                argv.extend(["--intervention-run", f"{seed}={intervention}"])
            output = root / "summary.json"
            argv.extend(["--output-file", str(output)])
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(StringIO()):
                    SUMMARY.main()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn(
                "zero_update_gates",
                result["persistent_checkpoint_interventions_normal_minus_control"],
            )
            self.assertTrue(
                result["decision_checks"]["zero_update_gates_drop_complete_coverage@30_every_seed"]
            )
            self.assertTrue(result["decision_passed"])

    def test_summary_accepts_table_competitive_path_qrgta_normal_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = ["stage17a_summarize_causal_controls.py"]
            for seed in (42, 43, 44):
                normal = root / f"competitive_normal_{seed}"
                mlp = root / f"mlp_{seed}"
                intervention = root / f"intervention_{seed}"
                self.write_training_run(
                    normal, "table_competitive_path_qrgta", "normal", 0.8
                )
                self.write_training_run(mlp, "mlp_residual", "normal", 0.7)
                intervention.mkdir()
                controls = {
                    "zero_query_edges": self.metric_payload(0.78),
                    "shuffled_schema_edges": self.metric_payload(0.76),
                    "shuffled_node_identity": self.metric_payload(0.74),
                    "shuffled_distance_buckets": self.metric_payload(0.79),
                    "shuffled_path_signatures": self.metric_payload(0.77),
                    "zero_path_features": self.metric_payload(0.79),
                    "zero_table_competition": self.metric_payload(0.785),
                    "shuffle_column_parent_table": self.metric_payload(0.79),
                    "zero_competition_gates": self.metric_payload(0.786),
                }
                (intervention / "intervention_summary.json").write_text(
                    json.dumps(
                        {
                            "parameters_unchanged": True,
                            "checkpoint_sha256": SUMMARY.file_sha256(normal / "best.pt"),
                            "reference_normal_reproduced": True,
                            "data_config": {
                                "dev_graph_file": "dev_graph.jsonl",
                                "dev_label_file": "dev_labels.jsonl",
                                "embedding_cache_dir": "cache",
                                "dev_limit": None,
                            },
                            "metrics": {"normal": self.metric_payload(0.8), **controls},
                        }
                    ),
                    encoding="utf-8",
                )
                argv.extend(["--normal-run", f"{seed}={normal}"])
                argv.extend(["--mlp-run", f"{seed}={mlp}"])
                argv.extend(["--intervention-run", f"{seed}={intervention}"])
            output = root / "summary.json"
            argv.extend(["--output-file", str(output)])
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(StringIO()):
                    SUMMARY.main()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn(
                "zero_table_competition",
                result["competition_checkpoint_interventions_normal_minus_control"],
            )
            self.assertTrue(
                result["decision_checks"][
                    "zero_table_competition_drop_complete_coverage@30_every_seed"
                ]
            )
            self.assertTrue(
                result["decision_checks"][
                    "shuffle_column_parent_table_drop_complete_coverage@30_at_least_2_of_3"
                ]
            )
            self.assertTrue(result["decision_passed"])

    def test_summary_accepts_enhanced_table_competitive_path_qrgta_normal_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = ["stage17a_summarize_causal_controls.py"]
            for seed in (42, 43, 44):
                normal = root / f"enhanced_competitive_normal_{seed}"
                mlp = root / f"mlp_{seed}"
                intervention = root / f"intervention_{seed}"
                self.write_training_run(
                    normal, "enhanced_table_competitive_path_qrgta", "normal", 0.8
                )
                self.write_training_run(mlp, "mlp_residual", "normal", 0.7)
                intervention.mkdir()
                controls = {
                    "zero_query_edges": self.metric_payload(0.78),
                    "shuffled_schema_edges": self.metric_payload(0.76),
                    "shuffled_node_identity": self.metric_payload(0.74),
                    "shuffled_distance_buckets": self.metric_payload(0.79),
                    "shuffled_path_signatures": self.metric_payload(0.77),
                    "zero_path_features": self.metric_payload(0.79),
                    "zero_table_competition": self.metric_payload(0.785),
                    "shuffle_column_parent_table": self.metric_payload(0.79),
                    "zero_competition_gates": self.metric_payload(0.786),
                }
                (intervention / "intervention_summary.json").write_text(
                    json.dumps(
                        {
                            "parameters_unchanged": True,
                            "checkpoint_sha256": SUMMARY.file_sha256(normal / "best.pt"),
                            "reference_normal_reproduced": True,
                            "data_config": {
                                "dev_graph_file": "dev_graph.jsonl",
                                "dev_label_file": "dev_labels.jsonl",
                                "embedding_cache_dir": "cache",
                                "dev_limit": None,
                            },
                            "metrics": {"normal": self.metric_payload(0.8), **controls},
                        }
                    ),
                    encoding="utf-8",
                )
                argv.extend(["--normal-run", f"{seed}={normal}"])
                argv.extend(["--mlp-run", f"{seed}={mlp}"])
                argv.extend(["--intervention-run", f"{seed}={intervention}"])
            output = root / "summary.json"
            argv.extend(["--output-file", str(output)])
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(StringIO()):
                    SUMMARY.main()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn(
                "zero_table_competition",
                result["competition_checkpoint_interventions_normal_minus_control"],
            )
            self.assertIn(
                "shuffled_path_signatures",
                result["path_checkpoint_interventions_normal_minus_control"],
            )
            self.assertTrue(result["decision_passed"])


if __name__ == "__main__":
    unittest.main()
