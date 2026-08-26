import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


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


if __name__ == "__main__":
    unittest.main()
