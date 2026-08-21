import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "stage10g_table_conditioned_completion",
        ROOT / "src/data/stage10g_table_conditioned_completion.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE10G = load_module()


class Stage10GCompletionTest(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is unavailable")
        self.np = np
        self.full_graph = {
            "inference_inputs": {
                "db_id": "demo",
                "schema_nodes": [
                    {"id": 0, "type": "table", "name": "schools"},
                    {"id": 1, "type": "column", "name": "schools.name", "table": "schools"},
                    {"id": 2, "type": "column", "name": "schools.county", "table": "schools"},
                    {"id": 3, "type": "table", "name": "districts"},
                    {"id": 4, "type": "column", "name": "districts.code", "table": "districts"},
                ],
                "schema_edges": [
                    {"src": 0, "dst": 1, "type": "table_to_column"},
                    {"src": 1, "dst": 0, "type": "column_to_table"},
                    {"src": 0, "dst": 2, "type": "table_to_column"},
                    {"src": 2, "dst": 0, "type": "column_to_table"},
                    {"src": 3, "dst": 4, "type": "table_to_column"},
                ],
            }
        }
        self.factor = {
            "record_index": 0,
            "db_id": "demo",
            "candidate_nodes": [
                {
                    "local_id": 0,
                    "schema_item_id": 0,
                    "schema_position": 0,
                    "name": "schools",
                    "type": "table",
                    "owner_table_id": 0,
                    "owner_local_id": 0,
                    "numeric_features": [1.0, 0.0, 1.0],
                    "priority": 3.0,
                },
                {
                    "local_id": 1,
                    "schema_item_id": 1,
                    "schema_position": 1,
                    "name": "schools.name",
                    "type": "column",
                    "owner_table_id": 0,
                    "owner_local_id": 0,
                    "numeric_features": [0.0, 1.0, 1.0],
                    "priority": 2.0,
                },
            ],
            "schema_edges": [],
            "factors": [],
            "factor_edges": [],
            "baseline_selected_ids": [0, 1],
            "gold_ids": [0, 2],
            "gold_table_ids": [0],
            "gold_column_ids": [2],
            "candidate_oracle_recall": 0.5,
        }
        self.embeddings = self.np.asarray(
            [[0.8, 0.2], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.1]],
            dtype=self.np.float32,
        )

    def test_recovers_missing_column_only_from_selected_table(self):
        result = STAGE10G.augment_factor_graph(
            self.factor,
            self.full_graph,
            {"relation_labels": {"PREDICATE_COLUMN": [2]}},
            self.np.asarray([1.0, 0.0], dtype=self.np.float32),
            self.embeddings,
            self.np,
            ["PREDICATE_COLUMN"],
            max_anchor_tables=1,
            columns_per_table=2,
            max_additions=2,
        )
        ids = [node["schema_item_id"] for node in result["candidate_nodes"]]
        self.assertEqual(ids, [0, 1, 2])
        self.assertNotIn(4, ids)
        self.assertEqual(result["candidate_oracle_recall"], 1.0)
        self.assertEqual(result["whole_labels"], [1.0, 0.0, 1.0])
        self.assertEqual(result["role_labels"][-1], [1.0])
        self.assertEqual(len(result["candidate_nodes"][0]["numeric_features"]), 7)
        self.assertEqual(result["candidate_nodes"][-1]["numeric_features"][-4], 1.0)
        self.assertIn(
            {"src": 0, "dst": 2, "type": "table_to_column"},
            result["schema_edges"],
        )

    def test_gold_does_not_affect_candidate_generation(self):
        kwargs = dict(
            full_graph=self.full_graph,
            relation_record={"relation_labels": {}},
            query_embedding=self.np.asarray([1.0, 0.0], dtype=self.np.float32),
            node_embeddings=self.embeddings,
            np=self.np,
            relation_types=["PREDICATE_COLUMN"],
            max_anchor_tables=1,
            columns_per_table=2,
            max_additions=2,
        )
        first = STAGE10G.augment_factor_graph(self.factor, **kwargs)
        changed_gold = dict(self.factor)
        changed_gold["gold_ids"] = [4]
        second = STAGE10G.augment_factor_graph(changed_gold, **kwargs)
        first_ids = [node["schema_item_id"] for node in first["candidate_nodes"]]
        second_ids = [node["schema_item_id"] for node in second["candidate_nodes"]]
        self.assertEqual(first_ids, second_ids)


if __name__ == "__main__":
    unittest.main()
