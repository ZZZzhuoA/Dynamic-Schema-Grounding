import importlib.util
import sqlite3
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


VALUE = load_module("stage9_value_index", "src/grounding/value_index.py")
COMPLETION = load_module(
    "stage9_value_join_completion",
    "src/grounding/stage9_value_join_completion.py",
)


class Stage9ValueIndexTest(unittest.TestCase):
    def test_value_index_maps_database_value_to_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "demo.sqlite"
            source = sqlite3.connect(source_path)
            source.execute('CREATE TABLE schools ("County Name" TEXT, "School" TEXT)')
            source.executemany(
                'INSERT INTO schools VALUES (?, ?)',
                [("Alameda", "A School"), ("Fresno", "B School")],
            )
            source.commit()
            source.close()

            index_path = root / "values.sqlite"
            index = sqlite3.connect(index_path)
            VALUE.initialize_index(index)
            nodes = [
                {
                    "id": 1,
                    "type": "column",
                    "table": "schools",
                    "column": "County Name",
                    "data_type": "text",
                },
                {
                    "id": 2,
                    "type": "column",
                    "table": "schools",
                    "column": "School",
                    "data_type": "text",
                },
            ]
            args = SimpleNamespace(
                include_numeric_values=False,
                max_values_per_column=100,
                max_value_chars=128,
            )
            VALUE.index_database(index, "demo", source_path, nodes, args)
            index.close()

            reader = VALUE.ValueIndex(index_path)
            try:
                matches = reader.query("demo", "schools in Alameda County")
            finally:
                reader.close()
            self.assertEqual(matches[0]["schema_item_id"], 1)
            self.assertEqual(matches[0]["matches"][0]["value"], "Alameda")
            self.assertEqual(matches[0]["score"], 1.0)


class Stage9JoinCompletionTest(unittest.TestCase):
    def test_value_matches_are_disambiguated_by_table_context(self):
        by_id = {
            0: {"id": 0, "type": "table", "name": "facts"},
            1: {"id": 1, "type": "table", "name": "dimension"},
            2: {"id": 2, "type": "column", "name": "facts.county", "table": "facts"},
            3: {"id": 3, "type": "column", "name": "dimension.county", "table": "dimension"},
            4: {"id": 4, "type": "column", "name": "facts.metric", "table": "facts"},
        }
        relation_rows = {
            "METRIC_TARGET": {"top_30": [{"id": 4, "score": 5.0}]},
            "PREDICATE_COLUMN": {
                "top_30": [{"id": 2, "score": 3.0}, {"id": 3, "score": 2.0}]
            },
        }
        matches = [
            {"schema_item_id": 3, "score": 1.0, "matches": []},
            {"schema_item_id": 2, "score": 1.0, "matches": []},
        ]
        args = SimpleNamespace(
            terminal_top_per_relation=3,
            value_table_context_weight=0.35,
            value_relation_context_weight=0.25,
        )
        ranked = COMPLETION.contextualize_value_matches(
            matches, relation_rows, by_id, args
        )
        self.assertEqual(ranked[0]["schema_item_id"], 2)

    def test_metric_closure_completes_intermediate_join_table(self):
        graph_example = {
            "inference_inputs": {
                "schema_nodes": [
                    {"id": 0, "type": "table", "name": "customers"},
                    {"id": 1, "type": "table", "name": "orders"},
                    {"id": 2, "type": "table", "name": "products"},
                    {"id": 3, "type": "column", "name": "customers.id", "table": "customers", "column": "id"},
                    {"id": 4, "type": "column", "name": "orders.customer_id", "table": "orders", "column": "customer_id"},
                    {"id": 5, "type": "column", "name": "orders.product_id", "table": "orders", "column": "product_id"},
                    {"id": 6, "type": "column", "name": "products.id", "table": "products", "column": "id"},
                ],
                "schema_edges": [
                    {"src": 3, "dst": 4, "type": "foreign_key_forward"},
                    {"src": 5, "dst": 6, "type": "foreign_key_forward"},
                ],
            }
        }
        relation_rows = {
            "OUTPUT_TARGET": {
                "top_30": [
                    {"id": 0, "score": 5.0},
                    {"id": 2, "score": 4.0},
                ]
            },
            "JOIN_BRIDGE": {"top_30": [{"id": 3, "score": 1.0}]},
        }
        args = SimpleNamespace(
            terminal_top_per_relation=3,
            max_terminal_tables=4,
            terminal_min_ratio=0.45,
            value_terminal_weight=1.0,
            join_support_weight=0.4,
        )
        candidates, debug = COMPLETION.complete_join_path(
            graph_example, relation_rows, [], args
        )
        candidate_ids = {item["schema_item_id"] for item in candidates}
        self.assertEqual(debug["terminal_tables"], ["customers", "products"])
        self.assertTrue({0, 1, 2, 3, 4, 5, 6}.issubset(candidate_ids))
        self.assertEqual(len(debug["paths"]), 1)


if __name__ == "__main__":
    unittest.main()
