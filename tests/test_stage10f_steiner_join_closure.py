import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLOSURE = load_module(
    "stage10f_steiner_join_closure",
    "src/grounding/stage10f_steiner_join_closure.py",
)
EVALUATION = load_module(
    "stage10f_evaluate_join_closure",
    "src/evaluation/stage10f_evaluate_join_closure.py",
)


def full_graph(include_second_fk=True):
    nodes = [
        {"id": 0, "type": "table", "name": "A"},
        {"id": 1, "type": "table", "name": "B"},
        {"id": 2, "type": "table", "name": "C"},
        {"id": 3, "type": "column", "name": "A.b_id", "table": "A", "column": "b_id"},
        {"id": 4, "type": "column", "name": "B.id", "table": "B", "column": "id"},
        {"id": 5, "type": "column", "name": "B.c_id", "table": "B", "column": "c_id"},
        {"id": 6, "type": "column", "name": "C.id", "table": "C", "column": "id"},
        {"id": 7, "type": "column", "name": "A.value", "table": "A", "column": "value"},
        {"id": 8, "type": "column", "name": "C.name", "table": "C", "column": "name"},
    ]
    edges = []
    for node in nodes:
        if node["type"] != "column":
            continue
        table_id = {"A": 0, "B": 1, "C": 2}[node["table"]]
        edges.extend(
            [
                {"src": table_id, "dst": node["id"], "type": "table_to_column"},
                {"src": node["id"], "dst": table_id, "type": "column_to_table"},
            ]
        )
    edges.extend(
        [
            {"src": 3, "dst": 4, "type": "foreign_key_forward"},
            {"src": 4, "dst": 3, "type": "foreign_key_backward"},
        ]
    )
    if include_second_fk:
        edges.extend(
            [
                {"src": 5, "dst": 6, "type": "foreign_key_forward"},
                {"src": 6, "dst": 5, "type": "foreign_key_backward"},
            ]
        )
    return {
        "record_index": 0,
        "inference_inputs": {
            "db_id": "demo",
            "question": "List A values with C names",
            "schema_nodes": nodes,
            "schema_edges": edges,
        },
    }


def prediction():
    return {
        "record_index": 0,
        "db_id": "demo",
        "question": "List A values with C names",
        "top_30": [
            {"schema_item_id": 0, "score": 1.0},
            {"schema_item_id": 2, "score": 0.9},
            {"schema_item_id": 7, "score": 0.8},
            {"schema_item_id": 8, "score": 0.7},
        ],
    }


def prior():
    return {
        "record_index": 0,
        "status": "ok",
        "node_priors": [
            {"schema_item_id": 7, "role_scores": {"OUTPUT_TARGET": 0.9}},
            {"schema_item_id": 8, "role_scores": {"OUTPUT_TARGET": 0.8}},
        ],
    }


class Stage10FSteinerJoinClosureTest(unittest.TestCase):
    def test_closure_adds_intermediate_table_and_fk_endpoints_outside_core(self):
        result = CLOSURE.complete_one(full_graph(), prediction(), prior())
        self.assertEqual(result["status"], "connected")
        self.assertEqual(set(result["semantic_core_ids"]), {0, 2, 7, 8})
        self.assertTrue({1, 3, 4, 5, 6}.issubset(result["structural_closure_ids"]))
        self.assertEqual(result["semantic_core_count"], 4)
        self.assertGreater(result["grounded_schema_count"], result["semantic_core_count"])
        self.assertEqual(result["terminal_components"], [["A", "C"]])

    def test_join_bridge_only_prior_does_not_create_semantic_terminal(self):
        bridge_prior = {
            "record_index": 0,
            "status": "ok",
            "node_priors": [
                {"schema_item_id": 3, "role_scores": {"JOIN_BRIDGE": 1.0}},
                {"schema_item_id": 6, "role_scores": {"JOIN_BRIDGE": 1.0}},
            ],
        }
        result = CLOSURE.complete_one(full_graph(), prediction(), bridge_prior)
        self.assertEqual(result["status"], "insufficient_semantic_terminals")
        self.assertEqual(result["structural_closure_ids"], [])

    def test_disconnected_declared_fk_graph_is_reported_without_inventing_edge(self):
        result = CLOSURE.complete_one(
            full_graph(include_second_fk=False), prediction(), prior()
        )
        self.assertEqual(result["status"], "declared_fk_disconnected")
        self.assertEqual(len(result["terminal_components"]), 2)
        self.assertNotIn(6, result["structural_closure_ids"])

    def test_full_graph_connectivity_requires_selected_path_nodes(self):
        graph = full_graph()
        core = {0, 2, 7, 8}
        closed = core | {1, 3, 4, 5, 6}
        self.assertFalse(
            EVALUATION.full_graph_connected(graph, core, required_tables={0, 2})
        )
        self.assertTrue(
            EVALUATION.full_graph_connected(graph, closed, required_tables={0, 2})
        )


if __name__ == "__main__":
    unittest.main()
