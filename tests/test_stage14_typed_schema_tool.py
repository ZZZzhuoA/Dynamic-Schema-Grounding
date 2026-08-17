import unittest


class Stage14TypedSchemaToolTest(unittest.TestCase):
    def test_oracle_trajectory_is_reduced_to_action_skeleton(self):
        from src.grounding.stage14_typed_schema_tool import normalize_requests

        source, requests = normalize_requests(
            {
                "teacher_steps": [
                    {
                        "action": "FILTER",
                        "column_pointer_ids": [99],
                        "operator_targets": ["="],
                    },
                    {"action": "STOP"},
                ]
            }
        )
        self.assertEqual(source, "oracle_action_skeleton_diagnostic")
        self.assertEqual([row["action"] for row in requests], ["FILTER", "STOP"])
        self.assertNotIn("column_pointer_ids", requests[0])

    def test_constraint_assembly_adds_owner_and_fk_path(self):
        from src.grounding.stage14_typed_schema_tool import assemble_tool_result

        nodes = [
            {"id": 0, "type": "table", "name": "users"},
            {"id": 1, "type": "table", "name": "orders"},
            {"id": 2, "type": "column", "name": "users.id", "table": "users"},
            {"id": 3, "type": "column", "name": "users.name", "table": "users"},
            {"id": 4, "type": "column", "name": "orders.user_id", "table": "orders"},
        ]
        edges = [
            {"src": 2, "dst": 4, "type": "foreign_key_forward"},
            {"src": 4, "dst": 2, "type": "foreign_key_backward"},
        ]
        steps = [
            {
                "request": {
                    "request_id": "project", "action": "PROJECT", "cardinality": 1,
                    "value_surface": None,
                },
                "table_candidates": [],
                "column_candidates": [
                    {"schema_id": 3, "type": "column", "name": "users.name",
                     "table": "users", "confidence": 0.9}
                ],
                "join_edge_candidates": [],
            },
            {
                "request": {
                    "request_id": "filter", "action": "FILTER", "cardinality": 1,
                    "value_surface": "Alice",
                },
                "table_candidates": [],
                "column_candidates": [
                    {"schema_id": 4, "type": "column", "name": "orders.user_id",
                     "table": "orders", "confidence": 0.8}
                ],
                "join_edge_candidates": [
                    {"left_schema_id": 2, "right_schema_id": 4, "confidence": 0.95}
                ],
            },
        ]
        assembled = assemble_tool_result(nodes, edges, steps, max_schema_items=10)
        selected = {row["schema_id"] for row in assembled["selected_schema"]}
        self.assertTrue({0, 1, 2, 3, 4}.issubset(selected))
        self.assertTrue(assembled["connected"])
        self.assertTrue(assembled["budget_feasible"])
        self.assertEqual(assembled["literal_surfaces"], ["Alice"])
        self.assertEqual(len(assembled["join_paths"]), 1)

    def test_literal_surface_is_not_normalized(self):
        from src.grounding.stage14_typed_schema_tool import assemble_tool_result

        value = "San Joaquin / 00D4"
        assembled = assemble_tool_result(
            [{"id": 0, "type": "table", "name": "x"}],
            [],
            [{
                "request": {
                    "request_id": "filter", "action": "FILTER", "cardinality": 1,
                    "value_surface": value,
                },
                "table_candidates": [], "column_candidates": [],
                "join_edge_candidates": [],
            }],
        )
        self.assertEqual(assembled["literal_surfaces"], [value])

    def test_diagnostic_evaluator_separates_pointer_and_assembly_metrics(self):
        from src.evaluation.stage14_evaluate_typed_schema_tool import evaluate

        target = [{
            "record_index": 1,
            "teacher_steps": [{
                "action": "PROJECT", "table_pointer_ids": [],
                "column_pointer_ids": [3], "join_edge_targets": [],
                "operator_targets": [], "value_routes": [],
            }],
        }]
        tool = [{
            "record_index": 1, "db_id": "x",
            "tool_steps": [{
                "table_candidates": [],
                "column_candidates": [{"schema_id": 3}],
                "join_edge_candidates": [], "operator_candidates": [],
                "value_route_candidates": [],
            }],
            "assembly": {"selected_schema": [{"schema_id": 3}]},
        }]
        metrics, missing = evaluate(tool, target)
        self.assertEqual(metrics["column_recall"], 1.0)
        self.assertEqual(metrics["assembled_complete_coverage"], 1.0)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
