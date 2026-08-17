import unittest


class Stage13BDataTest(unittest.TestCase):
    def test_clean_filter_and_stop_action_remove_gold_sql(self):
        from src.data.stage13b_prepare_typed_trajectories import exclusion_reasons, prepare_row

        row = {
            "split": "train", "record_index": 7, "question_id": None, "db_id": "x",
            "inference_inputs": {
                "question": "names", "evidence": "", "schema_items": [
                    {"id": 0, "type": "table", "name": "users"},
                    {"id": 1, "type": "column", "name": "users.name"},
                ],
                "schema_edges": [{"src": 0, "dst": 0, "type": "self_loop"}],
            },
            "training_targets": {
                "gold_sql": "SELECT name FROM users",
                "relational_algebra": {"version": "typed_ra_v1", "root_node_id": "project_1"},
                "action_sequence": [
                    {"action": "SCAN", "table_pointer_ids": [0], "column_pointer_ids": [],
                     "operator_targets": [], "value_targets": [], "input_node_ids": []},
                    {"action": "PROJECT", "table_pointer_ids": [], "column_pointer_ids": [1],
                     "operator_targets": [], "value_targets": [], "input_node_ids": ["scan_0"]},
                ],
                "join_path": {"table_pointer_ids": [0], "connected": True, "edge_targets": []},
            },
            "audit": {"parse_status": "supported_flat", "schema_label_coverage": 1.0},
        }
        self.assertEqual(exclusion_reasons(row), [])
        prepared = prepare_row(row)
        self.assertNotIn("gold_sql", str(prepared))
        self.assertEqual([step["action"] for step in prepared["teacher_steps"]],
                         ["SCAN", "PROJECT", "STOP"])

    def test_partial_or_incomplete_records_are_excluded(self):
        from src.data.stage13b_prepare_typed_trajectories import exclusion_reasons

        row = {
            "inference_inputs": {"schema_items": [{}], "schema_edges": [{}]},
            "training_targets": {"action_sequence": [{}], "join_path": {}},
            "audit": {"parse_status": "partial_nested", "schema_label_coverage": 0.8},
        }
        reasons = exclusion_reasons(row)
        self.assertIn("partial_nested", reasons)
        self.assertIn("incomplete_schema_assignment", reasons)


class Stage13BModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            cls.torch_available = True
        except ImportError:
            cls.torch_available = False

    def test_forward_backward_and_typed_masks(self):
        if not self.torch_available:
            self.skipTest("PyTorch is unavailable")
        import torch
        from src.modeling.typed_ra_decoder import TypedRAPointerDecoder

        model = TypedRAPointerDecoder(
            dense_dim=8, hidden_dim=16, relation_count=3, num_layers=1, dropout=0.0
        )
        dense = torch.randn(4, 8)
        query = torch.randn(1, 8)
        node_types = torch.tensor([0, 0, 1, 1])
        edge_index = torch.tensor([[0, 0, 1, 2, 3], [0, 2, 3, 2, 3]])
        edge_type = torch.tensor([0, 1, 1, 0, 0])
        join_edges = torch.tensor([[2], [3]])
        steps = [
            {"action": "SCAN", "table_pointer_ids": [0], "column_pointer_ids": [],
             "value_routes": []},
            {"action": "FILTER", "table_pointer_ids": [], "column_pointer_ids": [2],
             "value_routes": ["question"], "operator_targets": ["="]},
            {"action": "STOP", "table_pointer_ids": [], "column_pointer_ids": [],
             "value_routes": []},
        ]
        outputs = model.forward_trajectory(
            dense, query, node_types, edge_index, edge_type, join_edges, steps
        )
        self.assertEqual(len(outputs), 3)
        self.assertTrue((outputs[0]["table_logits"][2:] < -1e20).all())
        self.assertTrue((outputs[0]["column_logits"][:2] < -1e20).all())
        loss = sum(output["action_logits"].sum() + output["column_logits"][2:].sum()
                   for output in outputs)
        loss.backward()
        self.assertIsNotNone(model.action_head[-1].weight.grad)

    def test_optional_llm_plan_hidden_interface(self):
        if not self.torch_available:
            self.skipTest("PyTorch is unavailable")
        import torch
        from src.modeling.typed_ra_decoder import TypedRAPointerDecoder

        model = TypedRAPointerDecoder(
            dense_dim=8, hidden_dim=16, relation_count=1, num_layers=1,
            dropout=0.0, plan_hidden_dim=6,
        )
        outputs = model.forward_trajectory(
            torch.randn(2, 8), torch.randn(1, 8), torch.tensor([0, 1]),
            torch.tensor([[0, 1], [0, 1]]), torch.tensor([0, 0]),
            torch.empty((2, 0), dtype=torch.long),
            [{"action": "SCAN", "table_pointer_ids": [0], "column_pointer_ids": [],
              "operator_targets": [], "value_routes": []}],
            plan_hidden_states=torch.randn(1, 6),
        )
        outputs[0]["action_logits"].sum().backward()
        self.assertIsNotNone(model.plan_hidden.weight.grad)

    def test_forced_action_uses_predicted_transition_without_teacher_targets(self):
        if not self.torch_available:
            self.skipTest("PyTorch is unavailable")
        import torch
        from src.modeling.typed_ra_decoder import TypedRAPointerDecoder

        model = TypedRAPointerDecoder(
            dense_dim=8, hidden_dim=16, relation_count=1, num_layers=1, dropout=0.0
        )
        dense = torch.randn(3, 8)
        query = torch.randn(1, 8)
        node_types = torch.tensor([0, 1, 1])
        edges = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]])
        edge_types = torch.zeros(4, dtype=torch.long)
        base_nodes, query_state, state = model.initialize(dense, query, node_types)
        output = model.step(
            base_nodes, query_state, state, node_types, edges, edge_types,
            torch.empty((2, 0), dtype=torch.long), forced_action="FILTER",
        )
        self.assertEqual(output["forced_action"], "FILTER")
        self.assertEqual(output["controller_state"].shape, state.shape)
        with self.assertRaises(ValueError):
            model.step(
                base_nodes, query_state, state, node_types, edges, edge_types,
                torch.empty((2, 0), dtype=torch.long),
                teacher_step={"action": "FILTER"}, forced_action="FILTER",
            )


if __name__ == "__main__":
    unittest.main()
