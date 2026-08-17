import unittest


class Stage14BSemanticSlotDataTest(unittest.TestCase):
    def test_slot_text_uses_question_semantics_without_schema_answer(self):
        from src.data.stage14b_build_semantic_slots import build_row

        row = {
            "split": "dev", "record_index": 3, "question_id": 3, "db_id": "school",
            "inference_inputs": {
                "question": "List schools whose category is Continuation School.",
                "evidence": "", "schema_items": [], "schema_edges": [],
            },
            "teacher_steps": [
                {
                    "step_index": 0, "action": "FILTER", "table_pointer_ids": [],
                    "column_pointer_ids": [7], "join_edge_targets": [],
                    "operator_targets": ["="], "value_routes": ["question"],
                    "value_targets": [{
                        "kind": "text", "raw_sql_literal": "Continuation School",
                        "source": "question", "match": "exact", "start": 35, "end": 54,
                    }],
                }
            ],
        }
        result = build_row(row)
        request = result["inference_inputs"]["requests"][0]
        self.assertIn("Continuation School", request["slot_embedding_text"])
        self.assertNotIn("question:", request["slot_embedding_text"])
        self.assertEqual(request["value_embedding_text"], "literal values: Continuation School")
        self.assertNotIn("column_pointer_ids", str(result["inference_inputs"]))
        self.assertNotIn("Educational Option Type", request["slot_embedding_text"])
        self.assertEqual(result["training_targets"]["slot_targets"][0]["column_pointer_ids"], [7])

    def test_repeated_filters_have_value_specific_slot_text(self):
        from src.data.stage14b_build_semantic_slots import build_slot_request

        inputs = {"question": "Find charter schools in Alameda.", "evidence": ""}
        left = build_slot_request({
            "step_index": 0, "action": "FILTER", "column_pointer_ids": [1],
            "value_targets": [{"kind": "text", "raw_sql_literal": "charter", "source": "question"}],
        }, inputs)
        right = build_slot_request({
            "step_index": 1, "action": "FILTER", "column_pointer_ids": [2],
            "value_targets": [{"kind": "text", "raw_sql_literal": "Alameda", "source": "question"}],
        }, inputs)
        self.assertNotEqual(left["value_embedding_text"], right["value_embedding_text"])
        self.assertNotEqual(
            (left["slot_embedding_text"], left["value_embedding_text"]),
            (right["slot_embedding_text"], right["value_embedding_text"]),
        )


class Stage14BSemanticSlotModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            cls.torch_available = True
        except ImportError:
            cls.torch_available = False

    def test_slot_embedding_conditions_graph_and_pointer(self):
        if not self.torch_available:
            self.skipTest("PyTorch is unavailable")
        import torch
        from src.modeling.semantic_slot_binder import SemanticSlotGraphBinder, owner_table_indices

        torch.manual_seed(7)
        model = SemanticSlotGraphBinder(
            dense_dim=8, slot_dim=8, hidden_dim=16, relation_count=2,
            num_layers=1, dropout=0.0,
        )
        nodes = [
            {"id": 0, "type": "table", "name": "users"},
            {"id": 1, "type": "column", "name": "users.name", "table": "users"},
            {"id": 2, "type": "column", "name": "users.age", "table": "users"},
        ]
        dense = torch.randn(3, 8)
        query = torch.randn(1, 8)
        node_types = torch.tensor([0, 1, 1])
        edges = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]])
        edge_types = torch.tensor([0, 1, 0, 0])
        joins = torch.empty((2, 0), dtype=torch.long)
        owners = owner_table_indices(nodes, dense.device)
        left = model.forward_slot(
            dense, query, torch.randn(1, 8), node_types, edges, edge_types,
            joins, "FILTER", owners,
        )
        right = model.forward_slot(
            dense, query, torch.randn(1, 8), node_types, edges, edge_types,
            joins, "FILTER", owners,
        )
        self.assertFalse(torch.allclose(left["column_logits"][1:], right["column_logits"][1:]))
        left["column_logits"][1:].sum().backward()
        self.assertIsNotNone(model.slot_input.weight.grad)

    def test_stage13b_warm_start_is_shape_safe(self):
        if not self.torch_available:
            self.skipTest("PyTorch is unavailable")
        from src.modeling.semantic_slot_binder import SemanticSlotGraphBinder
        from src.modeling.typed_ra_decoder import TypedRAPointerDecoder

        base = TypedRAPointerDecoder(
            dense_dim=8, hidden_dim=16, relation_count=1, num_layers=1, dropout=0.0
        )
        model = SemanticSlotGraphBinder(
            dense_dim=8, slot_dim=8, hidden_dim=16, relation_count=1,
            num_layers=1, dropout=0.0,
        )
        report = model.load_stage13b_state(base.state_dict())
        self.assertGreater(report["loaded_parameter_count"], 20)
        self.assertIn("slot_input.weight", report["missing_keys"])

    def test_one_record_training_runtime(self):
        if not self.torch_available:
            self.skipTest("PyTorch is unavailable")
        from types import SimpleNamespace
        import numpy as np
        import torch
        import torch.nn.functional as F
        from src.modeling.semantic_slot_binder import SemanticSlotGraphBinder
        from src.training.stage14b_train_semantic_slot_binder import run_split

        graph = {
            "record_index": 0, "db_id": "x",
            "inference_inputs": {
                "schema_items": [
                    {"id": 0, "type": "table", "name": "users"},
                    {"id": 1, "type": "column", "name": "users.name", "table": "users"},
                    {"id": 2, "type": "column", "name": "users.age", "table": "users"},
                ],
                "schema_edges": [
                    {"src": 0, "dst": 0, "type": "self_loop"},
                    {"src": 1, "dst": 1, "type": "self_loop"},
                    {"src": 2, "dst": 2, "type": "self_loop"},
                    {"src": 0, "dst": 1, "type": "table_to_column"},
                    {"src": 0, "dst": 2, "type": "table_to_column"},
                ],
            },
        }
        slots = {
            "record_index": 0,
            "inference_inputs": {"requests": [
                {"step_index": 0, "action": "PROJECT"},
            ]},
            "training_targets": {"slot_targets": [
                {"column_pointer_ids": [1], "table_pointer_ids": [],
                 "join_edge_targets": [], "operator_targets": [], "value_routes": []},
            ]},
        }
        base_cache = {
            "node": np.random.randn(3, 8).astype("float32"),
            "query": np.random.randn(1, 8).astype("float32"),
            "by_index": {0: {"query_embedding_index": 0, "node_count": 3, "node_embedding_start": 0}},
        }
        slot_cache = {
            "focus": np.random.randn(1, 8).astype("float32"),
            "value": np.zeros((1, 8), dtype="float32"),
            "by_key": {(0, 0): {
                "embedding_index": 0, "focus_embedding_index": 0,
                "value_embedding_index": 0, "has_value": False,
            }},
            "by_action": {"PROJECT": [(0, 0)]},
        }
        args = SimpleNamespace(
            pointer_pos_weight=5.0, pointer_loss_weight=1.0,
            listwise_weight=0.3, owner_consistency_weight=0.2,
            hard_negative_weight=0.2, hard_negative_margin=0.5,
            contrastive_weight=0.3, contrastive_temperature=0.1,
            semantic_dropout=0.2,
            join_loss_weight=1.0, value_route_loss_weight=0.5,
            operator_loss_weight=0.75, table_top_k=2, column_top_k=2,
            join_top_k=1, max_grad_norm=1.0,
        )
        model = SemanticSlotGraphBinder(
            dense_dim=8, slot_dim=8, hidden_dim=16, relation_count=2,
            num_layers=1, dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics, _ = run_split(
            model, [(graph, slots)], base_cache, slot_cache,
            {"self_loop": 0, "table_to_column": 1}, args,
            {"torch": torch, "F": F}, torch.device("cpu"), optimizer,
        )
        self.assertGreater(metrics["loss"], 0.0)
        self.assertEqual(metrics["record_count"], 1)

    def test_backbone_freeze_keeps_only_slot_interface_trainable(self):
        if not self.torch_available:
            self.skipTest("PyTorch is unavailable")
        from src.modeling.semantic_slot_binder import SemanticSlotGraphBinder
        from src.training.stage14b_train_semantic_slot_binder import freeze_pretrained_backbone

        model = SemanticSlotGraphBinder(
            dense_dim=8, slot_dim=8, hidden_dim=16, relation_count=1,
            num_layers=1, dropout=0.0,
        )
        trainable = freeze_pretrained_backbone(model)
        self.assertIn("slot_input.weight", trainable)
        self.assertIn("semantic_scale", trainable)
        self.assertFalse(model.graph_layers[0].q.weight.requires_grad)
        self.assertFalse(model.column_key.weight.requires_grad)
        self.assertTrue(model.value_input.weight.requires_grad)

    def test_same_action_shuffle_uses_another_record(self):
        from src.training.stage14b_train_semantic_slot_binder import same_action_donor_key

        cache = {
            "by_action": {
                "FILTER": [(0, 1), (0, 2), (4, 1), (9, 3)],
            }
        }
        self.assertEqual(same_action_donor_key(cache, 0, 1, "FILTER"), (4, 1))


if __name__ == "__main__":
    unittest.main()
