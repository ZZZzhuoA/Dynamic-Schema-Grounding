import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DATA = load_module(
    "stage11_build_dynamic_grounding_trajectories",
    "src/data/stage11_build_dynamic_grounding_trajectories.py",
)
TRAINING = None


class Stage11TrajectoryTest(unittest.TestCase):
    def test_top_level_events_ignore_nested_query_clauses(self):
        sql = (
            "select name from users where id in "
            "(select user_id from orders where amount > 10) order by name"
        )
        events = DATA.clause_keyword_events(sql)
        self.assertEqual(
            [event["clause"] for event in events],
            ["select", "join", "where", "order_by"],
        )

    def test_trajectory_prefix_is_causal_and_targets_current_clause(self):
        graph = {
            "record_index": 0,
            "candidate_nodes": [
                {"local_id": 0, "schema_item_id": 10},
                {"local_id": 1, "schema_item_id": 11},
                {"local_id": 2, "schema_item_id": 12},
            ],
        }
        labels = {
            "sql": "select movie_title from movies where release_year = 1945",
            "clause_labels": {
                "select": [10], "join": [11], "where": [12],
                "group_by": [], "having": [], "order_by": [],
            },
        }
        row = DATA.build_trajectory(graph, labels)
        self.assertEqual([step["partial_sql"] for step in row["trajectory_steps"]], [
            "select", "select movie_title from", "select movie_title from movies where",
        ])
        self.assertEqual(
            [step["target_local_ids"] for step in row["trajectory_steps"]],
            [[0], [1], [2]],
        )
        self.assertEqual(
            [step["observed_local_ids"] for step in row["trajectory_steps"]],
            [[], [0], [0, 1]],
        )
        self.assertNotIn("release_year", row["trajectory_steps"][-1]["partial_sql"])


class Stage11ModelTest(unittest.TestCase):
    def test_trajectory_validation_skips_only_empty_target_free_graphs(self):
        try:
            from src.training.stage11_train_dynamic_grounding_controller import (
                validate_and_filter_trajectories,
            )
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        empty = {
            "record_index": 0,
            "candidate_nodes": [],
            "trajectory_steps": [{"target_schema_ids": []}],
        }
        valid = {
            "record_index": 1,
            "candidate_nodes": [
                {"numeric_features": [1.0, 0.0]},
            ],
            "trajectory_steps": [
                {
                    "target_schema_ids": [3],
                    "target_local_ids": [0],
                    "observed_local_ids": [],
                }
            ],
        }
        usable, diagnostics = validate_and_filter_trajectories(
            [empty, valid], "train"
        )
        self.assertEqual([row["record_index"] for row in usable], [1])
        self.assertEqual(diagnostics["skipped_count"], 1)
        self.assertEqual(diagnostics["numeric_dim"], 2)

    def test_trajectory_validation_rejects_empty_graph_with_targets(self):
        try:
            from src.training.stage11_train_dynamic_grounding_controller import (
                validate_and_filter_trajectories,
            )
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        with self.assertRaisesRegex(ValueError, "trajectory supervision"):
            validate_and_filter_trajectories(
                [
                    {
                        "record_index": 7,
                        "candidate_nodes": [],
                        "trajectory_steps": [{"target_schema_ids": [4]}],
                    }
                ],
                "train",
            )

    def test_recurrent_controller_changes_belief_and_backpropagates(self):
        try:
            import torch
            from src.modeling.dynamic_grounding_controller import (
                DynamicSchemaGroundingController,
                partial_sql_features,
            )
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        torch.manual_seed(7)
        model = DynamicSchemaGroundingController(
            dense_dim=8, numeric_dim=4, hidden_dim=16,
            relation_count=2, num_layers=2, dropout=0.0,
        )
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        edge_type = torch.tensor([0, 0, 1, 1])
        steps = [
            {
                "operation_id": torch.tensor([model.operation_to_id["PROJECT"]]),
                "sql_features": torch.tensor(partial_sql_features("select")),
                "observed_mask": torch.tensor([0.0, 0.0, 0.0, 0.0]),
            },
            {
                "operation_id": torch.tensor([model.operation_to_id["FILTER"]]),
                "sql_features": torch.tensor(partial_sql_features("select x from t where")),
                "observed_mask": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            },
        ]
        outputs = model.forward_trajectory(
            torch.randn(4, 8), torch.randn(4, 4), torch.randn(1, 8),
            (edge_index, edge_type), steps,
        )
        self.assertEqual(tuple(outputs[0]["grounding_tokens"].shape), (4, 16))
        self.assertEqual(tuple(outputs[0]["steering_state"].shape), (16,))
        self.assertGreater(
            float(torch.abs(outputs[1]["belief"] - outputs[0]["belief"]).sum().detach()),
            0.0,
        )
        (outputs[0]["logits"].sum() + outputs[1]["logits"].sum()).backward()
        self.assertIsNotNone(model.transition.weight_hh.grad)
        self.assertGreater(float(model.transition.weight_hh.grad.abs().sum()), 0.0)

    def test_llm_bridge_cross_attention_and_steering_shapes(self):
        try:
            import torch
            from src.modeling.dynamic_grounding_controller import GroundingLLMBridge
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        bridge = GroundingLLMBridge(grounding_dim=16, llm_dim=32)
        output = bridge(
            torch.randn(2, 5, 32), torch.randn(2, 7, 16), torch.randn(2, 1, 16)
        )
        self.assertEqual(tuple(output["hidden_states"].shape), (2, 5, 32))
        self.assertEqual(tuple(output["cross_attention"].shape), (2, 5, 7))
        self.assertEqual(tuple(output["gate"].shape), (2, 5, 32))
        unbatched_steering = bridge(
            torch.randn(5, 32), torch.randn(7, 16), torch.randn(16)
        )
        self.assertEqual(tuple(unbatched_steering["hidden_states"].shape), (5, 32))

    def test_uncertainty_residual_history_is_gated_and_backpropagates(self):
        try:
            import torch
            from src.modeling.dynamic_grounding_controller import (
                DynamicSchemaGroundingController,
                partial_sql_features,
            )
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        torch.manual_seed(11)
        model = DynamicSchemaGroundingController(
            dense_dim=8, numeric_dim=4, hidden_dim=16, relation_count=2,
            num_layers=1, dropout=0.0, history_mode="uncertainty_residual",
        )
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        edge_type = torch.tensor([0, 0, 1, 1])
        steps = [
            {
                "operation_id": torch.tensor([model.operation_to_id[operation]]),
                "sql_features": torch.tensor(partial_sql_features(sql)),
                "observed_mask": observed,
            }
            for operation, sql, observed in [
                ("PROJECT", "select", torch.zeros(4)),
                ("FILTER", "select x from t where", torch.tensor([1., 0., 0., 0.])),
            ]
        ]
        outputs = model.forward_trajectory(
            torch.randn(4, 8), torch.randn(4, 4), torch.randn(1, 8),
            (edge_index, edge_type), steps,
        )
        self.assertEqual(float(outputs[0]["history_gate"]), 0.0)
        history_gate = float(outputs[1]["history_gate"].detach())
        self.assertGreaterEqual(history_gate, 0.0)
        self.assertLessEqual(history_gate, 1.0)
        self.assertTrue(torch.isfinite(outputs[1]["provisional_entropy"]))
        outputs[1]["logits"].sum().backward()
        grad = model.history_delta[0].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_history_fusion_has_exact_independent_identity_at_zero_gate(self):
        try:
            import torch
            from src.modeling.dynamic_grounding_controller import (
                DynamicSchemaGroundingController,
            )
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        base = torch.randn(7)
        candidate = torch.randn(7)
        fused = DynamicSchemaGroundingController.mix_history(
            base, candidate, torch.tensor(0.0)
        )
        self.assertTrue(torch.equal(fused, base))
        full = DynamicSchemaGroundingController.mix_history(
            base, candidate, torch.tensor(1.0)
        )
        self.assertTrue(torch.allclose(full, candidate))

    def test_counterfactual_utility_is_positive_only_for_improvement(self):
        try:
            import torch
            from src.training.stage11_train_dynamic_grounding_controller import (
                counterfactual_utility,
            )
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        useful = counterfactual_utility(torch.tensor(0.5), torch.tensor(0.4), 0.05)
        harmful = counterfactual_utility(torch.tensor(0.4), torch.tensor(0.5), 0.05)
        tied = counterfactual_utility(torch.tensor(0.4), torch.tensor(0.4), 0.05)
        self.assertGreater(float(useful), 0.0)
        self.assertEqual(float(harmful), 0.0)
        self.assertEqual(float(tied), 0.0)

    def test_selection_regret_utility_tracks_topk_and_mrr_improvement(self):
        try:
            import torch
            from src.training.stage11_train_dynamic_grounding_controller import (
                selection_regret_utility,
            )
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
        base = torch.tensor([0.0, 2.0, 1.0, -1.0])
        candidate = torch.tensor([3.0, 2.0, 1.0, -1.0])
        useful, base_quality, candidate_quality = selection_regret_utility(
            base, candidate, labels, top_k=2, mrr_weight=0.1, temperature=0.05
        )
        harmful, _, _ = selection_regret_utility(
            candidate, base, labels, top_k=2, mrr_weight=0.1, temperature=0.05
        )
        self.assertGreater(float(candidate_quality), float(base_quality))
        self.assertGreater(float(useful), 0.0)
        self.assertEqual(float(harmful), 0.0)

    def test_straight_through_rejection_recovers_base_logits(self):
        try:
            import torch
            from src.modeling.dynamic_grounding_controller import (
                DynamicSchemaGroundingController,
                partial_sql_features,
            )
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        model = DynamicSchemaGroundingController(
            dense_dim=8, numeric_dim=4, hidden_dim=16, relation_count=1,
            num_layers=1, dropout=0.0, history_mode="uncertainty_residual",
            history_gate_policy="straight_through", history_gate_threshold=1.1,
        )
        model.eval()
        steps = [
            {
                "operation_id": torch.tensor([model.operation_to_id[operation]]),
                "sql_features": torch.tensor(partial_sql_features(sql)),
                "observed_mask": torch.zeros(3),
            }
            for operation, sql in [("PROJECT", "select"), ("FILTER", "select x where")]
        ]
        outputs = model.forward_trajectory(
            torch.randn(3, 8), torch.randn(3, 4), torch.randn(1, 8),
            (torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.long)),
            steps,
        )
        self.assertEqual(float(outputs[1]["history_gate"]), 0.0)
        self.assertTrue(torch.equal(outputs[1]["logits"], outputs[1]["provisional_logits"]))

    def test_safe_history_tuning_freezes_independent_path(self):
        try:
            from src.modeling.dynamic_grounding_controller import DynamicSchemaGroundingController
            from src.training.stage11_train_dynamic_grounding_controller import (
                configure_safe_history_tuning,
                initialize_rejecting_gate,
            )
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        model = DynamicSchemaGroundingController(
            dense_dim=8, numeric_dim=4, hidden_dim=16, relation_count=1,
            history_mode="uncertainty_residual",
        )
        trainable = configure_safe_history_tuning(model, freeze_base_controller=True)
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith(("history_delta.", "history_relevance.", "residual_norm.")) for name in trainable))
        self.assertFalse(model.node_input.weight.requires_grad)
        self.assertTrue(model.history_delta[0].weight.requires_grad)
        initialize_rejecting_gate(model)
        self.assertEqual(
            float(model.history_relevance[-1].weight.detach().abs().sum()), 0.0
        )
        self.assertEqual(float(model.history_relevance[-1].bias.detach()), -4.0)


if __name__ == "__main__":
    unittest.main()
