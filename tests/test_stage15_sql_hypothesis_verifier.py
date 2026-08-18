import unittest


def toy_trajectory():
    return {
        "split": "train",
        "record_index": 3,
        "question_id": 3,
        "db_id": "toy",
        "inference_inputs": {
            "question": "List user names with large paid orders",
            "evidence": "",
            "schema_items": [
                {"id": 0, "type": "table", "name": "users"},
                {"id": 1, "type": "table", "name": "orders"},
                {"id": 2, "type": "column", "name": "users.id", "table": "users", "data_type": "integer"},
                {"id": 3, "type": "column", "name": "users.name", "table": "users", "data_type": "text"},
                {"id": 4, "type": "column", "name": "users.email", "table": "users", "data_type": "text"},
                {"id": 5, "type": "column", "name": "orders.user_id", "table": "orders", "data_type": "integer"},
                {"id": 6, "type": "column", "name": "orders.amount", "table": "orders", "data_type": "real"},
                {"id": 7, "type": "column", "name": "orders.tax", "table": "orders", "data_type": "real"},
            ],
            "schema_edges": [
                {"src": index, "dst": index, "type": "self_loop"} for index in range(8)
            ] + [
                {"src": 2, "dst": 5, "type": "foreign_key_forward"},
                {"src": 5, "dst": 2, "type": "foreign_key_backward"},
            ],
        },
        "teacher_steps": [
            {"action": "SCAN", "table_pointer_ids": [0], "column_pointer_ids": [], "operator_targets": [], "value_routes": [], "join_edge_targets": []},
            {"action": "SCAN", "table_pointer_ids": [1], "column_pointer_ids": [], "operator_targets": [], "value_routes": [], "join_edge_targets": []},
            {"action": "JOIN", "table_pointer_ids": [], "column_pointer_ids": [2, 5], "operator_targets": ["="], "value_routes": [], "join_edge_targets": [{"left_column_id": 2, "right_column_id": 5}]},
            {"action": "FILTER", "table_pointer_ids": [], "column_pointer_ids": [6], "operator_targets": [">"], "value_routes": ["question"], "join_edge_targets": []},
            {"action": "PROJECT", "table_pointer_ids": [], "column_pointer_ids": [3], "operator_targets": [], "value_routes": [], "join_edge_targets": []},
            {"action": "STOP", "table_pointer_ids": [], "column_pointer_ids": [], "operator_targets": [], "value_routes": [], "join_edge_targets": []},
        ],
    }


class Stage15DataTest(unittest.TestCase):
    def test_builds_grouped_unique_hard_negatives(self):
        from src.data.stage15_build_sql_hypothesis_data import (
            build_candidate_group,
            candidate_signature,
        )

        group = build_candidate_group(toy_trajectory(), negatives_per_example=5, seed=42)
        self.assertIsNotNone(group)
        self.assertEqual(sum(candidate["label"] for candidate in group["candidates"]), 1)
        signatures = [candidate_signature(candidate["steps"]) for candidate in group["candidates"]]
        self.assertEqual(len(signatures), len(set(signatures)))
        corruptions = {
            candidate["corruption_type"] for candidate in group["candidates"] if not candidate["label"]
        }
        self.assertIn("same_table_column", corruptions)
        self.assertTrue(corruptions & {"operator", "value_route", "clause_role"})


class Stage15ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            cls.torch_available = True
        except ImportError:
            cls.torch_available = False

    def test_forward_and_grouped_loss_have_gradients(self):
        if not self.torch_available:
            self.skipTest("PyTorch unavailable")
        import torch
        from src.data.stage15_build_sql_hypothesis_data import build_candidate_group
        from src.modeling.sql_hypothesis_verifier import (
            SQLHypothesisGraphVerifier,
            grouped_ranking_loss,
        )

        group = build_candidate_group(toy_trajectory(), negatives_per_example=4, seed=7)
        model = SQLHypothesisGraphVerifier(
            dense_dim=8,
            hidden_dim=16,
            schema_relation_count=3,
            num_schema_layers=1,
            num_plan_layers=1,
            dropout=0.0,
        )
        edges = group["inference_inputs"]["schema_edges"]
        relation_to_id = {
            name: index for index, name in enumerate(
                sorted({edge["type"] for edge in edges})
            )
        }
        output = model(
            torch.randn(8, 8),
            torch.randn(1, 8),
            torch.tensor([0, 0, 1, 1, 1, 1, 1, 1]),
            torch.tensor([[edge["src"] for edge in edges], [edge["dst"] for edge in edges]]),
            torch.tensor([relation_to_id[edge["type"]] for edge in edges]),
            group["inference_inputs"]["schema_items"],
            edges,
            group["candidates"],
        )
        self.assertEqual(output["scores"].shape[0], len(group["candidates"]))
        labels = torch.tensor([candidate["label"] for candidate in group["candidates"]]).float()
        loss, components = grouped_ranking_loss(output["scores"], labels)
        loss.backward()
        self.assertGreater(float(components["listwise_loss"].detach()), 0.0)
        self.assertGreater(float(components["hardest_negative_loss"].detach()), 0.0)
        self.assertIsNotNone(model.global_score[-1].weight.grad)
        self.assertIsNotNone(model.consistency_encoder[0].weight.grad)

    def test_explicit_plan_schema_consistency_detects_scan_and_join_conflicts(self):
        from src.modeling.sql_hypothesis_verifier import plan_schema_consistency_features

        row = toy_trajectory()
        schema_items = row["inference_inputs"]["schema_items"]
        schema_edges = row["inference_inputs"]["schema_edges"]
        _, gold = plan_schema_consistency_features(
            schema_items, schema_edges, {"steps": row["teacher_steps"]}
        )
        scan_corrupted = {"steps": [dict(step) for step in row["teacher_steps"]]}
        scan_corrupted["steps"][1] = {
            **scan_corrupted["steps"][1], "table_pointer_ids": [0]
        }
        _, corrupted = plan_schema_consistency_features(
            schema_items, schema_edges, scan_corrupted
        )
        self.assertEqual(gold["owner_scan_coverage"], 1.0)
        self.assertLess(corrupted["owner_scan_coverage"], 1.0)
        join_corrupted = {"steps": [dict(step) for step in row["teacher_steps"]]}
        join_corrupted["steps"][2] = {
            **join_corrupted["steps"][2],
            "join_edge_targets": [{"left_column_id": 3, "right_column_id": 4}],
        }
        _, bad_join = plan_schema_consistency_features(
            schema_items, schema_edges, join_corrupted
        )
        self.assertEqual(bad_join["fk_validity"], 0.0)
        self.assertLess(bad_join["required_table_connectivity"], 1.0)


class Stage15MetricTest(unittest.TestCase):
    def test_grouped_metrics_are_corruption_aware(self):
        from src.evaluation.stage15_evaluate_sql_hypothesis_verifier import ranking_metrics

        metrics = ranking_metrics(
            [
                {
                    "record_index": 0,
                    "candidates": [
                        {"label": 1, "score": 2.0, "corruption_type": "gold"},
                        {"label": 0, "score": 1.0, "corruption_type": "operator"},
                        {"label": 0, "score": 3.0, "corruption_type": "join_edge"},
                    ],
                }
            ]
        )
        self.assertEqual(metrics["hits@1"], 0.0)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["pairwise_accuracy"], 0.5)
        self.assertEqual(metrics["by_corruption"]["operator"]["pairwise_accuracy"], 1.0)
        self.assertEqual(metrics["top1_error_by_corruption"]["join_edge"]["count"], 1)
        self.assertEqual(metrics["mean_hardest_margin"], -1.0)

    def test_ties_do_not_receive_hits_at_one(self):
        from src.evaluation.stage15_evaluate_sql_hypothesis_verifier import ranking_metrics

        metrics = ranking_metrics(
            [{"candidates": [
                {"label": 1, "score": 0.0, "corruption_type": "gold"},
                {"label": 0, "score": 0.0, "corruption_type": "operator"},
            ]}]
        )
        self.assertEqual(metrics["hits@1"], 0.0)
        self.assertAlmostEqual(metrics["mrr"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
