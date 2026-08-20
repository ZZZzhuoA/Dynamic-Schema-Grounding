import unittest
from types import SimpleNamespace


def schema_graph():
    items = [
        {"id": 0, "type": "table", "name": "users", "normalized_name": "users"},
        {"id": 1, "type": "table", "name": "orders", "normalized_name": "orders"},
        {
            "id": 2, "type": "column", "name": "users.id", "table": "users", "column": "id",
            "normalized_name": "users.id", "normalized_table": "users", "normalized_column": "id",
            "data_type": "integer",
        },
        {
            "id": 3, "type": "column", "name": "users.name", "table": "users", "column": "name",
            "normalized_name": "users.name", "normalized_table": "users", "normalized_column": "name",
            "data_type": "text",
        },
        {
            "id": 4, "type": "column", "name": "orders.user_id", "table": "orders", "column": "user_id",
            "normalized_name": "orders.user_id", "normalized_table": "orders", "normalized_column": "user_id",
            "data_type": "integer",
        },
        {
            "id": 5, "type": "column", "name": "orders.amount", "table": "orders", "column": "amount",
            "normalized_name": "orders.amount", "normalized_table": "orders", "normalized_column": "amount",
            "data_type": "real",
        },
    ]
    edges = [
        {"src": 2, "dst": 4, "type": "foreign_key_forward"},
        {"src": 4, "dst": 2, "type": "foreign_key_backward"},
    ]
    return {
        "record_index": 0,
        "db_id": "toy",
        "inference_inputs": {
            "question": "List users with orders above 10",
            "evidence": "",
            "schema_items": items,
            "schema_edges": edges,
        },
    }


class Stage15BParsingTest(unittest.TestCase):
    def test_candidate_plan_is_parsed_from_candidate_sql(self):
        from src.data.stage15b_prepare_real_sql_candidates import parse_candidate

        graph = schema_graph()
        base = {
            "db_id": "toy",
            "question": "List users with orders above 10",
            "evidence": "",
        }
        parsed, error = parse_candidate(
            base,
            graph,
            "SELECT u.name FROM users u JOIN orders o ON u.id=o.user_id WHERE o.amount > 10",
            0,
        )
        self.assertIsNone(error)
        self.assertEqual(parsed["parse_status"], "supported_flat")
        steps = parsed["steps"]
        project = next(step for step in steps if step["action"] == "PROJECT")
        filtering = next(step for step in steps if step["action"] == "FILTER")
        joining = next(step for step in steps if step["action"] == "JOIN")
        self.assertEqual(project["column_pointer_ids"], [3])
        self.assertEqual(filtering["column_pointer_ids"], [5])
        self.assertEqual({2, 4}, set(joining["column_pointer_ids"]))
        self.assertEqual(len(joining["join_edge_targets"]), 1)

    def test_greedy_request_overrides_sampling_temperature(self):
        from src.generation.stage15b_generate_sql_candidates import request_payload

        args = SimpleNamespace(
            temperature=0.7, top_p=0.95, max_tokens=128,
            request_logprobs=True, disable_thinking=True,
        )
        payload = request_payload(
            "model", "prompt", 1, args, 42, temperature=0.0, top_p=1.0
        )
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertEqual(payload["n"], 1)
        self.assertTrue(payload["logprobs"])


class Stage15BEvaluationTest(unittest.TestCase):
    def test_verifier_can_recover_an_available_llm_error(self):
        from src.evaluation.stage15b_evaluate_real_sql_reranking import evaluate

        rows = [
            {
                "record_index": 0,
                "candidates": [
                    {
                        "candidate_id": "llm_0", "llm_rank": 0, "mean_logprob": -0.1,
                        "execution_ok": True, "execution_correct": False, "parse_ok": True,
                        "verifier_score": 0.0,
                    },
                    {
                        "candidate_id": "llm_1", "llm_rank": 1, "mean_logprob": -0.2,
                        "execution_ok": True, "execution_correct": True, "parse_ok": True,
                        "verifier_score": 2.0,
                    },
                ],
            }
        ]
        metrics, details = evaluate(rows, [0.0, 1.0])
        self.assertEqual(metrics["methods"]["llm_top1"]["execution_accuracy"], 0.0)
        self.assertEqual(metrics["methods"]["verifier"]["execution_accuracy"], 1.0)
        self.assertEqual(metrics["methods"]["hybrid_1"]["recovered_count"], 1)
        self.assertEqual(metrics["methods"]["hybrid_0"]["execution_accuracy"], 0.0)
        self.assertEqual(details[0]["selections"]["verifier"]["candidate_index"], 1)

    def test_greedy_anchor_is_the_baseline_even_if_not_first(self):
        from src.evaluation.stage15b_evaluate_real_sql_reranking import evaluate

        rows = [{
            "record_index": 9,
            "candidates": [
                {"generation_mode": "sample", "execution_ok": True, "execution_correct": True,
                 "parse_ok": True, "verifier_score": 0.0, "llm_rank": 1},
                {"generation_mode": "greedy", "execution_ok": True, "execution_correct": False,
                 "parse_ok": True, "verifier_score": 1.0, "llm_rank": 0},
            ],
        }]
        metrics, _ = evaluate(rows, [1.0])
        self.assertEqual(metrics["methods"]["llm_top1"]["execution_accuracy"], 0.0)
        self.assertEqual(metrics["baseline_wrong_but_oracle_available_count"], 1)

    def test_control_scores_are_evaluated_as_separate_causal_methods(self):
        from src.evaluation.stage15b_evaluate_real_sql_reranking import evaluate

        rows = [{
            "record_index": 0,
            "candidates": [
                {"generation_mode": "greedy", "execution_ok": True, "execution_correct": False,
                 "parse_ok": True, "verifier_score": 0.0, "llm_rank": 0,
                 "control_scores": {"shuffled_fk": 2.0}},
                {"generation_mode": "sample", "execution_ok": True, "execution_correct": True,
                 "parse_ok": True, "verifier_score": 2.0, "llm_rank": 1,
                 "control_scores": {"shuffled_fk": 0.0}},
            ],
        }]
        metrics, _ = evaluate(rows, [1.0])
        self.assertEqual(metrics["methods"]["hybrid_1"]["execution_accuracy"], 1.0)
        self.assertEqual(
            metrics["methods"]["control_shuffled_fk_hybrid_1"]["execution_accuracy"], 0.0
        )

    def test_calibration_split_is_disjoint_and_deterministic(self):
        from src.evaluation.stage15b_evaluate_real_sql_reranking import calibration_split

        rows = [{"record_index": index} for index in range(10)]
        left, right = calibration_split(rows, 0.2, 42)
        self.assertEqual(len(left), 2)
        self.assertEqual(len(right), 8)
        self.assertFalse({x["record_index"] for x in left} & {x["record_index"] for x in right})
        left_again, _ = calibration_split(rows, 0.2, 42)
        self.assertEqual(left, left_again)


class Stage15BCausalControlTest(unittest.TestCase):
    def test_fk_control_changes_only_fk_destinations(self):
        from src.grounding.stage15b_score_real_sql_candidates import shuffled_fk_row

        row = schema_graph()
        row["inference_inputs"]["schema_edges"].append(
            {"src": 0, "dst": 0, "type": "self_loop"}
        )
        result = shuffled_fk_row(row, 42)
        original = row["inference_inputs"]["schema_edges"]
        changed = result["inference_inputs"]["schema_edges"]
        self.assertEqual(original[-1], changed[-1])
        self.assertNotEqual(
            [edge["dst"] for edge in original[:2]],
            [edge["dst"] for edge in changed[:2]],
        )


if __name__ == "__main__":
    unittest.main()
