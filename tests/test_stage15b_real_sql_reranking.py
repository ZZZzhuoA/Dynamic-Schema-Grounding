import unittest


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


if __name__ == "__main__":
    unittest.main()
