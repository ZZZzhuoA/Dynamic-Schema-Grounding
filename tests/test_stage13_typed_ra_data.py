import unittest


class Stage13TypedRADataTest(unittest.TestCase):
    def test_flat_query_builds_typed_actions_and_exact_value_copy(self):
        from src.data.stage13_build_typed_ra_data import build_typed_plan

        record = {
            "split": "train",
            "db_id": "movie_platform",
            "question": "Name movie titles released in year 1945, ordered by popularity.",
            "evidence": "released in year 1945 means movie_release_year = 1945",
            "sql": (
                "SELECT movie_title FROM movies WHERE movie_release_year = 1945 "
                "ORDER BY movie_popularity DESC LIMIT 1"
            ),
            "schema_items": [
                {"id": 0, "type": "table", "name": "movies", "normalized_name": "movies"},
                {
                    "id": 1, "type": "column", "name": "movies.movie_title",
                    "table": "movies", "column": "movie_title",
                    "normalized_name": "movies.movie_title", "normalized_table": "movies",
                    "normalized_column": "movie_title", "data_type": "text",
                },
                {
                    "id": 2, "type": "column", "name": "movies.movie_release_year",
                    "table": "movies", "column": "movie_release_year",
                    "normalized_name": "movies.movie_release_year", "normalized_table": "movies",
                    "normalized_column": "movie_release_year", "data_type": "integer",
                },
                {
                    "id": 3, "type": "column", "name": "movies.movie_popularity",
                    "table": "movies", "column": "movie_popularity",
                    "normalized_name": "movies.movie_popularity", "normalized_table": "movies",
                    "normalized_column": "movie_popularity", "data_type": "real",
                },
            ],
            "whole_sql_labels": [0, 1, 2, 3],
            "used_tables_from_sql": ["movies"],
            "label_sources": {
                "sql_parse": [
                    "movies", "movies.movie_title", "movies.movie_release_year",
                    "movies.movie_popularity",
                ],
                "foreign_key": [],
            },
        }
        row = build_typed_plan(record, None, 0)
        actions = [item["action"] for item in row["training_targets"]["action_sequence"]]
        self.assertEqual(actions, ["SCAN", "FILTER", "SORT", "LIMIT", "PROJECT"])
        self.assertEqual(row["audit"]["parse_status"], "supported_flat")
        self.assertEqual(row["audit"]["schema_label_coverage"], 1.0)
        values = row["training_targets"]["value_copy_targets"]
        year = next(value for value in values if value["canonical_value"] == "1945")
        self.assertEqual(year["source"], "question")
        self.assertEqual(year["match"], "exact")
        limit = next(value for value in values if value["clause"] == "limit")
        self.assertEqual(limit["source"], "operator_inference_required")
        self.assertEqual(limit["match"], "none")

    def test_numeric_copy_requires_a_complete_value_boundary(self):
        from src.data.stage13_build_typed_ra_data import source_match

        result = source_match("1", "released in 1945", "", "number")
        self.assertEqual(result["source"], "database_value_required")
        self.assertEqual(result["match"], "none")
        punctuated = source_match("1945", "released in 1945.", "", "number")
        self.assertEqual(punctuated["source"], "question")
        self.assertEqual(punctuated["match"], "exact")

    def test_limit_one_can_be_inferred_from_a_superlative(self):
        from src.data.stage13_build_typed_ra_data import literal_targets

        values = literal_targets("1", "Which movie has the highest popularity?", "", "limit")
        self.assertEqual(values[0]["source"], "semantic_inference")
        self.assertEqual(values[0]["match"], "inferred_superlative")

    def test_join_path_and_case_sensitive_value_are_separate_targets(self):
        from src.data.stage13_build_typed_ra_data import build_typed_plan

        schema = [
            {"id": 0, "type": "table", "name": "schools", "normalized_name": "schools"},
            {"id": 1, "type": "table", "name": "frpm", "normalized_name": "frpm"},
            {
                "id": 2, "type": "column", "name": "schools.CDSCode", "table": "schools",
                "column": "CDSCode", "normalized_name": "schools.cdscode",
                "normalized_table": "schools", "normalized_column": "cdscode", "data_type": "text",
            },
            {
                "id": 3, "type": "column", "name": "frpm.CDSCode", "table": "frpm",
                "column": "CDSCode", "normalized_name": "frpm.cdscode",
                "normalized_table": "frpm", "normalized_column": "cdscode", "data_type": "text",
            },
            {
                "id": 4, "type": "column", "name": "schools.City", "table": "schools",
                "column": "City", "normalized_name": "schools.city",
                "normalized_table": "schools", "normalized_column": "city", "data_type": "text",
            },
        ]
        record = {
            "split": "dev", "db_id": "schools", "question": "Schools in Hickman",
            "evidence": "", "sql": (
                "SELECT s.City FROM schools AS s JOIN frpm AS f ON s.CDSCode = f.CDSCode "
                "WHERE s.City = 'Hickman'"
            ),
            "schema_items": schema,
            "whole_sql_labels": [0, 1, 2, 3, 4],
            "used_tables_from_sql": ["schools", "frpm"],
            "label_sources": {
                "sql_parse": ["schools", "frpm", "schools.CDSCode", "frpm.CDSCode", "schools.City"],
                "foreign_key": ["schools.CDSCode", "frpm.CDSCode"],
            },
        }
        graph = {"inference_inputs": {"schema_edges": [
            {"src": 2, "dst": 3, "type": "foreign_key_forward"},
            {"src": 3, "dst": 2, "type": "foreign_key_backward"},
        ]}}
        row = build_typed_plan(record, graph, 0)
        join = row["training_targets"]["join_path"]
        self.assertEqual(join["table_pointer_ids"], [0, 1])
        self.assertTrue(join["connected"])
        self.assertEqual(len(join["edge_targets"]), 1)
        self.assertEqual(join["edge_targets"][0]["edge_type"], "sql_equality")
        value = row["training_targets"]["value_copy_targets"][0]
        self.assertEqual(value["canonical_value"], "Hickman")
        self.assertTrue(value["case_sensitive"])
        self.assertEqual(value["source"], "question")

    def test_fk_closure_does_not_require_both_endpoint_labels(self):
        from src.data.stage13_build_typed_ra_data import selected_join_edges

        schema = [
            {"id": 0, "type": "table", "name": "a", "normalized_name": "a"},
            {"id": 1, "type": "table", "name": "b", "normalized_name": "b"},
            {"id": 2, "type": "column", "name": "a.id", "table": "a", "column": "id",
             "normalized_name": "a.id", "normalized_table": "a", "normalized_column": "id"},
            {"id": 3, "type": "column", "name": "b.a_id", "table": "b", "column": "a_id",
             "normalized_name": "b.a_id", "normalized_table": "b", "normalized_column": "a_id"},
        ]
        graph = {"inference_inputs": {"schema_edges": [
            {"src": 2, "dst": 3, "type": "foreign_key_forward"},
            {"src": 3, "dst": 2, "type": "foreign_key_backward"},
        ]}}
        edges = selected_join_edges(graph, schema, [0, 1], [2], "SELECT * FROM a, b")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["edge_type"], "foreign_key")

    def test_nested_query_is_explicitly_marked_partial(self):
        from src.data.stage13_build_typed_ra_data import build_typed_plan

        record = {
            "split": "dev", "db_id": "x", "question": "q", "evidence": "",
            "sql": "SELECT name FROM users WHERE id IN (SELECT user_id FROM orders)",
            "schema_items": [
                {"id": 0, "type": "table", "name": "users", "normalized_name": "users"},
                {"id": 1, "type": "table", "name": "orders", "normalized_name": "orders"},
            ],
            "whole_sql_labels": [0, 1],
            "used_tables_from_sql": ["users", "orders"],
            "label_sources": {"sql_parse": ["users", "orders"], "foreign_key": []},
        }
        row = build_typed_plan(record, None, 0)
        self.assertEqual(row["audit"]["parse_status"], "partial_nested")
        self.assertIn("nested_subquery", row["audit"]["unsupported_features"])


if __name__ == "__main__":
    unittest.main()
