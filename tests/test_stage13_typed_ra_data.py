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
        self.assertEqual(actions, ["SCAN", "FILTER", "PROJECT", "SORT", "LIMIT"])
        self.assertEqual(row["audit"]["parse_status"], "supported_flat")
        self.assertEqual(row["audit"]["schema_label_coverage"], 1.0)
        values = row["training_targets"]["value_copy_targets"]
        year = next(value for value in values if value["canonical_value"] == "1945")
        self.assertEqual(year["source"], "question")
        self.assertEqual(year["match"], "exact")

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
        value = row["training_targets"]["value_copy_targets"][0]
        self.assertEqual(value["canonical_value"], "Hickman")
        self.assertTrue(value["case_sensitive"])
        self.assertEqual(value["source"], "question")

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
