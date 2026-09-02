import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GRAPH = load_module("stage5_build_dsg_data_metadata", "src/data/stage5_build_dsg_data.py")
EMBED = load_module(
    "stage8g_build_embedding_cache_metadata",
    "src/embedding/stage8g_build_embedding_cache.py",
)
DIAGNOSIS = load_module(
    "stage17b_diagnose_schema_coverage_metadata",
    "src/analysis/stage17b_diagnose_schema_coverage.py",
)


def table_entry():
    return {
        "db_id": "demo",
        "table_names_original": ["Schools", "Districts"],
        "column_names_original": [
            [-1, "*"],
            [0, "School_ID"],
            [0, "Name"],
            [1, "District_ID"],
        ],
        "column_types": ["text", "number", "text", "number"],
        "primary_keys": [1, [3]],
        "foreign_keys": [[1, 3]],
    }


class Stage17FOfficialMetadataTest(unittest.TestCase):
    def test_primary_key_edge_attributes_preserve_generic_relations(self):
        schema_items = [
            {"id": 0, "type": "table", "name": "Schools"},
            {"id": 1, "type": "table", "name": "Districts"},
            {
                "id": 2,
                "type": "column",
                "name": "Schools.School_ID",
                "table": "Schools",
                "column": "School_ID",
            },
            {
                "id": 3,
                "type": "column",
                "name": "Schools.Name",
                "table": "Schools",
                "column": "Name",
            },
            {
                "id": 4,
                "type": "column",
                "name": "Districts.District_ID",
                "table": "Districts",
                "column": "District_ID",
            },
        ]
        generic = GRAPH.schema_edges(schema_items, table_entry())
        attributed = GRAPH.schema_edges(
            schema_items,
            table_entry(),
            encode_primary_keys_as_edge_attributes=True,
        )
        self.assertEqual(
            [(row["src"], row["dst"], row["type"]) for row in generic],
            [(row["src"], row["dst"], row["type"]) for row in attributed],
        )
        self.assertTrue(all("is_primary_key_edge" not in row for row in generic))
        self.assertTrue(all("is_primary_key_edge" in row for row in attributed))
        marked = {
            (row["src"], row["dst"], row["type"])
            for row in attributed
            if row.get("is_primary_key_edge") is True
        }
        self.assertEqual(
            marked,
            {
                (0, 2, "table_to_column"),
                (2, 0, "column_to_table"),
                (1, 4, "table_to_column"),
                (4, 1, "column_to_table"),
            },
        )
        self.assertFalse(
            any(row.get("is_primary_key_edge") for row in attributed if row["src"] == 3)
        )

    def test_primary_key_encodings_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "cannot be encoded as both"):
            GRAPH.schema_edges(
                [],
                table_entry(),
                encode_primary_keys_as_relations=True,
                encode_primary_keys_as_edge_attributes=True,
            )

    def test_primary_keys_replace_generic_ownership_relations_without_node_text_changes(self):
        schema_items = [
            {"id": 0, "type": "table", "name": "Schools"},
            {"id": 1, "type": "table", "name": "Districts"},
            {
                "id": 2,
                "type": "column",
                "name": "Schools.School_ID",
                "table": "Schools",
                "column": "School_ID",
                "data_type": "number",
            },
            {
                "id": 3,
                "type": "column",
                "name": "Schools.Name",
                "table": "Schools",
                "column": "Name",
                "data_type": "text",
            },
            {
                "id": 4,
                "type": "column",
                "name": "Districts.District_ID",
                "table": "Districts",
                "column": "District_ID",
                "data_type": "number",
            },
        ]
        generic = GRAPH.schema_edges(schema_items, table_entry())
        typed = GRAPH.schema_edges(
            schema_items,
            table_entry(),
            encode_primary_keys_as_relations=True,
        )
        self.assertEqual(len(generic), len(typed))
        typed_rows = {(row["src"], row["dst"], row["type"]) for row in typed}
        self.assertIn((0, 2, "table_to_primary_key"), typed_rows)
        self.assertIn((2, 0, "primary_key_to_table"), typed_rows)
        self.assertIn((1, 4, "table_to_primary_key"), typed_rows)
        self.assertIn((4, 1, "primary_key_to_table"), typed_rows)
        self.assertIn((0, 3, "table_to_column"), typed_rows)
        self.assertNotIn((0, 2, "table_to_column"), typed_rows)
        nodes = GRAPH.schema_nodes(schema_items)
        self.assertTrue(all("is_primary_key" not in node for node in nodes))
        self.assertEqual(
            EMBED.node_embedding_text(nodes[2]),
            "schema item: Schools.School_ID | type: column | table: Schools | "
            "column: School_ID | data type: number",
        )

    def test_primary_key_relation_encoding_requires_valid_table_schema(self):
        with self.assertRaisesRegex(ValueError, "requires the matching BIRD tables entry"):
            GRAPH.schema_edges([], None, encode_primary_keys_as_relations=True)
        broken = table_entry()
        broken["primary_keys"] = [99]
        with self.assertRaisesRegex(ValueError, "Primary key index out of range"):
            GRAPH.schema_edges(
                [], broken, encode_primary_keys_as_relations=True
            )

    def test_diagnosis_infers_primary_key_identity_from_typed_edges(self):
        nodes = [
            {"id": 0, "type": "table", "name": "Schools"},
            {
                "id": 1,
                "type": "column",
                "name": "Schools.School_ID",
                "table": "Schools",
                "column": "School_ID",
            },
        ]
        edges = [
            {"src": 0, "dst": 1, "type": "table_to_primary_key"},
            {"src": 1, "dst": 0, "type": "primary_key_to_table"},
        ]
        node_by_id, _, _ = DIAGNOSIS.build_node_maps(nodes, edges)
        self.assertTrue(node_by_id[1]["is_primary_key"])
        self.assertNotIn("is_primary_key", nodes[1])

    def test_pk_fk_endpoint_and_direction_mapping(self):
        metadata = GRAPH.authoritative_schema_metadata(table_entry())
        school_id = metadata[("schools", "school_id")]
        district_id = metadata[("districts", "district_id")]
        self.assertTrue(school_id["is_primary_key"])
        self.assertTrue(school_id["is_foreign_key_endpoint"])
        self.assertEqual(school_id["foreign_key_outgoing_targets"], ["Districts.District_ID"])
        self.assertEqual(district_id["foreign_key_incoming_sources"], ["Schools.School_ID"])

    def test_csv_loading_normalizes_bom_whitespace_and_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "demo" / "database_description"
            directory.mkdir(parents=True)
            (directory / " schools .csv").write_text(
                "\ufefforiginal_column_name,column_name,column_description,data_format,value_description\n"
                " School_ID ,School identifier, Unique school key ,integer, Business identifier \n"
                "Name,School name,,,\n",
                encoding="utf-8",
            )
            (directory / "districts.csv").write_text(
                "original_column_name,column_name,column_description,data_format,value_description\n"
                "District_ID,District identifier,,,\n",
                encoding="utf-8",
            )
            metadata, summary = GRAPH.load_database_description(
                Path(tmp), {"demo": table_entry()}
            )
            row = metadata["demo"][("schools", "school_id")]
            self.assertEqual(row["official_column_name"], "School identifier")
            self.assertEqual(row["official_column_description"], "Unique school key")
            self.assertEqual(row["official_value_description"], "Business identifier")
            self.assertEqual(summary["matched_column_count"], 3)
            self.assertEqual(summary["unmatched_column_count"], 0)

    def test_duplicate_normalized_csv_column_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "demo" / "database_description"
            directory.mkdir(parents=True)
            (directory / "Schools.csv").write_text(
                "original_column_name,column_name\nSchool_ID,A\n school_id ,B\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Ambiguous duplicate column"):
                GRAPH.load_database_description(Path(tmp), {"demo": table_entry()})

    def test_old_graph_embedding_text_is_unchanged(self):
        node = {
            "id": 1,
            "type": "column",
            "name": "Schools.Name",
            "table": "Schools",
            "column": "Name",
            "data_type": "text",
            "semantic_name": "school name",
        }
        self.assertEqual(
            EMBED.node_embedding_text(node),
            "schema item: Schools.Name | type: column | table: Schools | column: Name | "
            "data type: text | semantic name: school name",
        )
        plain = GRAPH.schema_nodes([node])
        self.assertNotIn("is_primary_key", plain[0])
        self.assertNotIn("metadata_source", plain[0])

    def test_metadata_truncation_keeps_identity_first(self):
        node = {
            "type": "column",
            "name": "Schools.School_ID",
            "table": "Schools",
            "column": "School_ID",
            "data_type": "number",
            "is_primary_key": True,
            "official_column_name": "N" * 200,
            "official_column_description": "D" * 400,
            "official_value_description": "V" * 500,
            "semantic_text": "existing card",
        }
        text = EMBED.node_embedding_text(node)
        self.assertTrue(text.startswith("schema item: Schools.School_ID | type: column"))
        self.assertIn("official column name: " + "N" * 160, text)
        self.assertNotIn("N" * 161, text)
        self.assertLess(text.index("official description:"), text.index("semantic card:"))

    def test_identifier_detection_supports_snake_and_camel_case(self):
        self.assertTrue(
            DIAGNOSIS.is_identifier_column({"type": "column", "column": "school_id"})
        )
        self.assertTrue(
            DIAGNOSIS.is_identifier_column({"type": "column", "column": "raceId"})
        )
        self.assertFalse(
            DIAGNOSIS.is_identifier_column({"type": "column", "column": "paid"})
        )

    def test_diagnosis_reports_metadata_recall_and_paired_transitions(self):
        base_gold = {
            "schema_item_id": 1,
            "rank": 31,
            "is_primary_key": True,
            "is_foreign_key_endpoint": False,
            "is_identifier_column": True,
            "has_official_description": True,
            "has_official_value_description": True,
        }
        baseline = {
            "record_index": 0,
            "db_id": "demo",
            "question_id": 1,
            "complete_coverage": False,
            "recall": 0.5,
            "precision": 0.1,
            "schema_node_count": 50,
            "gold_count": 2,
            "topk_budget_ratio": 0.6,
            "missing_gold_count": 1,
            "false_positive_count": 29,
            "worst_gold_rank": 31,
            "gold_schema": [base_gold],
        }
        candidate = {
            **baseline,
            "complete_coverage": True,
            "recall": 1.0,
            "missing_gold_count": 0,
            "worst_gold_rank": 25,
            "gold_schema": [{**base_gold, "rank": 25}],
        }
        summary = DIAGNOSIS.summarize_group([baseline])
        self.assertEqual(summary["primary_key_recall@30"], 0.0)
        self.assertEqual(summary["primary_key_rank_31_40_missing_count"], 1)
        comparison = DIAGNOSIS.compare_runs([baseline], [candidate])
        self.assertEqual(comparison["recovered_complete_count"], 1)
        self.assertEqual(comparison["regressed_complete_count"], 0)


if __name__ == "__main__":
    unittest.main()
