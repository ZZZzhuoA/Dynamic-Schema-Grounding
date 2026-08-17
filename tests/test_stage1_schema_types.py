import unittest


class Stage1SchemaTypesTest(unittest.TestCase):
    def test_bird_column_types_keep_the_wildcard_index_alignment(self):
        from src.data.stage1_extract_bird_labels import schema_from_tables_entry

        schema = schema_from_tables_entry(
            {
                "table_names_original": ["movies"],
                "column_names_original": [[-1, "*"], [0, "title"], [0, "year"]],
                "column_types": ["text", "text", "integer"],
                "primary_keys": [],
                "foreign_keys": [],
            }
        )
        columns = {item["column"]: item for item in schema["schema_items"] if item["type"] == "column"}
        self.assertEqual(columns["title"]["data_type"], "text")
        self.assertEqual(columns["year"]["data_type"], "integer")


if __name__ == "__main__":
    unittest.main()
