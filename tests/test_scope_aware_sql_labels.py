import unittest

from src.data.stage1_extract_bird_labels import schema_from_tables_entry, sql_parse_labels
from src.data.stage5g_build_clause_labels import build_schema_index, clause_labels_from_sql


class ScopeAwareSqlLabelTest(unittest.TestCase):
    def schema(self, tables, columns):
        entry = {
            "table_names_original": tables,
            "column_names_original": [[-1, "*"]] + columns,
            "column_types": ["text"] * (len(columns) + 1),
            "foreign_keys": [],
        }
        return schema_from_tables_entry(entry)

    def clause_schema(self, schema):
        return build_schema_index(schema["schema_items"], schema["foreign_keys"])

    def test_qualified_same_name_belongs_only_to_alias_owner(self):
        schema = self.schema(
            ["cards", "legalities"],
            [[0, "id"], [0, "uuid"], [1, "id"], [1, "uuid"]],
        )
        labels, _ = sql_parse_labels(
            "SELECT T1.id FROM cards AS T1 JOIN legalities AS T2 ON T1.uuid = T2.uuid",
            schema,
        )
        self.assertIn(schema["column_item_ids"][("cards", "id")], labels)
        self.assertNotIn(schema["column_item_ids"][("legalities", "id")], labels)

    def test_short_name_does_not_match_inside_quoted_identifier(self):
        schema = self.schema(
            ["frpm", "schools"],
            [[0, "Charter School (Y/N)"], [1, "Charter"]],
        )
        labels, _ = sql_parse_labels(
            "SELECT * FROM frpm JOIN schools ON 1=1 WHERE frpm.`Charter School (Y/N)` = 1",
            schema,
        )
        self.assertIn(
            schema["column_item_ids"][("frpm", "charter_school_y_n")], labels
        )
        self.assertNotIn(schema["column_item_ids"][("schools", "charter")], labels)

    def test_nested_unqualified_columns_use_local_query_scope(self):
        schema = self.schema(
            ["cards", "set_translations"],
            [[0, "id"], [0, "name"], [1, "id"], [1, "language"]],
        )
        labels, _ = sql_parse_labels(
            "SELECT language FROM set_translations WHERE id IN "
            "(SELECT id FROM cards WHERE name = 'Angel')",
            schema,
        )
        expected = {
            schema["column_item_ids"][("set_translations", "language")],
            schema["column_item_ids"][("set_translations", "id")],
            schema["column_item_ids"][("cards", "id")],
            schema["column_item_ids"][("cards", "name")],
        }
        self.assertTrue(expected.issubset(labels))

    def test_clause_labels_preserve_owner_and_nested_clause(self):
        schema = self.schema(
            ["cards", "legalities"],
            [[0, "id"], [0, "uuid"], [1, "id"], [1, "uuid"], [1, "status"]],
        )
        clause_schema = self.clause_schema(schema)
        labels, _ = clause_labels_from_sql(
            "SELECT T1.id FROM cards AS T1 JOIN legalities AS T2 "
            "ON T1.uuid = T2.uuid WHERE T2.status = 'Banned'",
            clause_schema,
        )
        self.assertIn(schema["column_item_ids"][("cards", "id")], labels["select"])
        self.assertNotIn(schema["column_item_ids"][("legalities", "id")], labels["select"])
        self.assertIn(schema["column_item_ids"][("legalities", "status")], labels["where"])

    def test_table_named_order_is_not_added_from_order_by_keyword(self):
        schema = self.schema(
            ["loan", "account", "order"],
            [[0, "amount"], [0, "account_id"], [1, "account_id"], [2, "amount"]],
        )
        labels, used_tables = sql_parse_labels(
            "SELECT T2.account_id FROM loan AS T1 JOIN account AS T2 "
            "ON T1.account_id = T2.account_id ORDER BY T1.amount",
            schema,
        )
        self.assertEqual(used_tables, {"loan", "account"})
        self.assertNotIn(schema["table_item_ids"]["order"], labels)
        self.assertNotIn(schema["column_item_ids"][("order", "amount")], labels)


if __name__ == "__main__":
    unittest.main()
