import unittest


class Stage1ForeignKeyLabelPolicyTest(unittest.TestCase):
    def setUp(self):
        from src.data.stage1_extract_bird_labels import schema_from_tables_entry

        self.schema = schema_from_tables_entry(
            {
                "table_names_original": ["posts", "users"],
                "column_names_original": [
                    [-1, "*"],
                    [0, "Id"],
                    [0, "OwnerUserId"],
                    [0, "LastEditorUserId"],
                    [1, "Id"],
                ],
                "column_types": ["text", "integer", "integer", "integer", "integer"],
                "foreign_keys": [[2, 4], [3, 4]],
            }
        )

    def test_explicit_policy_does_not_add_parallel_unused_fk(self):
        from src.data.stage1_extract_bird_labels import fk_labels, sql_parse_labels

        sql = (
            "SELECT users.Id FROM posts JOIN users "
            "ON posts.OwnerUserId = users.Id"
        )
        sql_labels, used_tables = sql_parse_labels(sql, self.schema)
        labels = fk_labels(
            used_tables,
            self.schema,
            sql_labels=sql_labels,
            mode="explicit_sql",
        )

        owner = self.schema["column_item_ids"][("posts", "owneruserid")]
        editor = self.schema["column_item_ids"][("posts", "lasteditoruserid")]
        user_id = self.schema["column_item_ids"][("users", "id")]
        self.assertEqual(labels, {owner, user_id})
        self.assertNotIn(editor, labels)

    def test_legacy_policy_reproduces_all_used_table_closure(self):
        from src.data.stage1_extract_bird_labels import fk_labels, sql_parse_labels

        sql = (
            "SELECT users.Id FROM posts JOIN users "
            "ON posts.OwnerUserId = users.Id"
        )
        sql_labels, used_tables = sql_parse_labels(sql, self.schema)
        labels = fk_labels(
            used_tables,
            self.schema,
            sql_labels=sql_labels,
            mode="all_used_tables",
        )

        expected = {
            self.schema["column_item_ids"][("posts", "owneruserid")],
            self.schema["column_item_ids"][("posts", "lasteditoruserid")],
            self.schema["column_item_ids"][("users", "id")],
        }
        self.assertEqual(labels, expected)


if __name__ == "__main__":
    unittest.main()
