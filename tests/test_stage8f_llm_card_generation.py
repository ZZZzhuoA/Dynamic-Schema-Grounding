import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "data" / "stage8f_llm_card_generation.py"
SPEC = importlib.util.spec_from_file_location("stage8f_llm_card_generation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeClient:
    def chat(self, messages, **kwargs):
        prompt = json.loads(messages[-1]["content"])
        question = prompt["question"]
        return json.dumps(
            {
                "normalized_question": question.lower(),
                "intent": f"intent:{question}",
                "mentions": [],
                "operation_hints": [],
                "value_hints": [],
                "formula_hints": [],
                "ordering_hints": [],
            }
        )


class Stage8FParallelGenerationTest(unittest.TestCase):
    def test_extract_json_accepts_literal_control_characters(self):
        parsed = MODULE.extract_json('{"description": "line one\nline two"}')
        self.assertEqual(parsed["description"], "line one\nline two")

    def test_large_table_is_split_with_table_context(self):
        table_record = {
            "table_names_original": ["large_table"],
            "table_names": ["large table"],
            "column_names_original": [[-1, "*"]] + [[0, f"column_{i}"] for i in range(50)],
            "column_names": [[-1, "*"]] + [[0, f"column {i}"] for i in range(50)],
            "column_types": ["text"] * 51,
            "primary_keys": [],
            "foreign_keys": [],
        }
        chunks = MODULE.schema_item_chunks(table_record, "table", max_items=24)
        self.assertEqual([len(chunk) for chunk in chunks], [24, 24, 5])
        self.assertTrue(all(chunk[0]["node_type"] == "table" for chunk in chunks))

    def test_schema_card_quality_counts_fallbacks(self):
        quality = MODULE.schema_card_quality(
            [{"source": "llm"}, {"source": "llm_fallback"}, {"source": "llm_fallback"}]
        )
        self.assertEqual(quality["card_count"], 3)
        self.assertEqual(quality["fallback_card_count"], 2)
        self.assertAlmostEqual(quality["fallback_rate"], 2 / 3)

    def test_normalized_cache_content_ignores_case_and_whitespace(self):
        card = {"question": "How Many Users?", "evidence": " count   users "}
        record = {"question": "how many users?", "evidence": "COUNT users"}
        self.assertTrue(MODULE.resumed_question_content_matches(card, record))

        record["evidence"] = "count active users"
        self.assertFalse(MODULE.resumed_question_content_matches(card, record))

    def test_reuse_cache_is_loaded_before_output_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reuse_dir = root / "reuse"
            reuse_dir.mkdir()
            output_path = root / "cards.jsonl"
            MODULE.write_jsonl(reuse_dir / "train_question_cards.jsonl", [{"record_index": 0, "value": "old"}])
            MODULE.write_jsonl(output_path, [{"record_index": 0, "value": "new"}])

            records = MODULE.cached_jsonl_records(
                output_path,
                [reuse_dir],
                "train_question_cards.jsonl",
            )
            latest = MODULE.latest_by(records, lambda row: row["record_index"])
            self.assertEqual(latest[0]["value"], "new")

    def test_question_tasks_complete_concurrently_and_preserve_indices(self):
        args = SimpleNamespace(
            temperature=0.0,
            top_p=1.0,
            question_max_tokens=512,
            disable_thinking=True,
        )
        client = FakeClient()
        tasks = [
            (
                index,
                {
                    "db_id": "demo",
                    "question_id": index,
                    "question": f"Question {index}",
                    "evidence": "",
                },
                "train",
                client,
                args,
            )
            for index in range(12)
        ]

        results = list(MODULE.completed_results(tasks, MODULE.generate_question_task, workers=4))
        by_index = {index: (card, status) for index, card, status in results}

        self.assertEqual(set(by_index), set(range(12)))
        for index, (card, status) in by_index.items():
            self.assertEqual(card["record_index"], index)
            self.assertEqual(card["intent"], f"intent:Question {index}")
            self.assertIsNone(status["error"])


if __name__ == "__main__":
    unittest.main()
