import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "data" / "stage0_merge_train_corrections.py"
SPEC = importlib.util.spec_from_file_location("stage0_merge_train_corrections", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage0CorrectionMergeTest(unittest.TestCase):
    def test_exact_match_updates_sql_and_clears_stale_hit_info(self):
        originals = [
            {
                "db_name": "demo",
                "question": "How many users?",
                "answer": "SELECT name FROM users",
                "rep_answer": "SELECT name FROM users",
                "evidence": "old",
                "hit_info": {"users": ["name"]},
            }
        ]
        corrections = [
            {
                "db_id": "demo",
                "question": "How many users?",
                "SQL": "SELECT COUNT(*) FROM users",
                "evidence": "count users",
            }
        ]
        matches, unresolved = MODULE.match_corrections(originals, corrections, {}, 0.9, 0.05)
        merged, manifest, stats = MODULE.merge_records(originals, corrections, matches)

        self.assertFalse(unresolved)
        self.assertEqual(merged[0]["answer"], "SELECT COUNT(*) FROM users")
        self.assertEqual(merged[0]["rep_answer"], "SELECT COUNT(*) FROM users")
        self.assertEqual(merged[0]["hit_info"], {})
        self.assertTrue(manifest[0]["changes"]["sql"])
        self.assertEqual(stats["hit_info_cleared"], 1)

    def test_duplicate_exact_questions_are_matched_by_occurrence(self):
        originals = [
            {"db_name": "demo", "question": "Same question", "answer": "SELECT 1"},
            {"db_name": "demo", "question": "Same question", "answer": "SELECT 2"},
        ]
        corrections = [
            {"db_id": "demo", "question": "Same question", "SQL": "SELECT 10"},
            {"db_id": "demo", "question": "Same question", "SQL": "SELECT 20"},
        ]
        matches, unresolved = MODULE.match_corrections(originals, corrections, {}, 0.9, 0.05)

        self.assertFalse(unresolved)
        self.assertEqual(matches[0]["original_index"], 0)
        self.assertEqual(matches[1]["original_index"], 1)
        self.assertEqual(matches[0]["match_method"], "exact_occurrence")

    def test_low_similarity_question_stays_unresolved(self):
        originals = [
            {"db_name": "demo", "question": "Return customer names", "answer": "SELECT name FROM customer"}
        ]
        corrections = [
            {"db_id": "demo", "question": "Count all orders", "SQL": "SELECT COUNT(*) FROM orders"}
        ]
        matches, unresolved = MODULE.match_corrections(originals, corrections, {}, 0.9, 0.05)

        self.assertFalse(matches)
        self.assertEqual(unresolved[0]["reason"], "question_similarity_below_threshold")

    def test_manual_review_mapping_can_resolve_rewritten_question(self):
        originals = [
            {"db_name": "demo", "question": "Return customer names", "answer": "SELECT name FROM customer"}
        ]
        corrections = [
            {"db_id": "demo", "question": "Who are the clients?", "SQL": "SELECT name FROM customer"}
        ]
        matches, unresolved = MODULE.match_corrections(originals, corrections, {0: 0}, 0.9, 0.05)

        self.assertFalse(unresolved)
        self.assertEqual(matches[0]["original_index"], 0)
        self.assertEqual(matches[0]["match_method"], "manual_review")


if __name__ == "__main__":
    unittest.main()
