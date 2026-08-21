import unittest

from src.diagnosis.stage10f_organize_semantic_misses import (
    item_pipeline_reason,
    review_flags,
    sample_bucket,
)


class Stage10FSemanticMissOrganizationTest(unittest.TestCase):
    def test_pipeline_reason_separates_candidate_and_selection_failures(self):
        priors = {4: {"OUTPUT_TARGET": 0.8}, 5: {"OUTPUT_TARGET": 0.1}}
        self.assertEqual(
            item_pipeline_reason(3, 4, 30, {4, 5}, set(), priors, 0.5),
            "candidate_generation_gap",
        )
        self.assertEqual(
            item_pipeline_reason(4, 4, 30, {4, 5}, set(), priors, 0.5),
            "llm_supported_but_ranking_drop",
        )
        self.assertEqual(
            item_pipeline_reason(5, 4, 30, {4, 5}, {5}, priors, 0.5),
            "constrained_selector_drop",
        )

    def test_review_flags_are_candidates_not_annotation_verdicts(self):
        node = {"type": "column", "name": "orders.amount", "column": "amount", "data_type": "real"}
        flags, q_overlap, _ = review_flags(
            node, "Which customer spent the most?", "", "SELECT SUM(price) FROM orders"
        )
        self.assertFalse(q_overlap)
        self.assertIn("implicit_or_unexpressed_mapping", flags)
        self.assertIn("target_name_not_visible_in_gold_sql_surface", flags)

    def test_sample_bucket_keeps_mixed_failures_explicit(self):
        self.assertEqual(
            sample_bucket(
                ["candidate_generation_gap", "constrained_selector_drop"], []
            ),
            "mixed_pipeline_failures",
        )


if __name__ == "__main__":
    unittest.main()
