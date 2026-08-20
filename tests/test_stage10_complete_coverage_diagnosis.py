import unittest


class Stage10CompleteCoverageDiagnosisTest(unittest.TestCase):
    def test_failure_layers_and_policy_transition(self):
        from src.diagnosis.stage10_complete_coverage_diagnosis import (
            policy_transition,
            summarize_policy,
        )

        records = [
            {
                "record_index": 0,
                "db_id": "demo",
                "question": "q0",
                "legacy": {1, 2, 3},
                "exact": {1, 2},
                "candidate": {1, 2},
                "raw": {1, 2},
                "constrained": {1, 2},
                "baseline": {1},
                "names": {1: "t", 2: "t.x", 3: "t.unused_fk"},
            },
            {
                "record_index": 1,
                "db_id": "demo",
                "question": "q1",
                "legacy": {1, 2},
                "exact": {1, 2},
                "candidate": {1, 2, 4},
                "raw": {1, 4},
                "constrained": {1, 4},
                "baseline": {1},
                "names": {1: "t", 2: "t.x", 4: "t.y"},
            },
        ]

        legacy, failures = summarize_policy(records, "legacy", top_k=2)
        exact, _ = summarize_policy(records, "exact", top_k=2)
        transition = policy_transition(records, "legacy", "exact")

        self.assertEqual(legacy["candidate_complete_samples"], 1)
        self.assertEqual(legacy["constrained_complete_samples"], 0)
        self.assertEqual(exact["candidate_complete_samples"], 2)
        self.assertEqual(exact["constrained_complete_samples"], 1)
        self.assertEqual(legacy["failure_reasons"]["target_exceeds_top_k"], 1)
        self.assertEqual(legacy["failure_reasons"]["reranker_raw_missing"], 1)
        self.assertEqual(len(failures), 2)
        self.assertEqual(
            transition["counts"]["old_missing__new_complete"],
            1,
        )

    def test_alternate_fk_path_satisfies_structural_coverage(self):
        from src.diagnosis.stage10_complete_coverage_diagnosis import (
            summarize_structural_coverage,
        )

        selected = {0, 1, 2, 5, 6, 7, 8, 9}
        record = {
            "record_index": 0,
            "db_id": "demo",
            "question": "alternate path",
            "semantic": {0, 1, 9},
            "required_tables": {0, 1},
            "reference_join": {3, 4},
            "reference_join_edges": [],
            "candidate": set(range(10)),
            "raw": selected,
            "constrained": selected,
            "baseline": selected,
            "names": {9: "a.value"},
            "candidate_nodes": [
                {"local_id": index, "schema_item_id": index}
                for index in range(10)
            ],
            "schema_edges": [
                {"src": 0, "dst": 5, "type": "table_to_column"},
                {"src": 2, "dst": 6, "type": "table_to_column"},
                {"src": 2, "dst": 7, "type": "table_to_column"},
                {"src": 1, "dst": 8, "type": "table_to_column"},
                {"src": 5, "dst": 6, "type": "foreign_key_forward"},
                {"src": 7, "dst": 8, "type": "foreign_key_forward"},
            ],
        }

        metrics, failures = summarize_structural_coverage([record])
        constrained = metrics["constrained"]
        self.assertEqual(constrained["semantic_complete_coverage"], 1.0)
        self.assertEqual(constrained["reference_join_complete_coverage"], 0.0)
        self.assertEqual(constrained["join_connected_coverage"], 1.0)
        self.assertEqual(constrained["grounding_complete_coverage"], 1.0)
        self.assertEqual(
            metrics["constrained_outcomes"]["alternate_join_path_accepted"],
            1,
        )
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
