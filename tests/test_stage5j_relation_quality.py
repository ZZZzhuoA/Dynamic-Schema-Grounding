import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "data" / "stage5j_build_relation_labels.py"
SPEC = importlib.util.spec_from_file_location("stage5j_build_relation_labels", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage5JRelationQualityTest(unittest.TestCase):
    def test_parse_relation_list_rejects_unknown_relation(self):
        with self.assertRaises(ValueError):
            MODULE.parse_relation_list("METRIC_TARGET,NOT_A_RELATION")

    def test_empty_required_relations(self):
        summary = {
            "relation_stats": {
                "METRIC_TARGET": {"non_empty_samples": 3},
                "TEMPORAL_FILTER": {"non_empty_samples": 0},
            }
        }
        self.assertEqual(
            MODULE.empty_required_relations(summary, ["METRIC_TARGET", "TEMPORAL_FILTER"]),
            ["TEMPORAL_FILTER"],
        )


if __name__ == "__main__":
    unittest.main()
