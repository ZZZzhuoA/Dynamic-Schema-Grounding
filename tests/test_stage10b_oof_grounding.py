import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FOLDS = load_module(
    "stage10b_build_oof_folds",
    "src/data/stage10b_build_oof_folds.py",
)
TRAINING = load_module(
    "stage8g_train_dense_relation_grounder_oof",
    "src/training/stage8g_train_dense_relation_grounder.py",
)


class Stage10BOOFFoldTest(unittest.TestCase):
    def test_database_disjoint_balanced_assignment(self):
        records = []
        for db_id, count in [("large", 7), ("medium", 5), ("small", 3), ("tiny", 2)]:
            records.extend({"db_id": db_id} for _ in range(count))
        folds = FOLDS.assign_database_folds(records, fold_count=3, seed=42)
        integrity = FOLDS.validate_folds(records, folds)

        self.assertTrue(integrity["database_disjoint"])
        self.assertEqual(integrity["heldout_record_count"], len(records))
        heldout = [index for fold in folds for index in fold["record_indices"]]
        self.assertEqual(sorted(heldout), list(range(len(records))))
        for fold in folds:
            heldout_databases = set(fold["db_ids"])
            train_databases = {
                records[index]["db_id"] for index in fold["train_record_indices"]
            }
            self.assertFalse(heldout_databases & train_databases)

    def test_assignment_is_deterministic(self):
        records = [{"db_id": f"db_{index // 3}"} for index in range(30)]
        first = FOLDS.assign_database_folds(records, fold_count=5, seed=9)
        second = FOLDS.assign_database_folds(records, fold_count=5, seed=9)
        self.assertEqual(first, second)

    def test_record_index_filter_preserves_original_indices(self):
        aligned = [{"record_index": index} for index in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indices.json"
            path.write_text(
                json.dumps({"record_indices": [4, 1, 5]}), encoding="utf-8"
            )
            selected = TRAINING.select_aligned_records(aligned, path, limit=2)
        self.assertEqual([row["record_index"] for row in selected], [4, 1])

    def test_duplicate_record_indices_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indices.json"
            path.write_text(
                json.dumps({"record_indices": [1, 1]}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                TRAINING.load_record_index_filter(path)


if __name__ == "__main__":
    unittest.main()
