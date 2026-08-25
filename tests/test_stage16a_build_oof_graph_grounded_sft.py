import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "stage16a_build_oof_graph_grounded_sft",
        ROOT / "src/data/stage16a_build_oof_graph_grounded_sft.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE16A = load_module()


def load_merger():
    spec = importlib.util.spec_from_file_location(
        "stage16a_merge_oof_schema_predictions",
        ROOT / "src/evaluation/stage16a_merge_oof_schema_predictions.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MERGER = load_merger()


class Stage16AGraphGroundedSFTTest(unittest.TestCase):
    def graph(self):
        return {
            "example_id": "dev_0",
            "inference_inputs": {
                "db_id": "demo",
                "question": "Which school has the highest score?",
                "evidence": "highest means descending order",
                "schema_nodes": [
                    {"id": 0, "type": "table", "name": "schools"},
                    {
                        "id": 1,
                        "type": "column",
                        "name": "schools.name",
                        "table": "schools",
                        "column": "name",
                        "data_type": "text",
                    },
                    {
                        "id": 2,
                        "type": "column",
                        "name": "schools.score",
                        "table": "schools",
                        "column": "score",
                        "data_type": "real",
                    },
                    {"id": 3, "type": "table", "name": "districts"},
                    {
                        "id": 4,
                        "type": "column",
                        "name": "districts.school_id",
                        "table": "districts",
                        "column": "school_id",
                        "data_type": "integer",
                    },
                ],
                "schema_edges": [
                    {"src": 2, "dst": 4, "type": "foreign_key_forward"},
                    {"src": 4, "dst": 2, "type": "foreign_key_backward"},
                ],
            },
            "training_targets": {
                "sql": "SELECT name FROM schools ORDER BY score DESC LIMIT 1"
            },
            "metadata": {"record_index": 0, "question_id": 7},
        }

    def test_contract_keeps_gold_out_of_inference_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = {
                "graph": [self.graph()],
                "prediction": [
                    {
                        "record_index": 0,
                        "db_id": "demo",
                        "top_2": [
                            {"schema_item_id": 0, "score": 2.0},
                            {"schema_item_id": 1, "score": 1.5},
                        ],
                    }
                ],
                "prior": [
                    {
                        "record_index": 0,
                        "db_id": "demo",
                        "node_priors": [
                            {
                                "schema_item_id": 2,
                                "role_scores": {"ORDER_KEY": 0.9},
                            }
                        ],
                    }
                ],
                "closure": [
                    {
                        "record_index": 0,
                        "db_id": "demo",
                        "status": "connected",
                        "terminal_table_ids": [0, 3],
                        "terminal_tables": ["schools", "districts"],
                        "structural_closure_ids": [4],
                        "paths": [],
                    }
                ],
                "value": [
                    {
                        "record_index": 0,
                        "db_id": "demo",
                        "value_matches": [
                            {
                                "schema_item_id": 1,
                                "score": 1.0,
                                "matches": [
                                    {
                                        "value": "Lincoln High",
                                        "normalized_value": "lincoln high",
                                        "score": 1.0,
                                        "phrase_match": True,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
            paths = {}
            for key, values in rows.items():
                path = root / f"{key}.jsonl"
                path.write_text(
                    "\n".join(json.dumps(value) for value in values) + "\n",
                    encoding="utf-8",
                )
                paths[key] = path
            args = type(
                "Args",
                (),
                {
                    "dev_graph_file": str(paths["graph"]),
                    "dev_predictions": str(paths["prediction"]),
                    "dev_priors": str(paths["prior"]),
                    "dev_closure": str(paths["closure"]),
                    "dev_value_evidence": str(paths["value"]),
                    "limit": None,
                    "reserve_k": 10,
                    "reserve_min_role_score": 0.25,
                    "max_values_per_column": 3,
                },
            )()
            output, stats = STAGE16A.build_split(args, "dev", "cross_database_dev")
            self.assertEqual(stats["examples"], 1)
            record = output[0]
            self.assertNotIn("sql", record["inference_inputs"])
            self.assertEqual(
                record["training_targets"]["response"],
                self.graph()["training_targets"]["sql"],
            )
            reserve_ids = [
                item["id"]
                for item in record["inference_inputs"]["grounding_state"]["role_reserve"]
            ]
            self.assertEqual(reserve_ids, [2])
            self.assertEqual(
                record["inference_inputs"]["grounding_state"]["join_closure"][
                    "added_schema_items"
                ][0]["id"],
                4,
            )
            full_schema = record["inference_inputs"]["grounding_state"]["full_schema"]
            self.assertEqual(len(full_schema["foreign_keys"]), 1)
            self.assertEqual(
                [message["role"] for message in record["inference_inputs"]["prompt_messages"]],
                ["system", "user"],
            )

    def test_oof_summary_is_mandatory_and_verified(self):
        with self.assertRaisesRegex(ValueError, "train-oof-summary"):
            STAGE16A.validate_oof_summary(None, 1, "predictions.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "strict_oof": True,
                        "record_count": 1,
                        "integrity": {"database_disjoint": True},
                        "outputs": {"schema_predictions": "predictions.jsonl"},
                    }
                ),
                encoding="utf-8",
            )
            loaded = STAGE16A.validate_oof_summary(
                summary, 1, "/tmp/predictions.jsonl"
            )
            self.assertTrue(loaded["strict_oof"])

    def test_gold_keys_are_rejected_recursively(self):
        found = STAGE16A.find_forbidden_keys(
            {"grounding_state": {"debug": {"gold_ids": [1]}}}
        )
        self.assertEqual(found, ["inference_inputs.grounding_state.debug.gold_ids"])

    def test_final_oof_merger_strips_training_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold_root = root / "folds"
            manifest_folds = []
            for fold_id, indices in enumerate([[0], [1]]):
                manifest_dir = root / f"manifest_{fold_id}"
                manifest_dir.mkdir()
                heldout = manifest_dir / "heldout.json"
                heldout.write_text(
                    json.dumps(
                        {
                            "record_indices": indices,
                            "db_ids": [f"db_{fold_id}"],
                        }
                    ),
                    encoding="utf-8",
                )
                output_dir = fold_root / f"fold_{fold_id}"
                output_dir.mkdir(parents=True)
                (output_dir / "dev_predictions.jsonl").write_text(
                    json.dumps(
                        {
                            "record_index": indices[0],
                            "db_id": f"db_{fold_id}",
                            "gold_ids": [9],
                            "candidate_oracle_recall": 1.0,
                            "top_1": [{"schema_item_id": 1, "score": 1.0}],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                manifest_folds.append(
                    {"fold_id": fold_id, "heldout_index_file": str(heldout)}
                )
            predictions, _ = MERGER.merge_oof_predictions(
                {"record_count": 2, "folds": manifest_folds},
                fold_root,
                "dev_predictions.jsonl",
            )
            self.assertEqual([row["record_index"] for row in predictions], [0, 1])
            self.assertNotIn("gold_ids", predictions[0])
            self.assertEqual(predictions[0]["oof_fold_id"], 0)


if __name__ == "__main__":
    unittest.main()
