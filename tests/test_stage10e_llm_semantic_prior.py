import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = load_module(
    "stage10e_generate_llm_semantic_priors",
    "src/data/stage10e_generate_llm_semantic_priors.py",
)
ATTACH = load_module(
    "stage10e_attach_llm_semantic_priors",
    "src/data/stage10e_attach_llm_semantic_priors.py",
)
TRAINING = load_module(
    "stage10_train_factor_graph_reranker_stage10e",
    "src/training/stage10_train_factor_graph_reranker.py",
)


def example():
    return {
        "record_index": 7,
        "db_id": "demo",
        "question_id": 7,
        "question": "List school names in Alameda",
        "evidence": "Alameda is a county value",
        "gold_ids": [999],  # Must never enter the LLM prompt.
        "candidate_nodes": [
            {
                "local_id": 0,
                "schema_item_id": 10,
                "type": "table",
                "name": "schools",
                "owner_table_id": 10,
                "numeric_features": [1.0, 0.0],
            },
            {
                "local_id": 1,
                "schema_item_id": 11,
                "type": "column",
                "name": "schools.name",
                "owner_table_id": 10,
                "numeric_features": [0.0, 1.0],
            },
            {
                "local_id": 2,
                "schema_item_id": 12,
                "type": "column",
                "name": "schools.county",
                "owner_table_id": 10,
                "numeric_features": [0.0, 1.0],
            },
        ],
    }


def prior(row):
    return {
        "record_index": row["record_index"],
        "source_fingerprint": GENERATOR.source_fingerprint(row),
        "status": "ok",
        "node_priors": [
            {
                "schema_item_id": 11,
                "role_scores": {"OUTPUT_TARGET": 0.9},
            },
            {
                "schema_item_id": 12,
                "role_scores": {"PREDICATE_COLUMN": 0.8},
            },
        ],
    }


class Stage10ELLMSemanticPriorTest(unittest.TestCase):
    def test_prompt_excludes_gold_and_sql(self):
        messages = GENERATOR.build_messages(example())
        rendered = "\n".join(message["content"] for message in messages)
        self.assertNotIn("999", rendered)
        self.assertNotIn("gold_ids", rendered)
        self.assertIn("schools.county", rendered)

    def test_validator_rejects_hallucinated_ids_and_clamps_confidence(self):
        raw = {
            "roles": {
                "OUTPUT_TARGET": [
                    {"schema_id": 11, "confidence": 1.7},
                    {"schema_id": 999, "confidence": 0.8},
                ]
            }
        }
        node_priors, rejected = GENERATOR.validate_prior(raw, example())
        self.assertEqual(len(node_priors), 1)
        self.assertEqual(node_priors[0]["schema_item_id"], 11)
        self.assertEqual(node_priors[0]["role_scores"]["OUTPUT_TARGET"], 1.0)
        self.assertEqual(rejected[0]["reason"], "unknown_schema_id")

    def test_attach_normal_zero_and_shuffled_controls(self):
        row = example()
        source_prior = prior(row)
        normal = ATTACH.attach_one(row, source_prior, "normal", seed=42)
        zero = ATTACH.attach_one(row, source_prior, "zero", seed=42)
        shuffled = ATTACH.attach_one(
            row, source_prior, "shuffled_node_identity", seed=42
        )
        base_dim = 2
        self.assertEqual(
            len(normal["candidate_nodes"][0]["numeric_features"]),
            base_dim + len(ATTACH.FEATURE_NAMES),
        )
        self.assertTrue(
            all(
                value == 0.0
                for node in zero["candidate_nodes"]
                for value in node["numeric_features"][base_dim:]
            )
        )
        normal_vectors = [
            node["numeric_features"][base_dim:] for node in normal["candidate_nodes"]
        ]
        shuffled_vectors = [
            node["numeric_features"][base_dim:]
            for node in shuffled["candidate_nodes"]
        ]
        self.assertCountEqual(normal_vectors, shuffled_vectors)
        self.assertNotEqual(normal_vectors, shuffled_vectors)

    def test_fingerprint_detects_candidate_changes(self):
        row = example()
        changed = copy.deepcopy(row)
        changed["candidate_nodes"][1]["name"] = "schools.other_name"
        self.assertNotEqual(
            GENERATOR.source_fingerprint(row),
            GENERATOR.source_fingerprint(changed),
        )

    def test_control_alignment_checks_node_identity_and_dimension(self):
        source = example()
        row = ATTACH.attach_one(source, prior(source), "normal")
        control = ATTACH.attach_one(source, prior(source), "zero")
        dimension = len(row["candidate_nodes"][0]["numeric_features"])
        TRAINING.validate_control_alignment([row], [control], "zero", dimension)
        broken = copy.deepcopy(control)
        broken["candidate_nodes"][0]["schema_item_id"] = 77
        with self.assertRaises(ValueError):
            TRAINING.validate_control_alignment([row], [broken], "zero", dimension)


if __name__ == "__main__":
    unittest.main()
