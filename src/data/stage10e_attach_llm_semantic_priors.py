"""Attach frozen-LLM semantic priors to Stage 10 candidate node features."""

import argparse
import copy
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stage10e_generate_llm_semantic_priors import (
    SEMANTIC_ROLES,
    read_jsonl,
    source_fingerprint,
)


FEATURE_NAMES = ["llm_prior_present", "llm_prior_max", "llm_prior_mean"] + [
    f"llm_role::{role}" for role in SEMANTIC_ROLES
]


def prior_vectors(example, prior):
    by_id = {
        int(row["schema_item_id"]): row.get("role_scores", {})
        for row in prior.get("node_priors", [])
    }
    vectors = []
    valid_prior = prior.get("status") == "ok"
    for node in example.get("candidate_nodes", []):
        scores = [
            min(max(float(by_id.get(int(node["schema_item_id"]), {}).get(role, 0.0)), 0.0), 1.0)
            for role in SEMANTIC_ROLES
        ]
        vectors.append(
            [
                float(valid_prior),
                max(scores, default=0.0),
                sum(scores) / len(scores) if scores else 0.0,
                *scores,
            ]
        )
    return vectors


def attach_one(example, prior, control_mode="normal", seed=42):
    if prior.get("source_fingerprint") != source_fingerprint(example):
        raise ValueError(
            f"Prior/source mismatch at record_index={example.get('record_index')}"
        )
    vectors = prior_vectors(example, prior)
    if control_mode == "zero":
        vectors = [[0.0] * len(FEATURE_NAMES) for _ in vectors]
    elif control_mode == "shuffled_node_identity":
        order = list(range(len(vectors)))
        random.Random(seed + int(example["record_index"])).shuffle(order)
        vectors = [vectors[index] for index in order]
    elif control_mode != "normal":
        raise ValueError(f"Unsupported control_mode={control_mode}")

    result = copy.deepcopy(example)
    for node, vector in zip(result.get("candidate_nodes", []), vectors):
        node["numeric_features"] = [
            *[float(value) for value in node.get("numeric_features", [])],
            *vector,
        ]
    result["llm_semantic_prior"] = {
        "mode": control_mode,
        "source_status": prior.get("status"),
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "inference_only": True,
    }
    return result


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Attach Stage 10E LLM priors to factor graphs.")
    parser.add_argument("--factor-graph-file", required=True)
    parser.add_argument("--prior-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument(
        "--control-mode",
        choices=["normal", "zero", "shuffled_node_identity"],
        default="normal",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples = read_jsonl(args.factor_graph_file, args.limit)
    priors = {int(row["record_index"]): row for row in read_jsonl(args.prior_file)}
    missing = [int(row["record_index"]) for row in examples if int(row["record_index"]) not in priors]
    if missing:
        raise ValueError(f"Missing semantic priors for {len(missing)} records; first={missing[:10]}")
    augmented = [
        attach_one(example, priors[int(example["record_index"])], args.control_mode, args.seed)
        for example in examples
    ]
    write_jsonl(args.output_file, augmented)
    old_dim = (
        len(examples[0]["candidate_nodes"][0]["numeric_features"])
        if examples and examples[0].get("candidate_nodes")
        else 0
    )
    summary = {
        "config": vars(args),
        "sample_count": len(augmented),
        "control_mode": args.control_mode,
        "base_numeric_dim": old_dim,
        "llm_prior_dim": len(FEATURE_NAMES),
        "output_numeric_dim": old_dim + len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "error_prior_count": sum(
            priors[int(row["record_index"])].get("status") != "ok" for row in examples
        ),
        "gradient_policy": "LLM prior is cached inference output; only downstream graph modules are trainable.",
    }
    summary_path = Path(args.output_file).with_name(Path(args.output_file).stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {args.output_file}")


if __name__ == "__main__":
    main()
