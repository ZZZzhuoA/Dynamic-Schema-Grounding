import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5_train_dsg_grounder import write_json, write_jsonl  # noqa: E402


ROLE_ORDER = [
    "OUTPUT_TARGET",
    "ENTITY_NAME",
    "METRIC_TARGET",
    "PREDICATE_COLUMN",
    "VALUE_ANCHOR",
    "TEMPORAL_FILTER",
    "ORDER_KEY",
    "GROUP_KEY",
    "JOIN_BRIDGE",
    "FORMULA_COMPONENT",
]
FORBIDDEN_INFERENCE_KEYS = {
    "gold_ids",
    "gold_sql",
    "grounding_label_ids",
    "grounding_label_names",
    "training_targets",
    "whole_labels",
    "role_labels",
}


def read_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def index_rows(rows, label):
    indexed = {}
    for position, row in enumerate(rows):
        index = int(row.get("record_index", row.get("metadata", {}).get("record_index", position)))
        if index in indexed:
            raise ValueError(f"Duplicate {label} record_index={index}")
        indexed[index] = row
    return indexed


def source_digest(paths):
    digest = hashlib.sha256()
    for path in paths:
        if not path:
            continue
        resolved = Path(path)
        digest.update(str(resolved).encode("utf-8"))
        if resolved.exists():
            stat = resolved.stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()


def validate_oof_summary(summary_path, expected_count, prediction_path):
    if not summary_path:
        raise ValueError(
            "Train grounding requires --train-oof-summary. Use "
            "--allow-unverified-train-grounding only for debugging, never for SFT."
        )
    summary = read_json(summary_path)
    if summary.get("strict_oof") is not True:
        raise ValueError("OOF summary must contain strict_oof=true")
    count = summary.get("record_count")
    if count is not None and int(count) != int(expected_count):
        raise ValueError(
            f"OOF summary record_count={count} does not match train examples={expected_count}"
        )
    integrity = summary.get("integrity", {})
    false_checks = [key for key, value in integrity.items() if value is False]
    if false_checks:
        raise ValueError(f"OOF integrity checks failed: {false_checks}")
    outputs = summary.get("outputs", {})
    declared = outputs.get("schema_predictions") or outputs.get("predictions")
    if declared and Path(declared).name != Path(prediction_path).name:
        raise ValueError(
            "OOF summary prediction output does not match --train-predictions: "
            f"{declared} vs {prediction_path}"
        )
    return summary


def graph_parts(graph):
    inputs = graph.get("inference_inputs", graph)
    nodes = inputs.get("schema_nodes", [])
    edges = inputs.get("schema_edges", [])
    by_id = {int(node["id"]): node for node in nodes}
    return inputs, nodes, edges, by_id


def compact_node(node, score=None, role_scores=None, source=None):
    row = {
        "id": int(node["id"]),
        "type": node.get("type"),
        "name": node.get("name"),
    }
    for key in ["table", "column", "data_type"]:
        if node.get(key) is not None:
            row[key] = node.get(key)
    if score is not None:
        row["belief_score"] = round(float(score), 6)
    if role_scores:
        ranked = sorted(
            ((str(role), float(value)) for role, value in role_scores.items()),
            key=lambda item: (-item[1], item[0]),
        )
        row["role_scores"] = {
            role: round(value, 6) for role, value in ranked if value > 0.0
        }
    if source:
        row["source"] = source
    return row


def compact_full_schema(nodes, edges):
    tables = []
    columns_by_table = defaultdict(list)
    by_id = {int(node["id"]): node for node in nodes}
    for node in nodes:
        if node.get("type") == "table":
            tables.append({"id": int(node["id"]), "name": node.get("name")})
        elif node.get("type") == "column":
            columns_by_table[str(node.get("table") or "")].append(
                {
                    "id": int(node["id"]),
                    "name": node.get("name"),
                    "column": node.get("column"),
                    "data_type": node.get("data_type"),
                }
            )
    table_rows = []
    for table in sorted(tables, key=lambda item: item["id"]):
        table_rows.append(
            {
                **table,
                "columns": sorted(
                    columns_by_table.get(str(table["name"]), []),
                    key=lambda item: item["id"],
                ),
            }
        )
    foreign_keys = []
    seen = set()
    for edge in edges:
        if not str(edge.get("type", "")).startswith("foreign_key"):
            continue
        src, dst = int(edge["src"]), int(edge["dst"])
        pair = tuple(sorted((src, dst)))
        if pair in seen or src not in by_id or dst not in by_id:
            continue
        seen.add(pair)
        foreign_keys.append(
            {
                "left_id": src,
                "left": by_id[src].get("name"),
                "right_id": dst,
                "right": by_id[dst].get("name"),
            }
        )
    return {"tables": table_rows, "foreign_keys": foreign_keys}


def prior_map(prior):
    result = {}
    for item in (prior or {}).get("node_priors", []):
        item_id = int(item.get("schema_item_id", item.get("id")))
        result[item_id] = {
            str(role): float(score)
            for role, score in (item.get("role_scores") or {}).items()
        }
    return result


def prediction_items(prediction):
    keys = [key for key in prediction if key.startswith("top_")]
    if not keys:
        raise ValueError(
            f"Prediction record_index={prediction.get('record_index')} has no top_K field"
        )
    key = max(keys, key=lambda item: int(item.split("_", 1)[1]))
    return key, prediction[key]


def build_reserve(by_id, roles, core_ids, reserve_k, threshold):
    candidates = []
    for item_id, role_scores in roles.items():
        if item_id in core_ids or item_id not in by_id:
            continue
        max_role = max(role_scores.values(), default=0.0)
        if max_role < threshold:
            continue
        candidates.append((max_role, item_id, role_scores))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [
        compact_node(by_id[item_id], score=score, role_scores=role_scores, source="role_reserve")
        for score, item_id, role_scores in candidates[:reserve_k]
    ]


def compact_value_evidence(evidence, by_id, max_matches):
    rows = []
    for match in (evidence or {}).get("value_matches", []):
        item_id = int(match["schema_item_id"])
        node = by_id.get(item_id)
        if not node:
            continue
        values = []
        for value in match.get("matches", [])[:max_matches]:
            values.append(
                {
                    "value": value.get("value"),
                    "normalized_value": value.get("normalized_value"),
                    "score": round(float(value.get("score", 0.0)), 6),
                    "phrase_match": bool(value.get("phrase_match", False)),
                }
            )
        rows.append(
            {
                "schema_item_id": item_id,
                "schema_item": node.get("name"),
                "confidence": round(
                    float(match.get("value_confidence", match.get("score", 0.0))), 6
                ),
                "decision": match.get("value_decision"),
                "matches": values,
            }
        )
    return rows


def compact_join_closure(closure, by_id):
    closure = closure or {}
    added = [
        compact_node(by_id[item_id], source="join_closure")
        for item_id in map(int, closure.get("structural_closure_ids", []))
        if item_id in by_id
    ]
    return {
        "status": closure.get("status", "not_available"),
        "terminal_table_ids": [int(value) for value in closure.get("terminal_table_ids", [])],
        "terminal_tables": closure.get("terminal_tables", []),
        "added_schema_items": added,
        "paths": closure.get("paths", []),
    }


def build_user_content(inputs, grounding_state):
    payload = {
        "task": "Generate one valid SQLite query that answers the question.",
        "question": inputs.get("question"),
        "evidence": inputs.get("evidence") or "",
        "grounding_policy": {
            "semantic_core": "High-confidence graph belief; prefer it but do not treat it as a hard whitelist.",
            "role_reserve": "Role-conditioned fallback candidates for uncertain semantic mappings.",
            "join_closure": "Declared FK paths connecting grounded terminal tables.",
            "full_schema": "Final fallback; identifiers outside the semantic core remain legal.",
        },
        "grounding_state": grounding_state,
        "output_contract": "Return only the SQL query, without Markdown or explanation.",
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def find_forbidden_keys(value, path="inference_inputs"):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_INFERENCE_KEYS:
                found.append(f"{path}.{key}")
            found.extend(find_forbidden_keys(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_forbidden_keys(child, f"{path}[{index}]"))
    return found


def aligned_row(indexed, index, label, required=True):
    row = indexed.get(index) if indexed is not None else None
    if row is None and required:
        raise ValueError(f"Missing {label} row for record_index={index}")
    return row


def validate_identity(index, graph_inputs, *rows):
    db_id = graph_inputs.get("db_id")
    for label, row in rows:
        if row is None:
            continue
        if int(row.get("record_index", index)) != index:
            raise ValueError(f"Misaligned {label} record_index at {index}")
        row_db = row.get("db_id")
        if row_db is not None and str(row_db) != str(db_id):
            raise ValueError(
                f"Misaligned {label} db_id at record_index={index}: {row_db} != {db_id}"
            )


def build_split(args, split, provenance):
    graph_path = getattr(args, f"{split}_graph_file")
    prediction_path = getattr(args, f"{split}_predictions")
    prior_path = getattr(args, f"{split}_priors")
    closure_path = getattr(args, f"{split}_closure")
    value_path = getattr(args, f"{split}_value_evidence")
    graphs = read_jsonl(graph_path, args.limit)
    predictions = index_rows(read_jsonl(prediction_path), f"{split} predictions")
    priors = index_rows(read_jsonl(prior_path), f"{split} priors") if prior_path else {}
    closures = index_rows(read_jsonl(closure_path), f"{split} closure") if closure_path else {}
    values = index_rows(read_jsonl(value_path), f"{split} value evidence") if value_path else {}
    outputs = []
    stats = Counter()
    source_paths = [graph_path, prediction_path, prior_path, closure_path, value_path]
    digest = source_digest(source_paths)
    for position, graph in enumerate(graphs):
        inputs, nodes, edges, by_id = graph_parts(graph)
        index = int(graph.get("metadata", {}).get("record_index", position))
        prediction = aligned_row(predictions, index, "prediction")
        prior = aligned_row(priors, index, "prior", required=bool(prior_path))
        closure = aligned_row(closures, index, "closure", required=bool(closure_path))
        value = aligned_row(values, index, "value evidence", required=False)
        validate_identity(
            index,
            inputs,
            ("prediction", prediction),
            ("prior", prior),
            ("closure", closure),
            ("value evidence", value),
        )
        top_key, selected = prediction_items(prediction)
        roles = prior_map(prior)
        core = []
        core_ids = set()
        for item in selected:
            item_id = int(item.get("schema_item_id", item.get("id")))
            if item_id not in by_id:
                raise ValueError(f"Unknown core schema_item_id={item_id} at record_index={index}")
            core_ids.add(item_id)
            core.append(
                compact_node(
                    by_id[item_id],
                    score=item.get("score"),
                    role_scores=roles.get(item_id),
                    source="schema_rgta",
                )
            )
        reserve = build_reserve(
            by_id, roles, core_ids, args.reserve_k, args.reserve_min_role_score
        )
        grounding_state = {
            "semantic_core": core,
            "role_reserve": reserve,
            "value_bindings": compact_value_evidence(value, by_id, args.max_values_per_column),
            "join_closure": compact_join_closure(closure, by_id),
            "full_schema": compact_full_schema(nodes, edges),
        }
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Text-to-SQL model. Use graph grounding as "
                    "graded evidence, preserve exact SQLite identifiers and values, "
                    "and return only executable SQL."
                ),
            },
            {"role": "user", "content": build_user_content(inputs, grounding_state)},
        ]
        target = graph.get("training_targets", {})
        sql = target.get("sql")
        if not sql:
            raise ValueError(f"Missing training_targets.sql at record_index={index}")
        inference_inputs = {
            "db_id": inputs.get("db_id"),
            "question": inputs.get("question"),
            "evidence": inputs.get("evidence") or "",
            "prompt_messages": prompt_messages,
            "grounding_state": grounding_state,
        }
        forbidden = find_forbidden_keys(inference_inputs)
        if forbidden:
            raise ValueError(f"Gold leakage keys at record_index={index}: {forbidden}")
        serialized_prompt = json.dumps(prompt_messages, ensure_ascii=False)
        if str(sql).strip() and str(sql).strip() in serialized_prompt:
            raise ValueError(f"Gold SQL leaked into prompt at record_index={index}")
        outputs.append(
            {
                "example_id": graph.get("example_id", f"{split}_{index}"),
                "record_index": index,
                "db_id": inputs.get("db_id"),
                "question_id": graph.get("metadata", {}).get("question_id", index),
                "inference_inputs": inference_inputs,
                "training_targets": {"response": str(sql), "response_format": "sql"},
                "metadata": {
                    "split": split,
                    "grounding_provenance": provenance,
                    "semantic_top_key": top_key,
                    "semantic_core_count": len(core),
                    "role_reserve_count": len(reserve),
                    "join_closure_status": grounding_state["join_closure"]["status"],
                    "value_binding_count": len(grounding_state["value_bindings"]),
                    "source_digest": digest,
                    "inference_gold_leakage": False,
                },
            }
        )
        stats["examples"] += 1
        stats["semantic_core_items"] += len(core)
        stats["reserve_items"] += len(reserve)
        stats["value_bindings"] += len(grounding_state["value_bindings"])
        stats[f"closure::{grounding_state['join_closure']['status']}"] += 1
    expected = set(range(len(graphs)))
    observed = {int(row["record_index"]) for row in outputs}
    if observed != expected:
        raise ValueError(
            f"{split} graph record indices must be dense 0..N-1; "
            f"missing={sorted(expected-observed)[:10]}, extra={sorted(observed-expected)[:10]}"
        )
    return outputs, dict(stats)


def main():
    parser = argparse.ArgumentParser(
        description="Build leakage-safe Stage 16-A OOF Graph-grounded direct-SQL SFT data."
    )
    for split in ["train", "dev"]:
        parser.add_argument(f"--{split}-graph-file", required=True)
        parser.add_argument(f"--{split}-predictions", required=True)
        parser.add_argument(f"--{split}-priors", default=None)
        parser.add_argument(f"--{split}-closure", default=None)
        parser.add_argument(f"--{split}-value-evidence", default=None)
    parser.add_argument("--train-oof-summary", default=None)
    parser.add_argument("--allow-unverified-train-grounding", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reserve-k", type=int, default=20)
    parser.add_argument("--reserve-min-role-score", type=float, default=0.25)
    parser.add_argument("--max-values-per-column", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.reserve_k < 0:
        parser.error("--reserve-k must be non-negative")
    if not 0.0 <= args.reserve_min_role_score <= 1.0:
        parser.error("--reserve-min-role-score must be in [0,1]")
    if args.max_values_per_column < 1:
        parser.error("--max-values-per-column must be positive")

    train_graphs = read_jsonl(args.train_graph_file, args.limit)
    if args.allow_unverified_train_grounding:
        provenance = "unverified_debug_only"
        oof_summary = None
    else:
        oof_summary = validate_oof_summary(
            args.train_oof_summary, len(train_graphs), args.train_predictions
        )
        provenance = "strict_database_disjoint_oof"

    train_rows, train_stats = build_split(args, "train", provenance)
    dev_rows, dev_stats = build_split(args, "dev", "cross_database_dev")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train_sft.jsonl", train_rows)
    write_jsonl(output_dir / "dev_sft.jsonl", dev_rows)

    def summarize(rows, stats):
        count = max(len(rows), 1)
        return {
            "example_count": len(rows),
            "avg_semantic_core_items": stats.get("semantic_core_items", 0) / count,
            "avg_role_reserve_items": stats.get("reserve_items", 0) / count,
            "avg_value_bindings": stats.get("value_bindings", 0) / count,
            "closure_status_counts": {
                key.split("::", 1)[1]: value
                for key, value in stats.items()
                if key.startswith("closure::")
            },
            "gold_leakage_violations": 0,
        }

    summary = {
        "config": vars(args),
        "train": summarize(train_rows, train_stats),
        "dev": summarize(dev_rows, dev_stats),
        "strict_oof": not args.allow_unverified_train_grounding,
        "oof_summary": oof_summary,
        "data_contract": {
            "input": "inference_inputs.prompt_messages + grounding_state",
            "target": "training_targets.response",
            "target_language": "direct SQLite SQL (no IR)",
            "semantic_core_policy": "graded evidence, not a hard whitelist",
            "fallback_policy": "role reserve + complete schema + declared FK closure",
        },
        "leakage_note": (
            "Gold SQL is stored only under training_targets. Gold schema IDs, labels, "
            "candidate oracle statistics, and assistant answers are excluded from inference_inputs."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
