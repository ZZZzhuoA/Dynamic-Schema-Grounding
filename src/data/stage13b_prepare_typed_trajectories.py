"""Filter Stage 13-A records into clean, action-synchronous trajectories."""

import argparse
import json
from collections import Counter
from pathlib import Path


ACTIONS = [
    "SCAN", "JOIN", "FILTER", "AGGREGATE", "HAVING_FILTER",
    "SORT", "LIMIT", "PROJECT", "STOP",
]
VALUE_ROUTES = [
    "question", "evidence", "semantic_inference", "operator_inference_required",
    "expression_constant", "database_value_required", "casefold_lookup",
]
OPERATORS = [
    "COUNT", "SUM", "AVG", "MIN", "MAX", "GROUP_CONCAT", "CAST", "COALESCE",
    "STRFTIME", "DATE", "DATETIME", "JULIANDAY", "ROUND", "SUBSTR", "REPLACE",
    "LENGTH", "LOWER", "UPPER", "=", "!=", "<>", "<=", ">=", "<", ">",
    "BETWEEN", "LIKE", "IN", "IS NULL", "IS NOT NULL", "+", "-", "*", "/",
    "DISTINCT", "DESC", "ASC", "OTHER",
]


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def value_route(value):
    if value.get("match") == "casefold_only":
        return "casefold_lookup"
    source = value.get("source", "database_value_required")
    return source if source in VALUE_ROUTES else "database_value_required"


def normalized_operator(value):
    value = str(value).strip().upper()
    return value if value in OPERATORS else "OTHER"


def exclusion_reasons(row):
    audit = row.get("audit", {})
    reasons = []
    if audit.get("parse_status") != "supported_flat":
        reasons.append(str(audit.get("parse_status") or "unsupported_parse"))
    if float(audit.get("schema_label_coverage", 0.0)) < 1.0:
        reasons.append("incomplete_schema_assignment")
    join_path = row.get("training_targets", {}).get("join_path", {})
    if len(join_path.get("table_pointer_ids", [])) > 1 and not join_path.get("connected", False):
        reasons.append("disconnected_join_path")
    inputs = row.get("inference_inputs", {})
    if not inputs.get("schema_items") or not inputs.get("schema_edges"):
        reasons.append("empty_schema_graph")
    if not row.get("training_targets", {}).get("action_sequence"):
        reasons.append("empty_action_sequence")
    return list(dict.fromkeys(reasons))


def prepare_steps(row):
    steps = []
    for raw in row["training_targets"]["action_sequence"]:
        routes = sorted({value_route(value) for value in raw.get("value_targets", [])})
        step = {
            "step_index": len(steps),
            "action": raw["action"],
            "table_pointer_ids": sorted(set(raw.get("table_pointer_ids", []))),
            "column_pointer_ids": sorted(set(raw.get("column_pointer_ids", []))),
            "operator_targets": sorted({
                normalized_operator(value) for value in raw.get("operator_targets", [])
            }),
            "value_routes": routes,
            "value_targets": list(raw.get("value_targets", [])),
            "input_node_ids": list(raw.get("input_node_ids", [])),
            "join_edge_targets": [],
        }
        if raw["action"] == "JOIN":
            step["join_edge_targets"] = list(
                row["training_targets"].get("join_path", {}).get("edge_targets", [])
            )
        steps.append(step)
    steps.append(
        {
            "step_index": len(steps),
            "action": "STOP",
            "table_pointer_ids": [],
            "column_pointer_ids": [],
            "operator_targets": [],
            "value_routes": [],
            "value_targets": [],
            "input_node_ids": [row["training_targets"]["relational_algebra"]["root_node_id"]],
            "join_edge_targets": [],
        }
    )
    return steps


def prepare_row(row):
    result = {
        "split": row.get("split"),
        "record_index": int(row["record_index"]),
        "question_id": row.get("question_id"),
        "db_id": row.get("db_id"),
        "inference_inputs": row["inference_inputs"],
        "teacher_steps": prepare_steps(row),
        "metadata": {
            "typed_ra_version": row["training_targets"]["relational_algebra"].get("version"),
            "action_count_without_stop": len(row["training_targets"]["action_sequence"]),
        },
    }
    return result


def build_split(input_path, output_path):
    rows = read_jsonl(input_path)
    clean, excluded = [], []
    reason_counts = Counter()
    action_counts = Counter()
    operator_counts = Counter()
    join_target_count = 0
    join_candidate_target_count = 0
    for row in rows:
        reasons = exclusion_reasons(row)
        if reasons:
            reason_counts.update(reasons)
            excluded.append(
                {
                    "record_index": row.get("record_index"),
                    "db_id": row.get("db_id"),
                    "question": row.get("inference_inputs", {}).get("question"),
                    "reasons": reasons,
                }
            )
            continue
        prepared = prepare_row(row)
        clean.append(prepared)
        action_counts.update(step["action"] for step in prepared["teacher_steps"])
        operator_counts.update(
            operator for step in prepared["teacher_steps"]
            for operator in step.get("operator_targets", [])
        )
        fk_pairs = {
            tuple(sorted((int(edge["src"]), int(edge["dst"]))))
            for edge in prepared["inference_inputs"].get("schema_edges", [])
            if edge.get("type") in {"foreign_key_forward", "foreign_key_backward"}
        }
        for step in prepared["teacher_steps"]:
            for edge in step.get("join_edge_targets", []):
                pair = tuple(sorted((int(edge["left_column_id"]), int(edge["right_column_id"]))))
                join_target_count += 1
                join_candidate_target_count += int(pair in fk_pairs)
    write_jsonl(output_path, clean)
    write_jsonl(Path(output_path).with_name(Path(output_path).stem + "_excluded.jsonl"), excluded)
    return {
        "input_count": len(rows),
        "clean_count": len(clean),
        "clean_rate": len(clean) / len(rows) if rows else 0.0,
        "excluded_count": len(excluded),
        "exclusion_reason_counts": dict(reason_counts),
        "action_counts": dict(action_counts),
        "operator_counts": dict(operator_counts),
        "join_edge_target_count": join_target_count,
        "join_edge_candidate_target_count": join_candidate_target_count,
        "join_edge_candidate_coverage": (
            join_candidate_target_count / join_target_count if join_target_count else 1.0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="experiments/stage13a_typed_ra_typefix1/train_typed_ra.jsonl")
    parser.add_argument("--dev-file", default="experiments/stage13a_typed_ra_typefix1/dev_typed_ra.jsonl")
    parser.add_argument("--output-dir", default="experiments/stage13b_clean_typed_trajectories")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    summary = {
        "config": vars(args),
        "train": build_split(args.train_file, output_dir / "train_trajectories.jsonl"),
        "dev": build_split(args.dev_file, output_dir / "dev_trajectories.jsonl"),
        "action_vocabulary": ACTIONS,
        "value_route_vocabulary": VALUE_ROUTES,
        "operator_vocabulary": OPERATORS,
        "leakage_note": "Only test-time inputs and typed teacher targets are retained; gold SQL text is removed.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
