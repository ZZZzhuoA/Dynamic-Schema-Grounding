"""Build leakage-safe semantic slot requests for Stage 14B.

The source Stage 13B trajectories contain typed targets.  This builder keeps
those targets under ``training_targets`` while constructing each slot query
only from test-time available question/evidence text, the action name, and
literal surfaces already present in the question/evidence.  Schema names and
gold SQL expressions are never copied into the slot text.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ACTION_FOCUS = {
    "SCAN": "identify the entity or relation that must be read",
    "JOIN": "identify relations and keys needed to connect the requested entities",
    "FILTER": "identify the attribute constrained by this condition",
    "AGGREGATE": "identify the measure or entity consumed by the aggregation",
    "HAVING_FILTER": "identify the aggregated measure constrained after grouping",
    "SORT": "identify the attribute or measure that determines ordering",
    "LIMIT": "identify the requested rank or result count",
    "PROJECT": "identify the attribute or expression requested as output",
    "STOP": "finish the relational plan",
}

ACTION_CUES = {
    "PROJECT": ("list", "show", "give", "what", "which", "name", "find", "return"),
    "FILTER": ("where", "with", "whose", "that", "having", "between", "in ", "from "),
    "AGGREGATE": ("count", "number", "average", "total", "sum", "percentage", "rate"),
    "HAVING_FILTER": ("at least", "more than", "less than", "having"),
    "SORT": ("highest", "lowest", "top", "bottom", "most", "least", "ascending", "descending"),
    "LIMIT": ("top", "first", "last", "highest", "lowest"),
    "JOIN": ("whose", "with", "belong", "made by", "located", "associated"),
    "SCAN": (),
    "STOP": (),
}


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def sentence_fragments(text):
    return [clean_text(part) for part in re.split(r"(?<=[?.!;])\s+|\s+(?:and|but)\s+", text) if clean_text(part)]


def local_value_context(question, evidence, value_targets, radius=72):
    contexts = []
    for target in value_targets:
        surface = clean_text(target.get("raw_sql_literal") or target.get("canonical_value"))
        source = str(target.get("source") or "")
        if source not in {"question", "evidence"}:
            continue
        text = question if source == "question" else evidence
        start, end = target.get("start"), target.get("end")
        if text and isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            contexts.append(clean_text(text[max(0, start - radius) : min(len(text), end + radius)]))
            continue
        if surface and text:
            position = text.casefold().find(surface.casefold())
            if position >= 0:
                contexts.append(clean_text(text[max(0, position - radius) : position + len(surface) + radius]))
    return list(dict.fromkeys(contexts))


def cue_context(action, question):
    fragments = sentence_fragments(question)
    cues = ACTION_CUES.get(action, ())
    matched = [part for part in fragments if any(cue in part.casefold() for cue in cues)]
    return matched[:2] or fragments[:1] or [question]


def expected_value_type(action, value_targets):
    kinds = {str(value.get("kind") or "").casefold() for value in value_targets}
    if kinds & {"number", "integer", "float", "real"}:
        return "numeric"
    if kinds & {"date", "datetime", "time", "year"}:
        return "temporal"
    if value_targets:
        return "categorical"
    if action in {"AGGREGATE", "HAVING_FILTER"}:
        return "numeric_or_countable"
    if action == "LIMIT":
        return "integer"
    return "any"


def value_surfaces(value_targets):
    values = []
    for target in value_targets:
        value = target.get("raw_sql_literal")
        if value is None:
            value = target.get("canonical_value")
        if value is not None:
            values.append(str(value))
    return list(dict.fromkeys(values))


def build_slot_request(step, inputs):
    action = str(step["action"]).upper()
    question = clean_text(inputs.get("question"))
    evidence = clean_text(inputs.get("evidence"))
    values = value_surfaces(step.get("value_targets", []))
    value_context = local_value_context(question, evidence, step.get("value_targets", []))
    focus_context = value_context or cue_context(action, question)
    semantic_source = "value_local_context" if value_context else "action_cue_context"
    request = {
        "request_id": f"step_{int(step.get('step_index', 0))}",
        "step_index": int(step.get("step_index", 0)),
        "action": action,
        "semantic_text": " ".join(focus_context),
        "semantic_source": semantic_source,
        "role_description": ACTION_FOCUS[action],
        "expected_value_type": expected_value_type(action, step.get("value_targets", [])),
        "value_surfaces": values,
        "table_cardinality": len(step.get("table_pointer_ids", [])),
        "column_cardinality": len(step.get("column_pointer_ids", [])),
        "join_edge_cardinality": len(step.get("join_edge_targets", [])),
        "operator_cardinality": len(step.get("operator_targets", [])),
        "value_route_cardinality": len(step.get("value_routes", [])),
    }
    parts = [
        f"action: {action}",
        f"role: {request['role_description']}",
        f"semantic focus: {request['semantic_text']}",
        f"question: {question}",
    ]
    if evidence:
        parts.append(f"evidence: {evidence}")
    if values:
        parts.append("literal surfaces: " + "; ".join(values))
    parts.append(f"expected value type: {request['expected_value_type']}")
    request["slot_embedding_text"] = " | ".join(parts)
    return request


def build_row(row):
    requests, targets = [], []
    inputs = row["inference_inputs"]
    for step in row["teacher_steps"]:
        request = build_slot_request(step, inputs)
        requests.append(request)
        targets.append(
            {
                "request_id": request["request_id"],
                "step_index": request["step_index"],
                "action": request["action"],
                "table_pointer_ids": list(step.get("table_pointer_ids", [])),
                "column_pointer_ids": list(step.get("column_pointer_ids", [])),
                "join_edge_targets": list(step.get("join_edge_targets", [])),
                "operator_targets": list(step.get("operator_targets", [])),
                "value_routes": list(step.get("value_routes", [])),
            }
        )
    return {
        "split": row.get("split"),
        "record_index": int(row["record_index"]),
        "question_id": row.get("question_id"),
        "db_id": row.get("db_id"),
        "inference_inputs": {
            "question": inputs.get("question"),
            "evidence": inputs.get("evidence"),
            "requests": requests,
        },
        "training_targets": {"slot_targets": targets},
        "metadata": {
            "semantic_slot_version": "stage14b_semantic_slot_v1",
            "request_count": len(requests),
            "leakage_boundary": "no schema identity or gold SQL; this oracle-plan diagnostic may retain plan-derived action, arity, and literal semantics",
        },
    }


def build_split(input_path, output_path, limit=None):
    source = read_jsonl(input_path)
    if limit is not None:
        source = source[:limit]
    rows = [build_row(row) for row in source]
    write_jsonl(output_path, rows)
    counts = Counter(
        request["action"]
        for row in rows
        for request in row["inference_inputs"]["requests"]
    )
    return {
        "record_count": len(rows),
        "slot_count": sum(counts.values()),
        "action_counts": dict(counts),
        "output_file": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trajectories", default="experiments/stage13b_clean_typed_trajectories/train_trajectories.jsonl")
    parser.add_argument("--dev-trajectories", default="experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl")
    parser.add_argument("--output-dir", default="experiments/stage14b_semantic_slots")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": vars(args),
        "train": build_split(args.train_trajectories, output_dir / "train_semantic_slots.jsonl", args.train_limit),
        "dev": build_split(args.dev_trajectories, output_dir / "dev_semantic_slots.jsonl", args.dev_limit),
        "leakage_note": "No schema identity or gold SQL is present in inference_inputs. Action, arity, and literal semantics are oracle-plan inputs for this diagnostic and must later be replaced by an LLM planner.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
