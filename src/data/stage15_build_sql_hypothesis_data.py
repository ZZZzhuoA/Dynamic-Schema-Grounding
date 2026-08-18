"""Build grouped typed-plan hypotheses for the Stage 15A graph verifier.

The positive is the clean Stage 13B teacher trajectory. Negatives are controlled
structural corruptions. ``corruption_type`` is diagnostic metadata and must not
be consumed by the verifier.
"""

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stage13b_prepare_typed_trajectories import OPERATORS, VALUE_ROUTES


SEMANTIC_ACTIONS = {"FILTER", "AGGREGATE", "HAVING_FILTER", "SORT", "PROJECT"}


def read_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidate_signature(steps):
    compact = []
    for step in steps:
        compact.append(
            (
                step.get("action"),
                tuple(sorted(int(x) for x in step.get("table_pointer_ids", []))),
                tuple(sorted(int(x) for x in step.get("column_pointer_ids", []))),
                tuple(sorted(step.get("operator_targets", []))),
                tuple(sorted(step.get("value_routes", []))),
                tuple(
                    sorted(
                        tuple(sorted((int(edge["left_column_id"]), int(edge["right_column_id"]))))
                        for edge in step.get("join_edge_targets", [])
                    )
                ),
            )
        )
    return tuple(compact)


def schema_maps(row):
    items = row["inference_inputs"]["schema_items"]
    by_id = {int(item["id"]): item for item in items}
    table_ids = [item_id for item_id, item in by_id.items() if item.get("type") == "table"]
    column_ids = [item_id for item_id, item in by_id.items() if item.get("type") == "column"]
    fk_pairs = []
    seen = set()
    for edge in row["inference_inputs"].get("schema_edges", []):
        if edge.get("type") not in {"foreign_key_forward", "foreign_key_backward"}:
            continue
        pair = tuple(sorted((int(edge["src"]), int(edge["dst"]))))
        if pair not in seen:
            seen.add(pair)
            fk_pairs.append(pair)
    return by_id, table_ids, column_ids, fk_pairs


def replace_one_pointer(steps, candidates, predicate, rng):
    locations = []
    for step_index, step in enumerate(steps):
        for pointer_index, pointer in enumerate(step.get("column_pointer_ids", [])):
            replacements = [value for value in candidates if predicate(int(pointer), int(value))]
            if replacements:
                locations.append((step_index, pointer_index, replacements))
    if not locations:
        return None
    step_index, pointer_index, replacements = rng.choice(locations)
    result = copy.deepcopy(steps)
    result[step_index]["column_pointer_ids"][pointer_index] = int(rng.choice(replacements))
    result[step_index]["column_pointer_ids"] = sorted(
        set(result[step_index]["column_pointer_ids"])
    )
    return result


def same_table_column_swap(row, rng):
    by_id, _, columns, _ = schema_maps(row)

    def predicate(source, target):
        left, right = by_id[source], by_id[target]
        return (
            source != target
            and left.get("table") == right.get("table")
            and left.get("data_type") == right.get("data_type")
        )

    return replace_one_pointer(row["teacher_steps"], columns, predicate, rng)


def cross_table_same_type_swap(row, rng):
    by_id, _, columns, _ = schema_maps(row)

    def predicate(source, target):
        left, right = by_id[source], by_id[target]
        return (
            source != target
            and left.get("table") != right.get("table")
            and left.get("data_type") == right.get("data_type")
        )

    return replace_one_pointer(row["teacher_steps"], columns, predicate, rng)


def scan_table_swap(row, rng):
    _, tables, _, _ = schema_maps(row)
    locations = [
        index for index, step in enumerate(row["teacher_steps"])
        if step.get("action") == "SCAN" and step.get("table_pointer_ids")
    ]
    if not locations:
        return None
    index = rng.choice(locations)
    source = int(row["teacher_steps"][index]["table_pointer_ids"][0])
    replacements = [value for value in tables if value != source]
    if not replacements:
        return None
    result = copy.deepcopy(row["teacher_steps"])
    result[index]["table_pointer_ids"] = [int(rng.choice(replacements))]
    return result


def join_edge_swap(row, rng):
    _, _, _, fk_pairs = schema_maps(row)
    locations = [
        index for index, step in enumerate(row["teacher_steps"])
        if step.get("action") == "JOIN" and step.get("join_edge_targets")
    ]
    if not locations:
        return None
    index = rng.choice(locations)
    used = {
        tuple(sorted((int(edge["left_column_id"]), int(edge["right_column_id"]))))
        for edge in row["teacher_steps"][index]["join_edge_targets"]
    }
    replacements = [pair for pair in fk_pairs if pair not in used]
    if not replacements:
        return None
    left, right = rng.choice(replacements)
    result = copy.deepcopy(row["teacher_steps"])
    result[index]["join_edge_targets"] = [
        {"left_column_id": left, "right_column_id": right, "edge_type": "foreign_key"}
    ]
    result[index]["column_pointer_ids"] = sorted(set([left, right]))
    return result


def operator_swap(row, rng):
    locations = [
        (index, position)
        for index, step in enumerate(row["teacher_steps"])
        for position, _ in enumerate(step.get("operator_targets", []))
    ]
    if not locations:
        return None
    index, position = rng.choice(locations)
    source = row["teacher_steps"][index]["operator_targets"][position]
    families = [
        ["COUNT", "SUM", "AVG", "MIN", "MAX"],
        ["=", "!=", "<>", "<", ">", "<=", ">=", "LIKE", "IN"],
        ["ASC", "DESC"],
        ["+", "-", "*", "/"],
    ]
    family = next((values for values in families if source in values), OPERATORS)
    replacements = [value for value in family if value != source and value != "OTHER"]
    if not replacements:
        return None
    result = copy.deepcopy(row["teacher_steps"])
    result[index]["operator_targets"][position] = rng.choice(replacements)
    result[index]["operator_targets"] = sorted(set(result[index]["operator_targets"]))
    return result


def value_route_swap(row, rng):
    locations = [
        (index, position)
        for index, step in enumerate(row["teacher_steps"])
        for position, _ in enumerate(step.get("value_routes", []))
    ]
    if not locations:
        return None
    index, position = rng.choice(locations)
    source = row["teacher_steps"][index]["value_routes"][position]
    replacements = [value for value in VALUE_ROUTES if value != source]
    result = copy.deepcopy(row["teacher_steps"])
    result[index]["value_routes"][position] = rng.choice(replacements)
    result[index]["value_routes"] = sorted(set(result[index]["value_routes"]))
    return result


def clause_role_swap(row, rng):
    locations = [
        index for index, step in enumerate(row["teacher_steps"])
        if step.get("action") in SEMANTIC_ACTIONS and step.get("column_pointer_ids")
    ]
    pairs = [
        (left, right) for left in locations for right in locations
        if left < right
        and row["teacher_steps"][left]["action"] != row["teacher_steps"][right]["action"]
        and set(row["teacher_steps"][left]["column_pointer_ids"])
        != set(row["teacher_steps"][right]["column_pointer_ids"])
    ]
    if not pairs:
        return None
    left, right = rng.choice(pairs)
    result = copy.deepcopy(row["teacher_steps"])
    result[left]["column_pointer_ids"], result[right]["column_pointer_ids"] = (
        result[right]["column_pointer_ids"], result[left]["column_pointer_ids"]
    )
    return result


CORRUPTORS = [
    ("same_table_column", same_table_column_swap),
    ("cross_table_same_type_column", cross_table_same_type_swap),
    ("scan_table", scan_table_swap),
    ("join_edge", join_edge_swap),
    ("operator", operator_swap),
    ("value_route", value_route_swap),
    ("clause_role", clause_role_swap),
]


def build_candidate_group(row, negatives_per_example, seed):
    rng = random.Random(int(seed) * 1_000_003 + int(row["record_index"]))
    positive_steps = copy.deepcopy(row["teacher_steps"])
    candidates = [
        {"candidate_id": "gold", "label": 1, "corruption_type": "gold", "steps": positive_steps}
    ]
    seen = {candidate_signature(positive_steps)}
    corruptors = list(CORRUPTORS)
    rng.shuffle(corruptors)
    attempts = 0
    while len(candidates) - 1 < negatives_per_example and attempts < len(corruptors) * 4:
        name, function = corruptors[attempts % len(corruptors)]
        attempts += 1
        steps = function(row, rng)
        if steps is None:
            continue
        signature = candidate_signature(steps)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(
            {
                "candidate_id": f"negative_{len(candidates)}",
                "label": 0,
                "corruption_type": name,
                "steps": steps,
            }
        )
    if len(candidates) == 1:
        return None
    # Gold must not occupy a privileged fixed position. The verifier scores
    # candidates independently, and the evaluator uses labels rather than order.
    rng.shuffle(candidates)
    return {
        "split": row.get("split"),
        "record_index": int(row["record_index"]),
        "question_id": row.get("question_id"),
        "db_id": row.get("db_id"),
        "inference_inputs": row["inference_inputs"],
        "candidates": candidates,
        "metadata": {
            "candidate_count": len(candidates),
            "negative_count": len(candidates) - 1,
            "candidate_data_version": "stage15a_typed_hypothesis_v1",
        },
    }


def build_split(input_path, output_path, limit, negatives_per_example, seed):
    rows = read_jsonl(input_path, limit)
    output, skipped = [], []
    corruption_counts = Counter()
    negative_counts = []
    for row in rows:
        group = build_candidate_group(row, negatives_per_example, seed)
        if group is None:
            skipped.append({"record_index": row.get("record_index"), "reason": "no_valid_corruption"})
            continue
        output.append(group)
        negative_counts.append(len(group["candidates"]) - 1)
        corruption_counts.update(
            candidate["corruption_type"] for candidate in group["candidates"]
            if candidate["label"] == 0
        )
    write_jsonl(output_path, output)
    write_jsonl(Path(output_path).with_name(Path(output_path).stem + "_skipped.jsonl"), skipped)
    return {
        "input_count": len(rows),
        "output_count": len(output),
        "skipped_count": len(skipped),
        "avg_negative_count": sum(negative_counts) / len(negative_counts) if negative_counts else 0.0,
        "full_negative_budget_rate": (
            sum(value >= negatives_per_example for value in negative_counts) / len(negative_counts)
            if negative_counts else 0.0
        ),
        "corruption_counts": dict(corruption_counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trajectories", default="experiments/stage13b_clean_typed_trajectories/train_trajectories.jsonl")
    parser.add_argument("--dev-trajectories", default="experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl")
    parser.add_argument("--output-dir", default="experiments/stage15a_sql_hypothesis_data")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--negatives-per-example", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": vars(args),
        "train": build_split(
            args.train_trajectories, output_dir / "train_hypotheses.jsonl",
            args.train_limit, args.negatives_per_example, args.seed,
        ),
        "dev": build_split(
            args.dev_trajectories, output_dir / "dev_hypotheses.jsonl",
            args.dev_limit, args.negatives_per_example, args.seed,
        ),
        "leakage_note": (
            "corruption_type and labels are training/evaluation metadata only; verifier inputs are "
            "question/schema embeddings plus candidate typed-plan content."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
