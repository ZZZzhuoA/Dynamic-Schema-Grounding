"""Evaluate Stage 14 pointer/tool outputs against typed diagnostic targets."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ratio(hit, total):
    return hit / total if total else 1.0


def optional_ratio(hit, total):
    return hit / total if total else None


def candidate_ids(rows, key="schema_id"):
    return {int(row[key]) for row in rows if key in row}


def edge_ids(rows):
    return {
        tuple(sorted((int(row["left_schema_id"]), int(row["right_schema_id"]))))
        for row in rows
    }


def evaluate(tool_rows, target_rows):
    targets = {int(row["record_index"]): row for row in target_rows}
    counts = Counter()
    action_counts = defaultdict(Counter)
    transitions = []
    for tool in tool_rows:
        index = int(tool["record_index"])
        target = targets.get(index)
        if target is None:
            raise KeyError(f"Missing target trajectory for record_index={index}")
        predicted_steps = tool.get("tool_steps", [])
        gold_steps = target.get("teacher_steps", [])
        if len(predicted_steps) != len(gold_steps):
            raise ValueError(
                f"Step mismatch at record_index={index}: "
                f"tool={len(predicted_steps)} target={len(gold_steps)}"
            )
        sample_gold = set()
        for predicted, gold in zip(predicted_steps, gold_steps):
            action = gold["action"]
            bucket = action_counts[action]
            table_gold = {int(value) for value in gold.get("table_pointer_ids", [])}
            column_gold = {int(value) for value in gold.get("column_pointer_ids", [])}
            table_pred = candidate_ids(predicted.get("table_candidates", []))
            column_pred = candidate_ids(predicted.get("column_candidates", []))
            join_gold = {
                tuple(sorted((int(edge["left_column_id"]), int(edge["right_column_id"]))))
                for edge in gold.get("join_edge_targets", [])
            }
            join_pred = edge_ids(predicted.get("join_edge_candidates", []))
            operator_gold = set(gold.get("operator_targets", []))
            operator_pred = {
                row["name"] for row in predicted.get("operator_candidates", [])
            }
            route_gold = set(gold.get("value_routes", []))
            route_pred = {
                row["name"] for row in predicted.get("value_route_candidates", [])
            }
            groups = (
                ("table", table_gold, table_pred),
                ("column", column_gold, column_pred),
                ("join", join_gold, join_pred),
                ("operator", operator_gold, operator_pred),
                ("value_route", route_gold, route_pred),
            )
            complete = True
            for name, gold_set, predicted_set in groups:
                if not gold_set:
                    continue
                hit = len(gold_set & predicted_set)
                counts[f"{name}_hit"] += hit
                counts[f"{name}_total"] += len(gold_set)
                bucket[f"{name}_hit"] += hit
                bucket[f"{name}_total"] += len(gold_set)
                complete = complete and gold_set.issubset(predicted_set)
            counts["step_complete"] += int(complete)
            counts["step_total"] += 1
            if action != "STOP":
                counts["decision_step_complete"] += int(complete)
                counts["decision_step_total"] += 1
            if action not in {"SCAN", "STOP"}:
                counts["semantic_step_complete"] += int(complete)
                counts["semantic_step_total"] += 1
            bucket["step_complete"] += int(complete)
            bucket["step_total"] += 1
            sample_gold.update(table_gold)
            sample_gold.update(column_gold)
            for left, right in join_gold:
                sample_gold.update((left, right))
        assembled = {
            int(row["schema_id"])
            for row in tool.get("assembly", {}).get("selected_schema", [])
        }
        hit = len(sample_gold & assembled)
        counts["assembly_hit"] += hit
        counts["assembly_total"] += len(sample_gold)
        complete = sample_gold.issubset(assembled)
        counts["assembly_complete"] += int(complete)
        counts["sample_total"] += 1
        if not complete:
            transitions.append(
                {
                    "record_index": index,
                    "db_id": tool.get("db_id"),
                    "missing_schema_ids": sorted(sample_gold - assembled),
                }
            )
    metrics = {
        "sample_count": counts["sample_total"],
        "table_recall": ratio(counts["table_hit"], counts["table_total"]),
        "column_recall": ratio(counts["column_hit"], counts["column_total"]),
        "join_edge_recall": ratio(counts["join_hit"], counts["join_total"]),
        "operator_recall": ratio(counts["operator_hit"], counts["operator_total"]),
        "value_route_recall": ratio(
            counts["value_route_hit"], counts["value_route_total"]
        ),
        "step_complete_rate": ratio(counts["step_complete"], counts["step_total"]),
        "decision_step_complete_rate": ratio(
            counts["decision_step_complete"], counts["decision_step_total"]
        ),
        "semantic_step_complete_rate": ratio(
            counts["semantic_step_complete"], counts["semantic_step_total"]
        ),
        "assembled_schema_recall": ratio(counts["assembly_hit"], counts["assembly_total"]),
        "assembled_complete_coverage": ratio(
            counts["assembly_complete"], counts["sample_total"]
        ),
        "assembled_complete_samples": counts["assembly_complete"],
        "action_metrics": {},
    }
    for action, bucket in sorted(action_counts.items()):
        metrics["action_metrics"][action] = {
            "step_count": bucket["step_total"],
            "table_recall": optional_ratio(bucket["table_hit"], bucket["table_total"]),
            "column_recall": optional_ratio(bucket["column_hit"], bucket["column_total"]),
            "join_edge_recall": optional_ratio(bucket["join_hit"], bucket["join_total"]),
            "operator_recall": optional_ratio(bucket["operator_hit"], bucket["operator_total"]),
            "value_route_recall": optional_ratio(
                bucket["value_route_hit"], bucket["value_route_total"]
            ),
            "step_complete_rate": ratio(
                bucket["step_complete"], bucket["step_total"]
            ),
        }
    return metrics, transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-output", required=True)
    parser.add_argument("--target-trajectories", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    metrics, missing = evaluate(
        read_jsonl(args.tool_output), read_jsonl(args.target_trajectories)
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.tool_output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage14_tool_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "stage14_tool_missing.jsonl").open("w", encoding="utf-8") as handle:
        for row in missing:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
