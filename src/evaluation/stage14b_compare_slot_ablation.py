"""Compare correct semantic slots with action-only and shuffled-slot controls."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.stage14_evaluate_typed_schema_tool import evaluate


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def candidate_order(step, key):
    return [int(row["schema_id"]) for row in step.get(key, [])]


def compare_rankings(correct, control):
    control_by_index = {int(row["record_index"]): row for row in control}
    changed, total = 0, 0
    action_stats = {}
    for row in correct:
        other = control_by_index[int(row["record_index"])]
        for left, right in zip(row.get("tool_steps", []), other.get("tool_steps", [])):
            action = left["request"]["action"]
            bucket = action_stats.setdefault(action, {"changed": 0, "total": 0})
            for key in ("table_candidates", "column_candidates"):
                left_order, right_order = candidate_order(left, key), candidate_order(right, key)
                if not left_order and not right_order:
                    continue
                is_changed = left_order != right_order
                changed += int(is_changed); total += 1
                bucket["changed"] += int(is_changed); bucket["total"] += 1
    return {
        "ranking_change_rate": changed / total if total else 0.0,
        "changed_rankings": changed,
        "ranking_count": total,
        "action_ranking_change_rate": {
            action: values["changed"] / values["total"] if values["total"] else 0.0
            for action, values in sorted(action_stats.items())
        },
    }


def metric_delta(left, right):
    keys = [
        "table_recall", "column_recall", "join_edge_recall",
        "semantic_step_complete_rate", "assembled_schema_recall",
        "assembled_complete_coverage",
    ]
    return {key: left[key] - right[key] for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correct-output", required=True)
    parser.add_argument("--action-only-output", required=True)
    parser.add_argument("--shuffled-output", required=True)
    parser.add_argument("--same-action-shuffled-output")
    parser.add_argument("--target-trajectories", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    targets = read_jsonl(args.target_trajectories)
    correct = read_jsonl(args.correct_output)
    action_only = read_jsonl(args.action_only_output)
    shuffled = read_jsonl(args.shuffled_output)
    same_action_shuffled = (
        read_jsonl(args.same_action_shuffled_output)
        if args.same_action_shuffled_output else None
    )
    metrics = {}
    variants = [("correct", correct), ("action_only", action_only), ("shuffled", shuffled)]
    if same_action_shuffled is not None:
        variants.append(("same_action_shuffled", same_action_shuffled))
    for name, rows in variants:
        metrics[name], _ = evaluate(rows, targets)
    semantic_control_name = "same_action_shuffled" if same_action_shuffled is not None else "shuffled"
    semantic_control = same_action_shuffled if same_action_shuffled is not None else shuffled
    semantic_column_delta = (
        metrics["correct"]["column_recall"] - metrics[semantic_control_name]["column_recall"]
    )
    semantic_complete_delta = (
        metrics["correct"]["semantic_step_complete_rate"]
        - metrics[semantic_control_name]["semantic_step_complete_rate"]
    )
    summary = {
        "metrics": metrics,
        "correct_minus_action_only": metric_delta(metrics["correct"], metrics["action_only"]),
        "correct_minus_shuffled": metric_delta(metrics["correct"], metrics["shuffled"]),
        "semantic_control": semantic_control_name,
        "correct_minus_semantic_control": metric_delta(
            metrics["correct"], metrics[semantic_control_name]
        ),
        "ranking_effect_vs_action_only": compare_rankings(correct, action_only),
        "ranking_effect_vs_shuffled": compare_rankings(correct, shuffled),
        "ranking_effect_vs_semantic_control": compare_rankings(correct, semantic_control),
        "causal_gate": {
            "correct_beats_action_only_column_recall": metrics["correct"]["column_recall"] > metrics["action_only"]["column_recall"],
            "correct_beats_shuffled_column_recall": metrics["correct"]["column_recall"] > metrics["shuffled"]["column_recall"],
            "correct_beats_semantic_control_column_recall_by_5pct": semantic_column_delta >= 0.05,
            "correct_beats_semantic_control_semantic_complete_by_3pct": semantic_complete_delta >= 0.03,
            "correct_beats_action_only_semantic_complete": metrics["correct"]["semantic_step_complete_rate"] > metrics["action_only"]["semantic_step_complete_rate"],
            "ranking_change_vs_action_only_at_least_10pct": compare_rankings(correct, action_only)["ranking_change_rate"] >= 0.10,
        },
    }
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_file).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
