"""Diagnose real SQL groups where a correct candidate exists but reranking fails."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.stage15b_evaluate_real_sql_reranking import baseline_index, selections


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
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def plan_signature(candidate):
    steps = candidate.get("steps", [])
    return {
        "actions": tuple(step.get("action") for step in steps),
        "tables": tuple(sorted({int(value) for step in steps for value in step.get("table_pointer_ids", [])})),
        "columns": tuple(sorted({int(value) for step in steps for value in step.get("column_pointer_ids", [])})),
        "clause_columns": tuple(
            (step.get("action"), tuple(sorted(int(value) for value in step.get("column_pointer_ids", []))))
            for step in steps
        ),
        "operators": tuple(
            (step.get("action"), tuple(sorted(step.get("operator_targets", []))))
            for step in steps
        ),
        "value_routes": tuple(
            (step.get("action"), tuple(sorted(step.get("value_routes", []))))
            for step in steps
        ),
        "join_edges": tuple(
            sorted(
                tuple(sorted((int(edge["left_column_id"]), int(edge["right_column_id"]))))
                for step in steps for edge in step.get("join_edge_targets", [])
            )
        ),
    }


def difference_types(left, right):
    left_signature, right_signature = plan_signature(left), plan_signature(right)
    names = [name for name in left_signature if left_signature[name] != right_signature[name]]
    if left.get("parse_status") != right.get("parse_status"):
        names.append("parse_status")
    if left.get("execution_ok") != right.get("execution_ok"):
        names.append("execution_validity")
    return names or ["same_typed_plan_different_semantics"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--method", help="Defaults to hybrid_<alpha>")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    method = args.method or f"hybrid_{args.alpha:g}"
    rows = read_jsonl(args.scored_file, args.limit)
    categories = Counter()
    outcomes = Counter()
    cases = []
    for row in rows:
        candidates = row.get("candidates", [])
        if not candidates:
            continue
        baseline = baseline_index(candidates)
        selected = selections(row, [args.alpha]).get(method)
        correct_indices = [
            index for index, candidate in enumerate(candidates)
            if candidate.get("execution_correct")
        ]
        baseline_correct = bool(candidates[baseline].get("execution_correct"))
        selected_correct = bool(
            selected is not None and candidates[selected].get("execution_correct")
        )
        oracle = bool(correct_indices)
        if not baseline_correct and oracle and selected_correct:
            outcome = "recovered"
        elif not baseline_correct and oracle and not selected_correct:
            outcome = "oracle_reachable_unresolved"
        elif baseline_correct and not selected_correct:
            outcome = "regressed"
        else:
            continue
        outcomes[outcome] += 1
        if outcome == "recovered":
            # Explain what the reranker corrected: greedy wrong -> selected correct.
            comparison_index = baseline
            reference_index = selected
        elif outcome == "regressed":
            # Explain what the reranker broke: selected wrong -> greedy correct.
            comparison_index = selected
            reference_index = baseline
        else:
            # Explain the remaining opportunity: selected wrong -> best-scored available correct.
            comparison_index = selected if selected is not None else baseline
            reference_index = max(
                correct_indices,
                key=lambda index: (
                    float(candidates[index]["verifier_score"])
                    if candidates[index].get("verifier_score") is not None
                    else float("-inf")
                ),
            )
        differences = difference_types(candidates[comparison_index], candidates[reference_index])
        for name in differences:
            categories[f"{outcome}::{name}"] += 1
        cases.append(
            {
                "record_index": row.get("record_index"),
                "db_id": row.get("db_id"),
                "question": row.get("inference_inputs", {}).get("question"),
                "outcome": outcome,
                "method": method,
                "baseline_index": baseline,
                "selected_index": selected,
                "comparison_index": comparison_index,
                "reference_correct_index": reference_index,
                "difference_types": differences,
                "baseline_sql": candidates[baseline].get("generated_sql"),
                "selected_sql": candidates[selected].get("generated_sql") if selected is not None else None,
                "reference_correct_sql": candidates[reference_index].get("generated_sql"),
                "selected_verifier_score": candidates[selected].get("verifier_score") if selected is not None else None,
                "reference_verifier_score": candidates[reference_index].get("verifier_score"),
            }
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": vars(args),
        "method": method,
        "sample_count": len(rows),
        "outcome_counts": dict(outcomes),
        "difference_type_counts": dict(categories),
        "note": "All labels are used post-selection for diagnosis only; they do not affect candidate ranking.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(output / "cases.jsonl", cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
