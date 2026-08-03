import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grounding.stage5d_clause_aware_selector import (  # noqa: E402
    base_scores,
    clause_scores,
    gold_coverage,
    infer_intents,
    node_indexes,
    read_jsonl,
    select_schema,
    write_json,
    write_jsonl,
)
from src.grounding.stage5e_value_aware_selector import (  # noqa: E402
    VALUE_COLUMN_KEYWORDS,
    build_prompt,
    find_sqlite_index,
    is_forced_value_column,
    schema_text_from_rows,
    value_evidence_by_column,
)


METRIC_COLUMN_KEYWORDS = {
    "avg",
    "average",
    "score",
    "scr",
    "num",
    "number",
    "count",
    "total",
    "sum",
    "rate",
    "percent",
    "percentage",
    "enrollment",
    "meal",
    "frpm",
    "free",
    "date",
    "year",
    "age",
    "amount",
    "balance",
}


def normalize_tokens(text):
    from src.grounding.stage5d_clause_aware_selector import tokens

    return set(tokens(text))


def is_metric_or_operator_column(node, question, evidence):
    column_tokens = normalize_tokens(node.get("column") or node.get("name"))
    query_tokens = normalize_tokens(f"{question} {evidence}")
    dtype = str(node.get("data_type") or "").lower()
    if dtype in {"integer", "real", "number", "float", "double", "decimal", "date"}:
        return True
    if column_tokens & METRIC_COLUMN_KEYWORDS:
        return True
    if query_tokens & {"highest", "lowest", "greater", "less", "over", "under", "average", "avg", "count"}:
        return bool(column_tokens & (METRIC_COLUMN_KEYWORDS | query_tokens))
    return False


def selected_tables_from_rows(rows, by_id):
    tables = set()
    for row in rows:
        node = by_id.get(row["id"])
        if not node:
            continue
        if node["type"] == "table":
            tables.add(node["name"])
        elif node["type"] == "column":
            tables.add(node.get("table"))
    return tables


def candidate_value_columns(example, selected_tables, by_id, columns_by_table, question, evidence, args):
    candidates = []
    for table in selected_tables:
        for node in columns_by_table.get(table, []):
            if not is_forced_value_column(node):
                continue
            candidates.append(node)

    def priority(node):
        col_tokens = normalize_tokens(node.get("column") or node.get("name"))
        value_keyword_hit = len(col_tokens & VALUE_COLUMN_KEYWORDS)
        metric_penalty = 1 if is_metric_or_operator_column(node, question, evidence) else 0
        return (value_keyword_hit, -metric_penalty)

    return sorted(candidates, key=priority, reverse=True)[: args.value_search_columns]


def add_value_hints_to_rows(rows, value_evidence, args):
    augmented = []
    for row in rows:
        new_row = dict(row)
        hints = [
            hint
            for hint in value_evidence.get(row["id"], [])
            if hint.get("score", 0.0) >= args.hint_threshold
        ][: args.max_values_per_column]
        if hints:
            new_row["value_hints"] = hints
            new_row["value_score"] = max(hint["score"] for hint in hints)
        augmented.append(new_row)
    return augmented


def low_utility_replacement_ids(base_rows, by_id, protected_ids, args):
    candidates = []
    for index, row in enumerate(base_rows):
        if row["id"] in protected_ids:
            continue
        node = by_id.get(row["id"])
        if not node or node.get("type") != "column":
            continue
        score = float(row.get("score", 0.0))
        base_score = float(row.get("base_score", 0.0))
        value_score = float(row.get("value_score", 0.0))
        # Low utility means neither original scorer nor value evidence strongly
        # supports the column. These are safe replacement targets.
        candidates.append((score + base_score + value_score, index, row["id"]))
    candidates.sort()
    return candidates[: args.max_replacements]


def conservative_select(example, dsg_prediction, tcce_prediction, db_path, budget, args):
    by_id, _, columns_by_table = node_indexes(example)
    base_rows = tcce_prediction[f"top_{budget}"]
    question = example["inference_inputs"].get("question") or ""
    evidence = example["inference_inputs"].get("evidence") or ""
    selected_tables = selected_tables_from_rows(base_rows, by_id)
    value_candidates = candidate_value_columns(
        example, selected_tables, by_id, columns_by_table, question, evidence, args
    )
    value_evidence = value_evidence_by_column(db_path, value_candidates, question, evidence, args)
    augmented = add_value_hints_to_rows(base_rows, value_evidence, args)

    selected_ids = {row["id"] for row in augmented}
    protected_ids = set()
    for row in augmented:
        node = by_id.get(row["id"])
        if not node:
            continue
        if node["type"] == "table":
            protected_ids.add(row["id"])
        elif node["type"] == "column":
            if is_metric_or_operator_column(node, question, evidence):
                protected_ids.add(row["id"])
            if row.get("value_hints"):
                protected_ids.add(row["id"])

    replacements = []
    replacement_targets = low_utility_replacement_ids(augmented, by_id, protected_ids, args)
    if replacement_targets:
        _, _, columns_by_table = node_indexes(example)
        scores, ranks = base_scores(dsg_prediction, by_id)
        beliefs = defaultdict(float)
        for table in selected_tables:
            beliefs[table] = 1.0
        intents = infer_intents(question, evidence)
        value_candidates_by_score = []
        for node in value_candidates:
            if node["id"] in selected_ids:
                continue
            hints = [
                hint
                for hint in value_evidence.get(node["id"], [])
                if hint.get("score", 0.0) >= args.replacement_threshold
            ]
            if not hints:
                continue
            c_scores = clause_scores(node, question, evidence, intents)
            candidate_score = (
                max(hint["score"] for hint in hints)
                + 0.25 * c_scores["where"]
                + 0.15 * scores.get(node["id"], 0.0)
            )
            value_candidates_by_score.append((candidate_score, node, hints))
        value_candidates_by_score.sort(key=lambda item: item[0], reverse=True)

        for (_, target_index, target_id), (_, node, hints) in zip(
            replacement_targets, value_candidates_by_score
        ):
            if node["id"] in selected_ids:
                continue
            old_row = augmented[target_index]
            augmented[target_index] = {
                "id": int(node["id"]),
                "type": node["type"],
                "name": node["name"],
                "score": float(max(hint["score"] for hint in hints)),
                "base_score": float(scores.get(node["id"], 0.0)),
                "value_score": float(max(hint["score"] for hint in hints)),
                "value_hints": hints[: args.max_values_per_column],
                "selection_source": "conservative_value_replacement",
            }
            selected_ids.remove(target_id)
            selected_ids.add(node["id"])
            replacements.append(
                {
                    "removed": old_row["name"],
                    "added": node["name"],
                    "values": [hint["value"] for hint in hints[: args.max_values_per_column]],
                }
            )

    return augmented[:budget], {
        "selected_tables": sorted(selected_tables),
        "value_candidate_count": len(value_candidates),
        "value_hint_columns": sum(1 for row in augmented if row.get("value_hints")),
        "replacement_count": len(replacements),
        "replacements": replacements,
    }


def average(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsg-data", default="experiments/stage5_dsg_data_v2/dev_examples.jsonl")
    parser.add_argument(
        "--dsg-predictions",
        default="experiments/stage5_dsg_grounder_hardneg_v2_1000/dev_predictions.jsonl",
    )
    parser.add_argument(
        "--tcce-predictions",
        default="experiments/stage5d_clause_aware_selection_v2_100/dsg_tcce_predictions.jsonl",
    )
    parser.add_argument("--db-root", default="Data/BIRD/dev_databases")
    parser.add_argument("--output-dir", default="experiments/stage5f_conservative_value_tcce_v2_100")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--budgets", default="30")
    parser.add_argument("--value-search-columns", type=int, default=24)
    parser.add_argument("--max-value-query-tokens", type=int, default=10)
    parser.add_argument("--max-scan-values", type=int, default=20)
    parser.add_argument("--max-values-per-column", type=int, default=2)
    parser.add_argument("--min-value-score", type=float, default=0.45)
    parser.add_argument("--hint-threshold", type=float, default=0.65)
    parser.add_argument("--replacement-threshold", type=float, default=0.85)
    parser.add_argument("--max-replacements", type=int, default=2)
    args = parser.parse_args()

    examples = read_jsonl(Path(args.dsg_data), args.limit)
    dsg_predictions = read_jsonl(Path(args.dsg_predictions), args.limit)
    tcce_predictions = read_jsonl(Path(args.tcce_predictions), args.limit)
    if not (len(examples) == len(dsg_predictions) == len(tcce_predictions)):
        raise ValueError(
            f"Length mismatch: examples={len(examples)}, dsg={len(dsg_predictions)}, "
            f"tcce={len(tcce_predictions)}"
        )

    sqlite_index = find_sqlite_index(Path(args.db_root))
    budgets = [int(value.strip()) for value in args.budgets.split(",") if value.strip()]
    output_dir = Path(args.output_dir)
    prediction_rows = []
    prompts_by_budget = {budget: [] for budget in budgets}
    coverage_by_budget = {budget: defaultdict(list) for budget in budgets}
    debug_counter = Counter()

    for example, dsg_pred, tcce_pred in zip(examples, dsg_predictions, tcce_predictions):
        db_id = example["inference_inputs"]["db_id"]
        db_path = sqlite_index.get(db_id)
        row = {
            "example_id": example["example_id"],
            "db_id": db_id,
            "question": example["inference_inputs"].get("question"),
            "evidence": example["inference_inputs"].get("evidence"),
        }
        debug_by_budget = {}
        for budget in budgets:
            selected_rows, debug = conservative_select(example, dsg_pred, tcce_pred, db_path, budget, args)
            row[f"top_{budget}"] = selected_rows
            debug_by_budget[str(budget)] = debug
            debug_counter[f"value_hint_columns_top{budget}"] += debug["value_hint_columns"]
            debug_counter[f"replacements_top{budget}"] += debug["replacement_count"]

            coverage = gold_coverage(example, selected_rows)
            for key, value in coverage.items():
                if key != "missing_gold_names":
                    coverage_by_budget[budget][key].append(value)

            schema_text = schema_text_from_rows(example, selected_rows)
            prompts_by_budget[budget].append(
                {
                    "question_id": example.get("metadata", {}).get("question_id"),
                    "db_id": db_id,
                    "setting": f"conservative_value_tcce_top{budget}",
                    "question": example["inference_inputs"].get("question"),
                    "evidence": example["inference_inputs"].get("evidence"),
                    "selected_schema_item_count": len(selected_rows),
                    "schema_text": schema_text,
                    "prompt": build_prompt(
                        example["inference_inputs"].get("question"),
                        example["inference_inputs"].get("evidence"),
                        schema_text,
                    ),
                    "gold_sql": example.get("training_targets", {}).get("sql"),
                    "gold_labels": example.get("training_targets", {}).get("grounding_label_names", []),
                }
            )
        row["conservative_value_tcce_debug"] = debug_by_budget
        prediction_rows.append(row)

    statistics = {
        "config": vars(args),
        "sample_count": len(examples),
        "budgets": {},
        "debug_counts": dict(debug_counter),
        "note": (
            "Conservative Value-TCCE keeps TCCE's structural skeleton, adds high-confidence "
            "value hints to selected columns, and allows only limited replacement of low-utility columns."
        ),
    }
    for budget in budgets:
        statistics["budgets"][str(budget)] = {
            key: average(values) for key, values in sorted(coverage_by_budget[budget].items())
        }
        statistics["budgets"][str(budget)]["avg_value_hint_columns"] = (
            debug_counter[f"value_hint_columns_top{budget}"] / len(examples) if examples else 0.0
        )
        statistics["budgets"][str(budget)]["avg_replacements"] = (
            debug_counter[f"replacements_top{budget}"] / len(examples) if examples else 0.0
        )

    write_jsonl(output_dir / "conservative_value_tcce_predictions.jsonl", prediction_rows)
    for budget, prompts in prompts_by_budget.items():
        write_jsonl(output_dir / f"prompts_conservative_value_tcce_top{budget}_dev.jsonl", prompts)
    write_json(output_dir / "selection_statistics.json", statistics)
    with (output_dir / "examples.md").open("w", encoding="utf-8") as f:
        f.write("# Stage 5-F Conservative Value-TCCE examples\n\n")
        for item in prompts_by_budget[min(budgets)][:5]:
            f.write(f"## Question {item.get('question_id')}\n\n")
            f.write(f"{item['question']}\n\n")
            f.write("```text\n")
            f.write(item["schema_text"])
            f.write("\n```\n\n")

    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
