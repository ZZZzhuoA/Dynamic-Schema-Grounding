import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grounding.stage5e_value_aware_selector import build_prompt, schema_text_from_rows  # noqa: E402
from src.grounding.stage5f_conservative_value_tcce import is_metric_or_operator_column, normalize_tokens  # noqa: E402
from src.grounding.stage5g_clause_structural_assembly import coverage, read_jsonl, write_json, write_jsonl  # noqa: E402


DEFAULT_RELATION_BUDGETS = {
    "OUTPUT_TARGET": 2,
    "METRIC_TARGET": 2,
    "PREDICATE_COLUMN": 2,
    "ORDER_KEY": 1,
    "FORMULA_COMPONENT": 1,
}

JOIN_KEY_TOKENS = {
    "id",
    "code",
    "key",
    "cds",
    "cdscode",
    "account",
    "client",
    "customer",
    "order",
    "school",
}


def parse_budget_text(text):
    budgets = {}
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid budget entry {part!r}; expected RELATION:k")
        key, value = part.split(":", 1)
        budgets[key.strip()] = int(value)
    return budgets


def group_relation_predictions(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[int(row["record_index"])][row["clause"]] = row
    return grouped


def node_maps(label_record):
    by_id = {int(item["id"]): item for item in label_record.get("schema_items", [])}
    return by_id


def is_join_key_like_name(name):
    tokens = normalize_tokens(name)
    compact = "".join(tokens)
    return bool(tokens & JOIN_KEY_TOKENS) or any(token in compact for token in ["id", "code", "cds"])


def protected_base_ids(base_rows, by_id, question, evidence, args):
    protected = set()
    scored = []
    for row in base_rows:
        item_id = int(row["id"])
        item = by_id.get(item_id, {})
        if row.get("type") == "table" or item.get("type") == "table":
            protected.add(item_id)
            continue
        if row.get("value_hints"):
            protected.add(item_id)
        if row.get("source_clause") in {"table_backbone", "join"}:
            protected.add(item_id)
        if is_join_key_like_name(row.get("name", "")):
            protected.add(item_id)
        if item and is_metric_or_operator_column(item, question, evidence):
            protected.add(item_id)
        scored.append((float(row.get("score", 0.0)), item_id))
    scored.sort(reverse=True)
    protected.update(item_id for _, item_id in scored[: args.protect_top_base_columns])
    return protected


def relation_candidate_rows(relation_rows, budgets, args):
    candidates = []
    for relation, budget in budgets.items():
        row = relation_rows.get(relation)
        if not row:
            continue
        top_key = next((key for key in row if key.startswith("top_")), None)
        if not top_key:
            continue
        for rank, item in enumerate(row.get(top_key, [])[: args.relation_top_k], start=1):
            if item.get("type") != "column":
                continue
            rank_bonus = 1.0 - (rank - 1) / max(args.relation_top_k, 1)
            score = float(item.get("score", 0.0)) + args.rank_bonus_weight * rank_bonus
            candidates.append(
                {
                    "id": int(item["id"]),
                    "type": item.get("type"),
                    "name": item.get("name"),
                    "score": score,
                    "raw_relation_score": float(item.get("score", 0.0)),
                    "relation_type": relation,
                    "relation_rank": rank,
                    "relation_budget": budget,
                }
            )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def low_utility_slots(base_rows, by_id, protected, args):
    slots = []
    for index, row in enumerate(base_rows):
        item_id = int(row["id"])
        if item_id in protected:
            continue
        item = by_id.get(item_id, {})
        if item.get("type") != "column" and row.get("type") != "column":
            continue
        source = row.get("source_clause") or ""
        value_bonus = 1.0 if row.get("value_hints") else 0.0
        score = float(row.get("score", 0.0)) + 0.35 * float(row.get("table_belief", 0.0)) + value_bonus
        # Prefer replacing generic/fallback columns first.
        source_penalty = 0.2 if source in {"fallback", "global", "order_by"} else 0.0
        slots.append((score - source_penalty, index, item_id))
    slots.sort()
    return slots


def fuse_one(base_row, relation_rows, label_record, budget, args):
    by_id = node_maps(label_record)
    question = base_row.get("question") or label_record.get("question") or ""
    evidence = base_row.get("evidence") or label_record.get("evidence") or ""
    fused = [dict(row) for row in base_row[f"top_{budget}"]]
    selected_ids = {int(row["id"]) for row in fused}
    protected = protected_base_ids(fused, by_id, question, evidence, args)
    relation_budgets = parse_budget_text(args.relation_budgets)
    relation_used = Counter()
    replacements = []

    candidates = relation_candidate_rows(relation_rows, relation_budgets, args)
    slots = low_utility_slots(fused, by_id, protected, args)
    slot_cursor = 0
    for cand in candidates:
        item_id = int(cand["id"])
        relation = cand["relation_type"]
        if item_id in selected_ids:
            relation_used[relation] += 0
            continue
        if relation_used[relation] >= relation_budgets.get(relation, 0):
            continue
        if cand["raw_relation_score"] < args.min_raw_relation_score:
            continue
        item = by_id.get(item_id, {})
        if item.get("type") != "column":
            continue
        while slot_cursor < len(slots) and slots[slot_cursor][2] not in selected_ids:
            slot_cursor += 1
        if slot_cursor >= len(slots):
            break
        _, target_index, target_id = slots[slot_cursor]
        old = fused[target_index]
        fused[target_index] = {
            "id": item_id,
            "type": cand["type"],
            "name": cand["name"],
            "score": float(cand["score"]),
            "source_clause": f"relation_fusion:{relation}",
            "table_belief": float(old.get("table_belief", 0.0)),
            "relation_type": relation,
            "relation_rank": cand["relation_rank"],
            "raw_relation_score": cand["raw_relation_score"],
        }
        selected_ids.remove(target_id)
        selected_ids.add(item_id)
        relation_used[relation] += 1
        replacements.append(
            {
                "removed": old.get("name"),
                "added": cand["name"],
                "relation_type": relation,
                "raw_relation_score": cand["raw_relation_score"],
                "relation_rank": cand["relation_rank"],
            }
        )
        slot_cursor += 1
        if len(replacements) >= args.max_total_replacements:
            break

    return fused[:budget], {
        "protected_count": len(protected),
        "candidate_count": len(candidates),
        "replacement_count": len(replacements),
        "relation_used": dict(relation_used),
        "replacements": replacements,
    }


def average(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-predictions",
        default="experiments/stage5i_h2_value_gated_semantic_rgcn_1000_fast_columncache/value_gated_clause_assembly_predictions.jsonl",
    )
    parser.add_argument(
        "--relation-predictions",
        default="experiments/stage5j_relation_grounder_rgcn_1000/dev_clause_predictions.jsonl",
    )
    parser.add_argument("--clause-labels", default="experiments/stage5g_clause_labels/dev_clause_labels.jsonl")
    parser.add_argument("--dsg-data", default="experiments/stage5i_dsg_data_semantic_v1/dev_examples.jsonl")
    parser.add_argument("--output-dir", default="experiments/stage5j_relation_fusion_selector_rgcn_1000")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--budgets", default="30")
    parser.add_argument("--relation-budgets", default="OUTPUT_TARGET:2,METRIC_TARGET:2,PREDICATE_COLUMN:2,ORDER_KEY:1,FORMULA_COMPONENT:1")
    parser.add_argument("--relation-top-k", type=int, default=12)
    parser.add_argument("--rank-bonus-weight", type=float, default=0.15)
    parser.add_argument("--min-raw-relation-score", type=float, default=-0.5)
    parser.add_argument("--max-total-replacements", type=int, default=4)
    parser.add_argument("--protect-top-base-columns", type=int, default=10)
    args = parser.parse_args()

    base_rows = read_jsonl(Path(args.base_predictions), args.limit)
    labels = read_jsonl(Path(args.clause_labels), args.limit)
    examples = read_jsonl(Path(args.dsg_data), args.limit)
    relation_rows = read_jsonl(Path(args.relation_predictions))
    grouped_relations = group_relation_predictions(relation_rows)
    budgets = [int(value.strip()) for value in args.budgets.split(",") if value.strip()]
    if len(base_rows) != len(labels) or len(base_rows) != len(examples):
        raise ValueError(f"Length mismatch base={len(base_rows)} labels={len(labels)} examples={len(examples)}")

    output_rows = []
    prompts_by_budget = {budget: [] for budget in budgets}
    coverage_by_budget = {budget: defaultdict(list) for budget in budgets}
    debug_counts = Counter()
    for index, (base_row, label_record, example) in enumerate(zip(base_rows, labels, examples)):
        row = {
            "example_id": base_row.get("example_id"),
            "record_index": index,
            "db_id": base_row.get("db_id"),
            "question_id": base_row.get("question_id"),
            "question": base_row.get("question"),
            "evidence": base_row.get("evidence"),
            "gold_label_ids": label_record.get("whole_sql_labels", []),
            "gold_label_names": label_record.get("whole_sql_label_names", []),
        }
        debug = {}
        relation_for_index = grouped_relations.get(index, {})
        for budget in budgets:
            fused_rows, fusion_debug = fuse_one(base_row, relation_for_index, label_record, budget, args)
            row[f"top_{budget}"] = fused_rows
            debug[str(budget)] = fusion_debug
            debug_counts[f"replacements_top{budget}"] += fusion_debug["replacement_count"]
            for relation, count in fusion_debug["relation_used"].items():
                debug_counts[f"relation_used_{relation}_top{budget}"] += count
            cov = coverage(label_record, fused_rows)
            for key, value in cov.items():
                if key != "missing_gold_names":
                    coverage_by_budget[budget][key].append(value)
            schema_text = schema_text_from_rows(example, fused_rows)
            prompts_by_budget[budget].append(
                {
                    "question_id": label_record.get("question_id"),
                    "db_id": base_row.get("db_id"),
                    "setting": f"relation_fusion_top{budget}",
                    "question": base_row.get("question"),
                    "evidence": base_row.get("evidence"),
                    "selected_schema_item_count": len(fused_rows),
                    "schema_text": schema_text,
                    "prompt": build_prompt(base_row.get("question"), base_row.get("evidence"), schema_text),
                    "gold_sql": example.get("training_targets", {}).get("sql"),
                    "gold_labels": label_record.get("whole_sql_label_names", []),
                }
            )
        row["relation_fusion_debug"] = debug
        output_rows.append(row)

    statistics = {
        "config": vars(args),
        "sample_count": len(output_rows),
        "budgets": {},
        "debug_counts": dict(debug_counts),
        "note": (
            "Stage 5-J3 uses Semantic H2 as the protected skeleton and applies conservative "
            "relation-grounder replacements for selected relation types."
        ),
    }
    for budget in budgets:
        statistics["budgets"][str(budget)] = {
            key: average(values) for key, values in sorted(coverage_by_budget[budget].items())
        }
        statistics["budgets"][str(budget)]["missing_samples"] = sum(
            1 for value in coverage_by_budget[budget]["schema_recall"] if value is not None and value < 1.0
        )
        statistics["budgets"][str(budget)]["avg_replacements"] = (
            debug_counts[f"replacements_top{budget}"] / len(output_rows) if output_rows else 0.0
        )

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "relation_fusion_predictions.jsonl", output_rows)
    for budget, prompts in prompts_by_budget.items():
        write_jsonl(output_dir / f"prompts_relation_fusion_top{budget}_dev.jsonl", prompts)
    write_json(output_dir / "selection_statistics.json", statistics)
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
