import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


CLAUSES = ["select", "join", "where", "order_by"]
NUMERIC_TYPES = {"integer", "real", "number", "float", "double", "decimal"}


def read_jsonl(path: Path, limit=None):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text or ""))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text):
    return set(normalize_text(text).split())


def overlap(left, right):
    left = set(left)
    right = set(right)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def sigmoid(value):
    try:
        return 1.0 / (1.0 + math.exp(-float(value)))
    except OverflowError:
        return 0.0 if value < 0 else 1.0


def node_indexes(example):
    nodes = example["inference_inputs"]["schema_nodes"]
    by_id = {int(node["id"]): node for node in nodes}
    table_id_by_name = {}
    columns_by_table = defaultdict(list)
    for node in nodes:
        if node["type"] == "table":
            table_id_by_name[node["name"]] = int(node["id"])
        elif node["type"] == "column":
            columns_by_table[node.get("table")].append(node)
    return by_id, table_id_by_name, columns_by_table


def group_clause_predictions(predictions):
    grouped = defaultdict(dict)
    for row in predictions:
        grouped[int(row["record_index"])][row["clause"]] = row
    return grouped


def prediction_score_maps(clause_rows, node_ids):
    maps = {}
    ranks = {}
    for clause in CLAUSES:
        row = clause_rows.get(clause, {})
        top_key = next((key for key in row if key.startswith("top_")), None)
        top_rows = row.get(top_key, []) if top_key else []
        row_count = max(len(top_rows), 1)
        score_map = {item_id: 0.02 for item_id in node_ids}
        rank_map = {item_id: row_count + 999 for item_id in node_ids}
        for rank, item in enumerate(top_rows, start=1):
            item_id = int(item["id"])
            rank_score = 1.0 - ((rank - 1) / row_count)
            raw_score = sigmoid(float(item.get("score", 0.0)))
            score_map[item_id] = 0.70 * rank_score + 0.30 * raw_score
            rank_map[item_id] = rank
        maps[clause] = score_map
        ranks[clause] = rank_map
    return maps, ranks


def column_tokens(node):
    return token_set(f"{node.get('table', '')} {node.get('column', '')} {node.get('name', '')}")


def phrase_bonus(query_norm, node):
    column = normalize_text(node.get("column") or node.get("name"))
    table = normalize_text(node.get("table") or "")
    bonus = 0.0
    if column and f" {column} " in f" {query_norm} ":
        bonus += 1.0
    if table and f" {table} " in f" {query_norm} ":
        bonus += 0.15
    return bonus


def prior_scores(node, question, evidence):
    all_tokens = token_set(f"{question} {evidence}")
    question_tokens = token_set(question)
    evidence_tokens = token_set(evidence)
    col_tokens = column_tokens(node)
    query_norm = normalize_text(f"{question} {evidence}")
    data_type = str(node.get("data_type") or "").lower()
    numeric = 1.0 if data_type in NUMERIC_TYPES else 0.0
    date_like = 1.0 if col_tokens & {"date", "year", "time", "age"} else 0.0
    value_like = 1.0 if col_tokens & {
        "name",
        "type",
        "status",
        "county",
        "district",
        "city",
        "state",
        "charter",
        "virtual",
        "magnet",
        "school",
        "gender",
    } else 0.0
    text = overlap(all_tokens, col_tokens)
    q = overlap(question_tokens, col_tokens)
    e = overlap(evidence_tokens, col_tokens)
    phrase = phrase_bonus(query_norm, node)
    return {
        "select": 0.40 * q + 0.30 * text + 0.25 * phrase + 0.05 * value_like,
        "where": 0.35 * e + 0.25 * text + 0.25 * phrase + 0.15 * max(value_like, date_like),
        "order_by": 0.30 * text + 0.25 * e + 0.20 * phrase + 0.25 * max(numeric, date_like),
    }


def table_beliefs(by_id, columns_by_table, score_maps, question, evidence, args):
    q_tokens = token_set(f"{question} {evidence}")
    beliefs = {}
    for table, columns in columns_by_table.items():
        table_node = next(
            (node for node in by_id.values() if node.get("type") == "table" and node.get("name") == table),
            None,
        )
        table_id = int(table_node["id"]) if table_node else None
        join_table_score = score_maps["join"].get(table_id, 0.0) if table_id is not None else 0.0
        clause_supports = []
        for clause in CLAUSES:
            top_col_scores = sorted([score_maps[clause].get(int(col["id"]), 0.0) for col in columns], reverse=True)[:5]
            clause_supports.append(sum(top_col_scores) / len(top_col_scores) if top_col_scores else 0.0)
        lexical = overlap(q_tokens, token_set(table))
        beliefs[table] = (
            args.join_table_weight * join_table_score
            + args.column_support_weight * max(clause_supports)
            + args.table_lexical_weight * lexical
        )
    return beliefs


def fk_endpoint_ids(example, selected_tables, by_id):
    endpoints = set()
    for edge in example["inference_inputs"].get("schema_edges", []):
        if edge.get("type") not in {"foreign_key_forward", "foreign_key_backward"}:
            continue
        src = by_id.get(int(edge["src"]))
        dst = by_id.get(int(edge["dst"]))
        if not src or not dst:
            continue
        if src.get("type") == "column" and dst.get("type") == "column":
            if src.get("table") in selected_tables and dst.get("table") in selected_tables:
                endpoints.add(int(src["id"]))
                endpoints.add(int(dst["id"]))
    return endpoints


def add_unique(selected, item_ids, budget):
    seen = {int(row["id"]) if isinstance(row, dict) else int(row) for row in selected}
    for item_id in item_ids:
        item_id = int(item_id)
        if len(selected) >= budget:
            break
        if item_id in seen:
            continue
        selected.append(item_id)
        seen.add(item_id)


def ranked_columns_for_clause(clause, candidate_columns, score_maps, beliefs, question, evidence, args):
    ranked = []
    for node in candidate_columns:
        item_id = int(node["id"])
        priors = prior_scores(node, question, evidence)
        table_score = beliefs.get(node.get("table"), 0.0)
        model_score = score_maps[clause].get(item_id, 0.0)
        final_score = (
            args.model_weight * model_score
            + args.prior_weight * priors.get(clause, 0.0)
            + args.table_weight * table_score
        )
        ranked.append(
            {
                "id": item_id,
                "node": node,
                "score": final_score,
                "model_score": model_score,
                "prior_scores": priors,
                "table_belief": table_score,
            }
        )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def assemble_one(example, clause_rows, budget, args):
    by_id, table_id_by_name, columns_by_table = node_indexes(example)
    score_maps, _ = prediction_score_maps(clause_rows, by_id.keys())
    question = example["inference_inputs"].get("question") or ""
    evidence = example["inference_inputs"].get("evidence") or ""
    beliefs = table_beliefs(by_id, columns_by_table, score_maps, question, evidence, args)
    selected_table_names = [
        table
        for table, _ in sorted(beliefs.items(), key=lambda item: item[1], reverse=True)[: args.max_tables]
    ]
    if not selected_table_names and beliefs:
        selected_table_names = [max(beliefs, key=beliefs.get)]

    selected = []
    table_ids = [table_id_by_name[table] for table in selected_table_names if table in table_id_by_name]
    add_unique(selected, table_ids, budget)

    fk_ids = sorted(
        fk_endpoint_ids(example, set(selected_table_names), by_id),
        key=lambda item_id: score_maps["join"].get(item_id, 0.0),
        reverse=True,
    )
    add_unique(selected, fk_ids[: args.fk_budget], budget)

    candidate_columns = []
    for table in selected_table_names:
        candidate_columns.extend(columns_by_table.get(table, []))
    top_clause_column_ids = set()
    for clause in CLAUSES:
        row = clause_rows.get(clause, {})
        top_key = next((key for key in row if key.startswith("top_")), None)
        for item in row.get(top_key, [])[: args.global_clause_seed]:
            node = by_id.get(int(item["id"]))
            if node and node.get("type") == "column":
                top_clause_column_ids.add(int(item["id"]))
    for item_id in top_clause_column_ids:
        node = by_id.get(item_id)
        if node and node not in candidate_columns:
            candidate_columns.append(node)

    slots = [
        ("where", args.where_budget),
        ("select", args.select_budget),
        ("order_by", args.order_budget),
    ]
    column_debug = {}
    for clause, slot in slots:
        ranked = ranked_columns_for_clause(
            clause, candidate_columns, score_maps, beliefs, question, evidence, args
        )
        for item in ranked:
            column_debug[item["id"]] = {**item, "source_clause": clause}
        add_unique(selected, [item["id"] for item in ranked], min(budget, len(selected) + slot))

    all_ranked = []
    for clause in ["select", "where", "order_by", "join"]:
        all_ranked.extend(
            ranked_columns_for_clause(clause, candidate_columns, score_maps, beliefs, question, evidence, args)
        )
    best_by_id = {}
    for item in all_ranked:
        old = best_by_id.get(item["id"])
        if old is None or item["score"] > old["score"]:
            best_by_id[item["id"]] = item
    fallback = sorted(best_by_id.values(), key=lambda item: item["score"], reverse=True)
    for item in fallback:
        column_debug.setdefault(item["id"], {**item, "source_clause": "fallback"})
    add_unique(selected, [item["id"] for item in fallback], budget)

    if len(selected) < budget:
        global_ids = []
        for clause in CLAUSES:
            global_ids.extend(sorted(by_id, key=lambda item_id: score_maps[clause].get(item_id, 0.0), reverse=True))
        add_unique(selected, global_ids, budget)

    rows = []
    for item_id in selected[:budget]:
        node = by_id[int(item_id)]
        detail = column_debug.get(int(item_id), {})
        if node["type"] == "table":
            source = "table_backbone"
            score = beliefs.get(node["name"], score_maps["join"].get(int(item_id), 0.0))
            table_belief = beliefs.get(node["name"], 0.0)
        else:
            source = detail.get("source_clause", "global")
            score = detail.get("score", max(score_maps[c].get(int(item_id), 0.0) for c in CLAUSES))
            table_belief = detail.get("table_belief", beliefs.get(node.get("table"), 0.0))
        rows.append(
            {
                "id": int(item_id),
                "type": node["type"],
                "name": node["name"],
                "score": float(score),
                "source_clause": source,
                "table_belief": float(table_belief),
            }
        )
    return rows, {
        "selected_tables": selected_table_names,
        "table_beliefs": dict(sorted(beliefs.items(), key=lambda item: item[1], reverse=True)[:10]),
    }


def coverage(label_record, selected_rows):
    by_id = {item["id"]: item for item in label_record.get("schema_items", [])}
    gold = set(label_record.get("whole_sql_labels", []))
    selected = {int(row["id"]) for row in selected_rows}
    gold_tables = {item_id for item_id in gold if by_id.get(item_id, {}).get("type") == "table"}
    gold_columns = {item_id for item_id in gold if by_id.get(item_id, {}).get("type") == "column"}
    return {
        "schema_recall": len(gold & selected) / len(gold) if gold else None,
        "schema_precision": len(gold & selected) / len(selected) if selected else None,
        "table_recall": len(gold_tables & selected) / len(gold_tables) if gold_tables else None,
        "column_recall": len(gold_columns & selected) / len(gold_columns) if gold_columns else None,
        "missing_gold_names": [by_id[item_id]["name"] for item_id in sorted(gold - selected) if item_id in by_id],
    }


def average(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsg-data", default="experiments/stage5_dsg_data_v2/dev_examples.jsonl")
    parser.add_argument("--clause-labels", default="experiments/stage5g_clause_labels/dev_clause_labels.jsonl")
    parser.add_argument(
        "--clause-predictions",
        default="experiments/stage5g_clause_grounder_rgcn_1000/dev_clause_predictions.jsonl",
    )
    parser.add_argument("--output-dir", default="experiments/stage5g_structural_assembly_rgcn_1000")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--budgets", default="30")
    parser.add_argument("--max-tables", type=int, default=4)
    parser.add_argument("--fk-budget", type=int, default=8)
    parser.add_argument("--select-budget", type=int, default=8)
    parser.add_argument("--where-budget", type=int, default=10)
    parser.add_argument("--order-budget", type=int, default=4)
    parser.add_argument("--global-clause-seed", type=int, default=12)
    parser.add_argument("--model-weight", type=float, default=0.58)
    parser.add_argument("--prior-weight", type=float, default=0.27)
    parser.add_argument("--table-weight", type=float, default=0.15)
    parser.add_argument("--join-table-weight", type=float, default=0.60)
    parser.add_argument("--column-support-weight", type=float, default=0.32)
    parser.add_argument("--table-lexical-weight", type=float, default=0.08)
    args = parser.parse_args()

    examples = read_jsonl(Path(args.dsg_data), args.limit)
    labels = read_jsonl(Path(args.clause_labels), args.limit)
    clause_predictions = read_jsonl(Path(args.clause_predictions))
    grouped_predictions = group_clause_predictions(clause_predictions)
    budgets = [int(value.strip()) for value in args.budgets.split(",") if value.strip()]

    if len(examples) != len(labels):
        raise ValueError(f"Length mismatch examples={len(examples)} labels={len(labels)}")

    output_rows = []
    coverage_by_budget = {budget: defaultdict(list) for budget in budgets}
    examples_md = []
    for index, (example, label_record) in enumerate(zip(examples, labels)):
        row = {
            "example_id": example["example_id"],
            "record_index": index,
            "db_id": example["inference_inputs"]["db_id"],
            "question_id": label_record.get("question_id"),
            "question": example["inference_inputs"].get("question"),
            "evidence": example["inference_inputs"].get("evidence"),
            "gold_label_ids": label_record.get("whole_sql_labels", []),
            "gold_label_names": label_record.get("whole_sql_label_names", []),
        }
        debug = {}
        clause_rows = grouped_predictions.get(index, {})
        for budget in budgets:
            selected_rows, selected_debug = assemble_one(example, clause_rows, budget, args)
            row[f"top_{budget}"] = selected_rows
            debug[str(budget)] = selected_debug
            cov = coverage(label_record, selected_rows)
            for key, value in cov.items():
                if key != "missing_gold_names":
                    coverage_by_budget[budget][key].append(value)
            if len(examples_md) < 6 and budget == min(budgets):
                examples_md.append(
                    {
                        "question": row["question"],
                        "selected_tables": selected_debug["selected_tables"],
                        "selected": [item["name"] for item in selected_rows[:20]],
                        "missing_gold": cov["missing_gold_names"],
                    }
                )
        row["structural_debug"] = debug
        output_rows.append(row)

    statistics = {
        "config": vars(args),
        "sample_count": len(output_rows),
        "budgets": {},
        "note": (
            "Stage 5-G3 uses learned clause-conditioned scores plus structural priors. "
            "Gold labels are used only for offline diagnostics."
        ),
    }
    for budget in budgets:
        statistics["budgets"][str(budget)] = {
            key: average(values) for key, values in sorted(coverage_by_budget[budget].items())
        }
        statistics["budgets"][str(budget)]["missing_samples"] = sum(
            1 for value in coverage_by_budget[budget]["schema_recall"] if value is not None and value < 1.0
        )

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "clause_structural_predictions.jsonl", output_rows)
    write_json(output_dir / "selection_statistics.json", statistics)
    with (output_dir / "examples.md").open("w", encoding="utf-8") as f:
        f.write("# Stage 5-G3 structural assembly examples\n\n")
        for idx, item in enumerate(examples_md):
            f.write(f"## Example {idx}\n\n")
            f.write(f"Question: {item['question']}\n\n")
            f.write(f"Selected tables: {item['selected_tables']}\n\n")
            f.write("Selected schema:\n\n")
            for name in item["selected"]:
                f.write(f"- `{name}`\n")
            f.write("\nMissing gold:\n\n")
            for name in item["missing_gold"]:
                f.write(f"- `{name}`\n")
            f.write("\n")

    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
