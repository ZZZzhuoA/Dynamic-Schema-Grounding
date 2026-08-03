import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


SELECT_TRIGGERS = {
    "what",
    "which",
    "list",
    "show",
    "name",
    "names",
    "title",
    "titles",
    "phone",
    "email",
    "website",
    "address",
    "zip",
    "code",
    "number",
    "count",
}

WHERE_TRIGGERS = {
    "where",
    "with",
    "in",
    "from",
    "for",
    "whose",
    "after",
    "before",
    "between",
    "greater",
    "less",
    "over",
    "under",
    "equal",
    "equals",
    "opened",
    "closed",
    "county",
    "district",
    "type",
    "status",
    "school",
    "charter",
    "virtual",
    "magnet",
}

ORDER_TRIGGERS = {
    "highest",
    "lowest",
    "largest",
    "smallest",
    "maximum",
    "minimum",
    "max",
    "min",
    "most",
    "least",
    "top",
    "sort",
    "sorted",
    "order",
    "descending",
    "ascending",
    "average",
    "avg",
    "total",
    "sum",
    "rate",
    "score",
    "count",
    "number",
}

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


def tokens(text):
    return [token for token in normalize_text(text).split() if token]


def token_set(text):
    return set(tokens(text))


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


def available_top_rows(prediction):
    available = sorted(
        int(key.split("_", 1)[1])
        for key in prediction
        if key.startswith("top_") and key.split("_", 1)[1].isdigit()
    )
    if not available:
        return []
    return prediction[f"top_{max(available)}"]


def node_indexes(example):
    nodes = example["inference_inputs"]["schema_nodes"]
    by_id = {node["id"]: node for node in nodes}
    table_id_by_name = {}
    columns_by_table = defaultdict(list)
    for node in nodes:
        if node["type"] == "table":
            table_id_by_name[node["name"]] = node["id"]
        elif node["type"] == "column":
            columns_by_table[node.get("table")].append(node)
    return by_id, table_id_by_name, columns_by_table


def graph_indexes(example):
    by_src = defaultdict(list)
    by_dst = defaultdict(list)
    for edge in example["inference_inputs"].get("schema_edges", []):
        by_src[edge["src"]].append(edge)
        by_dst[edge["dst"]].append(edge)
    return by_src, by_dst


def base_scores(prediction, by_id):
    rows = available_top_rows(prediction)
    row_count = max(len(rows), 1)
    scores = {}
    ranks = {}
    min_score = None
    for rank, row in enumerate(rows, start=1):
        item_id = row["id"]
        raw = float(row.get("score", 0.0))
        rank_score = 1.0 - ((rank - 1) / row_count)
        calibrated = sigmoid(raw)
        scores[item_id] = 0.65 * rank_score + 0.35 * calibrated
        ranks[item_id] = rank
        min_score = raw if min_score is None else min(min_score, raw)
    default = 0.05
    for item_id in by_id:
        scores.setdefault(item_id, default)
        ranks.setdefault(item_id, row_count + 999)
    return scores, ranks


def column_text(node):
    return f"{node.get('table', '')} {node.get('column', '')} {node.get('name', '')} {node.get('data_type', '')}"


def exact_phrase_bonus(query_norm, node):
    column = normalize_text(node.get("column") or node.get("name"))
    table = normalize_text(node.get("table") or "")
    bonus = 0.0
    if column and f" {column} " in f" {query_norm} ":
        bonus += 1.0
    if table and f" {table} " in f" {query_norm} ":
        bonus += 0.2
    return bonus


def infer_intents(question, evidence):
    q_tokens = token_set(question)
    e_tokens = token_set(evidence)
    all_tokens = q_tokens | e_tokens
    return {
        "select": 1.0 if all_tokens & SELECT_TRIGGERS else 0.4,
        "where": 1.0 if all_tokens & WHERE_TRIGGERS or evidence else 0.3,
        "order": 1.0 if all_tokens & ORDER_TRIGGERS else 0.2,
    }


def clause_scores(node, question, evidence, intents):
    q_tokens = token_set(question)
    e_tokens = token_set(evidence)
    all_tokens = q_tokens | e_tokens
    node_tokens = token_set(column_text(node))
    col_tokens = token_set(node.get("column") or node.get("name"))
    query_norm = normalize_text(f"{question} {evidence}")
    data_type = str(node.get("data_type") or "").lower()

    text_overlap = overlap(all_tokens, node_tokens)
    question_overlap = overlap(q_tokens, col_tokens)
    evidence_overlap = overlap(e_tokens, col_tokens)
    phrase = exact_phrase_bonus(query_norm, node)
    numeric = 1.0 if data_type in NUMERIC_TYPES else 0.0
    date_like = 1.0 if any(t in col_tokens for t in {"date", "year", "time", "age"}) else 0.0
    value_like = 1.0 if any(
        t in col_tokens
        for t in {
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
        }
    ) else 0.0

    select_score = (
        0.35 * text_overlap
        + 0.25 * question_overlap
        + 0.20 * phrase
        + 0.20 * intents["select"] * (1.0 if value_like or "name" in col_tokens else 0.4)
    )
    where_score = (
        0.30 * text_overlap
        + 0.35 * evidence_overlap
        + 0.25 * phrase
        + 0.10 * intents["where"] * max(value_like, date_like, numeric * 0.5)
    )
    order_score = (
        0.25 * text_overlap
        + 0.15 * evidence_overlap
        + 0.25 * phrase
        + 0.35 * intents["order"] * max(numeric, date_like)
    )
    return {
        "select": select_score,
        "where": where_score,
        "order": order_score,
        "max_clause": max(select_score, where_score, order_score),
    }


def table_beliefs(by_id, columns_by_table, scores):
    beliefs = {}
    for table, columns in columns_by_table.items():
        table_node = next((node for node in by_id.values() if node.get("type") == "table" and node.get("name") == table), None)
        table_score = scores.get(table_node["id"], 0.0) if table_node else 0.0
        top_column_scores = sorted([scores.get(col["id"], 0.0) for col in columns], reverse=True)[:5]
        support = sum(top_column_scores) / len(top_column_scores) if top_column_scores else 0.0
        beliefs[table] = 0.55 * table_score + 0.45 * support
    return beliefs


def fk_endpoint_ids(example, selected_tables, by_id):
    endpoints = set()
    for edge in example["inference_inputs"].get("schema_edges", []):
        if edge.get("type") not in {"foreign_key_forward", "foreign_key_backward"}:
            continue
        src = by_id.get(edge["src"])
        dst = by_id.get(edge["dst"])
        if not src or not dst:
            continue
        if src.get("type") == "column" and dst.get("type") == "column":
            if src.get("table") in selected_tables and dst.get("table") in selected_tables:
                endpoints.add(src["id"])
                endpoints.add(dst["id"])
    return endpoints


def add_ranked_ids(selected, ranked_ids, budget):
    for item_id in ranked_ids:
        if len(selected) >= budget:
            break
        selected.append(item_id)


def select_schema(example, prediction, budget, args):
    by_id, table_id_by_name, columns_by_table = node_indexes(example)
    scores, ranks = base_scores(prediction, by_id)
    beliefs = table_beliefs(by_id, columns_by_table, scores)
    question = example["inference_inputs"].get("question") or ""
    evidence = example["inference_inputs"].get("evidence") or ""
    intents = infer_intents(question, evidence)

    selected_table_names = [
        table
        for table, _ in sorted(beliefs.items(), key=lambda item: item[1], reverse=True)[: args.max_tables]
    ]
    if not selected_table_names and beliefs:
        selected_table_names = [max(beliefs, key=beliefs.get)]

    columns = []
    for table in selected_table_names:
        for node in columns_by_table.get(table, []):
            c_scores = clause_scores(node, question, evidence, intents)
            base = scores.get(node["id"], 0.0)
            table_bonus = beliefs.get(table, 0.0)
            final = (
                args.base_weight * base
                + args.clause_weight * c_scores["max_clause"]
                + args.table_weight * table_bonus
            )
            columns.append(
                {
                    "id": node["id"],
                    "node": node,
                    "score": final,
                    "base": base,
                    "rank": ranks.get(node["id"], 10**9),
                    "clause_scores": c_scores,
                    "table_belief": table_bonus,
                }
            )

    table_ids = [table_id_by_name[table] for table in selected_table_names if table in table_id_by_name]
    selected = []
    add_ranked_ids(selected, table_ids, min(len(table_ids), budget))

    remaining_budget = budget - len(selected)
    fk_budget = min(args.fk_budget, max(0, remaining_budget // 3))
    fk_ids = sorted(
        fk_endpoint_ids(example, set(selected_table_names), by_id),
        key=lambda item_id: scores.get(item_id, 0.0),
        reverse=True,
    )
    add_ranked_ids(selected, [item_id for item_id in fk_ids if item_id not in selected], budget)

    remaining_budget = budget - len(selected)
    slot_plan = [
        ("where", min(args.where_budget, remaining_budget)),
        ("select", min(args.select_budget, max(0, remaining_budget - args.order_budget))),
        ("order", min(args.order_budget, max(0, budget - len(selected)))),
    ]
    for clause, slot in slot_plan:
        if slot <= 0:
            continue
        ranked = sorted(
            columns,
            key=lambda item: (item["clause_scores"][clause], item["score"], -item["rank"]),
            reverse=True,
        )
        add_ranked_ids(selected, [item["id"] for item in ranked if item["id"] not in selected], min(budget, len(selected) + slot))

    ranked_final = sorted(columns, key=lambda item: (item["score"], -item["rank"]), reverse=True)
    add_ranked_ids(selected, [item["id"] for item in ranked_final if item["id"] not in selected], budget)

    if len(selected) < budget:
        global_ids = sorted(by_id, key=lambda item_id: (scores.get(item_id, 0.0), -ranks.get(item_id, 10**9)), reverse=True)
        add_ranked_ids(selected, [item_id for item_id in global_ids if item_id not in selected], budget)

    column_debug = {item["id"]: item for item in columns}
    rows = []
    for item_id in selected[:budget]:
        node = by_id[item_id]
        detail = column_debug.get(item_id, {})
        rows.append(
            {
                "id": int(item_id),
                "type": node["type"],
                "name": node["name"],
                "score": float(detail.get("score", scores.get(item_id, 0.0))),
                "base_score": float(scores.get(item_id, 0.0)),
                "table_belief": float(detail.get("table_belief", beliefs.get(node.get("name"), 0.0))),
                "clause_scores": detail.get("clause_scores", {}),
                "selection_source": "tcce",
            }
        )
    return rows, {
        "selected_tables": selected_table_names,
        "table_beliefs": dict(sorted(beliefs.items(), key=lambda item: item[1], reverse=True)[:10]),
        "intent_scores": intents,
        "selected_count": len(rows),
        "table_node_count": sum(1 for row in rows if row["type"] == "table"),
        "column_node_count": sum(1 for row in rows if row["type"] == "column"),
    }


def gold_coverage(example, selected_rows):
    targets = example.get("training_targets", {})
    gold = set(targets.get("grounding_label_ids", []))
    selected = {row["id"] for row in selected_rows}
    nodes = {node["id"]: node for node in example["inference_inputs"]["schema_nodes"]}
    gold_columns = {item_id for item_id in gold if nodes.get(item_id, {}).get("type") == "column"}
    gold_tables = {item_id for item_id in gold if nodes.get(item_id, {}).get("type") == "table"}
    return {
        "schema_recall": len(gold & selected) / len(gold) if gold else None,
        "table_recall": len(gold_tables & selected) / len(gold_tables) if gold_tables else None,
        "column_recall": len(gold_columns & selected) / len(gold_columns) if gold_columns else None,
        "missing_gold_names": [nodes[item_id]["name"] for item_id in sorted(gold - selected) if item_id in nodes],
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
    parser.add_argument("--output-dir", default="experiments/stage5d_clause_aware_selection_v2_100")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--budgets", default="30,50,80")
    parser.add_argument("--max-tables", type=int, default=4)
    parser.add_argument("--fk-budget", type=int, default=6)
    parser.add_argument("--where-budget", type=int, default=10)
    parser.add_argument("--select-budget", type=int, default=8)
    parser.add_argument("--order-budget", type=int, default=4)
    parser.add_argument("--base-weight", type=float, default=0.52)
    parser.add_argument("--clause-weight", type=float, default=0.33)
    parser.add_argument("--table-weight", type=float, default=0.15)
    args = parser.parse_args()

    examples = read_jsonl(Path(args.dsg_data), args.limit)
    predictions = read_jsonl(Path(args.dsg_predictions), args.limit)
    if len(examples) != len(predictions):
        raise ValueError(f"Length mismatch: examples={len(examples)}, predictions={len(predictions)}")
    budgets = [int(value.strip()) for value in args.budgets.split(",") if value.strip()]
    max_budget = max(budgets)

    output_rows = []
    statistics = {
        "config": vars(args),
        "sample_count": len(examples),
        "budgets": {},
        "note": (
            "TCCE uses only inference-time information: question, evidence, schema graph, "
            "and DSG prediction scores. Gold labels are used here only for offline diagnostics."
        ),
    }
    coverage_by_budget = {budget: defaultdict(list) for budget in budgets}
    examples_md = []

    for example, prediction in zip(examples, predictions):
        row = {
            "example_id": example["example_id"],
            "db_id": example["inference_inputs"]["db_id"],
            "question": example["inference_inputs"].get("question"),
            "evidence": example["inference_inputs"].get("evidence"),
        }
        debug_by_budget = {}
        for budget in budgets:
            selected_rows, debug = select_schema(example, prediction, budget, args)
            row[f"top_{budget}"] = selected_rows
            debug_by_budget[str(budget)] = debug
            coverage = gold_coverage(example, selected_rows)
            for key, value in coverage.items():
                if key != "missing_gold_names":
                    coverage_by_budget[budget][key].append(value)
        row["tcce_debug"] = debug_by_budget
        output_rows.append(row)
        if len(examples_md) < 5:
            coverage = gold_coverage(example, row[f"top_{min(budgets)}"])
            examples_md.append(
                {
                    "question": row["question"],
                    "selected_tables": debug_by_budget[str(min(budgets))]["selected_tables"],
                    "top_names": [item["name"] for item in row[f"top_{min(budgets)}"][:20]],
                    "missing_gold_names": coverage["missing_gold_names"],
                }
            )

    for budget in budgets:
        statistics["budgets"][str(budget)] = {
            key: average(values) for key, values in sorted(coverage_by_budget[budget].items())
        }
        statistics["budgets"][str(budget)]["avg_selected_count"] = budget

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "dsg_tcce_predictions.jsonl", output_rows)
    write_json(output_dir / "selection_statistics.json", statistics)
    with (output_dir / "examples.md").open("w", encoding="utf-8") as f:
        f.write("# Stage 5-D TCCE examples\n\n")
        for index, item in enumerate(examples_md):
            f.write(f"## Example {index}\n\n")
            f.write(f"Question: {item['question']}\n\n")
            f.write(f"Selected tables: {item['selected_tables']}\n\n")
            f.write("Selected schema names:\n\n")
            for name in item["top_names"]:
                f.write(f"- `{name}`\n")
            f.write("\nMissing gold names for offline diagnosis:\n\n")
            for name in item["missing_gold_names"]:
                f.write(f"- `{name}`\n")
            f.write("\n")

    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
