import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grounding.stage5d_clause_aware_selector import (  # noqa: E402
    add_ranked_ids,
    base_scores,
    clause_scores,
    fk_endpoint_ids,
    gold_coverage,
    infer_intents,
    node_indexes,
    normalize_text,
    read_jsonl,
    table_beliefs,
    write_json,
    write_jsonl,
)


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "what",
    "which",
    "please",
    "list",
    "show",
    "all",
    "are",
    "was",
    "were",
    "from",
    "into",
    "than",
    "then",
    "sort",
    "order",
    "return",
    "students",
    "schools",
    "school",
    "count",
    "number",
    "rate",
    "rates",
}

VALUE_COLUMN_KEYWORDS = {
    "name",
    "type",
    "status",
    "county",
    "district",
    "city",
    "state",
    "charter",
    "funding",
    "option",
    "school",
    "virtual",
    "magnet",
}


def find_sqlite_index(db_root: Path):
    return {path.stem: path for path in db_root.rglob("*.sqlite")}


def quote_identifier(identifier):
    return '"' + str(identifier).replace('"', '""') + '"'


def stem_token(token):
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ly"):
        token = token[:-2]
    if len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def value_tokens(text):
    return [stem_token(token) for token in normalize_text(text).split() if token]


def salient_query_tokens(question, evidence, max_tokens):
    raw_tokens = value_tokens(f"{question} {evidence}")
    result = []
    seen = set()
    for token in raw_tokens:
        if token in STOPWORDS:
            continue
        if len(token) < 3 and not token.isdigit():
            continue
        if token not in seen:
            seen.add(token)
            result.append(token)
        if len(result) >= max_tokens:
            break
    return result


def value_match_score(query_token_set, query_norm, value):
    value_norm = normalize_text(value)
    if not value_norm:
        return 0.0
    if value_norm.isdigit():
        if re.search(rf"(=|equals|is)\s*{re.escape(value_norm)}\b", query_norm):
            return 0.9
        return 0.0
    if f" {value_norm} " in f" {query_norm} ":
        return 1.0
    v_tokens = set(value_tokens(value))
    if not v_tokens:
        return 0.0
    coverage = len(v_tokens & query_token_set) / len(v_tokens)
    hit_rate = len(v_tokens & query_token_set) / max(len(query_token_set), 1)
    return max(0.85 * coverage, 0.35 * hit_rate)


def is_value_searchable(node):
    dtype = str(node.get("data_type") or "").lower()
    column = normalize_text(node.get("column") or node.get("name"))
    if dtype in {"text", "date", "time"}:
        return True
    return any(token in column for token in ["name", "type", "status", "date", "year", "city", "county", "state"])


def is_forced_value_column(node):
    if not is_value_searchable(node):
        return False
    column_tokens = set(value_tokens(node.get("column") or node.get("name")))
    return bool(column_tokens & VALUE_COLUMN_KEYWORDS)


def fetch_matched_values(connection, table, column, query_tokens, max_scan_values):
    if not query_tokens:
        return []
    clauses = " OR ".join(["lower(cast({col} as text)) LIKE ?".format(col=quote_identifier(column)) for _ in query_tokens])
    sql = (
        f"SELECT DISTINCT {quote_identifier(column)} "
        f"FROM {quote_identifier(table)} "
        f"WHERE {quote_identifier(column)} IS NOT NULL AND ({clauses}) "
        f"LIMIT {int(max_scan_values)}"
    )
    params = [f"%{token}%" for token in query_tokens]
    try:
        cursor = connection.execute(sql, params)
        values = [row[0] for row in cursor.fetchall()]
    except sqlite3.Error:
        return []
    return [str(value) for value in values if value is not None and str(value).strip()]


def value_evidence_by_column(db_path, columns, question, evidence, args):
    if db_path is None:
        return {}
    query_tokens = salient_query_tokens(question, evidence, args.max_value_query_tokens)
    query_token_set = set(value_tokens(f"{question} {evidence}"))
    query_norm = normalize_text(f"{question} {evidence}")
    evidence = {}
    connection = sqlite3.connect(str(db_path))
    try:
        for node in columns:
            if not is_value_searchable(node):
                continue
            values = fetch_matched_values(
                connection,
                node.get("table"),
                node.get("column"),
                query_tokens,
                args.max_scan_values,
            )
            scored = []
            for value in values:
                score = value_match_score(query_token_set, query_norm, value)
                if score >= args.min_value_score:
                    scored.append({"value": value, "score": score})
            scored.sort(key=lambda item: item["score"], reverse=True)
            if scored:
                evidence[node["id"]] = scored[: args.max_values_per_column]
    finally:
        connection.close()
    return evidence


def select_schema_value_aware(example, prediction, db_path, budget, args, value_cache=None):
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

    candidate_columns = []
    for table in selected_table_names:
        candidate_columns.extend(columns_by_table.get(table, []))
    pre_value_ranked = []
    for node in candidate_columns:
        c_scores = clause_scores(node, question, evidence, intents)
        base = scores.get(node["id"], 0.0)
        table_bonus = beliefs.get(node.get("table"), 0.0)
        pre_value_ranked.append(
            {
                "node": node,
                "score": 0.45 * c_scores["where"] + 0.25 * c_scores["select"] + 0.20 * base + 0.10 * table_bonus,
            }
        )
    value_search_columns = [
        item["node"]
        for item in sorted(pre_value_ranked, key=lambda item: item["score"], reverse=True)[
            : args.value_search_columns
        ]
    ]
    selected_value_search_ids = {node["id"] for node in value_search_columns}
    pre_score_by_id = {item["node"]["id"]: item["score"] for item in pre_value_ranked}
    forced_value_columns = sorted(
        [
            node
            for node in candidate_columns
            if node["id"] not in selected_value_search_ids and is_forced_value_column(node)
        ],
        key=lambda node: (pre_score_by_id.get(node["id"], 0.0), scores.get(node["id"], 0.0)),
        reverse=True,
    )[: args.max_forced_value_columns]
    value_search_columns.extend(forced_value_columns)
    cache_key = (
        str(db_path) if db_path else "",
        normalize_text(question),
        normalize_text(evidence),
        tuple(sorted(node["id"] for node in value_search_columns)),
        args.max_value_query_tokens,
        args.max_scan_values,
        args.max_values_per_column,
        args.min_value_score,
    )
    if value_cache is not None and cache_key in value_cache:
        value_evidence = value_cache[cache_key]
    else:
        value_evidence = value_evidence_by_column(db_path, value_search_columns, question, evidence, args)
        if value_cache is not None:
            value_cache[cache_key] = value_evidence

    columns = []
    for node in candidate_columns:
        c_scores = clause_scores(node, question, evidence, intents)
        base = scores.get(node["id"], 0.0)
        table_bonus = beliefs.get(node.get("table"), 0.0)
        value_score = max([item["score"] for item in value_evidence.get(node["id"], [])], default=0.0)
        final = (
            args.base_weight * base
            + args.clause_weight * c_scores["max_clause"]
            + args.table_weight * table_bonus
            + args.value_weight * value_score
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
                "value_score": value_score,
                "value_hints": value_evidence.get(node["id"], []),
            }
        )

    selected = []
    table_ids = [table_id_by_name[table] for table in selected_table_names if table in table_id_by_name]
    add_ranked_ids(selected, table_ids, min(len(table_ids), budget))

    fk_ids = sorted(
        fk_endpoint_ids(example, set(selected_table_names), by_id),
        key=lambda item_id: scores.get(item_id, 0.0),
        reverse=True,
    )
    add_ranked_ids(selected, [item_id for item_id in fk_ids if item_id not in selected], budget)

    value_ranked = sorted(
        columns,
        key=lambda item: (item["value_score"], item["clause_scores"]["where"], item["score"], -item["rank"]),
        reverse=True,
    )
    add_ranked_ids(
        selected,
        [item["id"] for item in value_ranked if item["id"] not in selected and item["value_score"] > 0],
        min(budget, len(selected) + args.value_budget),
    )

    slot_plan = [
        ("where", args.where_budget),
        ("select", args.select_budget),
        ("order", args.order_budget),
    ]
    for clause, slot in slot_plan:
        ranked = sorted(
            columns,
            key=lambda item: (item["clause_scores"][clause], item["score"], item["value_score"], -item["rank"]),
            reverse=True,
        )
        add_ranked_ids(selected, [item["id"] for item in ranked if item["id"] not in selected], min(budget, len(selected) + slot))

    ranked_final = sorted(columns, key=lambda item: (item["score"], item["value_score"], -item["rank"]), reverse=True)
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
                "value_score": float(detail.get("value_score", 0.0)),
                "value_hints": detail.get("value_hints", []),
                "clause_scores": detail.get("clause_scores", {}),
                "selection_source": "value_tcce",
            }
        )
    debug = {
        "selected_tables": selected_table_names,
        "table_beliefs": dict(sorted(beliefs.items(), key=lambda item: item[1], reverse=True)[:10]),
        "intent_scores": intents,
        "selected_count": len(rows),
        "table_node_count": sum(1 for row in rows if row["type"] == "table"),
        "column_node_count": sum(1 for row in rows if row["type"] == "column"),
        "value_column_count": sum(1 for row in rows if row.get("value_hints")),
    }
    return rows, debug


def build_fk_lines(example, selected_ids):
    by_id = {node["id"]: node for node in example["inference_inputs"]["schema_nodes"]}
    selected_ids = set(selected_ids)
    lines = set()
    for edge in example["inference_inputs"].get("schema_edges", []):
        if edge.get("type") != "foreign_key_forward":
            continue
        src = by_id.get(edge["src"])
        dst = by_id.get(edge["dst"])
        if not src or not dst:
            continue
        if src["id"] in selected_ids and dst["id"] in selected_ids:
            lines.add(f"{src['name']} = {dst['name']}")
    return sorted(lines)


def schema_text_from_rows(example, rows):
    by_id = {node["id"]: node for node in example["inference_inputs"]["schema_nodes"]}
    selected_ids = {row["id"] for row in rows}
    row_by_id = {row["id"]: row for row in rows}
    grouped = defaultdict(list)
    table_order = []
    for row in rows:
        node = by_id[row["id"]]
        if node["type"] == "table":
            if node["name"] not in grouped:
                table_order.append(node["name"])
            grouped.setdefault(node["name"], [])
        elif node["type"] == "column":
            table = node.get("table")
            if table not in grouped:
                table_order.append(table)
            grouped[table].append((node, row_by_id[node["id"]]))

    lines = []
    for table in table_order:
        lines.append(f"Table {table}:")
        for node, row in sorted(grouped[table], key=lambda item: item[0].get("column", "")):
            dtype = f" ({node.get('data_type')})" if node.get("data_type") else ""
            hints = row.get("value_hints") or []
            if hints:
                values = ", ".join(f'"{hint["value"]}"' for hint in hints[:2])
                lines.append(f"- `{node.get('column')}`{dtype}; matched values: {values}")
            else:
                lines.append(f"- `{node.get('column')}`{dtype}")
        lines.append("")

    fks = build_fk_lines(example, selected_ids)
    if fks:
        lines.append("Foreign keys:")
        for fk in fks:
            lines.append(f"- {fk}")
        lines.append("")
    return "\n".join(lines).strip()


def build_prompt(question, evidence, schema_text):
    evidence_block = f"\nEvidence:\n{evidence}\n" if evidence else ""
    return (
        "You are an expert SQLite SQL generator.\n\n"
        "Rules:\n"
        "1. Use only the exact table and column names listed in the schema.\n"
        "2. Do not invent columns or tables.\n"
        "3. Use SQLite syntax only.\n"
        "4. Quote column names with backticks if they contain spaces, parentheses, %, /, or hyphens.\n"
        "5. If a column lists matched values, prefer those exact values when they match the question.\n"
        "6. Return only one SQL query, with no explanation.\n\n"
        "Given the database schema and question, generate a valid SQLite SQL query.\n\n"
        f"Database schema:\n{schema_text}\n\n"
        f"Question:\n{question}\n"
        f"{evidence_block}\n"
        "Return only the SQL query."
    )


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
    parser.add_argument("--db-root", default="Data/BIRD/dev_databases")
    parser.add_argument("--output-dir", default="experiments/stage5e_value_aware_prompts_v2_100")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--budgets", default="30,50,80")
    parser.add_argument("--max-tables", type=int, default=4)
    parser.add_argument("--fk-budget", type=int, default=6)
    parser.add_argument("--where-budget", type=int, default=8)
    parser.add_argument("--select-budget", type=int, default=7)
    parser.add_argument("--order-budget", type=int, default=4)
    parser.add_argument("--value-budget", type=int, default=6)
    parser.add_argument("--base-weight", type=float, default=0.45)
    parser.add_argument("--clause-weight", type=float, default=0.25)
    parser.add_argument("--table-weight", type=float, default=0.10)
    parser.add_argument("--value-weight", type=float, default=0.45)
    parser.add_argument("--max-value-query-tokens", type=int, default=10)
    parser.add_argument(
        "--value-search-columns",
        type=int,
        default=12,
        help="Only retrieve values for the top-N likely value/WHERE columns after table-conditioned scoring.",
    )
    parser.add_argument(
        "--max-forced-value-columns",
        type=int,
        default=24,
        help="Also retrieve values for up to this many typical value-bearing columns in selected tables.",
    )
    parser.add_argument("--max-scan-values", type=int, default=20)
    parser.add_argument("--max-values-per-column", type=int, default=2)
    parser.add_argument("--min-value-score", type=float, default=0.45)
    args = parser.parse_args()

    examples = read_jsonl(Path(args.dsg_data), args.limit)
    predictions = read_jsonl(Path(args.dsg_predictions), args.limit)
    if len(examples) != len(predictions):
        raise ValueError(f"Length mismatch: examples={len(examples)}, predictions={len(predictions)}")

    sqlite_index = find_sqlite_index(Path(args.db_root))
    budgets = [int(value.strip()) for value in args.budgets.split(",") if value.strip()]
    output_dir = Path(args.output_dir)

    prediction_rows = []
    prompts_by_budget = {budget: [] for budget in budgets}
    coverage_by_budget = {budget: defaultdict(list) for budget in budgets}
    value_hint_counts = Counter()
    value_cache = {}

    for example, prediction in zip(examples, predictions):
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
            selected_rows, debug = select_schema_value_aware(
                example, prediction, db_path, budget, args, value_cache=value_cache
            )
            row[f"top_{budget}"] = selected_rows
            debug_by_budget[str(budget)] = debug
            coverage = gold_coverage(example, selected_rows)
            for key, value in coverage.items():
                if key != "missing_gold_names":
                    coverage_by_budget[budget][key].append(value)
            schema_text = schema_text_from_rows(example, selected_rows)
            value_hint_counts[str(budget)] += sum(1 for item in selected_rows if item.get("value_hints"))
            prompts_by_budget[budget].append(
                {
                    "question_id": example.get("metadata", {}).get("question_id"),
                    "db_id": db_id,
                    "setting": f"value_tcce_top{budget}",
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
        row["value_tcce_debug"] = debug_by_budget
        prediction_rows.append(row)

    statistics = {
        "config": vars(args),
        "sample_count": len(examples),
        "budgets": {},
        "note": (
            "Value-TCCE uses test-time available database contents for value evidence. "
            "Gold labels are used only for offline diagnostics."
        ),
        "value_cache_entries": len(value_cache),
    }
    for budget in budgets:
        statistics["budgets"][str(budget)] = {
            key: average(values) for key, values in sorted(coverage_by_budget[budget].items())
        }
        statistics["budgets"][str(budget)]["avg_value_hint_columns"] = (
            value_hint_counts[str(budget)] / len(examples) if examples else 0.0
        )

    write_jsonl(output_dir / "value_tcce_predictions.jsonl", prediction_rows)
    for budget, prompts in prompts_by_budget.items():
        write_jsonl(output_dir / f"prompts_value_tcce_top{budget}_dev.jsonl", prompts)
    write_json(output_dir / "selection_statistics.json", statistics)

    with (output_dir / "examples.md").open("w", encoding="utf-8") as f:
        f.write("# Stage 5-E Value-TCCE examples\n\n")
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
