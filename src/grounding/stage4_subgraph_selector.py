import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path, limit=None):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def schema_indexes(schema_items):
    by_id = {item["id"]: item for item in schema_items}
    table_to_id = {}
    column_to_id = {}
    columns_by_table = defaultdict(list)
    for item in schema_items:
        if item["type"] == "table":
            table_to_id[item["name"]] = item["id"]
        elif item["type"] == "column":
            pair = (item.get("table"), item.get("column"))
            column_to_id[pair] = item["id"]
            columns_by_table[item.get("table")].append(item["id"])
    return by_id, table_to_id, column_to_id, columns_by_table


def ranked_items(prediction, required_k):
    available = sorted(
        int(key.split("_", 1)[1])
        for key in prediction
        if key.startswith("top_") and key.split("_", 1)[1].isdigit()
    )
    larger_or_equal = [value for value in available if value >= required_k]
    if larger_or_equal:
        key = f"top_{min(larger_or_equal)}"
    elif available:
        key = f"top_{max(available)}"
    else:
        key = "top_30"
    return prediction.get(key, [])[:required_k]


def add_owner_table(selected, item, table_to_id):
    table = item.get("table")
    if table in table_to_id:
        selected.add(table_to_id[table])


def add_fk_endpoint_closure(selected, schema_items, dev_table_entry):
    selected = set(selected)
    by_id, table_to_id, column_to_id, _ = schema_indexes(schema_items)
    selected_tables = set()
    selected_columns = set()
    for item_id in selected:
        item = by_id[item_id]
        if item["type"] == "table":
            selected_tables.add(item["name"])
        elif item["type"] == "column":
            selected_tables.add(item.get("table"))
            selected_columns.add((item.get("table"), item.get("column")))
    for table in list(selected_tables):
        if table in table_to_id:
            selected.add(table_to_id[table])

    if not dev_table_entry:
        return selected

    tables = dev_table_entry.get("table_names_original", [])
    columns = dev_table_entry.get("column_names_original", [])
    for left_idx, right_idx in dev_table_entry.get("foreign_keys", []):
        if left_idx >= len(columns) or right_idx >= len(columns):
            continue
        left_table_idx, left_col = columns[left_idx]
        right_table_idx, right_col = columns[right_idx]
        if left_table_idx < 0 or right_table_idx < 0:
            continue
        left_table = tables[left_table_idx]
        right_table = tables[right_table_idx]
        left_pair = (left_table, left_col)
        right_pair = (right_table, right_col)
        table_pair_selected = left_table in selected_tables and right_table in selected_tables
        column_pair_selected = left_pair in selected_columns or right_pair in selected_columns
        if table_pair_selected or column_pair_selected:
            for table, column in [left_pair, right_pair]:
                if table in table_to_id:
                    selected.add(table_to_id[table])
                column_id = column_to_id.get((table, column))
                if column_id is not None:
                    selected.add(column_id)
    return selected


def select_subgraph_ids(
    record,
    prediction,
    dev_table_entry,
    budget,
    source_top_k,
    seed_ratio,
    local_columns_per_table,
    fill_remaining,
):
    schema_items = record["schema_items"]
    by_id, table_to_id, _, columns_by_table = schema_indexes(schema_items)
    ranked = ranked_items(prediction, source_top_k)
    ranked_ids = [item["id"] for item in ranked]
    score_by_id = {item["id"]: item.get("score", 0.0) for item in ranked}
    ranked_columns = [
        item_id for item_id in ranked_ids if by_id.get(item_id, {}).get("type") == "column"
    ]
    ranked_tables = [
        item_id for item_id in ranked_ids if by_id.get(item_id, {}).get("type") == "table"
    ]

    selected = set()
    trace = {
        "seed_columns": [],
        "owner_tables": [],
        "local_context_columns": [],
        "filled_columns": [],
        "filled_tables": [],
        "fk_closure_added": [],
    }

    def add_column_with_owner(column_id, reason):
        item = by_id[column_id]
        before = set(selected)
        selected.add(column_id)
        add_owner_table(selected, item, table_to_id)
        if reason in trace:
            trace[reason].append(item["name"])
        for added in selected - before:
            added_item = by_id[added]
            if added_item["type"] == "table":
                trace["owner_tables"].append(added_item["name"])

    seed_budget = max(1, int(budget * seed_ratio))
    for column_id in ranked_columns:
        if len(selected) >= seed_budget:
            break
        add_column_with_owner(column_id, "seed_columns")

    selected_tables = {
        by_id[item_id]["name"] for item_id in selected if by_id[item_id]["type"] == "table"
    }
    for table in sorted(selected_tables):
        table_columns = sorted(
            columns_by_table.get(table, []),
            key=lambda item_id: score_by_id.get(item_id, -1e9),
            reverse=True,
        )
        local_added = 0
        for column_id in table_columns:
            if column_id in selected:
                continue
            if len(selected) >= budget:
                break
            add_column_with_owner(column_id, "local_context_columns")
            local_added += 1
            if local_added >= local_columns_per_table:
                break

    if fill_remaining:
        for column_id in ranked_columns:
            if len(selected) >= budget:
                break
            if column_id not in selected:
                add_column_with_owner(column_id, "filled_columns")

        for table_id in ranked_tables:
            if len(selected) >= budget:
                break
            if table_id not in selected:
                selected.add(table_id)
                trace["filled_tables"].append(by_id[table_id]["name"])

    before_closure = set(selected)
    selected = add_fk_endpoint_closure(selected, schema_items, dev_table_entry)
    for added in sorted(selected - before_closure):
        trace["fk_closure_added"].append(by_id[added]["name"])

    return sorted(selected), trace


def group_schema_items(schema_items, selected_ids):
    selected_ids = set(selected_ids)
    tables = defaultdict(list)
    table_names = set()
    for item in schema_items:
        if item["id"] not in selected_ids:
            continue
        if item["type"] == "table":
            table_names.add(item["name"])
        elif item["type"] == "column":
            tables[item.get("table", "")].append(
                {
                    "name": item.get("column", item.get("name", "")),
                    "type": item.get("data_type"),
                    "full_name": item.get("name"),
                }
            )
    # Do not emit table-only placeholders in the prompt. Join paths and owner
    # table names are still preserved whenever endpoint columns are selected.
    return dict(sorted(tables.items(), key=lambda x: x[0]))


def build_fk_text(dev_table_entry, selected_schema):
    if not dev_table_entry:
        return []
    tables = dev_table_entry.get("table_names_original", [])
    columns = dev_table_entry.get("column_names_original", [])
    selected_pairs = set()
    for table, cols in selected_schema.items():
        for col in cols:
            selected_pairs.add((table, col["name"]))
    selected_tables = set(selected_schema.keys())
    fk_lines = []
    for left_idx, right_idx in dev_table_entry.get("foreign_keys", []):
        if left_idx >= len(columns) or right_idx >= len(columns):
            continue
        left_table_idx, left_col = columns[left_idx]
        right_table_idx, right_col = columns[right_idx]
        if left_table_idx < 0 or right_table_idx < 0:
            continue
        left_table = tables[left_table_idx]
        right_table = tables[right_table_idx]
        left_selected = (left_table, left_col) in selected_pairs or left_table in selected_tables
        right_selected = (right_table, right_col) in selected_pairs or right_table in selected_tables
        if left_selected and right_selected:
            fk_lines.append(f"{left_table}.{left_col} = {right_table}.{right_col}")
    return sorted(set(fk_lines))


def schema_to_text(grouped_schema, foreign_keys, include_selection_notes=True):
    lines = []
    lines.append("Selected schema graph:")
    lines.append("")
    lines.append("Selected tables and columns:")
    for table, columns in grouped_schema.items():
        lines.append(f"Table {table}:")
        for column in sorted(columns, key=lambda x: x["name"]):
            dtype = f" ({column['type']})" if column.get("type") else ""
            lines.append(f"- `{column['name']}`{dtype}")
        lines.append("")
    if foreign_keys:
        lines.append("Join paths:")
        for fk in foreign_keys:
            lines.append(f"- {fk}")
        lines.append("")
    if include_selection_notes:
        lines.append("Selection notes:")
        lines.append("- Columns are selected by graph-aware schema grounding.")
        lines.append("- Owner tables are included for every selected column.")
        lines.append("- Foreign-key endpoint columns are included for valid joins.")
        lines.append("")
    return "\n".join(lines).strip()


def build_prompt(question, evidence, schema_text):
    evidence_block = f"\nEvidence:\n{evidence}\n" if evidence else ""
    return (
        "You are an expert SQLite SQL generator.\n\n"
        "Rules:\n"
        "1. Use only the exact table and column names listed in the schema graph.\n"
        "2. Do not invent columns or tables.\n"
        "3. Use SQLite syntax only.\n"
        "4. Quote column names with backticks if they contain spaces, parentheses, %, /, or hyphens.\n"
        "5. Use the join paths when a query needs multiple tables.\n"
        "6. Return only one SQL query, with no explanation.\n\n"
        "Given the database schema graph and question, generate a valid SQLite SQL query.\n\n"
        f"Database schema graph:\n{schema_text}\n\n"
        f"Question:\n{question}\n"
        f"{evidence_block}\n"
        "Return only the SQL query."
    )


def make_prompt_record(record, prediction, dev_table_entry, budget, args):
    selected_ids, trace = select_subgraph_ids(
        record,
        prediction,
        dev_table_entry,
        budget=budget,
        source_top_k=args.source_top_k,
        seed_ratio=args.seed_ratio,
        local_columns_per_table=args.local_columns_per_table,
        fill_remaining=args.fill_remaining,
    )
    grouped = group_schema_items(record["schema_items"], selected_ids)
    fks = build_fk_text(dev_table_entry, grouped)
    schema_text = schema_to_text(grouped, fks, include_selection_notes=not args.no_selection_notes)
    prompt = build_prompt(record.get("question"), record.get("evidence"), schema_text)
    return {
        "question_id": record.get("question_id"),
        "db_id": record["db_id"],
        "setting": f"rgta_subgraph_budget{budget}",
        "question": record.get("question"),
        "evidence": record.get("evidence"),
        "selected_schema": {"tables": grouped, "foreign_keys": fks},
        "selected_schema_item_count": len(set(selected_ids)),
        "budget": budget,
        "source_top_k": args.source_top_k,
        "selection_trace": trace,
        "schema_text": schema_text,
        "prompt": prompt,
        "gold_sql": record.get("sql"),
        "gold_labels": record.get("label_names", []),
    }


def load_dev_tables(path: Path):
    return {entry["db_id"]: entry for entry in read_json(path)}


def summarize_prompts(prompts):
    prompt_lengths = [len(item["prompt"]) for item in prompts]
    selected_counts = [item["selected_schema_item_count"] for item in prompts]
    empty_table_count = 0
    fk_counts = []
    schema_recalls = []
    table_recalls = []
    column_recalls = []
    missing_any_gold_column = 0
    for item in prompts:
        selected_tables = set(item["selected_schema"]["tables"].keys())
        selected_columns = set()
        for _, columns in item["selected_schema"]["tables"].items():
            if not columns:
                empty_table_count += 1
        for table, columns in item["selected_schema"]["tables"].items():
            for column in columns:
                selected_columns.add(f"{table}.{column['name']}")
        fk_counts.append(len(item["selected_schema"]["foreign_keys"]))
        gold = set(item.get("gold_labels", []))
        gold_tables = {name for name in gold if "." not in name}
        gold_columns = {name for name in gold if "." in name}
        gold_tables |= {name.split(".", 1)[0] for name in gold_columns}
        selected_all = selected_tables | selected_columns
        if gold:
            schema_recalls.append(len(gold & selected_all) / len(gold))
        if gold_tables:
            table_recalls.append(len(gold_tables & selected_tables) / len(gold_tables))
        if gold_columns:
            column_recalls.append(len(gold_columns & selected_columns) / len(gold_columns))
            if not gold_columns <= selected_columns:
                missing_any_gold_column += 1
    return {
        "count": len(prompts),
        "avg_prompt_chars": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
        "max_prompt_chars": max(prompt_lengths) if prompt_lengths else 0,
        "avg_selected_schema_items": sum(selected_counts) / len(selected_counts)
        if selected_counts
        else 0,
        "max_selected_schema_items": max(selected_counts) if selected_counts else 0,
        "empty_table_count": empty_table_count,
        "avg_fk_count": sum(fk_counts) / len(fk_counts) if fk_counts else 0,
        "schema_recall": sum(schema_recalls) / len(schema_recalls) if schema_recalls else 0,
        "table_recall": sum(table_recalls) / len(table_recalls) if table_recalls else 0,
        "column_recall": sum(column_recalls) / len(column_recalls) if column_recalls else 0,
        "missing_any_gold_column_count": missing_any_gold_column,
    }


def write_examples(path: Path, prompts_by_setting, max_examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for setting, prompts in prompts_by_setting.items():
            for index, item in enumerate(prompts[:max_examples], start=1):
                f.write(f"# {setting} example {index}\n\n")
                f.write(f"DB: `{item['db_id']}`\n\n")
                f.write(f"Question: {item['question']}\n\n")
                if item.get("evidence"):
                    f.write(f"Evidence: {item['evidence']}\n\n")
                f.write(f"Selected schema items: {item['selected_schema_item_count']}\n\n")
                f.write("```text\n")
                f.write(item["prompt"])
                f.write("\n```\n\n")


def parse_budgets(text):
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev-labels",
        default="experiments/stage1_label_extraction/bird_dev_grounding_labels.jsonl",
    )
    parser.add_argument("--dev-tables", default="Data/BIRD/dev_tables.json")
    parser.add_argument(
        "--rgta-predictions",
        default="experiments/stage2_rgta_torch_grounding_hybrid_500_top80/rgcn_torch_dev_predictions.jsonl",
    )
    parser.add_argument("--output-dir", default="experiments/stage4_subgraph_selection")
    parser.add_argument("--budgets", default="50,80")
    parser.add_argument("--source-top-k", type=int, default=80)
    parser.add_argument("--seed-ratio", type=float, default=0.55)
    parser.add_argument("--local-columns-per-table", type=int, default=4)
    parser.add_argument(
        "--fill-remaining",
        action="store_true",
        help="Fill unused budget with remaining high-score nodes. Default keeps budget as an upper bound.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument("--no-selection-notes", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(Path(args.dev_labels), limit=args.limit)
    predictions = read_jsonl(Path(args.rgta_predictions), limit=args.limit)
    dev_tables = load_dev_tables(Path(args.dev_tables))
    budgets = parse_budgets(args.budgets)

    if len(records) != len(predictions):
        raise ValueError(f"Input length mismatch: labels={len(records)}, predictions={len(predictions)}")

    prompts_by_setting = {}
    statistics = {
        "config": vars(args),
        "settings": {},
    }
    for budget in budgets:
        setting = f"rgta_subgraph_budget{budget}"
        prompts = []
        for record, prediction in zip(records, predictions):
            prompts.append(
                make_prompt_record(
                    record,
                    prediction,
                    dev_tables.get(record["db_id"]),
                    budget,
                    args,
                )
            )
        prompts_by_setting[setting] = prompts
        write_jsonl(output_dir / f"prompts_{setting}_dev.jsonl", prompts)
        statistics["settings"][setting] = summarize_prompts(prompts)

    write_json(output_dir / "subgraph_selection_statistics.json", statistics)
    write_examples(output_dir / "subgraph_examples.md", prompts_by_setting, args.max_examples)
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
