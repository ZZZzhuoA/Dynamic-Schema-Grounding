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
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_prediction_topk(path: Path, limit=None):
    records = read_jsonl(path, limit=limit)
    return records


def group_schema_items(schema_items, selected_ids):
    selected_ids = set(selected_ids)
    tables = defaultdict(list)
    table_names = set()
    for item in schema_items:
        if item["id"] not in selected_ids:
            continue
        if item["type"] == "table":
            table_names.add(item["name"])
        else:
            tables[item.get("table", "")].append(
                {
                    "name": item.get("column", item.get("name", "")),
                    "type": item.get("data_type"),
                    "full_name": item.get("name"),
                }
            )

    # Ensure selected table-only nodes still appear even if no selected column.
    for table_name in table_names:
        tables.setdefault(table_name, [])
    return dict(sorted(tables.items(), key=lambda x: x[0]))


def schema_item_indexes(schema_items):
    by_full_name = {}
    table_items = {}
    for item in schema_items:
        if item["type"] == "table":
            table_items[item["name"]] = item["id"]
        else:
            by_full_name[(item.get("table"), item.get("column"))] = item["id"]
    return table_items, by_full_name


def add_fk_endpoint_closure(schema_items, selected_ids, dev_table_entry):
    """Add owning tables and FK endpoint columns for selected schema.

    This avoids prompts that mention a foreign key while omitting the endpoint
    columns under their tables, and it prevents fake placeholders such as
    "[no selected columns]".
    """
    selected = set(selected_ids)
    table_items, column_items = schema_item_indexes(schema_items)
    selected_tables = set()
    selected_columns = set()
    for item in schema_items:
        if item["id"] not in selected:
            continue
        if item["type"] == "table":
            selected_tables.add(item["name"])
        else:
            selected_tables.add(item.get("table"))
            selected_columns.add((item.get("table"), item.get("column")))

    for table in list(selected_tables):
        if table in table_items:
            selected.add(table_items[table])

    if not dev_table_entry:
        return sorted(selected)

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
            for table, col in [left_pair, right_pair]:
                if table in table_items:
                    selected.add(table_items[table])
                col_id = column_items.get((table, col))
                if col_id is not None:
                    selected.add(col_id)

    return sorted(selected)


def selected_full_schema_ids(record):
    return [item["id"] for item in record["schema_items"]]


def selected_oracle_ids(record):
    return list(record["whole_sql_labels"])


def selected_prediction_ids(prediction, top_k):
    top_key = f"top_{top_k}"
    if top_key in prediction:
        return [item["id"] for item in prediction[top_key][:top_k]]
    if "top_30" in prediction and top_k <= 30:
        return [item["id"] for item in prediction["top_30"][:top_k]]
    available = sorted(
        int(key.split("_", 1)[1])
        for key in prediction
        if key.startswith("top_") and key.split("_", 1)[1].isdigit()
    )
    raise ValueError(
        f"Prediction does not contain top_{top_k}; available top-k fields: {available}"
    )


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


def schema_to_text(grouped_schema, foreign_keys):
    lines = []
    for table, columns in grouped_schema.items():
        lines.append(f"Table {table}:")
        if columns:
            for column in sorted(columns, key=lambda x: x["name"]):
                dtype = f" ({column['type']})" if column.get("type") else ""
                lines.append(f"- `{column['name']}`{dtype}")
        lines.append("")
    if foreign_keys:
        lines.append("Foreign keys:")
        for fk in foreign_keys:
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
        "5. Return only one SQL query, with no explanation.\n\n"
        "Given the database schema and question, generate a valid SQLite SQL query.\n\n"
        f"Database schema:\n{schema_text}\n\n"
        f"Question:\n{question}\n"
        f"{evidence_block}\n"
        "Return only the SQL query."
    )


def make_prompt_record(record, setting, selected_ids, dev_table_entry):
    selected_ids = add_fk_endpoint_closure(record["schema_items"], selected_ids, dev_table_entry)
    grouped = group_schema_items(record["schema_items"], selected_ids)
    fks = build_fk_text(dev_table_entry, grouped)
    schema_text = schema_to_text(grouped, fks)
    prompt = build_prompt(record.get("question"), record.get("evidence"), schema_text)
    return {
        "question_id": record.get("question_id"),
        "db_id": record["db_id"],
        "setting": setting,
        "question": record.get("question"),
        "evidence": record.get("evidence"),
        "selected_schema": {"tables": grouped, "foreign_keys": fks},
        "selected_schema_item_count": len(set(selected_ids)),
        "schema_text": schema_text,
        "prompt": prompt,
        "gold_sql": record.get("sql"),
        "gold_labels": record.get("label_names", []),
    }


def load_dev_tables(path: Path):
    return {entry["db_id"]: entry for entry in read_json(path)}


def build_setting_records(records, setting, selected_id_fn, dev_tables):
    prompts = []
    prompt_lengths = []
    selected_counts = []
    for idx, record in enumerate(records):
        selected_ids = selected_id_fn(idx, record)
        prompt_record = make_prompt_record(
            record,
            setting,
            selected_ids,
            dev_tables.get(record["db_id"]),
        )
        prompts.append(prompt_record)
        prompt_lengths.append(len(prompt_record["prompt"]))
        selected_counts.append(prompt_record["selected_schema_item_count"])
    return prompts, {
        "count": len(prompts),
        "avg_prompt_chars": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
        "max_prompt_chars": max(prompt_lengths) if prompt_lengths else 0,
        "avg_selected_schema_items": sum(selected_counts) / len(selected_counts)
        if selected_counts
        else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev-labels",
        default="experiments/stage1_label_extraction/bird_dev_grounding_labels.jsonl",
    )
    parser.add_argument("--dev-tables", default="Data/BIRD/dev_tables.json")
    parser.add_argument(
        "--lexical-predictions",
        default="experiments/stage2_static_grounding/lexical_dev_predictions.jsonl",
    )
    parser.add_argument(
        "--rgcn-predictions",
        default="experiments/stage2_rgcn_torch_grounding_hybrid_500/rgcn_torch_dev_predictions.jsonl",
    )
    parser.add_argument(
        "--rgta-predictions",
        default="experiments/stage2_rgta_torch_grounding_hybrid_500/rgcn_torch_dev_predictions.jsonl",
    )
    parser.add_argument("--output-dir", default="experiments/stage3_prompt_sql_generation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prediction-top-ks",
        default="20,30",
        help="Comma-separated prediction top-k settings to build, e.g. 20,30,50,80.",
    )
    parser.add_argument(
        "--prediction-methods",
        default="lexical,rgcn,rgta",
        help="Comma-separated prediction methods to build: lexical,rgcn,rgta.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(Path(args.dev_labels), limit=args.limit)
    dev_tables = load_dev_tables(Path(args.dev_tables))
    lexical = load_prediction_topk(Path(args.lexical_predictions), limit=args.limit)
    rgcn = load_prediction_topk(Path(args.rgcn_predictions), limit=args.limit)
    rgta = load_prediction_topk(Path(args.rgta_predictions), limit=args.limit)

    if not (len(records) == len(lexical) == len(rgcn) == len(rgta)):
        raise ValueError(
            f"Input length mismatch: labels={len(records)}, lexical={len(lexical)}, "
            f"rgcn={len(rgcn)}, rgta={len(rgta)}"
        )

    prediction_top_ks = [int(value.strip()) for value in args.prediction_top_ks.split(",") if value.strip()]
    prediction_methods = {
        value.strip()
        for value in args.prediction_methods.split(",")
        if value.strip()
    }

    settings = {
        "full_schema": lambda idx, rec: selected_full_schema_ids(rec),
        "oracle_schema": lambda idx, rec: selected_oracle_ids(rec),
    }
    for top_k in prediction_top_ks:
        if "lexical" in prediction_methods:
            settings[f"lexical_top{top_k}"] = (
                lambda idx, rec, top_k=top_k: selected_prediction_ids(lexical[idx], top_k)
            )
        if "rgcn" in prediction_methods:
            settings[f"rgcn_top{top_k}"] = (
                lambda idx, rec, top_k=top_k: selected_prediction_ids(rgcn[idx], top_k)
            )
        if "rgta" in prediction_methods:
            settings[f"rgta_top{top_k}"] = (
                lambda idx, rec, top_k=top_k: selected_prediction_ids(rgta[idx], top_k)
            )

    statistics = {}
    for setting, fn in settings.items():
        prompts, stats = build_setting_records(records, setting, fn, dev_tables)
        write_jsonl(output_dir / f"prompts_{setting}_dev.jsonl", prompts)
        statistics[setting] = stats

    with (output_dir / "prompt_statistics.json").open("w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)

    # Human-readable sample across key settings.
    sample_settings = [
        setting
        for setting in ["full_schema", "lexical_top30", "rgcn_top30", "rgta_top30", "rgta_top50", "rgta_top80", "oracle_schema"]
        if setting in settings
    ]
    with (output_dir / "prompt_examples.md").open("w", encoding="utf-8") as f:
        for setting in sample_settings:
            path = output_dir / f"prompts_{setting}_dev.jsonl"
            first = read_jsonl(path, limit=1)[0]
            f.write(f"# {setting}\n\n")
            f.write(f"DB: `{first['db_id']}`\n\n")
            f.write("```text\n")
            f.write(first["prompt"])
            f.write("\n```\n\n")

    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
