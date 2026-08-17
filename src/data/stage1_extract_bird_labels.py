import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SQL_KEYWORDS = {
    "select",
    "from",
    "where",
    "join",
    "inner",
    "left",
    "right",
    "full",
    "outer",
    "on",
    "group",
    "by",
    "order",
    "having",
    "limit",
    "union",
    "intersect",
    "except",
    "as",
    "and",
    "or",
    "desc",
    "asc",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_name(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = text.replace("`", "").replace('"', "").replace("[", "").replace("]", "")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_sql_for_match(sql):
    sql = sql.lower()
    sql = sql.replace("`", " ").replace('"', " ").replace("[", " ").replace("]", " ")
    sql = re.sub(r"'[^']*'", " STRING_LITERAL ", sql)
    sql = re.sub(r"[^a-z0-9_]+", " ", sql)
    sql = re.sub(r"\s+", " ", sql)
    return f" {sql.strip()} "


def schema_from_tables_entry(entry):
    tables_original = entry["table_names_original"]
    columns_original = entry["column_names_original"]
    column_types = entry.get("column_types", [])

    schema_items = []
    table_item_ids = {}
    column_item_ids = {}
    table_norm_to_original = {}
    column_norm_to_originals = defaultdict(list)

    for table_index, table_name in enumerate(tables_original):
        norm_table = normalize_name(table_name)
        item_id = len(schema_items)
        table_item_ids[norm_table] = item_id
        table_norm_to_original[norm_table] = table_name
        schema_items.append(
            {
                "id": item_id,
                "type": "table",
                "name": table_name,
                "normalized_name": norm_table,
            }
        )

    for column_index, pair in enumerate(columns_original):
        table_index, column_name = pair
        if table_index < 0 or column_name == "*":
            continue
        table_name = tables_original[table_index]
        norm_table = normalize_name(table_name)
        norm_column = normalize_name(column_name)
        full_norm = f"{norm_table}.{norm_column}"
        full_name = f"{table_name}.{column_name}"
        item_id = len(schema_items)
        column_item_ids[(norm_table, norm_column)] = item_id
        column_norm_to_originals[norm_column].append((norm_table, item_id))
        schema_items.append(
            {
                "id": item_id,
                "type": "column",
                "name": full_name,
                "table": table_name,
                "column": column_name,
                "normalized_name": full_norm,
                "normalized_table": norm_table,
                "normalized_column": norm_column,
                # BIRD keeps the wildcard entry in both column_names_original
                # and column_types, so their indices are aligned.  Subtracting
                # one shifts every real column onto the preceding column type.
                "data_type": column_types[column_index]
                if column_index < len(column_types)
                else None,
            }
        )

    fk_pairs = []
    for left_index, right_index in entry.get("foreign_keys", []):
        left_pair = columns_original[left_index]
        right_pair = columns_original[right_index]
        if left_pair[0] < 0 or right_pair[0] < 0:
            continue
        left_table = normalize_name(tables_original[left_pair[0]])
        right_table = normalize_name(tables_original[right_pair[0]])
        left_column = normalize_name(left_pair[1])
        right_column = normalize_name(right_pair[1])
        left_item = column_item_ids.get((left_table, left_column))
        right_item = column_item_ids.get((right_table, right_column))
        if left_item is not None and right_item is not None:
            fk_pairs.append(
                {
                    "left_id": left_item,
                    "right_id": right_item,
                    "left": f"{left_table}.{left_column}",
                    "right": f"{right_table}.{right_column}",
                    "left_table": left_table,
                    "right_table": right_table,
                }
            )

    return {
        "schema_items": schema_items,
        "table_item_ids": table_item_ids,
        "column_item_ids": column_item_ids,
        "table_norm_to_original": table_norm_to_original,
        "column_norm_to_originals": dict(column_norm_to_originals),
        "foreign_keys": fk_pairs,
    }


def load_table_schemas(path: Path):
    entries = read_json(path)
    return {entry["db_id"]: schema_from_tables_entry(entry) for entry in entries}


def normalize_train_record(record):
    return {
        "db_id": record.get("db_name"),
        "question": record.get("question"),
        "sql": record.get("answer") or record.get("rep_answer"),
        "evidence": record.get("evidence"),
        "hit_info": record.get("hit_info") or {},
    }


def normalize_dev_record(record):
    return {
        "db_id": record.get("db_id"),
        "question": record.get("question"),
        "sql": record.get("SQL"),
        "evidence": record.get("evidence"),
        "hit_info": {},
        "difficulty": record.get("difficulty"),
        "question_id": record.get("question_id"),
    }


def add_label(labels_by_source, source, item_id):
    if item_id is not None:
        labels_by_source[source].add(item_id)


def hit_info_labels(hit_info, schema):
    labels = set()
    unmapped = []
    for table_name, columns in hit_info.items():
        norm_table = normalize_name(table_name)
        table_id = schema["table_item_ids"].get(norm_table)
        if table_id is not None:
            labels.add(table_id)
        else:
            unmapped.append(f"table:{table_name}")

        for column_name in columns or []:
            norm_column = normalize_name(column_name)
            col_id = schema["column_item_ids"].get((norm_table, norm_column))
            if col_id is not None:
                labels.add(col_id)
            else:
                unmapped.append(f"column:{table_name}.{column_name}")
    return labels, unmapped


def extract_table_aliases(sql, schema):
    normalized_sql = normalize_sql_for_match(sql)
    used_tables = set()
    aliases = {}

    tokens = normalized_sql.strip().split()
    for i, token in enumerate(tokens):
        if token in {"from", "join"} and i + 1 < len(tokens):
            table_token = tokens[i + 1]
            norm_table = normalize_name(table_token)
            if norm_table in schema["table_item_ids"]:
                used_tables.add(norm_table)
                if i + 2 < len(tokens):
                    possible_alias = tokens[i + 2]
                    if possible_alias == "as" and i + 3 < len(tokens):
                        possible_alias = tokens[i + 3]
                    if possible_alias not in SQL_KEYWORDS:
                        aliases[normalize_name(possible_alias)] = norm_table

    # Some BIRD SQL uses comma joins or quoted identifiers. Add any explicit table
    # mention as a fallback, but only for exact normalized token matches.
    for norm_table in schema["table_item_ids"]:
        if re.search(rf"(?<![a-z0-9_]){re.escape(norm_table)}(?![a-z0-9_])", normalized_sql):
            used_tables.add(norm_table)

    for norm_table in used_tables:
        aliases[norm_table] = norm_table
    return used_tables, aliases, normalized_sql


def add_column_label_for_name(labels, schema, norm_column, candidate_tables):
    occurrences = schema["column_norm_to_originals"].get(norm_column, [])
    if not occurrences:
        return
    filtered = [(table, item_id) for table, item_id in occurrences if table in candidate_tables]
    if len(filtered) == 1:
        labels.add(filtered[0][1])
    elif len(filtered) > 1:
        for _, item_id in filtered:
            labels.add(item_id)


def add_delimited_identifier_labels(sql, schema, labels, used_tables, aliases):
    """Recover SQL-used columns quoted as `multi word column` or [multi word column].

    The earlier normalized-token matcher works for simple identifiers such as
    T1.AvgScrMath, but it loses multi-word quoted identifiers because
    `County Name` becomes separate tokens. This function parses those delimited
    identifiers before SQL normalization and maps them back to schema columns.
    """
    candidate_tables = used_tables or set(schema["table_item_ids"].keys())

    qualified_patterns = [
        r"(?i)([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*`([^`]+)`",
        r"(?i)([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*\[([^\]]+)\]",
    ]
    for pattern in qualified_patterns:
        for prefix, column_name in re.findall(pattern, sql or ""):
            norm_prefix = normalize_name(prefix)
            norm_table = aliases.get(norm_prefix, norm_prefix)
            norm_column = normalize_name(column_name)
            item_id = schema["column_item_ids"].get((norm_table, norm_column))
            if item_id is not None:
                labels.add(item_id)

    standalone_identifiers = []
    standalone_identifiers.extend(re.findall(r"`([^`]+)`", sql or ""))
    standalone_identifiers.extend(re.findall(r"\[([^\]]+)\]", sql or ""))
    for identifier in standalone_identifiers:
        norm_identifier = normalize_name(identifier)
        if norm_identifier in schema["table_item_ids"]:
            labels.add(schema["table_item_ids"][norm_identifier])
            continue
        add_column_label_for_name(labels, schema, norm_identifier, candidate_tables)


def sql_parse_labels(sql, schema):
    labels = set()
    used_tables, aliases, normalized_sql = extract_table_aliases(sql, schema)

    for norm_table in used_tables:
        labels.add(schema["table_item_ids"][norm_table])

    add_delimited_identifier_labels(sql, schema, labels, used_tables, aliases)

    # Fully qualified or alias-qualified column references.
    for (norm_table, norm_column), item_id in schema["column_item_ids"].items():
        table_candidates = [norm_table]
        table_candidates.extend(alias for alias, target in aliases.items() if target == norm_table)
        for table_candidate in table_candidates:
            patterns = [
                rf"(?<![a-z0-9_]){re.escape(table_candidate)}\s+{re.escape(norm_column)}(?![a-z0-9_])",
                rf"(?<![a-z0-9_]){re.escape(table_candidate)}_{re.escape(norm_column)}(?![a-z0-9_])",
            ]
            if any(re.search(pattern, normalized_sql) for pattern in patterns):
                labels.add(item_id)
                break

    # Unqualified column references. Restrict to used tables when possible.
    candidate_tables = used_tables or set(schema["table_item_ids"].keys())
    for norm_column, occurrences in schema["column_norm_to_originals"].items():
        if not re.search(rf"(?<![a-z0-9_]){re.escape(norm_column)}(?![a-z0-9_])", normalized_sql):
            continue
        # Ambiguous columns are still SQL-used signals, but we mark all
        # candidates among used tables. Later stages can refine this.
        add_column_label_for_name(labels, schema, norm_column, candidate_tables)

    return labels, used_tables


def fk_labels(used_tables, schema):
    labels = set()
    for fk in schema["foreign_keys"]:
        if fk["left_table"] in used_tables and fk["right_table"] in used_tables:
            labels.add(fk["left_id"])
            labels.add(fk["right_id"])
    return labels


def item_names(schema_items, ids):
    return [schema_items[item_id]["name"] for item_id in sorted(ids)]


def process_records(records, schemas, split):
    output = []
    failed = []
    stats = Counter()
    source_counts = Counter()

    for index, record in enumerate(records):
        db_id = record["db_id"]
        schema = schemas.get(db_id)
        if schema is None:
            failed.append({"index": index, "db_id": db_id, "reason": "missing_schema"})
            stats["missing_schema"] += 1
            continue

        labels_by_source = defaultdict(set)

        hit_labels, hit_unmapped = hit_info_labels(record.get("hit_info") or {}, schema)
        labels_by_source["hit_info"].update(hit_labels)

        sql_labels, used_tables = sql_parse_labels(record.get("sql") or "", schema)
        labels_by_source["sql_parse"].update(sql_labels)

        join_labels = fk_labels(used_tables, schema)
        labels_by_source["foreign_key"].update(join_labels)

        merged = set()
        for source, ids in labels_by_source.items():
            merged.update(ids)
            source_counts[source] += len(ids)

        if not merged:
            failed.append(
                {
                    "index": index,
                    "db_id": db_id,
                    "reason": "empty_labels",
                    "sql": record.get("sql"),
                    "hit_info": record.get("hit_info"),
                    "hit_unmapped": hit_unmapped,
                }
            )
            stats["empty_labels"] += 1

        if hit_unmapped:
            stats["hit_unmapped_samples"] += 1

        schema_items = schema["schema_items"]
        output_record = {
            "split": split,
            "db_id": db_id,
            "question": record.get("question"),
            "sql": record.get("sql"),
            "evidence": record.get("evidence"),
            "schema_items": schema_items,
            "whole_sql_labels": sorted(merged),
            "label_names": item_names(schema_items, merged),
            "label_sources": {
                source: item_names(schema_items, ids)
                for source, ids in sorted(labels_by_source.items())
            },
            "used_tables_from_sql": sorted(used_tables),
            "hit_info": record.get("hit_info") or {},
            "hit_unmapped": hit_unmapped,
        }
        if "difficulty" in record:
            output_record["difficulty"] = record["difficulty"]
        if "question_id" in record:
            output_record["question_id"] = record["question_id"]
        output.append(output_record)

    processed = len(output)
    non_empty = sum(1 for item in output if item["whole_sql_labels"])
    total_labels = sum(len(item["whole_sql_labels"]) for item in output)
    statistics = {
        "split": split,
        "processed_count": processed,
        "non_empty_label_count": non_empty,
        "non_empty_label_rate": non_empty / processed if processed else 0,
        "avg_labels_per_sample": total_labels / processed if processed else 0,
        "failed_count": len(failed),
        "stats": dict(stats),
        "source_label_counts": dict(source_counts),
    }
    return output, failed, statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird-dir", default="Data/BIRD")
    parser.add_argument(
        "--train-question-answer",
        default=None,
        help=(
            "Optional train_question_answer JSON override. Use the Stage 0 correction merge output "
            "without overwriting the original BIRD file."
        ),
    )
    parser.add_argument("--output-dir", default="experiments/stage1_label_extraction")
    parser.add_argument("--splits", default="train,dev")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    bird_dir = Path(args.bird_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_schemas = load_table_schemas(bird_dir / "train_databases" / "train_databases" / "train_tables.json")
    dev_schemas = load_table_schemas(bird_dir / "dev_tables.json")
    all_stats = {
        "config": {
            "bird_dir": str(bird_dir),
            "train_question_answer": args.train_question_answer,
            "output_dir": str(output_dir),
            "splits": args.splits,
            "limit": args.limit,
        },
        "schema_db_count": {
            "train": len(train_schemas),
            "dev": len(dev_schemas),
        }
    }

    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    for split in splits:
        if split == "train":
            train_question_answer_path = (
                Path(args.train_question_answer)
                if args.train_question_answer
                else bird_dir / "bird-schema" / "train_question_answer.json"
            )
            raw_records = read_json(train_question_answer_path)
            records = [normalize_train_record(item) for item in raw_records]
            schemas = train_schemas
            all_stats["config"]["resolved_train_question_answer"] = str(train_question_answer_path)
        elif split == "dev":
            raw_records = read_json(bird_dir / "dev.json")
            records = [normalize_dev_record(item) for item in raw_records]
            schemas = dev_schemas
        else:
            raise ValueError(f"Unsupported split: {split}")

        if args.limit is not None:
            records = records[: args.limit]

        output, failed, statistics = process_records(records, schemas, split)
        write_jsonl(output_dir / f"bird_{split}_grounding_labels.jsonl", output)
        write_jsonl(output_dir / f"bird_{split}_failed_label_cases.jsonl", failed)
        all_stats[split] = statistics

    stats_path = output_dir / "bird_label_statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(all_stats, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
