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


def split_direct_subqueries(sql):
    """Mask direct SELECT subqueries and return their SQL bodies.

    The masked parent keeps character positions stable, which lets downstream
    clause extraction operate on one query block at a time.  Nested subqueries
    are handled recursively by :func:`query_scopes`.
    """
    text = str(sql or "")
    masked = list(text)
    subqueries = []
    quote = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote:
                if quote == "'" and index + 1 < len(text) and text[index + 1] == "'":
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            continue
        if char == "[":
            quote = "]"
            index += 1
            continue
        if char != "(":
            index += 1
            continue

        depth = 1
        inner_quote = None
        end = index + 1
        while end < len(text) and depth:
            current = text[end]
            if inner_quote:
                if current == inner_quote:
                    if inner_quote == "'" and end + 1 < len(text) and text[end + 1] == "'":
                        end += 2
                        continue
                    inner_quote = None
                end += 1
                continue
            if current in {"'", '"', "`"}:
                inner_quote = current
            elif current == "[":
                inner_quote = "]"
            elif current == "(":
                depth += 1
            elif current == ")":
                depth -= 1
            end += 1
        if depth:
            index += 1
            continue
        body = text[index + 1 : end - 1]
        if re.match(r"(?is)^\s*(?:select|with)\b", body):
            subqueries.append(body)
            for position in range(index, end):
                masked[position] = " "
            index = end
        else:
            index += 1
    return "".join(masked), subqueries


def query_scopes(sql, schema, inherited_aliases=None):
    """Yield alias-aware SQL query blocks without nested-scope contamination."""
    masked, subqueries = split_direct_subqueries(sql)
    used_tables, local_aliases, _ = extract_table_aliases(masked, schema)
    aliases = dict(inherited_aliases or {})
    aliases.update(local_aliases)
    yield {
        "sql": masked,
        "used_tables": set(used_tables),
        "aliases": aliases,
        "local_aliases": local_aliases,
    }
    for subquery in subqueries:
        yield from query_scopes(subquery, schema, inherited_aliases=aliases)


def _qualified_column_references(text):
    pattern = re.compile(
        r"(?is)(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
        r"(?:`([^`]+)`|\[([^\]]+)\]|\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))"
    )
    references = []
    spans = []
    for match in pattern.finditer(text or ""):
        column = next(value for value in match.groups()[1:] if value is not None)
        references.append((normalize_name(match.group(1)), normalize_name(column)))
        spans.append(match.span())
    return references, spans


def _mask_spans_and_literals(text, spans):
    chars = list(text or "")
    for start, end in spans:
        for index in range(start, end):
            chars[index] = " "
    result = "".join(chars)
    # Values and delimited identifiers are processed separately and must not
    # leak short tokens (e.g. Charter inside `Charter School (Y/N)`).
    result = re.sub(r"'(?:''|[^'])*'", " ", result)
    result = re.sub(r"`[^`]*`|\[[^\]]*\]|\"[^\"]*\"", " ", result)
    return result


def exact_column_labels_in_scope(
    text,
    schema,
    used_tables,
    aliases,
    candidate_ids=None,
):
    """Resolve columns in one query scope without assigning ambiguous names.

    Qualified references are mapped through their alias owner.  Unqualified
    references are accepted only when exactly one table in the local scope owns
    that column.  This deliberately prefers missing a malformed/ambiguous SQL
    reference over creating false positive gold labels.
    """
    allowed = set(candidate_ids) if candidate_ids is not None else None
    labels = set()
    references, qualified_spans = _qualified_column_references(text)
    for prefix, norm_column in references:
        norm_table = aliases.get(prefix, prefix)
        item_id = schema["column_item_ids"].get((norm_table, norm_column))
        if item_id is not None and (allowed is None or item_id in allowed):
            labels.add(item_id)

    candidate_tables = set(used_tables)
    if not candidate_tables:
        candidate_tables.update(aliases.values())

    # Exact standalone quoted identifiers retain spaces and punctuation.
    qualified_positions = set()
    for start, end in qualified_spans:
        qualified_positions.update(range(start, end))
    for match in re.finditer(r"`([^`]+)`|\[([^\]]+)\]|\"([^\"]+)\"", text or ""):
        if any(position in qualified_positions for position in range(*match.span())):
            continue
        raw_name = next(value for value in match.groups() if value is not None)
        norm_column = normalize_name(raw_name)
        occurrences = [
            (table, item_id)
            for table, item_id in schema["column_norm_to_originals"].get(norm_column, [])
            if table in candidate_tables and (allowed is None or item_id in allowed)
        ]
        if len(occurrences) == 1:
            labels.add(occurrences[0][1])

    bare_text = normalize_sql_for_match(_mask_spans_and_literals(text, qualified_spans))
    for norm_column, occurrences in schema["column_norm_to_originals"].items():
        if not re.search(
            rf"(?<![a-z0-9_]){re.escape(norm_column)}(?![a-z0-9_])", bare_text
        ):
            continue
        local = [
            (table, item_id)
            for table, item_id in occurrences
            if table in candidate_tables and (allowed is None or item_id in allowed)
        ]
        if len(local) == 1:
            labels.add(local[0][1])
    return labels


def scope_aware_sql_labels(sql, schema):
    labels = set()
    used_tables = set()
    for scope in query_scopes(sql, schema):
        local_tables = scope["used_tables"]
        used_tables.update(local_tables)
        for norm_table in local_tables:
            table_id = schema["table_item_ids"].get(norm_table)
            if table_id is not None:
                labels.add(table_id)
        labels.update(
            exact_column_labels_in_scope(
                scope["sql"], schema, local_tables, scope["aliases"]
            )
        )
    return labels, used_tables


def sql_parse_labels(sql, schema):
    return scope_aware_sql_labels(sql, schema)


def fk_labels(used_tables, schema, sql_labels=None, mode="explicit_sql"):
    """Return FK endpoint labels under an explicit, auditable policy.

    ``all_used_tables`` preserves the original Stage 1 behavior: every FK between
    any two SQL-used tables is labeled, even if that edge is not present in the
    SQL.  It is retained only for reproducing legacy experiments.

    ``explicit_sql`` keeps an FK pair only when both endpoints were actually
    recovered from the SQL text.  The endpoints are already members of
    ``sql_labels``; the separate source records that they form a schema FK without
    expanding the gold target.  This avoids adding unrelated parallel FKs such as
    every ``Match.home_player_*`` edge merely because Match and Player co-occur.
    """
    if mode not in {"explicit_sql", "all_used_tables", "none"}:
        raise ValueError(f"Unsupported fk label mode: {mode}")
    if mode == "none":
        return set()

    sql_labels = set(sql_labels or [])
    labels = set()
    for fk in schema["foreign_keys"]:
        tables_used = fk["left_table"] in used_tables and fk["right_table"] in used_tables
        endpoints_explicit = fk["left_id"] in sql_labels and fk["right_id"] in sql_labels
        if tables_used and (mode == "all_used_tables" or endpoints_explicit):
            labels.add(fk["left_id"])
            labels.add(fk["right_id"])
    return labels


def item_names(schema_items, ids):
    return [schema_items[item_id]["name"] for item_id in sorted(ids)]


def process_records(records, schemas, split, fk_label_mode="explicit_sql"):
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

        join_labels = fk_labels(
            used_tables,
            schema,
            sql_labels=sql_labels,
            mode=fk_label_mode,
        )
        labels_by_source["foreign_key"].update(join_labels)

        # Keep the legacy expansion out of the target while exposing exactly how
        # many labels the former policy would have added.  This makes old/new
        # complete-coverage results directly auditable.
        legacy_join_labels = fk_labels(
            used_tables,
            schema,
            sql_labels=sql_labels,
            mode="all_used_tables",
        )
        legacy_fk_only = legacy_join_labels - sql_labels - hit_labels
        if legacy_fk_only:
            stats["legacy_fk_extra_samples"] += 1
            stats["legacy_fk_extra_labels"] += len(legacy_fk_only)

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
            "label_policy": {
                "fk_label_mode": fk_label_mode,
                "legacy_fk_extra_ids": sorted(legacy_fk_only),
                "legacy_fk_extra_names": item_names(schema_items, legacy_fk_only),
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
    parser.add_argument(
        "--fk-label-mode",
        choices=["explicit_sql", "all_used_tables", "none"],
        default="explicit_sql",
        help=(
            "How FK endpoints enter gold schema labels. explicit_sql is the corrected default; "
            "all_used_tables reproduces the legacy over-closure policy."
        ),
    )
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
            "fk_label_mode": args.fk_label_mode,
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

        output, failed, statistics = process_records(
            records,
            schemas,
            split,
            fk_label_mode=args.fk_label_mode,
        )
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
