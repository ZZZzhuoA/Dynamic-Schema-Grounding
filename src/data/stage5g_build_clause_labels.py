import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.stage1_extract_bird_labels import (  # noqa: E402
    SQL_KEYWORDS,
    add_column_label_for_name,
    add_delimited_identifier_labels,
    extract_table_aliases,
    exact_column_labels_in_scope,
    fk_labels,
    normalize_name,
    normalize_sql_for_match,
    query_scopes,
)


CLAUSES = ["select", "join", "where", "group_by", "having", "order_by"]
CLAUSE_KEYWORDS = {
    "select": ("select",),
    "from": ("from",),
    "join": ("join",),
    "where": ("where",),
    "group_by": ("group", "by"),
    "having": ("having",),
    "order_by": ("order", "by"),
    "limit": ("limit",),
    "union": ("union",),
    "intersect": ("intersect",),
    "except": ("except",),
}
CLAUSE_BOUNDARIES = {
    "select",
    "from",
    "join",
    "where",
    "group_by",
    "having",
    "order_by",
    "limit",
    "union",
    "intersect",
    "except",
}


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


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_schema_index(schema_items, foreign_keys=None):
    table_item_ids = {}
    column_item_ids = {}
    column_norm_to_originals = defaultdict(list)

    for item in schema_items:
        if item["type"] == "table":
            table_item_ids[item["normalized_name"]] = item["id"]
        elif item["type"] == "column":
            norm_table = item["normalized_table"]
            norm_column = item["normalized_column"]
            column_item_ids[(norm_table, norm_column)] = item["id"]
            column_norm_to_originals[norm_column].append((norm_table, item["id"]))

    return {
        "schema_items": schema_items,
        "items_by_id": {item["id"]: item for item in schema_items},
        "table_item_ids": table_item_ids,
        "column_item_ids": column_item_ids,
        "column_norm_to_originals": dict(column_norm_to_originals),
        "foreign_keys": foreign_keys or [],
    }


def recover_foreign_keys_from_edges(schema_items, edges):
    by_id = {item["id"]: item for item in schema_items}
    fk_edges = []
    seen = set()
    for edge in edges or []:
        if edge.get("type") not in {"foreign_key_forward", "foreign_key_backward"}:
            continue
        left_id = int(edge["src"])
        right_id = int(edge["dst"])
        key = tuple(sorted((left_id, right_id)))
        if key in seen:
            continue
        seen.add(key)
        left = by_id.get(left_id)
        right = by_id.get(right_id)
        if not left or not right:
            continue
        if left.get("type") != "column" or right.get("type") != "column":
            continue
        fk_edges.append(
            {
                "left_id": left_id,
                "right_id": right_id,
                "left": left["normalized_name"],
                "right": right["normalized_name"],
                "left_table": left["normalized_table"],
                "right_table": right["normalized_table"],
            }
        )
    return fk_edges


def read_word(sql, index):
    if index >= len(sql) or not (sql[index].isalnum() or sql[index] == "_"):
        return None, index
    end = index + 1
    while end < len(sql) and (sql[end].isalnum() or sql[end] == "_"):
        end += 1
    return sql[index:end].lower(), end


def next_nonspace(sql, index):
    while index < len(sql) and sql[index].isspace():
        index += 1
    return index


def scan_clause_events(sql, include_nested=False):
    events = []
    quote = None
    depth = 0
    index = 0
    while index < len(sql):
        char = sql[index]
        if quote:
            if char == quote:
                if quote == "'" and index + 1 < len(sql) and sql[index + 1] == "'":
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
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            index += 1
            continue
        if include_nested or depth == 0:
            word, end = read_word(sql, index)
            if word:
                clause = None
                clause_end = end
                if word == "group":
                    after = next_nonspace(sql, end)
                    word2, end2 = read_word(sql, after)
                    if word2 == "by":
                        clause = "group_by"
                        clause_end = end2
                elif word == "order":
                    after = next_nonspace(sql, end)
                    word2, end2 = read_word(sql, after)
                    if word2 == "by":
                        clause = "order_by"
                        clause_end = end2
                elif word in {"select", "from", "join", "where", "having", "limit", "union", "intersect", "except"}:
                    clause = word
                if clause:
                    events.append({"clause": clause, "start": index, "content_start": clause_end, "depth": depth})
                index = end
                continue
        index += 1
    return events


def top_level_clause_spans(sql):
    events = scan_clause_events(sql, include_nested=False)
    spans = defaultdict(list)
    for index, event in enumerate(events):
        clause = event["clause"]
        if clause not in CLAUSE_BOUNDARIES:
            continue
        next_start = len(sql)
        for later in events[index + 1 :]:
            if later["clause"] in CLAUSE_BOUNDARIES:
                next_start = later["start"]
                break
        target = "join" if clause in {"from", "join"} else clause
        if target not in CLAUSES:
            continue
        spans[target].append(sql[event["content_start"] : next_start].strip())
    return {clause: spans.get(clause, []) for clause in CLAUSES}


def table_mentions_in_text(text, schema):
    normalized = normalize_sql_for_match(text)
    labels = set()
    used_tables = set()
    for norm_table, item_id in schema["table_item_ids"].items():
        if re.search(rf"(?<![a-z0-9_]){re.escape(norm_table)}(?![a-z0-9_])", normalized):
            labels.add(item_id)
            used_tables.add(norm_table)
    return labels, used_tables


def candidate_column_entries(schema, candidate_ids=None):
    if candidate_ids is None:
        for (norm_table, norm_column), item_id in schema["column_item_ids"].items():
            yield norm_table, norm_column, item_id
        return

    for item_id in candidate_ids:
        item = schema["items_by_id"].get(item_id)
        if not item or item.get("type") != "column":
            continue
        yield item["normalized_table"], item["normalized_column"], item_id


def add_candidate_column_label_for_name(labels, schema, norm_column, candidate_tables, candidate_ids=None):
    occurrences = schema["column_norm_to_originals"].get(norm_column, [])
    if candidate_ids is not None:
        candidate_ids = set(candidate_ids)
        occurrences = [(table, item_id) for table, item_id in occurrences if item_id in candidate_ids]
    if not occurrences:
        return
    filtered = [(table, item_id) for table, item_id in occurrences if table in candidate_tables]
    if len(filtered) == 1:
        labels.add(filtered[0][1])
    elif len(filtered) > 1:
        for _, item_id in filtered:
            labels.add(item_id)


def column_labels_in_text(text, schema, full_used_tables, full_aliases, candidate_ids=None):
    labels = set()
    candidate_tables = full_used_tables or set(schema["table_item_ids"])
    normalized = normalize_sql_for_match(text)

    add_delimited_identifier_labels(text, schema, labels, full_used_tables, full_aliases)
    if candidate_ids is not None:
        labels.intersection_update(candidate_ids)

    candidate_norm_columns = set()
    for norm_table, norm_column, item_id in candidate_column_entries(schema, candidate_ids):
        candidate_norm_columns.add(norm_column)
        table_candidates = [norm_table]
        table_candidates.extend(alias for alias, target in full_aliases.items() if target == norm_table)
        for table_candidate in table_candidates:
            patterns = [
                rf"(?<![a-z0-9_]){re.escape(table_candidate)}\s+{re.escape(norm_column)}(?![a-z0-9_])",
                rf"(?<![a-z0-9_]){re.escape(table_candidate)}_{re.escape(norm_column)}(?![a-z0-9_])",
            ]
            if any(re.search(pattern, normalized) for pattern in patterns):
                labels.add(item_id)
                break

    for norm_column in candidate_norm_columns:
        if not re.search(rf"(?<![a-z0-9_]){re.escape(norm_column)}(?![a-z0-9_])", normalized):
            continue
        add_candidate_column_label_for_name(labels, schema, norm_column, candidate_tables, candidate_ids=candidate_ids)

    return labels


def clause_labels_from_sql(sql, schema, candidate_column_ids=None):
    labels_by_clause = {clause: set() for clause in CLAUSES}
    clause_spans = {clause: [] for clause in CLAUSES}
    all_used_tables = set()

    for scope in query_scopes(sql or "", schema):
        local_sql = scope["sql"]
        local_tables = scope["used_tables"]
        aliases = scope["aliases"]
        all_used_tables.update(local_tables)
        spans = top_level_clause_spans(local_sql)
        for clause in CLAUSES:
            clause_spans[clause].extend(spans.get(clause, []))
            for span in spans.get(clause, []):
                labels_by_clause[clause].update(
                    exact_column_labels_in_scope(
                        span,
                        schema,
                        local_tables,
                        aliases,
                        candidate_ids=candidate_column_ids,
                    )
                )
                if clause == "join":
                    table_labels, _ = table_mentions_in_text(span, schema)
                    labels_by_clause[clause].update(table_labels)
        for norm_table in local_tables:
            table_id = schema["table_item_ids"].get(norm_table)
            if table_id is not None:
                labels_by_clause["join"].add(table_id)

    labels_by_clause["join"].update(fk_labels(all_used_tables, schema))

    return labels_by_clause, clause_spans


def item_names(schema_items, ids):
    by_id = {item["id"]: item for item in schema_items}
    return [by_id[item_id]["name"] for item_id in sorted(ids) if item_id in by_id]


def table_column_counts(schema_items, ids):
    by_id = {item["id"]: item for item in schema_items}
    tables = 0
    columns = 0
    for item_id in ids:
        item = by_id.get(item_id)
        if not item:
            continue
        if item["type"] == "table":
            tables += 1
        elif item["type"] == "column":
            columns += 1
    return tables, columns


def transform_record(record):
    schema_items = record.get("schema_items", [])
    foreign_keys = recover_foreign_keys_from_edges(schema_items, record.get("schema_edges", []))
    schema = build_schema_index(schema_items, foreign_keys)
    sql_parse_labels = set()
    foreign_key_labels = set()
    label_sources = record.get("label_sources", {})
    name_to_id = {item["name"]: item["id"] for item in schema_items}
    for name in label_sources.get("sql_parse", []):
        if name in name_to_id:
            sql_parse_labels.add(name_to_id[name])
    for name in label_sources.get("foreign_key", []):
        if name in name_to_id:
            foreign_key_labels.add(name_to_id[name])

    sql_fk_labels = sql_parse_labels | foreign_key_labels
    candidate_column_ids = {
        item_id
        for item_id in sql_fk_labels
        if schema["items_by_id"].get(item_id, {}).get("type") == "column"
    }
    labels_by_clause, clause_spans = clause_labels_from_sql(
        record.get("sql") or "",
        schema,
        candidate_column_ids=candidate_column_ids,
    )
    labels_by_clause["join"].update(foreign_key_labels)

    union_clause_labels = set()
    for ids in labels_by_clause.values():
        union_clause_labels.update(ids)

    whole_labels = set(record.get("whole_sql_labels", []))
    unassigned_whole = whole_labels - union_clause_labels
    unassigned_sql_fk = sql_fk_labels - union_clause_labels

    return {
        "split": record.get("split"),
        "db_id": record.get("db_id"),
        "question_id": record.get("question_id"),
        "difficulty": record.get("difficulty"),
        "question": record.get("question"),
        "evidence": record.get("evidence"),
        "sql": record.get("sql"),
        "schema_items": schema_items,
        "whole_sql_labels": sorted(whole_labels),
        "whole_sql_label_names": item_names(schema_items, whole_labels),
        "clause_labels": {clause: sorted(ids) for clause, ids in labels_by_clause.items()},
        "clause_label_names": {clause: item_names(schema_items, ids) for clause, ids in labels_by_clause.items()},
        "clause_spans": clause_spans,
        "union_clause_labels": sorted(union_clause_labels),
        "union_clause_label_names": item_names(schema_items, union_clause_labels),
        "sql_fk_labels": sorted(sql_fk_labels),
        "sql_fk_label_names": item_names(schema_items, sql_fk_labels),
        "unassigned_whole_labels": sorted(unassigned_whole),
        "unassigned_whole_label_names": item_names(schema_items, unassigned_whole),
        "unassigned_sql_fk_labels": sorted(unassigned_sql_fk),
        "unassigned_sql_fk_label_names": item_names(schema_items, unassigned_sql_fk),
        "metadata": {
            "used_tables_from_sql": record.get("used_tables_from_sql", []),
            "label_sources": label_sources,
        },
    }


def summarize(records):
    stats = {
        "sample_count": len(records),
        "non_empty_clause_label_samples": 0,
        "avg_whole_labels": 0,
        "avg_union_clause_labels": 0,
        "whole_label_coverage_by_clause_union": 0,
        "sql_fk_label_coverage_by_clause_union": 0,
        "samples_with_unassigned_whole_labels": 0,
        "samples_with_unassigned_sql_fk_labels": 0,
        "clause_stats": {},
        "top_unassigned_whole_labels": [],
        "top_unassigned_sql_fk_labels": [],
    }
    if not records:
        return stats

    total_whole = 0
    total_union = 0
    total_whole_hit = 0
    total_sql_fk = 0
    total_sql_fk_hit = 0
    unassigned_whole_counter = Counter()
    unassigned_sql_fk_counter = Counter()
    clause_counts = {clause: Counter() for clause in CLAUSES}

    for record in records:
        whole = set(record["whole_sql_labels"])
        union = set(record["union_clause_labels"])
        sql_fk = set(record.get("sql_fk_labels", []))
        unassigned_sql_fk = set(record["unassigned_sql_fk_labels"])

        total_whole += len(whole)
        total_union += len(union)
        total_whole_hit += len(whole & union)
        total_sql_fk += len(sql_fk)
        total_sql_fk_hit += len(sql_fk & union)
        if union:
            stats["non_empty_clause_label_samples"] += 1
        if record["unassigned_whole_labels"]:
            stats["samples_with_unassigned_whole_labels"] += 1
            unassigned_whole_counter.update(record["unassigned_whole_label_names"])
        if record["unassigned_sql_fk_labels"]:
            stats["samples_with_unassigned_sql_fk_labels"] += 1
            unassigned_sql_fk_counter.update(record["unassigned_sql_fk_label_names"])

        for clause in CLAUSES:
            ids = set(record["clause_labels"].get(clause, []))
            table_count, column_count = table_column_counts(record["schema_items"], ids)
            clause_counts[clause]["label_total"] += len(ids)
            clause_counts[clause]["table_total"] += table_count
            clause_counts[clause]["column_total"] += column_count
            if ids:
                clause_counts[clause]["non_empty_samples"] += 1

    stats["avg_whole_labels"] = total_whole / len(records)
    stats["avg_union_clause_labels"] = total_union / len(records)
    stats["whole_label_coverage_by_clause_union"] = total_whole_hit / total_whole if total_whole else 0
    stats["sql_fk_label_coverage_by_clause_union"] = total_sql_fk_hit / total_sql_fk if total_sql_fk else 0
    stats["non_empty_clause_label_rate"] = stats["non_empty_clause_label_samples"] / len(records)

    for clause in CLAUSES:
        counter = clause_counts[clause]
        stats["clause_stats"][clause] = {
            "non_empty_samples": counter["non_empty_samples"],
            "non_empty_rate": counter["non_empty_samples"] / len(records),
            "avg_labels": counter["label_total"] / len(records),
            "avg_table_labels": counter["table_total"] / len(records),
            "avg_column_labels": counter["column_total"] / len(records),
        }

    stats["top_unassigned_whole_labels"] = unassigned_whole_counter.most_common(30)
    stats["top_unassigned_sql_fk_labels"] = unassigned_sql_fk_counter.most_common(30)
    return stats


def build_split(input_file: Path, output_file: Path, limit=None):
    input_records = read_jsonl(input_file, limit=limit)
    output_records = [transform_record(record) for record in input_records]
    write_jsonl(output_file, output_records)
    return summarize(output_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-labels", default="experiments/stage1_label_extraction_v2/bird_train_grounding_labels.jsonl")
    parser.add_argument("--dev-labels", default="experiments/stage1_label_extraction_v2/bird_dev_grounding_labels.jsonl")
    parser.add_argument("--output-dir", default="experiments/stage5g_clause_labels")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_summary = build_split(Path(args.train_labels), output_dir / "train_clause_labels.jsonl", limit=args.limit)
    dev_summary = build_split(Path(args.dev_labels), output_dir / "dev_clause_labels.jsonl", limit=args.limit)
    summary = {
        "config": {
            "train_labels": args.train_labels,
            "dev_labels": args.dev_labels,
            "output_dir": str(output_dir),
            "limit": args.limit,
            "clauses": CLAUSES,
        },
        "train": train_summary,
        "dev": dev_summary,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
