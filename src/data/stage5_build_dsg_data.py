import argparse
import csv
import io
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


REL_SELF_LOOP = "self_loop"
REL_TABLE_TO_COLUMN = "table_to_column"
REL_COLUMN_TO_TABLE = "column_to_table"
REL_FK_FORWARD = "foreign_key_forward"
REL_FK_BACKWARD = "foreign_key_backward"
REL_SAME_TABLE = "same_table_column"


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


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_table_entries(path: Path):
    return {entry["db_id"]: entry for entry in read_json(path)}


def normalize_metadata_name(value):
    text = unicodedata.normalize("NFKC", str(value or "")).replace("\ufeff", "")
    return text.strip().casefold()


def clean_metadata_value(value):
    return unicodedata.normalize("NFKC", str(value or "")).replace("\ufeff", "").strip()


def read_database_description_csv(path: Path):
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return csv.DictReader(io.StringIO(payload.decode(encoding))), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "utf-8/cp1252", payload, 0, len(payload), f"Could not decode {path}"
    )


def flatten_primary_keys(values):
    flattened = set()
    for value in values or []:
        if isinstance(value, list):
            flattened.update(int(item) for item in value)
        else:
            flattened.add(int(value))
    return flattened


def authoritative_schema_metadata(table_entry):
    tables = table_entry.get("table_names_original", [])
    columns = table_entry.get("column_names_original", [])
    primary_keys = flatten_primary_keys(table_entry.get("primary_keys", []))
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    fk_endpoints = set()
    for left, right in table_entry.get("foreign_keys", []):
        if left >= len(columns) or right >= len(columns):
            raise ValueError(f"Foreign key index out of range for db={table_entry.get('db_id')}")
        left_table, left_column = columns[left]
        right_table, right_column = columns[right]
        if left_table < 0 or right_table < 0:
            continue
        left_name = f"{tables[left_table]}.{left_column}"
        right_name = f"{tables[right_table]}.{right_column}"
        outgoing[left].append(right_name)
        incoming[right].append(left_name)
        fk_endpoints.update((left, right))

    result = {}
    for column_index, (table_index, column_name) in enumerate(columns):
        if table_index < 0 or column_name == "*":
            continue
        key = (
            normalize_metadata_name(tables[table_index]),
            normalize_metadata_name(column_name),
        )
        if key in result:
            raise ValueError(
                f"Ambiguous normalized schema column for db={table_entry.get('db_id')}: {key}"
            )
        result[key] = {
            "is_primary_key": column_index in primary_keys,
            "is_foreign_key_endpoint": column_index in fk_endpoints,
            "foreign_key_outgoing_targets": sorted(set(outgoing[column_index]))[:8],
            "foreign_key_incoming_sources": sorted(set(incoming[column_index]))[:8],
        }
    return result


def load_database_description(database_root: Path, table_entries):
    metadata_by_db = {}
    summary = {
        "database_root": str(database_root),
        "database_count": 0,
        "matched_table_count": 0,
        "unmatched_tables": [],
        "matched_column_count": 0,
        "unmatched_columns": [],
        "extra_csv_columns": [],
        "csv_encoding_counts": Counter(),
    }
    for db_id, table_entry in sorted(table_entries.items()):
        description_dir = database_root / db_id / "database_description"
        table_map = {}
        for table_name in table_entry.get("table_names_original", []):
            key = normalize_metadata_name(table_name)
            if key in table_map:
                raise ValueError(f"Ambiguous normalized table name for db={db_id}: {key!r}")
            table_map[key] = table_name

        csv_map = {}
        if description_dir.exists():
            for csv_path in sorted(description_dir.glob("*.csv"), key=lambda path: path.name.casefold()):
                key = normalize_metadata_name(csv_path.stem)
                if key in csv_map:
                    raise ValueError(f"Ambiguous database-description CSV for db={db_id}: {key!r}")
                csv_map[key] = csv_path
        db_metadata = {}
        summary["database_count"] += 1
        schema_columns = authoritative_schema_metadata(table_entry)
        for key, structural in schema_columns.items():
            db_metadata[key] = {
                **structural,
                "official_column_name": "",
                "official_column_description": "",
                "official_data_format": "",
                "official_value_description": "",
                "metadata_source": "bird_database_description",
            }
        for table_key, table_name in table_map.items():
            csv_path = csv_map.get(table_key)
            expected_columns = {
                column_key
                for current_table, column_key in schema_columns
                if current_table == table_key
            }
            if csv_path is None:
                summary["unmatched_tables"].append({"db_id": db_id, "table": table_name})
                for column_key in sorted(expected_columns):
                    summary["unmatched_columns"].append(
                        {"db_id": db_id, "table": table_name, "column": column_key}
                    )
                continue
            summary["matched_table_count"] += 1
            reader, encoding = read_database_description_csv(csv_path)
            summary["csv_encoding_counts"][encoding] += 1
            try:
                normalized_headers = {
                    normalize_metadata_name(header): header for header in (reader.fieldnames or [])
                }
                original_header = normalized_headers.get("original_column_name")
                if original_header is None:
                    raise ValueError(f"Missing original_column_name header: {csv_path}")
                seen_columns = set()
                for row_number, row in enumerate(reader, start=2):
                    column_original = clean_metadata_value(row.get(original_header))
                    column_key = normalize_metadata_name(column_original)
                    if not column_key:
                        continue
                    if column_key in seen_columns:
                        raise ValueError(
                            f"Ambiguous duplicate column in {csv_path}:{row_number}: {column_original!r}"
                        )
                    seen_columns.add(column_key)
                    key = (table_key, column_key)
                    if key not in schema_columns:
                        summary["extra_csv_columns"].append(
                            {"db_id": db_id, "table": table_name, "column": column_original}
                        )
                        continue

                    def field(name):
                        header = normalized_headers.get(name)
                        return clean_metadata_value(row.get(header)) if header else ""

                    db_metadata[key] = {
                        **schema_columns[key],
                        "official_column_name": field("column_name"),
                        "official_column_description": field("column_description"),
                        "official_data_format": field("data_format"),
                        "official_value_description": field("value_description"),
                        "metadata_source": "bird_database_description",
                    }
                    summary["matched_column_count"] += 1
                for column_key in sorted(expected_columns - seen_columns):
                    summary["unmatched_columns"].append(
                        {"db_id": db_id, "table": table_name, "column": column_key}
                    )
            finally:
                del reader
        metadata_by_db[db_id] = db_metadata
    summary["unmatched_table_count"] = len(summary["unmatched_tables"])
    summary["unmatched_column_count"] = len(summary["unmatched_columns"])
    summary["extra_csv_column_count"] = len(summary["extra_csv_columns"])
    summary["csv_encoding_counts"] = dict(sorted(summary["csv_encoding_counts"].items()))
    return metadata_by_db, summary


def load_schema_semantic_cards(path: Path | None):
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(
            f"Schema semantic card file not found: {path}. "
            "Stage 8F writes files like dev_schema_semantic_cards.jsonl."
        )
    cards_by_db = defaultdict(dict)
    for card in read_jsonl(path):
        db_id = card.get("db_id")
        item_id = card.get("schema_item_id")
        if db_id is None or item_id is None:
            continue
        cards_by_db[db_id][int(item_id)] = card
    return cards_by_db


def load_question_cards(path: Path | None):
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(
            f"Question card file not found: {path}. "
            "Stage 8F writes files like dev_question_cards.jsonl."
        )
    cards = {}
    for index, card in enumerate(read_jsonl(path)):
        db_id = card.get("db_id")
        question_id = card.get("question_id")
        record_index = card.get("record_index", index)
        if db_id is not None and question_id is not None:
            cards[("qid", db_id, question_id)] = card
        cards[("idx", record_index)] = card
    return cards


def question_card_key(record, record_index):
    if record.get("db_id") is not None and record.get("question_id") is not None:
        return ("qid", record.get("db_id"), record.get("question_id"))
    return ("idx", record_index)


def question_semantic_text(card):
    if not card:
        return None
    if card.get("compact_text"):
        return str(card.get("compact_text"))
    parts = []
    for key in ["normalized_question", "intent"]:
        value = card.get(key)
        if value:
            parts.append(str(value))
    mentions = []
    for mention in card.get("mentions", [])[:12]:
        if not isinstance(mention, dict):
            continue
        fields = [
            mention.get("phrase"),
            mention.get("semantic_hint"),
            mention.get("operation"),
            mention.get("relation_type"),
            mention.get("value_hint"),
        ]
        text = " / ".join(str(x) for x in fields if x)
        if text:
            mentions.append(text)
    if mentions:
        parts.append("mentions: " + "; ".join(mentions))
    for key, label in [
        ("operation_hints", "operations"),
        ("value_hints", "values"),
        ("formula_hints", "formulas"),
        ("ordering_hints", "ordering"),
    ]:
        values = card.get(key) or []
        if values:
            parts.append(label + ": " + "; ".join(str(x) for x in values[:12]))
    return " | ".join(parts)


def semantic_text_from_card(card):
    if not card:
        return None
    if card.get("compact_text"):
        return str(card.get("compact_text"))
    parts = []
    for key in ["qualified_name", "semantic_name", "description", "value_type"]:
        value = card.get(key)
        if value:
            parts.append(str(value))
    aliases = card.get("aliases") or []
    if aliases:
        parts.append("aliases: " + "; ".join(str(x) for x in aliases[:8]))
    roles = card.get("likely_sql_roles") or []
    if isinstance(roles, str):
        roles = [roles]
    if roles:
        parts.append("roles: " + "; ".join(str(x) for x in roles))
    relations = card.get("relation_types") or []
    if isinstance(relations, str):
        relations = [relations]
    if relations:
        parts.append("relations: " + "; ".join(str(x) for x in relations))
    return " | ".join(parts)


def attach_semantic_card(node, card):
    if not card:
        return node
    node = dict(node)
    node.update(
        {
            "semantic_name": card.get("semantic_name"),
            "semantic_description": card.get("description"),
            "semantic_aliases": card.get("aliases", []),
            "value_type": card.get("value_type"),
            "likely_sql_roles": card.get("likely_sql_roles", []),
            "relation_types": card.get("relation_types", []),
            "semantic_source": card.get("source"),
            "semantic_text": semantic_text_from_card(card),
        }
    )
    return node


def schema_nodes(schema_items, semantic_cards=None, official_metadata=None):
    semantic_cards = semantic_cards or {}
    official_metadata = official_metadata or {}
    nodes = []
    for item in schema_items:
        node = {
            "id": item["id"],
            "type": item["type"],
            "name": item["name"],
            "normalized_name": item.get("normalized_name"),
        }
        if item["type"] == "column":
            node.update(
                {
                    "table": item.get("table"),
                    "column": item.get("column"),
                    "normalized_table": item.get("normalized_table"),
                    "normalized_column": item.get("normalized_column"),
                    "data_type": item.get("data_type"),
                }
            )
            metadata_key = (
                normalize_metadata_name(item.get("table")),
                normalize_metadata_name(item.get("column")),
            )
            if metadata_key in official_metadata:
                node.update(official_metadata[metadata_key])
        nodes.append(attach_semantic_card(node, semantic_cards.get(item["id"])))
    return nodes


def schema_indexes(schema_items):
    table_to_id = {}
    column_pair_to_id = {}
    columns_by_table = defaultdict(list)
    for item in schema_items:
        if item["type"] == "table":
            table_to_id[item["name"]] = item["id"]
        elif item["type"] == "column":
            pair = (item.get("table"), item.get("column"))
            column_pair_to_id[pair] = item["id"]
            columns_by_table[item.get("table")].append(item["id"])
    return table_to_id, column_pair_to_id, columns_by_table


def add_edge(edges, seen, src, dst, edge_type):
    key = (int(src), int(dst), edge_type)
    if key in seen:
        return
    seen.add(key)
    edges.append({"src": int(src), "dst": int(dst), "type": edge_type})


def schema_edges(schema_items, table_entry, include_same_table_edges=False):
    table_to_id, column_pair_to_id, columns_by_table = schema_indexes(schema_items)
    edges = []
    seen = set()

    for item in schema_items:
        add_edge(edges, seen, item["id"], item["id"], REL_SELF_LOOP)
        if item["type"] != "column":
            continue
        table = item.get("table")
        table_id = table_to_id.get(table)
        if table_id is not None:
            add_edge(edges, seen, table_id, item["id"], REL_TABLE_TO_COLUMN)
            add_edge(edges, seen, item["id"], table_id, REL_COLUMN_TO_TABLE)

    if include_same_table_edges:
        for _, column_ids in columns_by_table.items():
            for src in column_ids:
                for dst in column_ids:
                    if src != dst:
                        add_edge(edges, seen, src, dst, REL_SAME_TABLE)

    if table_entry:
        tables = table_entry.get("table_names_original", [])
        columns = table_entry.get("column_names_original", [])
        for left_idx, right_idx in table_entry.get("foreign_keys", []):
            if left_idx >= len(columns) or right_idx >= len(columns):
                continue
            left_table_idx, left_col = columns[left_idx]
            right_table_idx, right_col = columns[right_idx]
            if left_table_idx < 0 or right_table_idx < 0:
                continue
            left_table = tables[left_table_idx]
            right_table = tables[right_table_idx]
            left_id = column_pair_to_id.get((left_table, left_col))
            right_id = column_pair_to_id.get((right_table, right_col))
            if left_id is None or right_id is None:
                continue
            add_edge(edges, seen, left_id, right_id, REL_FK_FORWARD)
            add_edge(edges, seen, right_id, left_id, REL_FK_BACKWARD)

    return edges


def grounding_vector(record):
    node_count = len(record.get("schema_items", []))
    vector = [0] * node_count
    for item_id in record.get("whole_sql_labels", []):
        if 0 <= item_id < node_count:
            vector[item_id] = 1
    return vector


def table_column_label_splits(record):
    by_id = {item["id"]: item for item in record.get("schema_items", [])}
    table_ids = []
    column_ids = []
    for item_id in record.get("whole_sql_labels", []):
        item = by_id.get(item_id)
        if not item:
            continue
        if item["type"] == "table":
            table_ids.append(item_id)
        elif item["type"] == "column":
            column_ids.append(item_id)
    return table_ids, column_ids


def make_example(
    record,
    table_entry,
    record_index,
    include_same_table_edges=False,
    semantic_cards_by_id=None,
    question_card=None,
    official_metadata=None,
):
    nodes = schema_nodes(record["schema_items"], semantic_cards_by_id, official_metadata)
    edges = schema_edges(record["schema_items"], table_entry, include_same_table_edges)
    label_vector = grounding_vector(record)
    table_label_ids, column_label_ids = table_column_label_splits(record)

    # Important generalization boundary:
    # inference_inputs must be usable at test time without gold SQL/schema labels.
    inference_inputs = {
        "db_id": record["db_id"],
        "question": record.get("question"),
        "evidence": record.get("evidence"),
        "schema_nodes": nodes,
        "schema_edges": edges,
    }
    if question_card:
        inference_inputs["question_card"] = question_card
        inference_inputs["question_semantic_text"] = question_semantic_text(question_card)
    training_targets = {
        "sql": record.get("sql"),
        "grounding_label_ids": record.get("whole_sql_labels", []),
        "grounding_label_names": record.get("label_names", []),
        "grounding_table_label_ids": table_label_ids,
        "grounding_column_label_ids": column_label_ids,
        "grounding_label_vector": label_vector,
        "label_sources": record.get("label_sources", {}),
    }
    metadata = {
        "split": record.get("split"),
        "question_id": record.get("question_id"),
        "record_index": record_index,
        "difficulty": record.get("difficulty"),
        "used_tables_from_sql": record.get("used_tables_from_sql", []),
        "hit_unmapped": record.get("hit_unmapped", []),
        "schema_semantic_cards_attached": bool(semantic_cards_by_id),
        "question_card_attached": bool(question_card),
        "official_metadata_attached": bool(official_metadata),
    }
    stable_id = record.get("question_id")
    if stable_id is None:
        stable_id = record_index
    return {
        "example_id": f"{record.get('split')}::{record['db_id']}::{stable_id}",
        "inference_inputs": inference_inputs,
        "training_targets": training_targets,
        "metadata": metadata,
    }


def build_split(
    records,
    table_entries,
    include_same_table_edges=False,
    semantic_cards_by_db=None,
    question_cards=None,
    official_metadata_by_db=None,
):
    semantic_cards_by_db = semantic_cards_by_db or {}
    examples = []
    for index, record in enumerate(records):
        table_entry = table_entries.get(record["db_id"])
        examples.append(
            make_example(
                record,
                table_entry,
                index,
                include_same_table_edges,
                semantic_cards_by_db.get(record["db_id"], {}),
                (question_cards or {}).get(question_card_key(record, index)) or (question_cards or {}).get(("idx", index)),
                (official_metadata_by_db or {}).get(record["db_id"]),
            )
        )
    return examples


def summarize_examples(examples):
    node_counts = []
    edge_counts = []
    label_counts = []
    table_label_counts = []
    column_label_counts = []
    relation_counter = Counter()
    db_ids = set()
    missing_question_ids = 0
    leakage_violations = []
    semantic_node_counts = []
    question_card_count = 0

    forbidden_inference_keys = {
        "sql",
        "gold_sql",
        "grounding_label_ids",
        "grounding_label_names",
        "grounding_label_vector",
        "whole_sql_labels",
        "label_names",
        "label_sources",
    }

    for index, example in enumerate(examples):
        inputs = example["inference_inputs"]
        targets = example["training_targets"]
        nodes = inputs.get("schema_nodes", [])
        edges = inputs.get("schema_edges", [])
        node_counts.append(len(nodes))
        semantic_node_counts.append(sum(1 for node in nodes if node.get("semantic_text")))
        if inputs.get("question_semantic_text"):
            question_card_count += 1
        edge_counts.append(len(edges))
        label_counts.append(len(targets.get("grounding_label_ids", [])))
        table_label_counts.append(len(targets.get("grounding_table_label_ids", [])))
        column_label_counts.append(len(targets.get("grounding_column_label_ids", [])))
        db_ids.add(inputs.get("db_id"))
        if example.get("metadata", {}).get("question_id") is None:
            missing_question_ids += 1
        for edge in edges:
            relation_counter[edge["type"]] += 1
        leaked = sorted(forbidden_inference_keys & set(inputs.keys()))
        if leaked:
            leakage_violations.append({"index": index, "keys": leaked})

    def avg(values):
        return sum(values) / len(values) if values else 0

    return {
        "count": len(examples),
        "unique_db_count": len(db_ids),
        "avg_node_count": avg(node_counts),
        "max_node_count": max(node_counts) if node_counts else 0,
        "avg_edge_count": avg(edge_counts),
        "max_edge_count": max(edge_counts) if edge_counts else 0,
        "avg_grounding_labels": avg(label_counts),
        "avg_table_labels": avg(table_label_counts),
        "avg_column_labels": avg(column_label_counts),
        "empty_label_count": sum(1 for count in label_counts if count == 0),
        "missing_question_id_count": missing_question_ids,
        "avg_semantic_node_count": avg(semantic_node_counts),
        "semantic_node_attachment_rate": (
            sum(semantic_node_counts) / sum(node_counts) if sum(node_counts) else 0
        ),
        "question_card_attachment_rate": question_card_count / len(examples) if examples else 0,
        "edge_type_counts": dict(sorted(relation_counter.items())),
        "inference_target_leakage_violation_count": len(leakage_violations),
        "inference_target_leakage_examples": leakage_violations[:5],
    }


def write_example_markdown(path: Path, examples_by_split):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for split, examples in examples_by_split.items():
            if not examples:
                continue
            example = examples[0]
            inputs = example["inference_inputs"]
            targets = example["training_targets"]
            f.write(f"# {split} example\n\n")
            f.write(f"Example id: `{example['example_id']}`\n\n")
            f.write("## Inference inputs\n\n")
            f.write(f"- DB: `{inputs['db_id']}`\n")
            f.write(f"- Question: {inputs.get('question')}\n")
            if inputs.get("evidence"):
                f.write(f"- Evidence: {inputs.get('evidence')}\n")
            if inputs.get("question_semantic_text"):
                f.write(f"- Question semantic text: {inputs.get('question_semantic_text')[:300]}\n")
            f.write(f"- Schema nodes: {len(inputs.get('schema_nodes', []))}\n")
            f.write(f"- Schema edges: {len(inputs.get('schema_edges', []))}\n\n")
            f.write("First schema nodes:\n\n")
            for node in inputs.get("schema_nodes", [])[:8]:
                semantic = node.get("semantic_text")
                if semantic:
                    f.write(f"- `{node['id']}` {node['type']} `{node['name']}` — {semantic[:180]}\n")
                else:
                    f.write(f"- `{node['id']}` {node['type']} `{node['name']}`\n")
            f.write("\nFirst schema edges:\n\n")
            for edge in inputs.get("schema_edges", [])[:12]:
                f.write(f"- {edge['src']} -> {edge['dst']} ({edge['type']})\n")
            f.write("\n## Training targets\n\n")
            f.write("These fields are for training supervision only and must not be used as test-time inputs.\n\n")
            f.write(f"- SQL: `{targets.get('sql')}`\n")
            f.write(f"- Grounding labels: {targets.get('grounding_label_names', [])[:20]}\n")
            f.write(f"- Label vector length: {len(targets.get('grounding_label_vector', []))}\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-labels",
        default="experiments/stage1_label_extraction/bird_train_grounding_labels.jsonl",
    )
    parser.add_argument(
        "--dev-labels",
        default="experiments/stage1_label_extraction/bird_dev_grounding_labels.jsonl",
    )
    parser.add_argument(
        "--train-tables",
        default="Data/BIRD/train_databases/train_databases/train_tables.json",
    )
    parser.add_argument("--dev-tables", default="Data/BIRD/dev_tables.json")
    parser.add_argument(
        "--train-schema-semantic-cards",
        "--train-schema-cards",
        dest="train_schema_semantic_cards",
        default=None,
    )
    parser.add_argument(
        "--dev-schema-semantic-cards",
        "--dev-schema-cards",
        dest="dev_schema_semantic_cards",
        default=None,
    )
    parser.add_argument(
        "--train-question-cards",
        "--train-question-semantic-cards",
        dest="train_question_cards",
        default=None,
    )
    parser.add_argument(
        "--dev-question-cards",
        "--dev-question-semantic-cards",
        dest="dev_question_cards",
        default=None,
    )
    parser.add_argument(
        "--train-database-root",
        default=None,
        help="Optional BIRD train_databases root containing <db>/database_description/*.csv.",
    )
    parser.add_argument(
        "--dev-database-root",
        default=None,
        help="Optional BIRD dev_databases root containing <db>/database_description/*.csv.",
    )
    parser.add_argument("--output-dir", default="experiments/stage5_dsg_data")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--include-same-table-edges", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_records = read_jsonl(Path(args.train_labels), args.train_limit)
    dev_records = read_jsonl(Path(args.dev_labels), args.dev_limit)
    train_tables = load_table_entries(Path(args.train_tables))
    dev_tables = load_table_entries(Path(args.dev_tables))
    train_semantic_cards = load_schema_semantic_cards(
        Path(args.train_schema_semantic_cards) if args.train_schema_semantic_cards else None
    )
    dev_semantic_cards = load_schema_semantic_cards(
        Path(args.dev_schema_semantic_cards) if args.dev_schema_semantic_cards else None
    )
    train_question_cards = load_question_cards(Path(args.train_question_cards) if args.train_question_cards else None)
    dev_question_cards = load_question_cards(Path(args.dev_question_cards) if args.dev_question_cards else None)
    train_official_metadata, train_metadata_summary = ({}, None)
    dev_official_metadata, dev_metadata_summary = ({}, None)
    if args.train_database_root:
        train_official_metadata, train_metadata_summary = load_database_description(
            Path(args.train_database_root), train_tables
        )
    if args.dev_database_root:
        dev_official_metadata, dev_metadata_summary = load_database_description(
            Path(args.dev_database_root), dev_tables
        )

    train_examples = build_split(
        train_records,
        train_tables,
        args.include_same_table_edges,
        train_semantic_cards,
        train_question_cards,
        train_official_metadata,
    )
    dev_examples = build_split(
        dev_records,
        dev_tables,
        args.include_same_table_edges,
        dev_semantic_cards,
        dev_question_cards,
        dev_official_metadata,
    )

    write_jsonl(output_dir / "train_examples.jsonl", train_examples)
    write_jsonl(output_dir / "dev_examples.jsonl", dev_examples)

    statistics = {
        "config": vars(args),
        "train": summarize_examples(train_examples),
        "dev": summarize_examples(dev_examples),
        "official_metadata": {
            "train": train_metadata_summary,
            "dev": dev_metadata_summary,
        },
        "generalization_note": (
            "Gold SQL and gold schema labels are stored only in training_targets. "
            "Test-time inference should consume inference_inputs only."
        ),
    }
    write_json(output_dir / "data_statistics.json", statistics)
    write_example_markdown(output_dir / "example.md", {"train": train_examples, "dev": dev_examples})

    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
