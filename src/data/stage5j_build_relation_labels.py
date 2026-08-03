import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


RELATION_TYPES = [
    "OUTPUT_TARGET",
    "ENTITY_NAME",
    "METRIC_TARGET",
    "PREDICATE_COLUMN",
    "VALUE_ANCHOR",
    "TEMPORAL_FILTER",
    "ORDER_KEY",
    "GROUP_KEY",
    "JOIN_BRIDGE",
    "FORMULA_COMPONENT",
]


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


def load_schema_cards(path: Path | None):
    if path is None:
        return {}
    cards = defaultdict(dict)
    for card in read_jsonl(path):
        db_id = card.get("db_id")
        item_id = card.get("schema_item_id")
        if db_id is None or item_id is None:
            continue
        cards[db_id][int(item_id)] = card
    return cards


def normalize_list(value):
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def item_value_type(item, card):
    if card and card.get("value_type"):
        return card["value_type"]
    if item.get("type") == "table":
        return "table"
    dtype = str(item.get("data_type") or "").lower()
    name = str(item.get("name") or "").lower()
    if dtype in {"integer", "real", "number", "float", "double", "decimal"}:
        return "numeric_metric"
    if dtype in {"date", "time"} or any(token in name for token in ["date", "year", "time"]):
        return "temporal"
    if any(token in name for token in ["id", "code", "key", "cds"]):
        return "identifier"
    if any(token in name for token in ["type", "status", "county", "state", "category", "funding"]):
        return "categorical"
    if any(token in name for token in ["name", "title", "school", "phone", "email", "website", "street", "city"]):
        return "entity_text"
    return "text"


def card_relations(card):
    return set(normalize_list(card.get("relation_types")) if card else [])


def add_relation(labels, relation, item_id):
    if relation in labels:
        labels[relation].add(int(item_id))


def relation_labels_for_item(item, card, clause):
    value_type = item_value_type(item, card)
    relations = card_relations(card)
    item_type = item.get("type")
    output = set()

    if clause == "join":
        output.add("JOIN_BRIDGE")
        return output

    if clause == "select":
        if item_type == "table":
            return output
        output.add("OUTPUT_TARGET")
        if value_type in {"entity_text", "text", "identifier", "categorical"} or "ENTITY_NAME" in relations:
            output.add("ENTITY_NAME")
        if value_type == "numeric_metric" or "METRIC_TARGET" in relations:
            output.add("METRIC_TARGET")
        if "FORMULA_COMPONENT" in relations:
            output.add("FORMULA_COMPONENT")
        return output

    if clause == "where":
        if item_type == "table":
            return output
        output.add("PREDICATE_COLUMN")
        if value_type in {"categorical", "entity_text", "text"} or "VALUE_ANCHOR" in relations:
            output.add("VALUE_ANCHOR")
        if value_type == "numeric_metric" or "METRIC_TARGET" in relations:
            output.add("METRIC_TARGET")
        if value_type == "temporal" or "TEMPORAL_FILTER" in relations:
            output.add("TEMPORAL_FILTER")
        if "FORMULA_COMPONENT" in relations:
            output.add("FORMULA_COMPONENT")
        return output

    if clause == "order_by":
        if item_type != "table":
            output.add("ORDER_KEY")
            if value_type == "numeric_metric" or "METRIC_TARGET" in relations:
                output.add("METRIC_TARGET")
            if value_type == "temporal":
                output.add("TEMPORAL_FILTER")
        return output

    if clause == "group_by":
        if item_type != "table":
            output.add("GROUP_KEY")
            if value_type in {"entity_text", "text", "categorical"}:
                output.add("ENTITY_NAME")
        return output

    if clause == "having":
        if item_type != "table":
            output.add("PREDICATE_COLUMN")
            if value_type == "numeric_metric":
                output.add("METRIC_TARGET")
        return output

    return output


def transform_record(record, cards_by_id):
    by_id = {item["id"]: item for item in record.get("schema_items", [])}
    relation_labels = {relation: set() for relation in RELATION_TYPES}
    relation_sources = defaultdict(lambda: defaultdict(list))

    for clause, item_ids in record.get("clause_labels", {}).items():
        for item_id in item_ids:
            item = by_id.get(int(item_id))
            if not item:
                continue
            card = cards_by_id.get(int(item_id), {})
            for relation in relation_labels_for_item(item, card, clause):
                add_relation(relation_labels, relation, item_id)
                relation_sources[relation][str(item_id)].append(clause)

    # Formula components often appear in evidence-derived SQL labels but may be distributed across SELECT/ORDER.
    # Keep this conservative: only add FORMULA_COMPONENT if the item is already a gold SQL label.
    whole = set(record.get("whole_sql_labels", []))
    for item_id in whole:
        item = by_id.get(int(item_id))
        if not item or item.get("type") == "table":
            continue
        card = cards_by_id.get(int(item_id), {})
        if "FORMULA_COMPONENT" in card_relations(card):
            relation_labels["FORMULA_COMPONENT"].add(int(item_id))
            relation_sources["FORMULA_COMPONENT"][str(item_id)].append("schema_card")

    relation_label_names = {}
    for relation, ids in relation_labels.items():
        relation_label_names[relation] = [by_id[item_id]["name"] for item_id in sorted(ids) if item_id in by_id]

    union = set()
    for ids in relation_labels.values():
        union.update(ids)

    return {
        "split": record.get("split"),
        "db_id": record.get("db_id"),
        "question_id": record.get("question_id"),
        "difficulty": record.get("difficulty"),
        "question": record.get("question"),
        "evidence": record.get("evidence"),
        "sql": record.get("sql"),
        "schema_items": record.get("schema_items", []),
        "whole_sql_labels": sorted(whole),
        "whole_sql_label_names": record.get("whole_sql_label_names", []),
        "clause_labels": record.get("clause_labels", {}),
        "clause_label_names": record.get("clause_label_names", {}),
        "relation_labels": {relation: sorted(ids) for relation, ids in relation_labels.items()},
        "relation_label_names": relation_label_names,
        "relation_sources": {rel: dict(items) for rel, items in relation_sources.items()},
        "union_relation_labels": sorted(union),
        "union_relation_label_names": [by_id[item_id]["name"] for item_id in sorted(union) if item_id in by_id],
        "unassigned_whole_labels": sorted(whole - union),
        "unassigned_whole_label_names": [by_id[item_id]["name"] for item_id in sorted(whole - union) if item_id in by_id],
    }


def summarize(records):
    stats = {
        "sample_count": len(records),
        "avg_whole_labels": 0,
        "avg_union_relation_labels": 0,
        "whole_label_coverage_by_relation_union": 0,
        "samples_with_unassigned_whole_labels": 0,
        "relation_stats": {},
        "top_unassigned_whole_labels": [],
    }
    if not records:
        return stats

    relation_counts = {relation: Counter() for relation in RELATION_TYPES}
    unassigned = Counter()
    total_whole = 0
    total_union = 0
    total_hit = 0
    for record in records:
        whole = set(record.get("whole_sql_labels", []))
        union = set(record.get("union_relation_labels", []))
        total_whole += len(whole)
        total_union += len(union)
        total_hit += len(whole & union)
        if record.get("unassigned_whole_labels"):
            stats["samples_with_unassigned_whole_labels"] += 1
            unassigned.update(record.get("unassigned_whole_label_names", []))
        by_id = {item["id"]: item for item in record.get("schema_items", [])}
        for relation in RELATION_TYPES:
            ids = set(record.get("relation_labels", {}).get(relation, []))
            relation_counts[relation]["label_total"] += len(ids)
            relation_counts[relation]["table_total"] += sum(1 for item_id in ids if by_id.get(item_id, {}).get("type") == "table")
            relation_counts[relation]["column_total"] += sum(1 for item_id in ids if by_id.get(item_id, {}).get("type") == "column")
            if ids:
                relation_counts[relation]["non_empty_samples"] += 1

    stats["avg_whole_labels"] = total_whole / len(records)
    stats["avg_union_relation_labels"] = total_union / len(records)
    stats["whole_label_coverage_by_relation_union"] = total_hit / total_whole if total_whole else 0
    for relation in RELATION_TYPES:
        counter = relation_counts[relation]
        stats["relation_stats"][relation] = {
            "non_empty_samples": counter["non_empty_samples"],
            "non_empty_rate": counter["non_empty_samples"] / len(records),
            "avg_labels": counter["label_total"] / len(records),
            "avg_table_labels": counter["table_total"] / len(records),
            "avg_column_labels": counter["column_total"] / len(records),
        }
    stats["top_unassigned_whole_labels"] = unassigned.most_common(30)
    return stats


def build_split(input_file, output_file, schema_cards, split, limit=None):
    records = read_jsonl(input_file, limit=limit)
    outputs = [transform_record(record, schema_cards.get(record.get("db_id"), {})) for record in records]
    write_jsonl(output_file, outputs)
    return summarize(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-clause-labels", default="experiments/stage5g_clause_labels/train_clause_labels.jsonl")
    parser.add_argument("--dev-clause-labels", default="experiments/stage5g_clause_labels/dev_clause_labels.jsonl")
    parser.add_argument("--train-schema-semantic-cards", default="experiments/stage5i_schema_semantic_cards_heuristic_v2/train_schema_semantic_cards.jsonl")
    parser.add_argument("--dev-schema-semantic-cards", default="experiments/stage5i_schema_semantic_cards_heuristic_v2/dev_schema_semantic_cards.jsonl")
    parser.add_argument("--output-dir", default="experiments/stage5j_relation_labels")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_cards = load_schema_cards(Path(args.train_schema_semantic_cards))
    dev_cards = load_schema_cards(Path(args.dev_schema_semantic_cards))
    train_summary = build_split(
        Path(args.train_clause_labels),
        output_dir / "train_relation_labels.jsonl",
        train_cards,
        "train",
        limit=args.limit,
    )
    dev_summary = build_split(
        Path(args.dev_clause_labels),
        output_dir / "dev_relation_labels.jsonl",
        dev_cards,
        "dev",
        limit=args.limit,
    )
    summary = {
        "config": vars(args),
        "relation_types": RELATION_TYPES,
        "train": train_summary,
        "dev": dev_summary,
        "generalization_boundary": (
            "Relation labels are derived from gold SQL/clause labels only for training/evaluation. "
            "Test-time inputs will use question, evidence, schema semantic graph, and relation tokens."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
