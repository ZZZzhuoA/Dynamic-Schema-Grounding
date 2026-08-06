import argparse
import json
import re
from pathlib import Path


ALLOWED_SQL_ROLES = {"SELECT", "WHERE", "JOIN", "ORDER_BY", "GROUP_BY", "HAVING", "FORMULA", "VALUE"}
ALLOWED_RELATION_TYPES = {
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
}
ALLOWED_OPERATIONS = {
    "PROJECT",
    "FILTER",
    "JOIN",
    "GROUP",
    "ORDER",
    "AGGREGATE",
    "COMPUTE",
    "LIMIT",
    "WINDOW",
    "COMPARE",
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


def normalize_space(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def truncate_text(text, max_chars):
    text = normalize_space(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def dedup_clean(values, limit=None, allowed=None, lower_key=False):
    seen = set()
    out = []
    for value in as_list(values):
        text = normalize_space(value)
        if not text:
            continue
        if allowed is not None and text not in allowed:
            continue
        key = text.lower() if lower_key else text
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def schema_name_aliases(card):
    aliases = []
    for key in ["qualified_name", "raw_name", "semantic_name"]:
        if card.get(key):
            aliases.append(card[key])
    aliases.extend(as_list(card.get("aliases")))
    table = card.get("table_name_original")
    column = card.get("column_name_original")
    if column:
        aliases.append(column)
    if table and column:
        aliases.append(f"{table}.{column}")
    return aliases


def compact_schema_card(card, max_aliases, max_description_chars):
    aliases = dedup_clean(schema_name_aliases(card), limit=max_aliases, lower_key=True)
    roles = dedup_clean(card.get("likely_sql_roles"), limit=8, allowed=ALLOWED_SQL_ROLES)
    relations = dedup_clean(card.get("relation_types"), limit=8, allowed=ALLOWED_RELATION_TYPES)
    semantic_name = truncate_text(card.get("semantic_name") or card.get("qualified_name") or card.get("raw_name"), 96)
    description = truncate_text(card.get("description"), max_description_chars)
    value_type = normalize_space(card.get("value_type"))
    compact_parts = [
        normalize_space(card.get("qualified_name")),
        semantic_name,
    ]
    if aliases:
        compact_parts.append("aliases: " + ", ".join(aliases[:max_aliases]))
    if value_type:
        compact_parts.append(value_type)
    if roles:
        compact_parts.append("roles: " + ", ".join(roles))
    if relations:
        compact_parts.append("relations: " + ", ".join(relations))
    if description:
        compact_parts.append("hint: " + description)
    compact = dict(card)
    compact.update(
        {
            "semantic_name": semantic_name,
            "description": description,
            "aliases": aliases,
            "likely_sql_roles": roles,
            "relation_types": relations,
            "compact_aliases": aliases,
            "compact_text": " | ".join(part for part in compact_parts if part),
            "compact_source": "stage8f_compact_llm_cards",
        }
    )
    return compact


def mention_text(mention):
    phrase = normalize_space(mention.get("phrase"))
    semantic = normalize_space(mention.get("semantic_hint"))
    operation = normalize_space(mention.get("operation"))
    relation = normalize_space(mention.get("relation_type"))
    value = normalize_space(mention.get("value_hint"))
    left = phrase
    if semantic and semantic.lower() != phrase.lower():
        left = f"{left} -> {semantic}" if left else semantic
    tags = []
    if operation in ALLOWED_OPERATIONS:
        tags.append(operation)
    if relation in ALLOWED_RELATION_TYPES:
        tags.append(relation)
    if value and value.lower() not in {"none", "null"}:
        tags.append(f"value={value}")
    if tags:
        left = f"{left} ({', '.join(tags)})" if left else ", ".join(tags)
    return left


def compact_question_card(card, max_mentions, max_intent_chars):
    normalized_question = truncate_text(card.get("normalized_question") or card.get("question"), 180)
    intent = truncate_text(card.get("intent") or normalized_question, max_intent_chars)
    mentions = []
    clean_mentions = []
    for mention in as_list(card.get("mentions")):
        if not isinstance(mention, dict):
            continue
        text = mention_text(mention)
        if text:
            mentions.append(text)
        clean = dict(mention)
        if clean.get("operation") not in ALLOWED_OPERATIONS:
            clean["operation"] = None
        if clean.get("relation_type") not in ALLOWED_RELATION_TYPES:
            clean["relation_type"] = None
        clean_mentions.append(clean)
        if len(mentions) >= max_mentions:
            break
    operations = dedup_clean(card.get("operation_hints"), limit=10, allowed=ALLOWED_OPERATIONS)
    values = dedup_clean(card.get("value_hints"), limit=12, lower_key=True)
    formulas = dedup_clean(card.get("formula_hints"), limit=8, lower_key=True)
    ordering = dedup_clean(card.get("ordering_hints"), limit=8, lower_key=True)
    compact_parts = [normalized_question, intent]
    if mentions:
        compact_parts.append("mentions: " + "; ".join(mentions[:max_mentions]))
    if operations:
        compact_parts.append("operations: " + ", ".join(operations))
    if values:
        compact_parts.append("values: " + ", ".join(values))
    if formulas:
        compact_parts.append("formulas: " + "; ".join(formulas))
    if ordering:
        compact_parts.append("ordering: " + ", ".join(ordering))
    compact = dict(card)
    compact.update(
        {
            "normalized_question": normalized_question,
            "intent": intent,
            "mentions": clean_mentions[:max_mentions],
            "operation_hints": operations,
            "value_hints": values,
            "formula_hints": formulas,
            "ordering_hints": ordering,
            "compact_mentions": mentions[:max_mentions],
            "compact_text": " | ".join(part for part in compact_parts if part),
            "compact_source": "stage8f_compact_llm_cards",
        }
    )
    return compact


def compact_file(input_path, output_path, card_type, args):
    records = read_jsonl(Path(input_path), args.limit)
    if card_type == "schema":
        compacted = [
            compact_schema_card(record, args.max_schema_aliases, args.max_schema_description_chars)
            for record in records
        ]
    elif card_type == "question":
        compacted = [
            compact_question_card(record, args.max_question_mentions, args.max_question_intent_chars)
            for record in records
        ]
    else:
        raise ValueError(f"Unsupported card_type: {card_type}")
    write_jsonl(Path(output_path), compacted)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "card_type": card_type,
        "count": len(compacted),
        "avg_compact_text_chars": (
            sum(len(record.get("compact_text") or "") for record in compacted) / len(compacted)
            if compacted
            else 0
        ),
        "empty_compact_text_count": sum(1 for record in compacted if not record.get("compact_text")),
    }


def add_if_present(tasks, path, output_dir, output_name, card_type):
    if path:
        tasks.append((Path(path), output_dir / output_name, card_type))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-schema-cards", default=None)
    parser.add_argument("--train-question-cards", default=None)
    parser.add_argument("--dev-schema-cards", default=None)
    parser.add_argument("--dev-question-cards", default=None)
    parser.add_argument("--output-dir", default="experiments/stage8f_compact_llm_cards")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-schema-aliases", type=int, default=6)
    parser.add_argument("--max-schema-description-chars", type=int, default=96)
    parser.add_argument("--max-question-mentions", type=int, default=10)
    parser.add_argument("--max-question-intent-chars", type=int, default=160)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    add_if_present(tasks, args.train_schema_cards, output_dir, "train_schema_semantic_cards.jsonl", "schema")
    add_if_present(tasks, args.train_question_cards, output_dir, "train_question_cards.jsonl", "question")
    add_if_present(tasks, args.dev_schema_cards, output_dir, "dev_schema_semantic_cards.jsonl", "schema")
    add_if_present(tasks, args.dev_question_cards, output_dir, "dev_question_cards.jsonl", "question")
    if not tasks:
        raise ValueError("No input card files provided.")
    reports = []
    for input_path, output_path, card_type in tasks:
        if not input_path.exists():
            raise FileNotFoundError(f"Input card file not found: {input_path}")
        reports.append(compact_file(input_path, output_path, card_type, args))
    summary = {
        "config": vars(args),
        "files": reports,
        "note": (
            "Compacted cards keep test-time available semantics but shorten free-form LLM text into "
            "structured alignment fields. Gold SQL/schema labels are not used."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
