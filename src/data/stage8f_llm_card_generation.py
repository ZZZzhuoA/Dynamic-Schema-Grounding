import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


ALLOWED_VALUE_TYPES = [
    "table",
    "identifier",
    "numeric_metric",
    "temporal",
    "categorical",
    "entity_text",
    "boolean_flag",
    "text",
]

ALLOWED_SQL_ROLES = ["SELECT", "WHERE", "JOIN", "ORDER_BY", "GROUP_BY", "HAVING", "FORMULA", "VALUE"]

ALLOWED_RELATION_TYPES = [
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

ALLOWED_OPERATIONS = [
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
]


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


class OpenAICompatibleClient:
    def __init__(self, base_url, api_key, model, timeout):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, messages, temperature=0.0, top_p=1.0, max_tokens=4096, extra_body=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if extra_body:
            payload.update(extra_body)
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]


def extract_json(text):
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start_candidates = [pos for pos in [text.find("{"), text.find("[")] if pos >= 0]
    if start_candidates:
        text = text[min(start_candidates) :]
    return json.loads(text)


def table_record_to_schema_items(table_record):
    table_names = table_record["table_names_original"]
    normalized_tables = table_record.get("table_names", table_names)
    column_names = table_record["column_names_original"]
    normalized_columns = table_record.get("column_names", column_names)
    column_types = table_record.get("column_types", [])
    primary_keys = flatten_primary_keys(table_record.get("primary_keys", []))
    fk_column_ids = {idx for pair in table_record.get("foreign_keys", []) for idx in pair}
    items = []
    for table_id, table_name in enumerate(table_names):
        items.append(
            {
                "schema_item_id": table_id,
                "node_type": "table",
                "qualified_name": table_name,
                "table_name": table_name,
                "normalized_name": normalized_tables[table_id] if table_id < len(normalized_tables) else table_name,
            }
        )
    offset = len(table_names)
    for column_index, (table_id, column_name) in enumerate(column_names):
        if table_id < 0 or column_name == "*":
            continue
        table_name = table_names[table_id]
        normalized_column = normalized_columns[column_index][1] if column_index < len(normalized_columns) else column_name
        items.append(
            {
                "schema_item_id": offset + column_index - 1,
                "node_type": "column",
                "qualified_name": f"{table_name}.{column_name}",
                "table_name": table_name,
                "column_name": column_name,
                "normalized_name": normalized_column,
                "data_type": column_types[column_index] if column_index < len(column_types) else "",
                "is_primary_key": column_index in primary_keys,
                "is_foreign_key_endpoint": column_index in fk_column_ids,
            }
        )
    return items


def flatten_primary_keys(primary_keys):
    out = set()
    for item in primary_keys:
        if isinstance(item, list):
            out.update(item)
        else:
            out.add(item)
    return out


def fk_descriptions(table_record):
    tables = table_record.get("table_names_original", [])
    columns = table_record.get("column_names_original", [])
    descriptions = []
    for left, right in table_record.get("foreign_keys", []):
        if left >= len(columns) or right >= len(columns):
            continue
        lt, lc = columns[left]
        rt, rc = columns[right]
        if lt < 0 or rt < 0:
            continue
        descriptions.append(
            {
                "left": f"{tables[lt]}.{lc}",
                "right": f"{tables[rt]}.{rc}",
            }
        )
    return descriptions


def schema_prompt(table_record):
    return {
        "task": (
            "Create semantic cards for text-to-SQL schema grounding. Use only the provided schema names, "
            "types, primary/foreign-key flags, and foreign-key graph. Do not use any question, SQL, answer, "
            "label, or dataset-specific gold information. Infer likely meanings from names only."
        ),
        "db_id": table_record["db_id"],
        "allowed_value_types": ALLOWED_VALUE_TYPES,
        "allowed_sql_roles": ALLOWED_SQL_ROLES,
        "allowed_relation_types": ALLOWED_RELATION_TYPES,
        "schema_items": table_record_to_schema_items(table_record),
        "foreign_keys": fk_descriptions(table_record),
        "output_format": {
            "items": [
                {
                    "schema_item_id": 0,
                    "semantic_name": "short natural-language meaning",
                    "description": "one concise sentence",
                    "aliases": ["2-8 natural-language aliases or paraphrases"],
                    "value_type": "one allowed value type",
                    "likely_sql_roles": ["allowed SQL roles"],
                    "relation_types": ["allowed relation types"],
                }
            ]
        },
    }


def sanitize_schema_card(card, raw_by_id, split, db_id):
    item_id = int(card.get("schema_item_id"))
    raw = raw_by_id[item_id]
    aliases = card.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    roles = [x for x in (card.get("likely_sql_roles") or []) if x in ALLOWED_SQL_ROLES]
    relations = [x for x in (card.get("relation_types") or []) if x in ALLOWED_RELATION_TYPES]
    value_type = card.get("value_type")
    if value_type not in ALLOWED_VALUE_TYPES:
        value_type = "text" if raw["node_type"] == "column" else "table"
    return {
        "split": split,
        "db_id": db_id,
        "schema_item_id": item_id,
        "node_type": raw["node_type"],
        "qualified_name": raw["qualified_name"],
        "raw_name": raw.get("column_name") or raw.get("table_name") or raw["qualified_name"],
        "table_name_original": raw.get("table_name"),
        "column_name_original": raw.get("column_name"),
        "data_type": raw.get("data_type", ""),
        "semantic_name": str(card.get("semantic_name") or raw["qualified_name"]).strip(),
        "description": str(card.get("description") or "").strip(),
        "aliases": [str(x).strip() for x in aliases if str(x).strip()][:12],
        "value_type": value_type,
        "likely_sql_roles": roles,
        "relation_types": relations,
        "is_primary_key": bool(raw.get("is_primary_key", False)),
        "is_foreign_key_endpoint": bool(raw.get("is_foreign_key_endpoint", False)),
        "source": "llm",
    }


def generate_schema_cards_for_db(table_record, split, client, args):
    raw_items = table_record_to_schema_items(table_record)
    raw_by_id = {item["schema_item_id"]: item for item in raw_items}
    messages = [
        {
            "role": "system",
            "content": "You are a database semantic parser for text-to-SQL. Return strict JSON only.",
        },
        {"role": "user", "content": json.dumps(schema_prompt(table_record), ensure_ascii=False)},
    ]
    text = client.chat(
        messages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        extra_body={"enable_thinking": False} if args.disable_thinking else None,
    )
    parsed = extract_json(text)
    items = parsed.get("items", parsed if isinstance(parsed, list) else [])
    cards = []
    seen = set()
    for item in items:
        if not isinstance(item, dict) or item.get("schema_item_id") not in raw_by_id:
            continue
        cards.append(sanitize_schema_card(item, raw_by_id, split, table_record["db_id"]))
        seen.add(int(item["schema_item_id"]))
    # Fill missing items with source=llm_fallback, still no handcrafted domain aliases.
    for item_id, raw in raw_by_id.items():
        if item_id in seen:
            continue
        cards.append(
            {
                "split": split,
                "db_id": table_record["db_id"],
                "schema_item_id": item_id,
                "node_type": raw["node_type"],
                "qualified_name": raw["qualified_name"],
                "raw_name": raw.get("column_name") or raw.get("table_name") or raw["qualified_name"],
                "table_name_original": raw.get("table_name"),
                "column_name_original": raw.get("column_name"),
                "data_type": raw.get("data_type", ""),
                "semantic_name": raw["qualified_name"],
                "description": "",
                "aliases": [raw["qualified_name"]],
                "value_type": "table" if raw["node_type"] == "table" else "text",
                "likely_sql_roles": [],
                "relation_types": [],
                "is_primary_key": bool(raw.get("is_primary_key", False)),
                "is_foreign_key_endpoint": bool(raw.get("is_foreign_key_endpoint", False)),
                "source": "llm_fallback",
            }
        )
    return sorted(cards, key=lambda x: x["schema_item_id"])


def question_prompt(record):
    return {
        "task": (
            "Create a question card for text-to-SQL schema grounding. Extract only test-time available "
            "information from the natural-language question and evidence. Do not use gold SQL, answer, "
            "or schema labels. Keep phrases faithful to the input."
        ),
        "db_id": record.get("db_id"),
        "question": record.get("question"),
        "evidence": record.get("evidence"),
        "allowed_operations": ALLOWED_OPERATIONS,
        "allowed_relation_types": ALLOWED_RELATION_TYPES,
        "output_format": {
            "normalized_question": "short normalized restatement",
            "intent": "one sentence",
            "mentions": [
                {
                    "phrase": "exact or near-exact phrase from question/evidence",
                    "semantic_hint": "meaning useful for schema grounding",
                    "operation": "one allowed operation",
                    "relation_type": "one allowed relation type",
                    "value_hint": "literal value or null",
                }
            ],
            "operation_hints": ["allowed operations likely needed"],
            "value_hints": ["literal values or normalized values"],
            "formula_hints": ["formula or derived metric hints"],
            "ordering_hints": ["highest/lowest/top/rank/limit hints"],
        },
    }


def sanitize_question_card(parsed, record, split, record_index):
    mentions = parsed.get("mentions") or []
    clean_mentions = []
    for mention in mentions:
        if not isinstance(mention, dict):
            continue
        operation = mention.get("operation")
        relation_type = mention.get("relation_type")
        clean_mentions.append(
            {
                "phrase": str(mention.get("phrase") or "").strip(),
                "semantic_hint": str(mention.get("semantic_hint") or "").strip(),
                "operation": operation if operation in ALLOWED_OPERATIONS else None,
                "relation_type": relation_type if relation_type in ALLOWED_RELATION_TYPES else None,
                "value_hint": mention.get("value_hint"),
            }
        )
    return {
        "split": split,
        "db_id": record.get("db_id"),
        "question_id": record.get("question_id"),
        "record_index": record_index,
        "question": record.get("question"),
        "evidence": record.get("evidence"),
        "normalized_question": str(parsed.get("normalized_question") or record.get("question") or "").strip(),
        "intent": str(parsed.get("intent") or "").strip(),
        "mentions": clean_mentions[:20],
        "operation_hints": [x for x in (parsed.get("operation_hints") or []) if x in ALLOWED_OPERATIONS],
        "value_hints": [str(x).strip() for x in (parsed.get("value_hints") or []) if str(x).strip()][:20],
        "formula_hints": [str(x).strip() for x in (parsed.get("formula_hints") or []) if str(x).strip()][:12],
        "ordering_hints": [str(x).strip() for x in (parsed.get("ordering_hints") or []) if str(x).strip()][:12],
        "source": "llm",
    }


def generate_question_card(record, split, record_index, client, args):
    messages = [
        {
            "role": "system",
            "content": "You extract question semantics for text-to-SQL grounding. Return strict JSON only.",
        },
        {"role": "user", "content": json.dumps(question_prompt(record), ensure_ascii=False)},
    ]
    text = client.chat(
        messages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.question_max_tokens,
        extra_body={"enable_thinking": False} if args.disable_thinking else None,
    )
    parsed = extract_json(text)
    return sanitize_question_card(parsed, record, split, record_index)


def table_path_for_split(args, split):
    return Path(args.train_tables if split == "train" else args.dev_tables)


def label_path_for_split(args, split):
    return Path(args.train_labels if split == "train" else args.dev_labels)


def build_client(args):
    api_key = os.environ.get(args.api_key_env, "dummy")
    return OpenAICompatibleClient(args.base_url, api_key, args.model, args.timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tables", default="Data/BIRD/train_databases/train_databases/train_tables.json")
    parser.add_argument("--dev-tables", default="Data/BIRD/dev_tables.json")
    parser.add_argument("--train-labels", default="experiments/stage1_label_extraction_v2/bird_train_grounding_labels.jsonl")
    parser.add_argument("--dev-labels", default="experiments/stage1_label_extraction_v2/bird_dev_grounding_labels.jsonl")
    parser.add_argument("--output-dir", default="experiments/stage8f_llm_cards_qwen25")
    parser.add_argument("--splits", default="train,dev")
    parser.add_argument("--card-types", choices=["schema", "question", "both"], default="both")
    parser.add_argument("--db-limit", type=int, default=None)
    parser.add_argument("--question-limit", type=int, default=None)
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen2.5-coder-32b"))
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--question-max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = build_client(args)
    summary = {
        "config": vars(args),
        "splits": {},
        "note": (
            "Cards are generated by the local LLM from test-time available schema/question/evidence only. "
            "No handcrafted domain alias rules are used in this script."
        ),
    }
    for split in [x.strip() for x in args.splits.split(",") if x.strip()]:
        split_summary = {}
        if args.card_types in {"schema", "both"}:
            table_records = read_json(table_path_for_split(args, split))
            if args.db_limit is not None:
                table_records = table_records[: args.db_limit]
            schema_cards = []
            status_rows = []
            for index, table_record in enumerate(table_records, start=1):
                error = None
                try:
                    cards = generate_schema_cards_for_db(table_record, split, client, args)
                except Exception as exc:  # noqa: BLE001 - record and continue for long jobs.
                    cards = []
                    error = repr(exc)
                    if isinstance(exc, urllib.error.HTTPError):
                        try:
                            error = f"{error} body={exc.read().decode('utf-8', errors='replace')}"
                        except Exception:
                            pass
                schema_cards.extend(cards)
                status = {"split": split, "kind": "schema", "index": index, "db_id": table_record["db_id"], "card_count": len(cards), "error": error}
                status_rows.append(status)
                print(json.dumps(status, ensure_ascii=False))
                if args.sleep > 0:
                    time.sleep(args.sleep)
            write_jsonl(output_dir / f"{split}_schema_semantic_cards.jsonl", schema_cards)
            write_jsonl(output_dir / f"{split}_schema_status.jsonl", status_rows)
            split_summary["schema_card_count"] = len(schema_cards)
            split_summary["schema_error_count"] = sum(1 for row in status_rows if row["error"])
        if args.card_types in {"question", "both"}:
            records = read_jsonl(label_path_for_split(args, split), args.question_limit)
            question_cards = []
            status_rows = []
            for index, record in enumerate(records):
                error = None
                try:
                    card = generate_question_card(record, split, index, client, args)
                except Exception as exc:  # noqa: BLE001
                    card = {
                        "split": split,
                        "db_id": record.get("db_id"),
                        "question_id": record.get("question_id"),
                        "record_index": index,
                        "question": record.get("question"),
                        "evidence": record.get("evidence"),
                        "normalized_question": record.get("question"),
                        "intent": "",
                        "mentions": [],
                        "operation_hints": [],
                        "value_hints": [],
                        "formula_hints": [],
                        "ordering_hints": [],
                        "source": "llm_fallback",
                        "llm_error": repr(exc),
                    }
                    error = repr(exc)
                question_cards.append(card)
                status = {"split": split, "kind": "question", "index": index, "db_id": record.get("db_id"), "question_id": record.get("question_id"), "error": error}
                status_rows.append(status)
                print(json.dumps(status, ensure_ascii=False))
                if args.sleep > 0:
                    time.sleep(args.sleep)
            write_jsonl(output_dir / f"{split}_question_cards.jsonl", question_cards)
            write_jsonl(output_dir / f"{split}_question_status.jsonl", status_rows)
            split_summary["question_card_count"] = len(question_cards)
            split_summary["question_error_count"] = sum(1 for row in status_rows if row["error"])
        summary["splits"][split] = split_summary
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
