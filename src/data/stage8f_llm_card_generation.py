import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def read_jsonl_if_exists(path: Path):
    return read_jsonl(path) if path.exists() else []


def append_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def latest_by(records, key_fn):
    result = {}
    for record in records:
        result[key_fn(record)] = record
    return result


def normalized_cache_text(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def resumed_question_content_matches(card, record):
    return (
        normalized_cache_text(card.get("question"))
        == normalized_cache_text(record.get("question"))
        and normalized_cache_text(card.get("evidence"))
        == normalized_cache_text(record.get("evidence"))
    )


def validate_resumed_question(card, record, index):
    if int(card.get("record_index", -1)) != index:
        raise ValueError(f"Resume question record_index mismatch at {index}")
    if card.get("db_id") != record.get("db_id"):
        raise ValueError(
            f"Resume question db_id mismatch at {index}: "
            f"cached={card.get('db_id')} source={record.get('db_id')}"
        )
    cached_question_id = card.get("question_id")
    source_question_id = record.get("question_id")
    if cached_question_id is not None and source_question_id is not None and cached_question_id != source_question_id:
        raise ValueError(
            f"Resume question_id mismatch at {index}: "
            f"cached={cached_question_id} source={source_question_id}"
        )


def cached_jsonl_records(output_path, reuse_card_dirs, filename):
    records = []
    for reuse_card_dir in reuse_card_dirs:
        records.extend(read_jsonl_if_exists(reuse_card_dir / filename))
    records.extend(read_jsonl_if_exists(output_path))
    return records


def completed_results(tasks, worker_fn, workers):
    if workers <= 1:
        for task in tasks:
            yield worker_fn(*task)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stage8f-card") as executor:
        futures = [executor.submit(worker_fn, *task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()


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
    # Some OpenAI-compatible servers return literal newlines/control characters
    # inside otherwise valid JSON strings. strict=False accepts those without
    # attempting unsafe structural repair of malformed or truncated JSON.
    return json.loads(text, strict=False)


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
                "table_id": table_id,
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
                "table_id": table_id,
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


def column_schema_item_id(table_record, column_index):
    return len(table_record.get("table_names_original", [])) + column_index - 1


def fk_descriptions(table_record, allowed_item_ids=None):
    allowed_item_ids = set(allowed_item_ids or [])
    tables = table_record.get("table_names_original", [])
    columns = table_record.get("column_names_original", [])
    descriptions = []
    for left, right in table_record.get("foreign_keys", []):
        if left >= len(columns) or right >= len(columns):
            continue
        left_item_id = column_schema_item_id(table_record, left)
        right_item_id = column_schema_item_id(table_record, right)
        if allowed_item_ids and left_item_id not in allowed_item_ids and right_item_id not in allowed_item_ids:
            continue
        lt, lc = columns[left]
        rt, rc = columns[right]
        if lt < 0 or rt < 0:
            continue
        descriptions.append(
            {
                "left": f"{tables[lt]}.{lc}",
                "right": f"{tables[rt]}.{rc}",
                "left_schema_item_id": left_item_id,
                "right_schema_item_id": right_item_id,
            }
        )
    return descriptions


def schema_prompt(table_record, schema_items=None):
    schema_items = schema_items or table_record_to_schema_items(table_record)
    schema_item_ids = {item["schema_item_id"] for item in schema_items}
    return {
        "task": (
            "Create semantic cards for text-to-SQL schema grounding. Use only the provided schema names, "
            "types, primary/foreign-key flags, and foreign-key graph. Do not use any question, SQL, answer, "
            "label, or dataset-specific gold information. Infer likely meanings from names only. "
            "Return exactly one item for each provided schema_item_id."
        ),
        "db_id": table_record["db_id"],
        "allowed_value_types": ALLOWED_VALUE_TYPES,
        "allowed_sql_roles": ALLOWED_SQL_ROLES,
        "allowed_relation_types": ALLOWED_RELATION_TYPES,
        "schema_items": schema_items,
        "foreign_keys": fk_descriptions(table_record, schema_item_ids),
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


def fallback_schema_card(raw, split, db_id):
    return {
        "split": split,
        "db_id": db_id,
        "schema_item_id": raw["schema_item_id"],
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


def generate_schema_cards_for_items(table_record, split, client, args, schema_items):
    raw_items = table_record_to_schema_items(table_record)
    raw_by_id = {item["schema_item_id"]: item for item in raw_items}
    requested_ids = {item["schema_item_id"] for item in schema_items}
    messages = [
        {
            "role": "system",
            "content": "You are a database semantic parser for text-to-SQL. Return strict JSON only.",
        },
        {"role": "user", "content": json.dumps(schema_prompt(table_record, schema_items), ensure_ascii=False)},
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
        if not isinstance(item, dict) or item.get("schema_item_id") not in requested_ids:
            continue
        cards.append(sanitize_schema_card(item, raw_by_id, split, table_record["db_id"]))
        seen.add(int(item["schema_item_id"]))
    # Fill missing items with source=llm_fallback, still no handcrafted domain aliases.
    for item_id in requested_ids:
        if item_id in seen:
            continue
        cards.append(fallback_schema_card(raw_by_id[item_id], split, table_record["db_id"]))
    return sorted(cards, key=lambda x: x["schema_item_id"])


def schema_item_chunks(table_record, mode, max_items=None):
    raw_items = table_record_to_schema_items(table_record)
    if mode == "db":
        if not max_items or len(raw_items) <= max_items:
            return [raw_items]
        return [raw_items[index : index + max_items] for index in range(0, len(raw_items), max_items)]
    table_items = [item for item in raw_items if item["node_type"] == "table"]
    chunks = []
    for table_item in table_items:
        table_id = table_item.get("table_id")
        columns = [
            item
            for item in raw_items
            if item["node_type"] == "column" and item.get("table_id") == table_id
        ]
        if not max_items or len(columns) + 1 <= max_items:
            chunks.append([table_item] + columns)
            continue
        column_budget = max(max_items - 1, 1)
        for index in range(0, len(columns), column_budget):
            # Repeat the table item so every column shard retains table context.
            chunks.append([table_item] + columns[index : index + column_budget])
    return chunks


def generate_schema_cards_for_db(table_record, split, client, args):
    cards = []
    seen = set()
    chunk_errors = []
    chunks = schema_item_chunks(
        table_record,
        args.schema_card_mode,
        getattr(args, "schema_chunk_max_items", None),
    )
    for chunk_index, chunk in enumerate(chunks, start=1):
        try:
            chunk_cards = generate_schema_cards_for_items(table_record, split, client, args, chunk)
        except Exception as exc:  # noqa: BLE001 - fallback only for the failed chunk.
            chunk_cards = [fallback_schema_card(item, split, table_record["db_id"]) for item in chunk]
            chunk_errors.append({"chunk_index": chunk_index, "error": repr(exc)})
        for card in chunk_cards:
            item_id = card["schema_item_id"]
            if item_id in seen:
                continue
            seen.add(item_id)
            cards.append(card)
        if args.sleep > 0:
            time.sleep(args.sleep)
    if chunk_errors:
        for card in cards:
            if card.get("source") == "llm_fallback":
                card["chunk_errors"] = chunk_errors[:3]
    cards = sorted(cards, key=lambda x: x["schema_item_id"])
    diagnostics = {
        "chunk_count": len(chunks),
        "chunk_error_count": len(chunk_errors),
        "chunk_errors": chunk_errors,
        "fallback_card_count": sum(1 for card in cards if card.get("source") == "llm_fallback"),
    }
    diagnostics["fallback_rate"] = (
        diagnostics["fallback_card_count"] / len(cards) if cards else 1.0
    )
    return cards, diagnostics


def schema_card_quality(cards):
    fallback_count = sum(1 for card in cards if card.get("source") == "llm_fallback")
    return {
        "card_count": len(cards),
        "fallback_card_count": fallback_count,
        "fallback_rate": fallback_count / len(cards) if cards else 1.0,
    }


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


def exception_message(exc):
    error = repr(exc)
    if isinstance(exc, urllib.error.HTTPError):
        try:
            error = f"{error} body={exc.read().decode('utf-8', errors='replace')}"
        except Exception:
            pass
    return error


def generate_schema_task(index, table_record, split, client, args):
    db_id = table_record["db_id"]
    error = None
    try:
        cards, diagnostics = generate_schema_cards_for_db(table_record, split, client, args)
    except Exception as exc:  # noqa: BLE001 - preserve long-running batch progress.
        cards = []
        diagnostics = {
            "chunk_count": 0,
            "chunk_error_count": 0,
            "chunk_errors": [],
            "fallback_card_count": 0,
            "fallback_rate": 1.0,
        }
        error = exception_message(exc)
    if error is None and diagnostics["chunk_error_count"]:
        error = f"{diagnostics['chunk_error_count']} schema-card chunk(s) used fallback"
    status = {
        "split": split,
        "kind": "schema",
        "index": index,
        "db_id": db_id,
        "card_count": len(cards),
        **diagnostics,
        "error": error,
    }
    return index, cards, status


def fallback_question_card(record, split, index, error):
    return {
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
        "llm_error": error,
    }


def generate_question_task(index, record, split, client, args):
    error = None
    try:
        card = generate_question_card(record, split, index, client, args)
    except Exception as exc:  # noqa: BLE001 - preserve long-running batch progress.
        error = exception_message(exc)
        card = fallback_question_card(record, split, index, error)
    if getattr(args, "sleep", 0) > 0:
        time.sleep(args.sleep)
    status = {
        "split": split,
        "kind": "question",
        "index": index,
        "db_id": record.get("db_id"),
        "question_id": record.get("question_id"),
        "error": error,
    }
    return index, card, status


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
    parser.add_argument("--schema-card-mode", choices=["table", "db"], default="table")
    parser.add_argument(
        "--schema-chunk-max-items",
        type=int,
        default=24,
        help=(
            "Maximum schema items per LLM request. Large tables are split into column shards "
            "with the table item repeated as context."
        ),
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent OpenAI-compatible requests. vLLM can batch these continuously.",
    )
    parser.add_argument(
        "--reuse-card-dir",
        action="append",
        default=[],
        help=(
            "Existing Stage 8F output directory used as a read-only cache seed. "
            "Repeat the option to combine train/dev caches from multiple directories."
        ),
    )
    parser.add_argument(
        "--refresh-mismatched-question-cards",
        dest="refresh_mismatched_question_cards",
        action="store_true",
        help="Regenerate cached cards when normalized question/evidence changed (default).",
    )
    parser.add_argument(
        "--no-refresh-mismatched-question-cards",
        dest="refresh_mismatched_question_cards",
        action="store_false",
        help="Reuse cached cards without checking question/evidence content.",
    )
    parser.set_defaults(refresh_mismatched_question_cards=True)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing card/status JSONL files and append each completed item immediately.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume, regenerate records whose latest status contains an error.",
    )
    parser.add_argument(
        "--max-schema-fallback-rate",
        type=float,
        default=0.25,
        help="Fail quality validation when a split exceeds this schema-card fallback fraction.",
    )
    parser.add_argument(
        "--allow-excessive-schema-fallback",
        action="store_true",
        help="Write outputs and continue even when schema fallback exceeds the configured threshold.",
    )
    args = parser.parse_args()
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not 0.0 <= args.max_schema_fallback_rate <= 1.0:
        parser.error("--max-schema-fallback-rate must be in [0, 1]")
    if args.schema_chunk_max_items < 2:
        parser.error("--schema-chunk-max-items must be at least 2")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reuse_card_dirs = [Path(path) for path in args.reuse_card_dir]
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
            card_path = output_dir / f"{split}_schema_semantic_cards.jsonl"
            status_path = output_dir / f"{split}_schema_status.jsonl"
            if not args.resume:
                write_jsonl(card_path, [])
                write_jsonl(status_path, [])
            schema_by_key = latest_by(
                cached_jsonl_records(
                    card_path,
                    reuse_card_dirs,
                    f"{split}_schema_semantic_cards.jsonl",
                ),
                lambda row: (row.get("db_id"), int(row.get("schema_item_id", -1))),
            )
            status_by_db = latest_by(
                cached_jsonl_records(
                    status_path,
                    reuse_card_dirs,
                    f"{split}_schema_status.jsonl",
                ),
                lambda row: row.get("db_id"),
            )
            generated_count = 0
            reused_count = 0
            pending_schema_tasks = []
            for index, table_record in enumerate(table_records, start=1):
                db_id = table_record["db_id"]
                previous_status = status_by_db.get(db_id)
                expected_item_ids = {
                    int(item["schema_item_id"])
                    for item in table_record_to_schema_items(table_record)
                }
                cached_item_ids = {
                    item_id for cached_db_id, item_id in schema_by_key if cached_db_id == db_id
                }
                cached_cards = [
                    card for (cached_db_id, _), card in schema_by_key.items() if cached_db_id == db_id
                ]
                cached_quality = schema_card_quality(cached_cards)
                cache_complete = expected_item_ids.issubset(cached_item_ids)
                reuse = previous_status is not None and cache_complete and not (
                    args.retry_errors
                    and (
                        previous_status.get("error")
                        or cached_quality["fallback_rate"] > args.max_schema_fallback_rate
                    )
                )
                if reuse:
                    reused_count += 1
                    print(
                        json.dumps(
                            {"split": split, "kind": "schema", "index": index, "db_id": db_id, "event": "resume_skip"},
                            ensure_ascii=False,
                        )
                    )
                    continue
                pending_schema_tasks.append((index, table_record, split, client, args))
            for _, cards, status in completed_results(
                pending_schema_tasks,
                generate_schema_task,
                args.workers,
            ):
                append_jsonl(card_path, cards)
                append_jsonl(status_path, [status])
                for card in cards:
                    schema_by_key[(status["db_id"], int(card["schema_item_id"]))] = card
                status_by_db[status["db_id"]] = status
                generated_count += 1
                print(json.dumps(status, ensure_ascii=False))
            db_order = {record["db_id"]: index for index, record in enumerate(table_records)}
            schema_cards = sorted(
                [card for (db_id, _), card in schema_by_key.items() if db_id in db_order],
                key=lambda card: (db_order[card["db_id"]], int(card["schema_item_id"])),
            )
            status_rows = [status_by_db[record["db_id"]] for record in table_records if record["db_id"] in status_by_db]
            write_jsonl(card_path, schema_cards)
            write_jsonl(status_path, status_rows)
            split_summary["schema_db_target_count"] = len(table_records)
            split_summary["schema_db_completed_count"] = len(status_rows)
            split_summary["schema_db_generated_this_run"] = generated_count
            split_summary["schema_db_reused_count"] = reused_count
            split_summary["schema_card_count"] = len(schema_cards)
            split_summary["schema_error_count"] = sum(1 for row in status_rows if row.get("error"))
            schema_quality = schema_card_quality(schema_cards)
            split_summary["schema_fallback_card_count"] = schema_quality["fallback_card_count"]
            split_summary["schema_fallback_rate"] = schema_quality["fallback_rate"]
            split_summary["schema_chunk_error_count"] = sum(
                int(row.get("chunk_error_count", 0)) for row in status_rows
            )
            split_summary["schema_quality_passed"] = (
                schema_quality["fallback_rate"] <= args.max_schema_fallback_rate
            )
        if args.card_types in {"question", "both"}:
            records = read_jsonl(label_path_for_split(args, split), args.question_limit)
            card_path = output_dir / f"{split}_question_cards.jsonl"
            status_path = output_dir / f"{split}_question_status.jsonl"
            if not args.resume:
                write_jsonl(card_path, [])
                write_jsonl(status_path, [])
            question_by_index = latest_by(
                cached_jsonl_records(
                    card_path,
                    reuse_card_dirs,
                    f"{split}_question_cards.jsonl",
                ),
                lambda row: int(row.get("record_index", -1)),
            )
            status_by_index = latest_by(
                cached_jsonl_records(
                    status_path,
                    reuse_card_dirs,
                    f"{split}_question_status.jsonl",
                ),
                lambda row: int(row.get("index", -1)),
            )
            generated_count = 0
            reused_count = 0
            mismatched_count = 0
            pending_question_tasks = []
            for index, record in enumerate(records):
                previous_card = question_by_index.get(index)
                previous_status = status_by_index.get(index)
                reuse = previous_card is not None and previous_status is not None and not (
                    args.retry_errors and previous_status.get("error")
                )
                if reuse:
                    validate_resumed_question(previous_card, record, index)
                    if (
                        args.refresh_mismatched_question_cards
                        and not resumed_question_content_matches(previous_card, record)
                    ):
                        reuse = False
                        mismatched_count += 1
                if reuse:
                    reused_count += 1
                    print(
                        json.dumps(
                            {"split": split, "kind": "question", "index": index, "db_id": record.get("db_id"), "question_id": record.get("question_id"), "event": "resume_skip"},
                            ensure_ascii=False,
                        )
                    )
                    continue
                pending_question_tasks.append((index, record, split, client, args))
            for index, card, status in completed_results(
                pending_question_tasks,
                generate_question_task,
                args.workers,
            ):
                append_jsonl(card_path, [card])
                append_jsonl(status_path, [status])
                question_by_index[index] = card
                status_by_index[index] = status
                generated_count += 1
                print(json.dumps(status, ensure_ascii=False))
            question_cards = [question_by_index[index] for index in range(len(records)) if index in question_by_index]
            status_rows = [status_by_index[index] for index in range(len(records)) if index in status_by_index]
            write_jsonl(card_path, question_cards)
            write_jsonl(status_path, status_rows)
            split_summary["question_target_count"] = len(records)
            split_summary["question_completed_count"] = len(question_cards)
            split_summary["question_generated_this_run"] = generated_count
            split_summary["question_reused_count"] = reused_count
            split_summary["question_cache_mismatch_count"] = mismatched_count
            split_summary["question_card_count"] = len(question_cards)
            split_summary["question_error_count"] = sum(1 for row in status_rows if row["error"])
        summary["splits"][split] = split_summary
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")
    failed_schema_splits = [
        split
        for split, split_summary in summary["splits"].items()
        if split_summary.get("schema_quality_passed") is False
    ]
    if failed_schema_splits and not args.allow_excessive_schema_fallback:
        rates = {
            split: summary["splits"][split]["schema_fallback_rate"]
            for split in failed_schema_splits
        }
        raise RuntimeError(
            "Schema-card fallback quality gate failed: "
            f"rates={rates}, max={args.max_schema_fallback_rate}. "
            "Use --retry-errors to regenerate fallback caches; override only with "
            "--allow-excessive-schema-fallback."
        )


if __name__ == "__main__":
    main()
