import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


RELATION_TYPES = [
    "OUTPUT_TARGET",
    "ENTITY_NAME",
    "METRIC_TARGET",
    "PREDICATE_COLUMN",
    "VALUE_ANCHOR",
    "ORDER_KEY",
    "GROUP_KEY",
    "JOIN_BRIDGE",
    "FORMULA_COMPONENT",
]


ABBREVIATIONS = [
    (r"\bavg\b", "average"),
    (r"\bscr\b", "score"),
    (r"\bnum\b", "number"),
    (r"\btst\b", "test"),
    (r"\btakr\b", "takers"),
    (r"\badm\b", "administrator"),
    (r"\bfname\b", "first name"),
    (r"\blname\b", "last name"),
    (r"\bstr\b", "street"),
    (r"\babr\b", "abbreviated"),
    (r"\bnces\b", "National Center for Education Statistics"),
    (r"\bcds\b", "county district school"),
    (r"\bfrpm\b", "free and reduced-price meals"),
    (r"\bnslp\b", "National School Lunch Program"),
    (r"\bge\b", "greater than or equal to"),
]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_identifier(name: str) -> str:
    text = name.replace("_", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d+)", r"\1 \2", text)
    text = re.sub(r"(\d+)([A-Za-z])", r"\1 \2", text)
    text = text.replace("(Y/N)", " yes no ")
    text = text.replace("%", " percent ")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def expand_abbreviations(text: str) -> str:
    expanded = f" {text.lower()} "
    for pattern, replacement in ABBREVIATIONS:
        expanded = re.sub(pattern, replacement, expanded)
    expanded = re.sub(r"\badministrator f name\b", "administrator first name", expanded)
    expanded = re.sub(r"\badministrator l name\b", "administrator last name", expanded)
    expanded = re.sub(r"\bmail street abbreviated\b", "abbreviated mailing street address", expanded)
    expanded = re.sub(r"\bmail street\b", "mailing street address", expanded)
    expanded = re.sub(r"\bstreet abbreviated\b", "abbreviated street address", expanded)
    return re.sub(r"\s+", " ", expanded).strip()


def infer_value_type(raw_name: str, expanded_name: str, data_type: str, node_type: str) -> str:
    if node_type == "table":
        return "table"
    text = f"{raw_name} {expanded_name}".lower()
    if any(x in text for x in ["id", "code", "cds", "key"]):
        return "identifier"
    if data_type in {"integer", "real", "number", "float"} or any(
        x in text
        for x in [
            "count",
            "number",
            "score",
            "rate",
            "percent",
            "enrollment",
            "amount",
            "price",
            "average",
            "total",
            "latitude",
            "longitude",
        ]
    ):
        return "numeric_metric"
    if data_type in {"date", "time"} or any(x in text for x in ["date", "year", "time"]):
        return "temporal"
    if any(
        x in text
        for x in [
            "type",
            "status",
            "category",
            "county",
            "district",
            "state",
            "charter",
            "virtual",
            "magnet",
            "gender",
            "level",
            "funding",
        ]
    ):
        return "categorical"
    if any(
        x in text
        for x in [
            "name",
            "school",
            "title",
            "phone",
            "email",
            "website",
            "address",
            "street",
            "zip",
            "city",
        ]
    ):
        return "entity_text"
    return "text"


def infer_roles(raw_name: str, expanded_name: str, value_type: str, node_type: str):
    if node_type == "table":
        return ["FROM", "JOIN"], ["JOIN_BRIDGE"]
    text = f"{raw_name} {expanded_name}".lower()
    roles = set()
    relations = set()
    if value_type == "identifier":
        roles.update(["JOIN", "WHERE", "SELECT"])
        relations.add("JOIN_BRIDGE")
    if value_type in {"entity_text", "text"}:
        roles.update(["SELECT", "WHERE"])
        relations.update(["OUTPUT_TARGET", "ENTITY_NAME"])
    if value_type in {"categorical", "temporal"}:
        roles.update(["WHERE", "GROUP_BY", "SELECT"])
        relations.update(["PREDICATE_COLUMN", "VALUE_ANCHOR", "GROUP_KEY"])
    if value_type == "numeric_metric":
        roles.update(["SELECT", "WHERE", "ORDER_BY"])
        relations.update(["METRIC_TARGET", "PREDICATE_COLUMN", "ORDER_KEY"])
        if any(x in text for x in ["rate", "percent", "count", "enrollment", "amount", "score", "average"]):
            relations.add("FORMULA_COMPONENT")
    if any(x in text for x in ["name", "title", "phone", "email", "website", "address", "street", "zip"]):
        relations.update(["OUTPUT_TARGET", "ENTITY_NAME"])
    if any(x in text for x in ["date", "year", "time"]):
        relations.update(["PREDICATE_COLUMN", "ORDER_KEY"])
    return sorted(roles), sorted(relations, key=RELATION_TYPES.index)


def make_aliases(raw_name: str, normalized_name: str, expanded_name: str, table_name: str):
    aliases = []
    for item in [raw_name, normalized_name, expanded_name]:
        if item and item not in aliases:
            aliases.append(item)
    low = expanded_name.lower()
    if "average score math" in low or "average score in math" in low:
        aliases.extend(["average SAT math score", "math average score", "average score in Math"])
    if "average score read" in low or "average score reading" in low:
        aliases.extend(["average SAT reading score", "reading average score", "average score in Reading"])
    if "average score write" in low or "average score writing" in low:
        aliases.extend(["average SAT writing score", "writing average score", "average score in Writing"])
    if "number test takers" in low:
        aliases.extend(["number of SAT test takers", "SAT test takers", "number of test takers"])
    if "administrator last name" in low:
        aliases.extend(["administrator last name", "admin surname", "principal last name"])
    if "administrator first name" in low:
        aliases.extend(["administrator first name", "principal first name"])
    if "administrator email" in low:
        aliases.extend(["administrator email", "principal email"])
    if "mailing street address" in low:
        aliases.extend(["mailing street address", "mail address", "mail street"])
    if "abbreviated mailing street address" in low:
        aliases.extend(["abbreviated mailing street address", "abbreviated mailing street"])
    if "educational option type" in low:
        aliases.extend(["educational option", "school educational option", "continuation school"])
    if "charter funding type" in low:
        aliases.extend(["charter funding type", "direct charter-funded", "directly funded charter"])
    if "free and reduced price meals" in table_name.lower() or "free reduced price meals" in low:
        aliases.extend(["free or reduced price meal", "FRPM"])
    dedup = []
    for alias in aliases:
        alias = re.sub(r"\s+", " ", alias).strip()
        if alias and alias.lower() not in {x.lower() for x in dedup}:
            dedup.append(alias)
    return dedup[:12]


def build_schema_cards(table_record, split: str):
    db_id = table_record["db_id"]
    table_original = table_record["table_names_original"]
    table_normalized = table_record.get("table_names", table_original)
    column_original = table_record["column_names_original"]
    column_normalized = table_record.get("column_names", column_original)
    column_types = table_record.get("column_types", [])
    primary_keys = flatten_primary_keys(table_record.get("primary_keys", []))
    fk_column_ids = {x for pair in table_record.get("foreign_keys", []) for x in pair}

    cards = []
    for table_id, raw_table in enumerate(table_original):
        norm_table = table_normalized[table_id] if table_id < len(table_normalized) else raw_table
        split_name = split_identifier(raw_table)
        expanded = expand_abbreviations(split_name)
        value_type = infer_value_type(raw_table, expanded, "", "table")
        roles, relations = infer_roles(raw_table, expanded, value_type, "table")
        cards.append(
            {
                "split": split,
                "db_id": db_id,
                "node_id": f"{db_id}::table::{table_id}",
                "schema_item_id": table_id,
                "node_type": "table",
                "table_id": table_id,
                "table_name_original": raw_table,
                "table_name_normalized": norm_table,
                "column_name_original": None,
                "column_name_normalized": None,
                "raw_name": raw_table,
                "qualified_name": raw_table,
                "semantic_name": expanded,
                "description": f"Table containing records related to {expanded}.",
                "aliases": make_aliases(raw_table, norm_table, expanded, expanded),
                "value_type": value_type,
                "likely_sql_roles": roles,
                "relation_types": relations,
                "is_primary_key": False,
                "is_foreign_key_endpoint": False,
                "source": "heuristic",
            }
        )

    column_item_offset = len(table_original)
    for col_index, (table_id, raw_col) in enumerate(column_original):
        if table_id < 0 or raw_col == "*":
            continue
        norm_col = column_normalized[col_index][1] if col_index < len(column_normalized) else raw_col
        raw_table = table_original[table_id]
        norm_table = table_normalized[table_id] if table_id < len(table_normalized) else raw_table
        dtype = column_types[col_index] if col_index < len(column_types) else ""
        split_name = split_identifier(raw_col)
        expanded = expand_abbreviations(split_name)
        table_sem = expand_abbreviations(split_identifier(norm_table))
        value_type = infer_value_type(raw_col, expanded, dtype, "column")
        roles, relations = infer_roles(raw_col, expanded, value_type, "column")
        schema_item_id = column_item_offset + col_index - 1
        cards.append(
            {
                "split": split,
                "db_id": db_id,
                "node_id": f"{db_id}::column::{schema_item_id}",
                "schema_item_id": schema_item_id,
                "node_type": "column",
                "table_id": table_id,
                "table_name_original": raw_table,
                "table_name_normalized": norm_table,
                "column_name_original": raw_col,
                "column_name_normalized": norm_col,
                "raw_name": raw_col,
                "qualified_name": f"{raw_table}.{raw_col}",
                "semantic_name": expanded,
                "description": f"Column '{raw_col}' in table '{raw_table}', interpreted as {expanded}.",
                "aliases": make_aliases(raw_col, norm_col, expanded, table_sem),
                "value_type": value_type,
                "likely_sql_roles": roles,
                "relation_types": relations,
                "data_type": dtype,
                "is_primary_key": col_index in primary_keys,
                "is_foreign_key_endpoint": col_index in fk_column_ids,
                "source": "heuristic",
            }
        )
    return cards


def flatten_primary_keys(primary_keys):
    out = set()
    for item in primary_keys:
        if isinstance(item, list):
            out.update(item)
        else:
            out.add(item)
    return out


class OpenAICompatibleClient:
    def __init__(self, base_url, api_key, model, timeout):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, prompt, temperature, top_p, max_tokens, enable_thinking):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You explain database schema names for text-to-SQL. Return strict JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
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


def extract_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = min([p for p in [text.find("{"), text.find("[")] if p >= 0], default=0)
    text = text[start:]
    return json.loads(text)


def llm_enhance_cards(cards, client, args):
    prompt = {
        "task": "For each schema item, improve semantic_name, description, aliases, value_type, likely_sql_roles, and relation_types. Do not use any question, SQL, answer, or label.",
        "allowed_value_types": ["table", "identifier", "numeric_metric", "temporal", "categorical", "entity_text", "text"],
        "allowed_relation_types": RELATION_TYPES,
        "schema_items": [
            {
                "schema_item_id": c["schema_item_id"],
                "node_type": c["node_type"],
                "qualified_name": c["qualified_name"],
                "data_type": c.get("data_type", ""),
                "heuristic_semantic_name": c["semantic_name"],
                "heuristic_aliases": c["aliases"],
            }
            for c in cards
        ],
        "output_format": {
            "items": [
                {
                    "schema_item_id": "integer",
                    "semantic_name": "short English phrase",
                    "description": "one sentence",
                    "aliases": ["short aliases"],
                    "value_type": "one allowed value type",
                    "likely_sql_roles": ["SELECT|WHERE|JOIN|ORDER_BY|GROUP_BY|HAVING"],
                    "relation_types": RELATION_TYPES[:3],
                }
            ]
        },
    }
    text = client.chat(
        json.dumps(prompt, ensure_ascii=False),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
    )
    parsed = extract_json_object(text)
    items = parsed.get("items", parsed if isinstance(parsed, list) else [])
    by_id = {item.get("schema_item_id"): item for item in items if isinstance(item, dict)}
    enhanced = []
    for card in cards:
        item = by_id.get(card["schema_item_id"])
        merged = dict(card)
        if item:
            for key in ["semantic_name", "description", "aliases", "value_type", "likely_sql_roles", "relation_types"]:
                if item.get(key):
                    merged[key] = item[key]
            merged["source"] = "llm"
        enhanced.append(merged)
    return enhanced


def summarize(cards):
    by_split = {}
    high_risk = {}
    risk_patterns = [
        "AvgScrMath",
        "AvgScrRead",
        "AvgScrWrite",
        "NumTstTakr",
        "NumGE1500",
        "AdmLName1",
        "AdmFName1",
        "AdmEmail1",
        "MailStreet",
        "MailStrAbr",
        "FRPM Count",
        "Educational Option Type",
        "Charter Funding Type",
    ]
    for card in cards:
        stat = by_split.setdefault(card["split"], {"db_count": set(), "table_count": 0, "column_count": 0, "source": {}})
        stat["db_count"].add(card["db_id"])
        if card["node_type"] == "table":
            stat["table_count"] += 1
        else:
            stat["column_count"] += 1
        stat["source"][card["source"]] = stat["source"].get(card["source"], 0) + 1
        if any(p.lower() in card["qualified_name"].lower() for p in risk_patterns):
            high_risk.setdefault(card["db_id"], []).append(
                {
                    "qualified_name": card["qualified_name"],
                    "semantic_name": card["semantic_name"],
                    "aliases": card["aliases"][:8],
                    "value_type": card["value_type"],
                    "relation_types": card["relation_types"],
                }
            )
    for stat in by_split.values():
        stat["db_count"] = len(stat["db_count"])
    return {"by_split": by_split, "high_risk_audit": high_risk}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tables", default="Data/BIRD/train_databases/train_databases/train_tables.json")
    parser.add_argument("--dev-tables", default="Data/BIRD/dev_tables.json")
    parser.add_argument("--output-dir", default="experiments/stage5i_schema_semantic_cards")
    parser.add_argument("--splits", default="train,dev")
    parser.add_argument("--db-limit", type=int, default=None)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3-32b"))
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    split_paths = {
        "train": Path(args.train_tables),
        "dev": Path(args.dev_tables),
    }
    all_cards = []
    client = None
    if args.use_llm:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key. Set environment variable: {args.api_key_env}")
        client = OpenAICompatibleClient(args.base_url, api_key, args.model, args.timeout)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        table_path = split_paths[split]
        table_records = read_json(table_path)
        if args.db_limit is not None:
            table_records = table_records[: args.db_limit]
        split_cards = []
        for db_index, table_record in enumerate(table_records, start=1):
            cards = build_schema_cards(table_record, split)
            if client:
                try:
                    cards = llm_enhance_cards(cards, client, args)
                    error = None
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                    error = repr(exc)
                    if isinstance(exc, urllib.error.HTTPError):
                        try:
                            error = f"{repr(exc)} body={exc.read().decode('utf-8', errors='replace')}"
                        except Exception:
                            pass
                    for card in cards:
                        card["llm_error"] = error
                print(json.dumps({"split": split, "db_id": table_record["db_id"], "index": db_index, "error": error}, ensure_ascii=False))
                if args.sleep > 0:
                    time.sleep(args.sleep)
            split_cards.extend(cards)
        write_jsonl(out_dir / f"{split}_schema_semantic_cards.jsonl", split_cards)
        all_cards.extend(split_cards)

    summary = summarize(all_cards)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary["by_split"], ensure_ascii=False, indent=2))
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
