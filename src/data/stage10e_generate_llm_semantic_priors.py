"""Generate frozen-LLM semantic priors for Stage 10 candidate schema graphs.

The LLM is an inference-only semantic assistant.  Prompts contain only the
question, evidence, and candidate schema metadata; gold labels and SQL are never
exposed.  The validated output is a sparse role-by-schema score table that can be
cached and injected into a trainable graph reranker without back-propagating into
the LLM.
"""

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SEMANTIC_ROLES = [
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


def read_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def extract_json(text):
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    if start >= 0:
        text = text[start:]
    return json.loads(text, strict=False)


def source_fingerprint(example):
    payload = {
        "record_index": int(example["record_index"]),
        "db_id": example.get("db_id"),
        "question": example.get("question") or "",
        "evidence": example.get("evidence") or "",
        "candidates": [
            [
                int(node["schema_item_id"]),
                node.get("type"),
                node.get("name"),
                node.get("owner_table_id"),
            ]
            for node in example.get("candidate_nodes", [])
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_prompt_rows(example):
    rows = []
    id_to_name = {
        int(node["schema_item_id"]): node.get("name")
        for node in example.get("candidate_nodes", [])
    }
    for node in example.get("candidate_nodes", []):
        owner_id = node.get("owner_table_id")
        rows.append(
            {
                "schema_id": int(node["schema_item_id"]),
                "type": node.get("type"),
                "name": node.get("name"),
                "owner_table": id_to_name.get(int(owner_id)) if owner_id is not None else None,
            }
        )
    return rows


def build_messages(example):
    schema_rows = candidate_prompt_rows(example)
    contract = {role: [] for role in SEMANTIC_ROLES}
    return [
        {
            "role": "system",
            "content": (
                "You are a semantic schema-grounding assistant for SQLite. Return strict JSON only. "
                "Use only schema_id values listed by the user. Do not write SQL. Infer semantic relevance, "
                "not foreign-key closure. Omit uncertain nodes rather than guessing. Confidence must be in [0,1]."
            ),
        },
        {
            "role": "user",
            "content": (
                "Assign candidate schema nodes to text-to-SQL semantic roles. A node may have multiple roles.\n\n"
                f"Database: {example.get('db_id')}\n"
                f"Question: {example.get('question') or ''}\n"
                f"Evidence: {example.get('evidence') or ''}\n\n"
                "Candidate schema nodes:\n"
                + json.dumps(schema_rows, ensure_ascii=False, indent=2)
                + "\n\nEach role array contains objects shaped as "
                "{\"schema_id\": <integer from the candidate list>, \"confidence\": <0..1>}. "
                "Return exactly this top-level shape (empty arrays are allowed):\n"
                + json.dumps({"roles": contract}, ensure_ascii=False, indent=2)
            ),
        },
    ]


class OpenAICompatibleClient:
    def __init__(self, base_url, api_key, model, timeout):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, messages, temperature, top_p, max_tokens, disable_thinking):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if disable_thinking:
            payload["enable_thinking"] = False
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        return result["choices"][0]["message"]["content"]


def validate_prior(raw, example):
    allowed_ids = {
        int(node["schema_item_id"]) for node in example.get("candidate_nodes", [])
    }
    raw_roles = raw.get("roles", raw) if isinstance(raw, dict) else {}
    role_scores = {role: {} for role in SEMANTIC_ROLES}
    rejected = []
    for role in SEMANTIC_ROLES:
        entries = raw_roles.get(role, []) if isinstance(raw_roles, dict) else []
        if not isinstance(entries, list):
            rejected.append({"role": role, "reason": "not_a_list"})
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                rejected.append({"role": role, "reason": "not_an_object"})
                continue
            try:
                schema_id = int(entry.get("schema_id"))
                confidence = float(entry.get("confidence", 0.0))
            except (TypeError, ValueError):
                rejected.append({"role": role, "entry": entry, "reason": "invalid_value"})
                continue
            if schema_id not in allowed_ids:
                rejected.append({"role": role, "schema_id": schema_id, "reason": "unknown_schema_id"})
                continue
            confidence = min(max(confidence, 0.0), 1.0)
            role_scores[role][schema_id] = max(
                role_scores[role].get(schema_id, 0.0), confidence
            )
    node_priors = []
    for schema_id in sorted(allowed_ids):
        scores = {role: role_scores[role].get(schema_id, 0.0) for role in SEMANTIC_ROLES}
        if max(scores.values(), default=0.0) > 0.0:
            node_priors.append({"schema_item_id": schema_id, "role_scores": scores})
    return node_priors, rejected


def generate_one(example, client, args):
    base = {
        "record_index": int(example["record_index"]),
        "db_id": example.get("db_id"),
        "question_id": example.get("question_id"),
        "question": example.get("question"),
        "evidence": example.get("evidence"),
        "source_fingerprint": source_fingerprint(example),
        "semantic_roles": SEMANTIC_ROLES,
        "model": args.model,
    }
    try:
        raw_output = client.chat(
            build_messages(example),
            args.temperature,
            args.top_p,
            args.max_tokens,
            args.disable_thinking,
        )
        parsed = extract_json(raw_output)
        node_priors, rejected = validate_prior(parsed, example)
        return {
            **base,
            "status": "ok",
            "node_priors": node_priors,
            "rejected_entries": rejected,
            "raw_output": raw_output if args.keep_raw_output else None,
            "error": None,
        }
    except Exception as exc:  # Persist failures so resume/retry is deterministic.
        return {
            **base,
            "status": "error",
            "node_priors": [],
            "rejected_entries": [],
            "raw_output": None,
            "error": repr(exc),
        }


def main():
    parser = argparse.ArgumentParser(description="Generate frozen-LLM semantic priors for Stage 10E.")
    parser.add_argument("--factor-graph-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:9009/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--flush-every",
        type=int,
        default=20,
        help="Append completed records every N calls so interrupted runs can resume.",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--keep-raw-output", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.flush_every < 1:
        parser.error("--flush-every must be at least 1")
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")

    examples = read_jsonl(args.factor_graph_file, args.limit)
    output_path = Path(args.output_file)
    previous = read_jsonl(output_path) if args.resume and output_path.exists() else []
    cached = {int(row["record_index"]): row for row in previous}
    pending = []
    reused = 0
    for example in examples:
        index = int(example["record_index"])
        old = cached.get(index)
        reusable = bool(
            old
            and old.get("source_fingerprint") == source_fingerprint(example)
            and (old.get("status") == "ok" or not args.retry_errors)
        )
        if reusable:
            reused += 1
        else:
            pending.append(example)
    if not args.resume:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    client = OpenAICompatibleClient(
        args.base_url,
        os.environ.get(args.api_key_env, "dummy"),
        args.model,
        args.timeout,
    )
    generated_count = 0
    buffer = []

    def record_result(result):
        nonlocal generated_count
        buffer.append(result)
        generated_count += 1
        if len(buffer) >= args.flush_every:
            append_jsonl(output_path, buffer)
            buffer.clear()
        if generated_count % max(args.flush_every, 1) == 0:
            print(f"generated {generated_count}/{len(pending)}", flush=True)

    if args.workers == 1:
        for example in pending:
            record_result(generate_one(example, client, args))
    else:
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="stage10e-llm") as executor:
            futures = [executor.submit(generate_one, example, client, args) for example in pending]
            for future in as_completed(futures):
                record_result(future.result())
    if buffer:
        append_jsonl(output_path, buffer)

    latest = {int(row["record_index"]): row for row in read_jsonl(output_path)}
    selected = [latest[int(example["record_index"])] for example in examples if int(example["record_index"]) in latest]
    summary = {
        "config": vars(args),
        "target_count": len(examples),
        "completed_count": len(selected),
        "generated_this_run": generated_count,
        "reused_count": reused,
        "ok_count": sum(row.get("status") == "ok" for row in selected),
        "error_count": sum(row.get("status") != "ok" for row in selected),
        "non_empty_prior_count": sum(bool(row.get("node_priors")) for row in selected),
        "rejected_entry_count": sum(len(row.get("rejected_entries", [])) for row in selected),
        "leakage_note": "Prompts use question, evidence, and candidate schema metadata only; gold labels/SQL are excluded.",
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_path}")


if __name__ == "__main__":
    main()
