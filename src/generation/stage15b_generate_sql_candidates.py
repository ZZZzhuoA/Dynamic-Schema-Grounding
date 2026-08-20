"""Generate diverse real SQL hypotheses with an OpenAI-compatible endpoint."""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.stage3_online_llm_generate import clean_sql


def read_prompt_rows(path, limit=None, offset=0, allowed_indices=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for record_index, line in enumerate(handle):
            if record_index < offset or not line.strip():
                continue
            if allowed_indices is not None and record_index not in allowed_indices:
                continue
            row = json.loads(line)
            row["record_index"] = record_index
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def read_allowed_indices(path):
    if not path:
        return None
    result = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result.add(int(row.get("record_index", row.get("metadata", {}).get("record_index"))))
    return result


def sql_signature(sql):
    return " ".join(str(sql or "").strip().rstrip(";").split()).casefold()


def choice_logprob(choice):
    content = (choice.get("logprobs") or {}).get("content") or []
    values = [float(token["logprob"]) for token in content if token.get("logprob") is not None]
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def parse_choices(response):
    parsed = []
    for choice in response.get("choices", []):
        message = choice.get("message") or {}
        raw = message.get("content") or ""
        mean_logprob, token_count = choice_logprob(choice)
        parsed.append(
            {
                "raw_output": raw,
                "generated_sql": clean_sql(raw),
                "mean_logprob": mean_logprob,
                "logprob_token_count": token_count,
                "finish_reason": choice.get("finish_reason"),
            }
        )
    return parsed


def request_payload(model, prompt, n, args, seed, temperature=None, top_p=None):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert SQLite SQL generator. Return only SQL."},
            {"role": "user", "content": prompt},
        ],
        "n": int(n),
        "temperature": float(args.temperature if temperature is None else temperature),
        "top_p": float(args.top_p if top_p is None else top_p),
        "max_tokens": int(args.max_tokens),
        "stream": False,
        "seed": int(seed),
    }
    if args.request_logprobs:
        payload["logprobs"] = True
    if args.disable_thinking:
        payload["enable_thinking"] = False
    return payload


class CandidateClient:
    def __init__(self, base_url, api_key, model, timeout):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, prompt, n, args, seed, temperature=None, top_p=None):
        payload = request_payload(
            self.model, prompt, n, args, seed, temperature=temperature, top_p=top_p
        )
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def done_indices(path):
    if not Path(path).exists():
        return set()
    result = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.add(int(json.loads(line)["record_index"]))
    return result


def append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def error_text(exc):
    text = repr(exc)
    if isinstance(exc, urllib.error.HTTPError):
        try:
            text += " body=" + exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--record-index-file", help="Optional JSONL whose record_index values define the subset")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:9009/v1"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen2.5-coder-32b"))
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument(
        "--greedy-anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reserve candidate 0 for a separate temperature=0 request.",
    )
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-logprobs", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "EMPTY")
    allowed = read_allowed_indices(args.record_index_file)
    prompts = read_prompt_rows(args.prompt_file, args.limit, args.offset, allowed)
    completed = done_indices(args.output_file) if args.resume else set()
    client = CandidateClient(args.base_url, api_key, args.model, args.timeout)
    config = vars(args).copy()
    config.pop("api_key_env", None)

    for position, item in enumerate(prompts, start=1):
        record_index = int(item["record_index"])
        if record_index in completed:
            continue
        candidates, seen, errors = [], set(), []
        if args.greedy_anchor:
            try:
                response = client.generate(
                    item["prompt"], 1, args, args.seed + record_index * 1009,
                    temperature=0.0, top_p=1.0,
                )
                greedy = parse_choices(response)
                if greedy:
                    candidate = greedy[0]
                    signature = sql_signature(candidate["generated_sql"])
                    if signature:
                        seen.add(signature)
                        candidate.update(
                            {
                                "candidate_id": "greedy",
                                "llm_rank": 0,
                                "generation_mode": "greedy",
                                "generation_round": -1,
                            }
                        )
                        candidates.append(candidate)
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                errors.append("greedy: " + error_text(exc))
        for round_index in range(args.max_rounds):
            remaining = args.candidate_count - len(candidates)
            if remaining <= 0:
                break
            try:
                response = client.generate(
                    item["prompt"], remaining, args,
                    args.seed + record_index * 1009 + round_index,
                )
                for candidate in parse_choices(response):
                    signature = sql_signature(candidate["generated_sql"])
                    if not signature or signature in seen:
                        continue
                    seen.add(signature)
                    candidate["candidate_id"] = f"sample_{len(candidates)}"
                    candidate["llm_rank"] = len(candidates)
                    candidate["generation_mode"] = "sample"
                    candidate["generation_round"] = round_index
                    candidates.append(candidate)
                    if len(candidates) >= args.candidate_count:
                        break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                errors.append(error_text(exc))
                break
        row = {
            "record_index": record_index,
            "question_id": item.get("question_id"),
            "db_id": item.get("db_id"),
            "question": item.get("question"),
            "evidence": item.get("evidence"),
            "gold_sql": item.get("gold_sql"),
            "setting": item.get("setting"),
            "candidates": candidates,
            "generation_errors": errors,
            "generation_config": config,
        }
        append_row(args.output_file, row)
        print(f"[{position}/{len(prompts)}] index={record_index} unique={len(candidates)} errors={len(errors)}")
        if args.sleep:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
