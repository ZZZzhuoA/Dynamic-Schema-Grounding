import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def read_jsonl(path: Path, limit=None, offset=0):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index < offset:
                continue
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def append_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_sql(text):
    if not text:
        return ""
    text = text.strip()
    if "```" in text:
        chunks = text.split("```")
        sql_chunks = []
        for chunk in chunks:
            cleaned = chunk.strip()
            if cleaned.lower().startswith("sql"):
                cleaned = cleaned[3:].strip()
            if "select" in cleaned.lower() or "with" in cleaned.lower():
                sql_chunks.append(cleaned)
        if sql_chunks:
            text = sql_chunks[0]
    lowered = text.lower()
    starts = [pos for token in ["select", "with"] if (pos := lowered.find(token)) >= 0]
    if starts:
        text = text[min(starts):]
    # Drop common trailing explanation markers.
    for marker in ["\n\n", "\nExplanation:", "\nNote:", "\nThe SQL"]:
        marker_pos = text.find(marker)
        if marker_pos > 0:
            text = text[:marker_pos]
    return text.strip().rstrip(";") + ";" if text.strip() else ""


class OpenAICompatibleClient:
    def __init__(self, base_url, api_key, model, timeout):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, prompt, temperature, top_p, max_tokens, enable_thinking):
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert SQLite SQL generator. Return only SQL.",
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
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read().decode("utf-8")
            result = json.loads(body)
        return result["choices"][0]["message"]["content"], result


def load_done_keys(path: Path):
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            done.add((item.get("setting"), item.get("question_id"), item.get("db_id")))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3-32b"))
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen thinking mode. Default is disabled for DashScope non-streaming compatibility.",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Set environment variable: {args.api_key_env}")

    prompt_file = Path(args.prompt_file)
    output_file = Path(args.output_file)
    prompts = read_jsonl(prompt_file, limit=args.limit, offset=args.offset)
    done_keys = load_done_keys(output_file) if args.resume else set()
    client = OpenAICompatibleClient(args.base_url, api_key, args.model, args.timeout)

    config = {
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "prompt_file": str(prompt_file),
        "output_file": str(output_file),
        "limit": args.limit,
        "offset": args.offset,
    }

    print(json.dumps({"generation_config": config, "prompt_count": len(prompts)}, ensure_ascii=False, indent=2))
    for idx, item in enumerate(prompts, start=1):
        key = (item.get("setting"), item.get("question_id"), item.get("db_id"))
        if key in done_keys:
            continue
        try:
            raw_output, raw_response = client.chat(
                item["prompt"],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                enable_thinking=args.enable_thinking,
            )
            output = {
                "question_id": item.get("question_id"),
                "db_id": item.get("db_id"),
                "setting": item.get("setting"),
                "question": item.get("question"),
                "evidence": item.get("evidence"),
                "prompt": item.get("prompt"),
                "raw_output": raw_output,
                "generated_sql": clean_sql(raw_output),
                "gold_sql": item.get("gold_sql"),
                "generation_config": config,
                "error": None,
            }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            error_text = repr(exc)
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                    error_text = f"{repr(exc)} body={error_body}"
                except Exception as body_exc:
                    error_text = f"{repr(exc)} body_read_error={repr(body_exc)}"
            output = {
                "question_id": item.get("question_id"),
                "db_id": item.get("db_id"),
                "setting": item.get("setting"),
                "question": item.get("question"),
                "evidence": item.get("evidence"),
                "prompt": item.get("prompt"),
                "raw_output": "",
                "generated_sql": "",
                "gold_sql": item.get("gold_sql"),
                "generation_config": config,
                "error": error_text,
            }
        append_jsonl(output_file, [output])
        print(f"[{idx}/{len(prompts)}] {output['setting']} {output['db_id']} error={output['error'] is not None}")
        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
