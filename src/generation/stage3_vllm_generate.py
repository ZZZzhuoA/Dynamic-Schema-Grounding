import argparse
import json
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


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
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
    for marker in ["\n\n", "\nExplanation:", "\nNote:", "\nThe SQL"]:
        marker_pos = text.find(marker)
        if marker_pos > 0:
            text = text[:marker_pos]
    return text.strip().rstrip(";") + ";" if text.strip() else ""


def build_chat_prompt(tokenizer, prompt):
    messages = [
        {
            "role": "system",
            "content": "You are an expert SQLite SQL generator. Return only SQL.",
        },
        {"role": "user", "content": prompt},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        "System: You are an expert SQLite SQL generator. Return only SQL.\n\n"
        f"User: {prompt}\n\nAssistant:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed in this environment. Install vLLM on the server first."
        ) from exc

    prompts = read_jsonl(Path(args.prompt_file), limit=args.limit, offset=args.offset)
    config = vars(args).copy()
    print(json.dumps({"vllm_generation_config": config, "prompt_count": len(prompts)}, ensure_ascii=False, indent=2))

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    outputs = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        batch_prompts = [build_chat_prompt(tokenizer, item["prompt"]) for item in batch]
        batch_outputs = llm.generate(batch_prompts, sampling_params)
        for item, result in zip(batch, batch_outputs):
            raw_output = result.outputs[0].text if result.outputs else ""
            outputs.append(
                {
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
            )
        write_jsonl(Path(args.output_file), outputs)
        print(f"generated {min(start + len(batch), len(prompts))}/{len(prompts)}")

    print(f"Outputs written to: {args.output_file}")


if __name__ == "__main__":
    main()
