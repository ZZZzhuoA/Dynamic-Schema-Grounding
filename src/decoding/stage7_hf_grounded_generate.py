import argparse
import json
from pathlib import Path

import torch

from stage7_grounded_logits import GroundedSchemaLogitsProcessor, summarize_token_sequences
from stage7_schema_token_map import (
    apply_intervention,
    build_schema_token_sequences,
    items_from_prediction_record,
    items_from_selected_schema,
)


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
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return (
        "System: You are an expert SQLite SQL generator. Return only SQL.\n\n"
        f"User: {prompt}\n\nAssistant:"
    )


def key_for(record):
    return (record.get("question_id"), record.get("db_id"))


def load_grounding_records(path: Path | None):
    if path is None:
        return {}
    records = read_jsonl(path)
    keyed = {}
    for index, record in enumerate(records):
        keyed[key_for(record)] = record
        keyed[(index, record.get("db_id"))] = record
    return keyed


def get_grounding_record(item, index, grounding_by_key):
    return grounding_by_key.get(key_for(item)) or grounding_by_key.get((index, item.get("db_id")))


def build_items_for_prompt(item, index, grounding_by_key, top_k, intervention, seed):
    grounding = get_grounding_record(item, index, grounding_by_key)
    if grounding:
        items = items_from_prediction_record(grounding, top_k=top_k)
    else:
        items = items_from_selected_schema(item)
    if not items:
        items = items_from_selected_schema(item)
    return apply_intervention(items, mode=intervention, seed=seed + index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--grounding-file", default=None)
    parser.add_argument("--grounding-top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--boost", type=float, default=1.5)
    parser.add_argument("--first-token-boost-ratio", type=float, default=0.35)
    parser.add_argument("--max-bias-per-token", type=float, default=6.0)
    parser.add_argument("--max-schema-items", type=int, default=80)
    parser.add_argument("--min-grounding-score", type=float, default=0.0)
    parser.add_argument("--intervention", choices=["none", "zero", "random", "reverse"], default="none")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--debug-token-map", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    prompts = read_jsonl(Path(args.prompt_file), limit=args.limit, offset=args.offset)
    grounding_by_key = load_grounding_records(Path(args.grounding_file) if args.grounding_file else None)
    config = vars(args).copy()
    print(json.dumps({"stage7_generation_config": config, "prompt_count": len(prompts)}, ensure_ascii=False, indent=2))

    for index, item in enumerate(prompts):
        chat_prompt = build_chat_prompt(tokenizer, item["prompt"])
        encoded = tokenizer(chat_prompt, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_len = int(encoded["input_ids"].shape[1])

        grounded_items = build_items_for_prompt(
            item,
            index + args.offset,
            grounding_by_key,
            args.grounding_top_k,
            args.intervention,
            args.seed,
        )
        token_sequences = build_schema_token_sequences(
            tokenizer,
            grounded_items,
            max_items=args.max_schema_items,
            min_score=args.min_grounding_score,
        )
        processor = GroundedSchemaLogitsProcessor(
            [token_sequences],
            [prompt_len],
            boost=args.boost,
            first_token_boost_ratio=args.first_token_boost_ratio,
            max_bias_per_token=args.max_bias_per_token,
        )
        processors = LogitsProcessorList([processor])

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                logits_processor=processors,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0, prompt_len:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        output = {
            "question_id": item.get("question_id"),
            "db_id": item.get("db_id"),
            "setting": f"stage7_grounded_{item.get('setting')}",
            "question": item.get("question"),
            "evidence": item.get("evidence"),
            "prompt": item.get("prompt"),
            "raw_output": raw_output,
            "generated_sql": clean_sql(raw_output),
            "gold_sql": item.get("gold_sql"),
            "generation_config": config,
            "error": None,
            "grounding_control": {
                "item_count": len(grounded_items),
                "token_sequence_count": len(token_sequences),
                "intervention": args.intervention,
                "boost": args.boost,
                "processor_diagnostics": processor.diagnostics(),
                "top_sequences": summarize_token_sequences(token_sequences) if args.debug_token_map else [],
            },
        }
        append_jsonl(Path(args.output_file), [output])
        print(
            f"[{index + 1}/{len(prompts)}] {output['db_id']} "
            f"items={len(grounded_items)} seqs={len(token_sequences)} sql={bool(output['generated_sql'])}"
        )

    print(f"Outputs written to: {args.output_file}")


if __name__ == "__main__":
    main()
