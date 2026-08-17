"""Embed Stage 14B semantic slot requests with the schema embedding model."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embedding.stage8g_build_embedding_cache import (  # noqa: E402
    embed_texts,
    import_embedding_runtime,
    read_jsonl,
    resolve_dtype,
    write_json,
    write_jsonl,
)


def collect_slots(rows):
    texts, index_rows, text_rows = [], [], []
    for example_index, row in enumerate(rows):
        record_index = int(row["record_index"])
        for request in row["inference_inputs"]["requests"]:
            embedding_index = len(texts)
            text = request["slot_embedding_text"]
            texts.append(text)
            index_rows.append(
                {
                    "record_index": record_index,
                    "step_index": int(request["step_index"]),
                    "request_id": request["request_id"],
                    "embedding_index": embedding_index,
                    "action": request["action"],
                }
            )
            text_rows.append(
                {
                    "embedding_index": embedding_index,
                    "record_index": record_index,
                    "step_index": int(request["step_index"]),
                    "action": request["action"],
                    "text": text,
                }
            )
    return texts, index_rows, text_rows


def build_split(split, path, output_dir, tokenizer, model, runtime, args, limit=None):
    rows = read_jsonl(Path(path), limit)
    texts, index_rows, text_rows = collect_slots(rows)
    embeddings = embed_texts(texts, tokenizer, model, runtime, args)
    runtime["np"].save(output_dir / f"{split}_slot_embeddings.npy", embeddings)
    write_json(output_dir / f"{split}_slot_index.json", index_rows)
    if args.write_texts:
        write_jsonl(output_dir / f"{split}_slot_texts.jsonl", text_rows)
    return {
        "split": split,
        "record_count": len(rows),
        "slot_count": len(texts),
        "embedding_shape": list(embeddings.shape),
        "embedding_file": str(output_dir / f"{split}_slot_embeddings.npy"),
        "index_file": str(output_dir / f"{split}_slot_index.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-slots")
    parser.add_argument("--dev-slots")
    parser.add_argument("--splits", default="train,dev")
    parser.add_argument("--output-dir", default="experiments/stage14b_slot_embedding_cache")
    parser.add_argument("--model-path", default="/data/1_pretrained_models/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", choices=["last", "mean", "cls"], default="last")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--write-texts", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = import_embedding_runtime()
    torch = runtime["torch"]
    if args.device == "auto":
        args.resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        args.resolved_device = args.device
    dtype = resolve_dtype(torch, args.dtype)
    kwargs = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    tokenizer = runtime["AutoTokenizer"].from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    model = runtime["AutoModel"].from_pretrained(args.model_path, **kwargs).to(args.resolved_device)
    paths = {"train": args.train_slots, "dev": args.dev_slots}
    summaries = {}
    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        if not paths.get(split):
            raise ValueError(f"Missing --{split}-slots for requested split {split}")
        split_limit = getattr(args, f"{split}_limit")
        if split_limit is None:
            split_limit = args.limit
        summaries[split] = build_split(
            split, paths[split], output_dir, tokenizer, model, runtime, args, split_limit
        )
    summary = {
        "config": {
            **vars(args),
            "innovation": "Embed each semantic SQL slot independently instead of reusing one question embedding for every repeated action.",
        },
        "splits": summaries,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
