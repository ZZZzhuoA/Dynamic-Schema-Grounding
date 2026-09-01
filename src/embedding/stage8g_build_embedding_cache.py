import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def compact_list(values, limit=12):
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    return [str(value).strip() for value in values if str(value).strip()][:limit]


def truncated_text(value, char_limit):
    return str(value or "").strip()[:char_limit]


def node_embedding_text(node):
    """Build the text actually sent to the dense embedding model.

    The text intentionally keeps structured field names.  Qwen/BGE-style
    embedding models usually benefit from explicit facets, and this also makes
    the cache auditable.
    """
    parts = [
        f"schema item: {node.get('name', '')}",
        f"type: {node.get('type', '')}",
    ]
    if node.get("type") == "column":
        parts.extend(
            [
                f"table: {node.get('table', '')}",
                f"column: {node.get('column', '')}",
                f"data type: {node.get('data_type', '')}",
            ]
        )
        if node.get("is_primary_key"):
            parts.append("primary key: yes")
        if node.get("is_foreign_key_endpoint"):
            parts.append("foreign key endpoint: yes")
        outgoing = compact_list(node.get("foreign_key_outgoing_targets"), 8)
        if outgoing:
            parts.append("foreign key outgoing targets: " + "; ".join(outgoing))
        incoming = compact_list(node.get("foreign_key_incoming_sources"), 8)
        if incoming:
            parts.append("foreign key incoming sources: " + "; ".join(incoming))
        official_name = truncated_text(node.get("official_column_name"), 160)
        if official_name:
            parts.append(f"official column name: {official_name}")
        official_description = truncated_text(
            node.get("official_column_description"), 320
        )
        if official_description:
            parts.append(f"official description: {official_description}")
        official_format = str(node.get("official_data_format") or "").strip()
        if official_format:
            parts.append(f"official data format: {official_format}")
        official_value_description = truncated_text(
            node.get("official_value_description"), 384
        )
        if official_value_description:
            parts.append(
                f"official value description: {official_value_description}"
            )
    for key, label in [
        ("semantic_name", "semantic name"),
        ("semantic_description", "description"),
        ("value_type", "value type"),
    ]:
        value = node.get(key)
        if value:
            parts.append(f"{label}: {value}")
    aliases = compact_list(node.get("semantic_aliases"), 12)
    if aliases:
        parts.append("aliases: " + "; ".join(aliases))
    roles = compact_list(node.get("likely_sql_roles"), 10)
    if roles:
        parts.append("likely SQL roles: " + "; ".join(roles))
    relations = compact_list(node.get("relation_types"), 10)
    if relations:
        parts.append("relation types: " + "; ".join(relations))
    semantic_text = node.get("semantic_text")
    if semantic_text:
        parts.append(f"semantic card: {semantic_text}")
    return " | ".join(str(part) for part in parts if part)


def query_embedding_text(inference_inputs):
    parts = [
        f"question: {inference_inputs.get('question') or ''}",
    ]
    if inference_inputs.get("evidence"):
        parts.append(f"evidence: {inference_inputs.get('evidence')}")
    if inference_inputs.get("question_semantic_text"):
        parts.append(f"question semantic card: {inference_inputs.get('question_semantic_text')}")
    question_card = inference_inputs.get("question_card") or {}
    if isinstance(question_card, dict):
        if question_card.get("intent"):
            parts.append(f"intent: {question_card.get('intent')}")
        operation_hints = compact_list(question_card.get("operation_hints"), 12)
        if operation_hints:
            parts.append("operation hints: " + "; ".join(operation_hints))
        value_hints = compact_list(question_card.get("value_hints"), 20)
        if value_hints:
            parts.append("value hints: " + "; ".join(value_hints))
        formula_hints = compact_list(question_card.get("formula_hints"), 12)
        if formula_hints:
            parts.append("formula hints: " + "; ".join(formula_hints))
        ordering_hints = compact_list(question_card.get("ordering_hints"), 8)
        if ordering_hints:
            parts.append("ordering hints: " + "; ".join(ordering_hints))
    return " | ".join(str(part) for part in parts if part)


def import_embedding_runtime():
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Stage 8G-A requires numpy, torch, and transformers. "
            "Install them in the server environment or run inside the model environment."
        ) from exc
    return {
        "np": np,
        "torch": torch,
        "F": F,
        "AutoModel": AutoModel,
        "AutoTokenizer": AutoTokenizer,
    }


def resolve_dtype(torch, dtype_text):
    if dtype_text == "auto":
        return None
    if dtype_text == "float16":
        return torch.float16
    if dtype_text == "bfloat16":
        return torch.bfloat16
    if dtype_text == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_text}")


def last_token_pool(last_hidden_state, attention_mask):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = last_hidden_state.new_tensor(range(last_hidden_state.shape[0]), dtype=sequence_lengths.dtype)
    return last_hidden_state[batch_indices.long(), sequence_lengths.long()]


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-8)


def embed_texts(texts, tokenizer, model, runtime, args):
    np = runtime["np"]
    torch = runtime["torch"]
    F = runtime["F"]
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start : start + args.batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(args.resolved_device) for key, value in encoded.items()}
            output = model(**encoded)
            hidden = output.last_hidden_state
            if args.pooling == "mean":
                pooled = mean_pool(hidden, encoded["attention_mask"])
            elif args.pooling == "cls":
                pooled = hidden[:, 0]
            elif args.pooling == "last":
                pooled = last_token_pool(hidden, encoded["attention_mask"])
            else:
                raise ValueError(f"Unsupported pooling: {args.pooling}")
            if args.normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.detach().float().cpu().numpy())
            print(
                json.dumps(
                    {
                        "event": "embedded_batch",
                        "start": start,
                        "end": min(start + len(batch_texts), len(texts)),
                        "total": len(texts),
                    },
                    ensure_ascii=False,
                )
            )
    if not all_embeddings:
        return np.empty((0, 0), dtype="float32")
    return np.concatenate(all_embeddings, axis=0).astype("float32")


def collect_split_texts(examples, deduplicate_node_texts=False):
    query_texts = []
    node_texts = []
    index_rows = []
    text_rows = []
    node_cursor = 0
    node_text_to_index = {}
    node_occurrence_count = 0
    for example_index, example in enumerate(examples):
        inputs = example["inference_inputs"]
        query_text = query_embedding_text(inputs)
        query_texts.append(query_text)
        nodes = inputs.get("schema_nodes", [])
        start = node_cursor
        node_embedding_indices = []
        for node in nodes:
            text = node_embedding_text(node)
            node_occurrence_count += 1
            if deduplicate_node_texts:
                embedding_index = node_text_to_index.get(text)
                if embedding_index is None:
                    embedding_index = len(node_texts)
                    node_text_to_index[text] = embedding_index
                    node_texts.append(text)
                    text_rows.append(
                        {
                            "embedding_index": embedding_index,
                            "first_example_index": example_index,
                            "first_example_id": example.get("example_id"),
                            "first_db_id": inputs.get("db_id"),
                            "first_node_id": node.get("id"),
                            "node_name": node.get("name"),
                            "node_type": node.get("type"),
                            "text": text,
                        }
                    )
                node_embedding_indices.append(embedding_index)
            else:
                node_texts.append(text)
                text_rows.append(
                    {
                        "embedding_index": node_cursor,
                        "example_index": example_index,
                        "example_id": example.get("example_id"),
                        "db_id": inputs.get("db_id"),
                        "node_id": node.get("id"),
                        "node_name": node.get("name"),
                        "node_type": node.get("type"),
                        "text": text,
                    }
                )
                node_cursor += 1
        index_row = {
            "example_index": example_index,
            "example_id": example.get("example_id"),
            "record_index": example.get("metadata", {}).get("record_index"),
            "db_id": inputs.get("db_id"),
            "question_id": example.get("metadata", {}).get("question_id"),
            "query_embedding_index": example_index,
            "node_count": len(nodes),
        }
        if deduplicate_node_texts:
            index_row["node_embedding_indices"] = node_embedding_indices
        else:
            index_row["node_embedding_start"] = start
        index_rows.append(index_row)
    return query_texts, node_texts, index_rows, text_rows, node_occurrence_count


def build_split_cache(split, example_path, output_dir, tokenizer, model, runtime, args):
    np = runtime["np"]
    examples = read_jsonl(Path(example_path), args.limit)
    query_texts, node_texts, index_rows, text_rows, node_occurrence_count = collect_split_texts(
        examples, deduplicate_node_texts=args.deduplicate_node_texts
    )
    print(
        json.dumps(
            {
                "event": "build_split_start",
                "split": split,
                "example_count": len(examples),
                "query_text_count": len(query_texts),
                "node_text_count": len(node_texts),
                "node_occurrence_count": node_occurrence_count,
                "deduplicate_node_texts": args.deduplicate_node_texts,
            },
            ensure_ascii=False,
        )
    )
    query_embeddings = embed_texts(query_texts, tokenizer, model, runtime, args)
    node_embeddings = embed_texts(node_texts, tokenizer, model, runtime, args)
    np.save(output_dir / f"{split}_query_embeddings.npy", query_embeddings)
    np.save(output_dir / f"{split}_node_embeddings.npy", node_embeddings)
    write_json(output_dir / f"{split}_index.json", index_rows)
    if args.write_texts:
        write_jsonl(output_dir / f"{split}_embedding_texts.jsonl", text_rows)
        write_jsonl(
            output_dir / f"{split}_query_texts.jsonl",
            [
                {
                    "example_index": index,
                    "example_id": examples[index].get("example_id"),
                    "text": text,
                }
                for index, text in enumerate(query_texts)
            ],
        )
    return {
        "split": split,
        "example_count": len(examples),
        "query_embedding_shape": list(query_embeddings.shape),
        "node_embedding_shape": list(node_embeddings.shape),
        "node_occurrence_count": node_occurrence_count,
        "unique_node_text_count": len(node_texts),
        "node_deduplication_ratio": (
            1.0 - len(node_texts) / node_occurrence_count if node_occurrence_count else 0.0
        ),
        "index_file": str(output_dir / f"{split}_index.json"),
        "query_embedding_file": str(output_dir / f"{split}_query_embeddings.npy"),
        "node_embedding_file": str(output_dir / f"{split}_node_embeddings.npy"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-examples", default=None)
    parser.add_argument("--dev-examples", default=None)
    parser.add_argument("--splits", default="train,dev")
    parser.add_argument("--output-dir", default="experiments/stage8g_embedding_cache")
    parser.add_argument("--model-path", default="/data/1_pretrained_models/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", choices=["last", "mean", "cls"], default="last")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write-texts", action="store_true")
    parser.add_argument(
        "--deduplicate-node-texts",
        action="store_true",
        help="Embed each distinct schema-node text once and store per-example embedding indices.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = import_embedding_runtime()
    torch = runtime["torch"]
    AutoTokenizer = runtime["AutoTokenizer"]
    AutoModel = runtime["AutoModel"]
    if args.device == "auto":
        args.resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        args.resolved_device = args.device
    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    dtype = resolve_dtype(torch, args.dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(args.model_path, **model_kwargs).to(args.resolved_device)

    split_to_path = {"train": args.train_examples, "dev": args.dev_examples}
    summaries = {}
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        example_path = split_to_path.get(split)
        if not example_path:
            raise ValueError(f"Missing --{split}-examples for requested split {split!r}")
        summaries[split] = build_split_cache(split, example_path, output_dir, tokenizer, model, runtime, args)
    summary = {
        "config": {
            "model_path": args.model_path,
            "output_dir": str(output_dir),
            "splits": args.splits,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "pooling": args.pooling,
            "normalize": args.normalize,
            "device": args.device,
            "resolved_device": args.resolved_device,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
            "limit": args.limit,
            "deduplicate_node_texts": args.deduplicate_node_texts,
        },
        "splits": summaries,
        "note": (
            "Stage 8G-A only builds dense embedding caches from test-time available "
            "question/evidence/schema semantic-card fields. Gold SQL/schema labels are not embedded."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
