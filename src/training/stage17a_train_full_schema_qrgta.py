"""Train the Stage 17-A0 full-schema binary QRGTA grounder."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


QUERY_TO_TABLE = "query_to_table"
QUERY_TO_COLUMN = "query_to_column"
TOP_K_VALUES = (10, 20, 30, 50)


def read_jsonl(path, limit=None):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    return records


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def import_runtime():
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("Stage 17-A0 requires numpy and PyTorch.") from exc
    from src.modeling.full_schema_qrgta import FullSchemaQRGTA, balanced_binary_loss

    return {
        "np": np,
        "torch": torch,
        "F": F,
        "model": FullSchemaQRGTA,
        "loss": balanced_binary_loss,
    }


def load_embedding_cache(cache_dir, split, runtime):
    np = runtime["np"]
    cache_dir = Path(cache_dir)
    query = np.load(cache_dir / f"{split}_query_embeddings.npy", mmap_mode="r")
    nodes = np.load(cache_dir / f"{split}_node_embeddings.npy", mmap_mode="r")
    rows = json.loads((cache_dir / f"{split}_index.json").read_text(encoding="utf-8"))
    by_record = {}
    by_example = {}
    for row in rows:
        example_index = int(row["example_index"])
        by_example[example_index] = row
        record_index = row.get("record_index")
        if record_index is not None:
            record_index = int(record_index)
            if record_index in by_record:
                raise ValueError(f"Duplicate cache record_index={record_index} in {split}")
            by_record[record_index] = row
    return {
        "query": query,
        "nodes": nodes,
        "by_record": by_record,
        "by_example": by_example,
        "dense_dim": int(query.shape[1]),
    }


def cache_row(cache, example_index, record_index):
    row = cache["by_record"].get(int(record_index))
    if row is None and not cache["by_record"]:
        row = cache["by_example"].get(int(example_index))
    if row is None:
        raise KeyError(
            f"Embedding cache has no row for record_index={record_index}, "
            f"example_index={example_index}"
        )
    return row


def node_embeddings(cache, row):
    if "node_embedding_indices" in row:
        indices = [int(value) for value in row["node_embedding_indices"]]
        return cache["nodes"][indices]
    start = int(row["node_embedding_start"])
    count = int(row["node_count"])
    return cache["nodes"][start : start + count]


def _key(db_id, question_id):
    if question_id is None:
        return None
    return str(db_id), str(question_id)


def align_graphs_and_labels(graph_records, label_records, split):
    labels_by_key = {}
    for label_index, label in enumerate(label_records):
        key = _key(label.get("db_id"), label.get("question_id"))
        if key is not None:
            if key in labels_by_key:
                raise ValueError(f"Duplicate label key in {split}: {key}")
            labels_by_key[key] = (label_index, label)

    aligned = []
    skipped_empty = []
    used_label_indices = set()
    for graph_index, graph in enumerate(graph_records):
        inputs = graph.get("inference_inputs", {})
        metadata = graph.get("metadata", {})
        record_index = metadata.get("record_index", graph_index)
        record_index = int(record_index)
        key = _key(inputs.get("db_id"), metadata.get("question_id"))
        pair = labels_by_key.get(key) if key is not None else None
        if pair is None:
            if not 0 <= record_index < len(label_records):
                raise ValueError(
                    f"No label row for {split} record_index={record_index}, key={key}"
                )
            pair = (record_index, label_records[record_index])
        label_index, label = pair
        if label_index in used_label_indices:
            raise ValueError(f"Label row {label_index} is aligned more than once in {split}")
        used_label_indices.add(label_index)
        if str(label.get("db_id")) != str(inputs.get("db_id")):
            raise ValueError(
                f"Database mismatch at {split} record_index={record_index}: "
                f"graph={inputs.get('db_id')} label={label.get('db_id')}"
            )
        graph_question = " ".join(str(inputs.get("question") or "").split())
        label_question = " ".join(str(label.get("question") or "").split())
        if graph_question != label_question:
            raise ValueError(
                f"Question mismatch at {split} record_index={record_index}: "
                f"graph={graph_question!r} label={label_question!r}"
            )
        nodes = inputs.get("schema_nodes", [])
        label_nodes = label.get("schema_items", [])
        if len(nodes) != len(label_nodes):
            raise ValueError(
                f"Schema length mismatch at {split} record_index={record_index}: "
                f"graph={len(nodes)} label={len(label_nodes)}"
            )
        for position, (node, label_node) in enumerate(zip(nodes, label_nodes)):
            if int(node.get("id", -1)) != int(label_node.get("id", -2)) or str(
                node.get("name")
            ) != str(label_node.get("name")):
                raise ValueError(
                    f"Schema identity mismatch at {split} record_index={record_index}, "
                    f"position={position}: graph=({node.get('id')}, {node.get('name')}) "
                    f"label=({label_node.get('id')}, {label_node.get('name')})"
                )
        gold_ids = sorted({int(value) for value in label.get("whole_sql_labels", [])})
        node_ids = {int(node["id"]) for node in nodes}
        unknown = sorted(set(gold_ids) - node_ids)
        if unknown:
            raise ValueError(
                f"Gold ids outside full schema at {split} record_index={record_index}: {unknown}"
            )
        item = {
            "split": split,
            "graph_index": graph_index,
            "record_index": record_index,
            "db_id": inputs.get("db_id"),
            "question_id": metadata.get("question_id"),
            "question": inputs.get("question"),
            "nodes": nodes,
            "schema_edges": inputs.get("schema_edges", []),
            "gold_ids": gold_ids,
        }
        if gold_ids:
            aligned.append(item)
        else:
            skipped_empty.append(
                {
                    "record_index": record_index,
                    "db_id": inputs.get("db_id"),
                    "question_id": metadata.get("question_id"),
                    "reason": "empty_gold_schema",
                }
            )
    return aligned, {
        "split": split,
        "graph_count": len(graph_records),
        "label_count": len(label_records),
        "usable_count": len(aligned),
        "skipped_empty_gold_count": len(skipped_empty),
        "skipped_examples": skipped_empty,
    }


def relation_mapping(train_examples, dev_examples):
    names = {
        str(edge["type"])
        for example in train_examples + dev_examples
        for edge in example.get("schema_edges", [])
    }
    names.update({QUERY_TO_TABLE, QUERY_TO_COLUMN})
    return {name: index for index, name in enumerate(sorted(names))}


def deterministic_permutation(count, seed, runtime):
    torch = runtime["torch"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(count, generator=generator)


def schema_edge_tensors(example, relations, control_mode, seed, runtime, device):
    torch = runtime["torch"]
    nodes = example["nodes"]
    id_to_local = {int(node["id"]): index for index, node in enumerate(nodes)}
    if len(id_to_local) != len(nodes):
        raise ValueError(f"Duplicate schema node ids at record_index={example['record_index']}")
    sources, destinations, types = [], [], []
    for edge in example.get("schema_edges", []):
        src = int(edge["src"])
        dst = int(edge["dst"])
        if src not in id_to_local or dst not in id_to_local:
            raise ValueError(
                f"Schema edge references unknown node at record_index={example['record_index']}"
            )
        sources.append(id_to_local[src])
        destinations.append(id_to_local[dst])
        types.append(relations[str(edge["type"])])
    if control_mode == "shuffled_schema_edges" and sources:
        grouped = defaultdict(list)
        self_loop_id = relations.get("self_loop")
        for index, relation_id in enumerate(types):
            if relation_id != self_loop_id:
                grouped[relation_id].append(index)
        for relation_id, edge_ids in grouped.items():
            permutation = deterministic_permutation(
                len(edge_ids), seed + example["record_index"] * 101 + relation_id, runtime
            ).tolist()
            old_destinations = [destinations[index] for index in edge_ids]
            for target_position, source_position in enumerate(permutation):
                destinations[edge_ids[target_position]] = old_destinations[source_position]
    edge_index = torch.tensor(
        [sources, destinations], dtype=torch.long, device=device
    )
    if not sources:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    edge_type = torch.tensor(types, dtype=torch.long, device=device)
    return edge_index, edge_type


def within_type_identity_permutation(nodes, seed, runtime):
    """Shuffle semantic identities only among nodes with the same schema type."""
    torch = runtime["torch"]
    permutation = torch.arange(len(nodes), dtype=torch.long)
    by_type = defaultdict(list)
    for index, node in enumerate(nodes):
        by_type[str(node.get("type"))].append(index)
    for offset, node_type in enumerate(sorted(by_type)):
        indices = by_type[node_type]
        local = deterministic_permutation(len(indices), seed + offset * 1009, runtime)
        source_indices = torch.tensor(indices, dtype=torch.long)[local]
        permutation[torch.tensor(indices, dtype=torch.long)] = source_indices
    return permutation


def example_to_tensors(example, cache, relations, args, runtime, device):
    torch = runtime["torch"]
    F = runtime["F"]
    row = cache_row(cache, example["graph_index"], example["record_index"])
    dense_array = node_embeddings(cache, row)
    if int(row["node_count"]) != len(example["nodes"]) or len(dense_array) != len(
        example["nodes"]
    ):
        raise ValueError(
            f"Embedding node count mismatch at record_index={example['record_index']}: "
            f"cache={len(dense_array)} graph={len(example['nodes'])}"
        )
    query_index = int(row["query_embedding_index"])
    dense = torch.tensor(dense_array, dtype=torch.float32, device=device)
    query = torch.tensor(
        cache["query"][query_index], dtype=torch.float32, device=device
    ).unsqueeze(0)
    node_types = torch.tensor(
        [0 if node.get("type") == "table" else 1 for node in example["nodes"]],
        dtype=torch.long,
        device=device,
    )
    if args.control_mode == "shuffled_node_identity":
        permutation = within_type_identity_permutation(
            example["nodes"],
            args.seed + example["record_index"] * 131,
            runtime,
        ).to(device)
        dense = dense[permutation]
    edge_index, edge_type = schema_edge_tensors(
        example, relations, args.control_mode, args.seed, runtime, device
    )
    if args.control_mode == "zero_query_edges":
        query_destination = torch.empty(0, dtype=torch.long, device=device)
        query_type = torch.empty(0, dtype=torch.long, device=device)
        query_similarity = torch.empty(0, dtype=torch.float32, device=device)
    else:
        query_destination = torch.arange(len(example["nodes"]), device=device)
        query_type = torch.tensor(
            [
                relations[
                    QUERY_TO_TABLE if node.get("type") == "table" else QUERY_TO_COLUMN
                ]
                for node in example["nodes"]
            ],
            dtype=torch.long,
            device=device,
        )
        query_similarity = F.cosine_similarity(
            dense, query.expand_as(dense), dim=-1
        ).to(torch.float32)
    id_to_local = {int(node["id"]): index for index, node in enumerate(example["nodes"])}
    labels = torch.zeros(len(example["nodes"]), dtype=torch.float32, device=device)
    labels[[id_to_local[value] for value in example["gold_ids"]]] = 1.0
    return {
        "dense_nodes": dense,
        "node_types": node_types,
        "query_embedding": query,
        "schema_edge_index": edge_index,
        "schema_edge_type": edge_type,
        "query_edge_destination": query_destination,
        "query_edge_type": query_type,
        "query_edge_similarity": query_similarity,
        "labels": labels,
    }


def forward_model(model, tensors):
    return model(
        tensors["dense_nodes"],
        tensors["node_types"],
        tensors["query_embedding"],
        tensors["schema_edge_index"],
        tensors["schema_edge_type"],
        tensors["query_edge_destination"],
        tensors["query_edge_type"],
        tensors["query_edge_similarity"],
    )


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def ranking_metrics(examples, rankings, losses=None, split="dev"):
    metrics = {"split": split, "sample_count": len(examples)}
    if losses is not None:
        metrics["loss"] = _mean(losses)
    by_database = defaultdict(list)
    by_size = defaultdict(list)
    accumulated = defaultdict(list)
    for example in examples:
        ranked_ids = rankings[int(example["record_index"])]
        gold = set(example["gold_ids"])
        node_by_id = {int(node["id"]): node for node in example["nodes"]}
        gold_tables = {item for item in gold if node_by_id[item].get("type") == "table"}
        gold_columns = gold - gold_tables
        first_rank = next(
            (rank for rank, item_id in enumerate(ranked_ids, start=1) if item_id in gold),
            None,
        )
        accumulated["mrr"].append(1.0 / first_rank if first_rank else 0.0)
        sample_metrics = {}
        for k in TOP_K_VALUES:
            selected = set(ranked_ids[:k])
            denominator = min(k, len(ranked_ids))
            recall = len(selected & gold) / len(gold)
            sample_metrics[f"schema_recall@{k}"] = recall
            sample_metrics[f"complete_coverage@{k}"] = float(gold.issubset(selected))
            accumulated[f"schema_recall@{k}"].append(recall)
            accumulated[f"complete_coverage@{k}"].append(float(gold.issubset(selected)))
            if k in {10, 20, 30}:
                accumulated[f"schema_precision@{k}"].append(
                    len(selected & gold) / max(denominator, 1)
                )
            if gold_tables:
                accumulated[f"table_recall@{k}"].append(
                    len(selected & gold_tables) / len(gold_tables)
                )
            if gold_columns:
                accumulated[f"column_recall@{k}"].append(
                    len(selected & gold_columns) / len(gold_columns)
                )
        by_database[str(example["db_id"])].append(sample_metrics)
        size = len(example["nodes"])
        bucket = "le_50" if size <= 50 else "51_100" if size <= 100 else "101_200" if size <= 200 else "gt_200"
        by_size[bucket].append(sample_metrics)
    metrics.update({key: _mean(values) for key, values in sorted(accumulated.items())})
    metrics["by_database"] = {
        key: {
            "sample_count": len(rows),
            "schema_recall@30": _mean([row["schema_recall@30"] for row in rows]),
            "complete_coverage@30": _mean([row["complete_coverage@30"] for row in rows]),
        }
        for key, rows in sorted(by_database.items())
    }
    metrics["by_schema_size"] = {
        key: {
            "sample_count": len(rows),
            "schema_recall@30": _mean([row["schema_recall@30"] for row in rows]),
            "complete_coverage@30": _mean([row["complete_coverage@30"] for row in rows]),
        }
        for key, rows in sorted(by_size.items())
    }
    return metrics


def evaluate(model, examples, cache, relations, args, runtime, device, split, predictions=False):
    torch = runtime["torch"]
    model.eval()
    losses = []
    rankings = {}
    rows = []
    with torch.no_grad():
        for example in examples:
            tensors = example_to_tensors(example, cache, relations, args, runtime, device)
            output = forward_model(model, tensors)
            losses.append(float(runtime["loss"](output["logits"], tensors["labels"]).cpu()))
            logits = output["logits"].detach().float().cpu().tolist()
            probabilities = output["probabilities"].detach().float().cpu().tolist()
            order = sorted(range(len(logits)), key=lambda index: (-logits[index], index))
            ranked_ids = [int(example["nodes"][index]["id"]) for index in order]
            rankings[int(example["record_index"])] = ranked_ids
            if predictions:
                ranked_schema = []
                for rank, local_index in enumerate(order, start=1):
                    node = example["nodes"][local_index]
                    ranked_schema.append(
                        {
                            "schema_item_id": int(node["id"]),
                            "name": node.get("name"),
                            "type": node.get("type"),
                            "logit": float(logits[local_index]),
                            "probability": float(probabilities[local_index]),
                            "rank": rank,
                        }
                    )
                rows.append(
                    {
                        "record_index": int(example["record_index"]),
                        "db_id": example.get("db_id"),
                        "question_id": example.get("question_id"),
                        "schema_node_count": len(example["nodes"]),
                        "ranked_schema": ranked_schema,
                        **{
                            f"top_{k}_ids": ranked_ids[:k]
                            for k in TOP_K_VALUES
                        },
                    }
                )
    return ranking_metrics(examples, rankings, losses, split), rows


def cosine_baseline(examples, cache, runtime):
    F = runtime["F"]
    torch = runtime["torch"]
    rankings = {}
    for example in examples:
        row = cache_row(cache, example["graph_index"], example["record_index"])
        dense = torch.tensor(node_embeddings(cache, row), dtype=torch.float32)
        query = torch.tensor(
            cache["query"][int(row["query_embedding_index"])], dtype=torch.float32
        ).unsqueeze(0)
        scores = F.cosine_similarity(dense, query.expand_as(dense), dim=-1).tolist()
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        rankings[int(example["record_index"])] = [
            int(example["nodes"][index]["id"]) for index in order
        ]
    return ranking_metrics(examples, rankings, split="cosine_baseline")


def train_epoch(model, examples, cache, relations, args, runtime, device, optimizer, epoch):
    model.train()
    shuffled = list(examples)
    random.Random(args.seed + epoch).shuffle(shuffled)
    accumulation = max(int(args.gradient_accumulation_steps), 1)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for position, example in enumerate(shuffled, start=1):
        tensors = example_to_tensors(example, cache, relations, args, runtime, device)
        output = forward_model(model, tensors)
        loss = runtime["loss"](output["logits"], tensors["labels"])
        (loss / accumulation).backward()
        losses.append(float(loss.detach().cpu()))
        if position % accumulation == 0 or position == len(shuffled):
            if args.max_grad_norm > 0:
                runtime["torch"].nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return {"train_loss": _mean(losses), "train_sample_count": len(shuffled)}


def cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def better_checkpoint(metrics, best_metrics, selection_metric):
    if best_metrics is None:
        return True
    current = (
        float(metrics[selection_metric]),
        float(metrics["schema_recall@30"]),
        -float(metrics["loss"]),
    )
    best = (
        float(best_metrics[selection_metric]),
        float(best_metrics["schema_recall@30"]),
        -float(best_metrics["loss"]),
    )
    return current > best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-graph-file", required=True)
    parser.add_argument("--dev-graph-file", required=True)
    parser.add_argument("--train-label-file", required=True)
    parser.add_argument("--dev-label-file", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-type", choices=["qrgta", "mlp", "mlp_residual"], default="qrgta"
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--selection-metric", default="complete_coverage@30")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--control-mode",
        choices=["normal", "zero_query_edges", "shuffled_schema_edges", "shuffled_node_identity"],
        default="normal",
    )
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runtime = import_runtime()
    torch = runtime["torch"]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if args.hidden_dim % args.num_heads:
        raise ValueError("--hidden-dim must be divisible by --num-heads")

    train_graphs = read_jsonl(args.train_graph_file, args.train_limit)
    dev_graphs = read_jsonl(args.dev_graph_file, args.dev_limit)
    train_labels = read_jsonl(args.train_label_file)
    dev_labels = read_jsonl(args.dev_label_file)
    train_examples, train_alignment = align_graphs_and_labels(
        train_graphs, train_labels, "train"
    )
    dev_examples, dev_alignment = align_graphs_and_labels(dev_graphs, dev_labels, "dev")
    train_cache = load_embedding_cache(args.embedding_cache_dir, "train", runtime)
    dev_cache = load_embedding_cache(args.embedding_cache_dir, "dev", runtime)
    if train_cache["dense_dim"] != dev_cache["dense_dim"]:
        raise ValueError("Train/dev embedding dimensions differ")
    relations = relation_mapping(train_examples, dev_examples)
    model_config = {
        "dense_dim": train_cache["dense_dim"],
        "relation_count": len(relations),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "model_type": args.model_type,
        "relations": relations,
        "control_mode": args.control_mode,
        "control_semantics": {
            "zero_query_edges": "Remove Query-to-Schema graph messages but retain the final query-conditioned scorer.",
            "shuffled_schema_edges": "Permute non-self edge destinations within relation type.",
            "shuffled_node_identity": "Permute dense semantic identities within table/column type.",
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = runtime["model"](
        dense_dim=model_config["dense_dim"],
        relation_count=model_config["relation_count"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        model_type=args.model_type,
    ).to(device)
    model_config["trainable_parameter_count"] = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    write_json(output_dir / "model_config.json", model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    cosine_metrics = cosine_baseline(dev_examples, dev_cache, runtime)
    history = []
    best_metrics = None
    best_state = None
    best_epoch = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_examples, train_cache, relations, args, runtime, device, optimizer, epoch
        )
        dev_metrics, _ = evaluate(
            model, dev_examples, dev_cache, relations, args, runtime, device, "dev"
        )
        if args.selection_metric not in dev_metrics:
            raise ValueError(
                f"Unknown selection metric {args.selection_metric!r}; "
                f"available={sorted(dev_metrics)}"
            )
        row = {"epoch": epoch, **train_metrics, **dev_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        write_jsonl(output_dir / "training_history.jsonl", history)
        if better_checkpoint(dev_metrics, best_metrics, args.selection_metric):
            best_metrics = dev_metrics
            best_state = cpu_state_dict(model)
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {"model_state_dict": best_state, "model_config": model_config, "epoch": epoch},
                output_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        torch.save(
            {"model_state_dict": cpu_state_dict(model), "model_config": model_config, "epoch": epoch},
            output_dir / "last.pt",
        )
        if args.patience > 0 and stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training completed without a checkpoint")
    model.load_state_dict(best_state)
    final_metrics, predictions = evaluate(
        model, dev_examples, dev_cache, relations, args, runtime, device, "dev", predictions=True
    )
    final_metrics.update(
        {
            "best_epoch": best_epoch,
            "selection_metric": args.selection_metric,
            "selection_value": final_metrics[args.selection_metric],
        }
    )
    write_json(output_dir / "best_metrics.json", final_metrics)
    write_jsonl(output_dir / "dev_predictions.jsonl", predictions)
    summary = {
        "best_epoch": best_epoch,
        "last_epoch": history[-1]["epoch"],
        "stopped_early": history[-1]["epoch"] < args.epochs,
        "selection_metric": args.selection_metric,
        "selection_value": final_metrics[args.selection_metric],
        "model_config": model_config,
        "train_alignment": train_alignment,
        "dev_alignment": dev_alignment,
        "cosine_baseline_metrics": cosine_metrics,
        "dev_metrics": final_metrics,
        "config": vars(args),
        "leakage_note": "Gold schema labels are used only for training loss, checkpoint selection, and evaluation; prediction rows contain no gold labels.",
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
