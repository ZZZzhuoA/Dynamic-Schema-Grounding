import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grounding.stage10_constrained_selector import constrained_topk  # noqa: E402
from src.training.stage5_train_dsg_grounder import (  # noqa: E402
    clone_state_dict_to_cpu,
    get_metric,
    is_better_metric,
    read_jsonl,
    write_json,
    write_jsonl,
)


def import_runtime():
    try:
        import numpy as np
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("Stage 10-A requires numpy and PyTorch.") from exc
    from src.modeling.factor_graph_reranker import HeterogeneousFactorGraphReranker

    return {
        "np": np,
        "torch": torch,
        "nn": nn,
        "model": HeterogeneousFactorGraphReranker,
    }


def load_cache(cache_dir, split, runtime):
    np = runtime["np"]
    cache_dir = Path(cache_dir)
    query = np.load(cache_dir / f"{split}_query_embeddings.npy", mmap_mode="r")
    node = np.load(cache_dir / f"{split}_node_embeddings.npy", mmap_mode="r")
    index_rows = json.loads(
        (cache_dir / f"{split}_index.json").read_text(encoding="utf-8")
    )
    return {
        "query": query,
        "node": node,
        "index": {int(row["example_index"]): row for row in index_rows},
        "dense_dim": int(query.shape[1]),
    }


def full_node_embeddings(cache, record_index):
    row = cache["index"].get(int(record_index))
    if row is None:
        raise KeyError(f"Embedding cache has no record_index={record_index}")
    if "node_embedding_indices" in row:
        return cache["node"][[int(index) for index in row["node_embedding_indices"]]]
    start = int(row["node_embedding_start"])
    count = int(row["node_count"])
    return cache["node"][start : start + count]


def collect_schema_relations(examples):
    return sorted(
        {
            edge["type"]
            for example in examples
            for edge in example.get("schema_edges", [])
        }
    )


def example_to_tensors(example, cache, maps, runtime, device):
    torch = runtime["torch"]
    full_nodes = full_node_embeddings(cache, example["record_index"])
    positions = [int(node["schema_position"]) for node in example["candidate_nodes"]]
    dense_nodes = torch.tensor(full_nodes[positions], dtype=torch.float32, device=device)
    query_row = cache["index"][int(example["record_index"])]
    query_index = int(query_row["query_embedding_index"])
    query_embedding = torch.tensor(
        cache["query"][query_index : query_index + 1],
        dtype=torch.float32,
        device=device,
    )
    numeric = torch.tensor(
        [node["numeric_features"] for node in example["candidate_nodes"]],
        dtype=torch.float32,
        device=device,
    )
    schema_pairs = [
        (int(edge["src"]), int(edge["dst"]))
        for edge in example.get("schema_edges", [])
        if edge["type"] in maps["schema_relation_to_id"]
    ]
    schema_types = [
        maps["schema_relation_to_id"][edge["type"]]
        for edge in example.get("schema_edges", [])
        if edge["type"] in maps["schema_relation_to_id"]
    ]
    factor_pairs = [
        (int(edge["schema"]), int(edge["factor"]))
        for edge in example.get("factor_edges", [])
    ]
    factor_types = [int(edge["type_id"]) for edge in example.get("factor_edges", [])]
    factor_weights = [float(edge.get("weight", 1.0)) for edge in example.get("factor_edges", [])]

    def edge_index(pairs):
        if not pairs:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()

    factors = example.get("factors", [])
    return {
        "dense_nodes": dense_nodes,
        "numeric_features": numeric,
        "query_embedding": query_embedding,
        "factor_kind": torch.tensor(
            [int(factor["kind_id"]) for factor in factors],
            dtype=torch.long,
            device=device,
        ),
        "factor_relation": torch.tensor(
            [int(factor.get("relation_id", -1)) for factor in factors],
            dtype=torch.long,
            device=device,
        ),
        "factor_numeric": torch.tensor(
            [factor["numeric_features"] for factor in factors],
            dtype=torch.float32,
            device=device,
        ).reshape(len(factors), maps["factor_numeric_dim"]),
        "schema_edge_index": edge_index(schema_pairs),
        "schema_edge_type": torch.tensor(schema_types, dtype=torch.long, device=device),
        "factor_edge_index": edge_index(factor_pairs),
        "factor_edge_type": torch.tensor(factor_types, dtype=torch.long, device=device),
        "factor_edge_weight": torch.tensor(factor_weights, dtype=torch.float32, device=device),
        "whole_labels": torch.tensor(example["whole_labels"], dtype=torch.float32, device=device),
        "role_labels": torch.tensor(example["role_labels"], dtype=torch.float32, device=device),
    }


def forward_model(model, tensors):
    return model(
        tensors["dense_nodes"],
        tensors["numeric_features"],
        tensors["query_embedding"],
        tensors["factor_kind"],
        tensors["factor_relation"],
        tensors["factor_numeric"],
        tensors["schema_edge_index"],
        tensors["schema_edge_type"],
        tensors["factor_edge_index"],
        tensors["factor_edge_type"],
        tensors["factor_edge_weight"],
    )


def reranker_loss(output, tensors, args, runtime):
    torch = runtime["torch"]
    labels = tensors["whole_labels"]
    logits = output["logits"]
    positive_weight = torch.tensor(args.pos_weight, dtype=logits.dtype, device=logits.device)
    node_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=positive_weight
    )
    role_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["role_logits"], tensors["role_labels"], pos_weight=positive_weight
    )
    positive = logits[labels > 0.5]
    negative = logits[labels <= 0.5]
    if positive.numel() and negative.numel():
        hard_negative = negative.topk(min(args.hard_negative_k, negative.numel())).values
        pair_loss = torch.relu(
            args.pairwise_margin - positive.unsqueeze(1) + hard_negative.unsqueeze(0)
        ).mean()
    else:
        pair_loss = logits.new_zeros(())
    total = node_loss + args.role_loss_weight * role_loss + args.pairwise_loss_weight * pair_loss
    return total, {
        "node_loss": float(node_loss.detach().cpu()),
        "role_loss": float(role_loss.detach().cpu()),
        "pairwise_loss": float(pair_loss.detach().cpu()),
    }


def selection_metrics(examples, selected_by_index, top_k, prefix):
    recalls = []
    precisions = []
    table_recalls = []
    column_recalls = []
    complete = 0
    for example in examples:
        selected = set(selected_by_index[int(example["record_index"])])
        gold = set(int(item_id) for item_id in example.get("gold_ids", []))
        recalls.append(len(selected & gold) / len(gold) if gold else 1.0)
        precisions.append(len(selected & gold) / max(min(top_k, len(selected)), 1))
        complete += int(gold.issubset(selected))
        gold_tables = set(int(item_id) for item_id in example.get("gold_table_ids", []))
        gold_columns = set(int(item_id) for item_id in example.get("gold_column_ids", []))
        if gold_tables:
            table_recalls.append(len(selected & gold_tables) / len(gold_tables))
        if gold_columns:
            column_recalls.append(len(selected & gold_columns) / len(gold_columns))
    count = len(examples)
    return {
        f"{prefix}_schema_recall@{top_k}": sum(recalls) / count if count else 0.0,
        f"{prefix}_schema_precision@{top_k}": sum(precisions) / count if count else 0.0,
        f"{prefix}_complete_samples@{top_k}": complete,
        f"{prefix}_complete_coverage@{top_k}": complete / count if count else 0.0,
        f"{prefix}_table_recall@{top_k}": sum(table_recalls) / len(table_recalls) if table_recalls else 0.0,
        f"{prefix}_column_recall@{top_k}": sum(column_recalls) / len(column_recalls) if column_recalls else 0.0,
    }


def evaluate(model, examples, cache, maps, args, runtime, device, split):
    torch = runtime["torch"]
    model.eval()
    losses = []
    selected_raw = {}
    selected_constrained = {}
    selected_baseline = {}
    predictions = []
    with torch.no_grad():
        for example in examples:
            tensors = example_to_tensors(example, cache, maps, runtime, device)
            output = forward_model(model, tensors)
            loss, _ = reranker_loss(output, tensors, args, runtime)
            losses.append(float(loss.detach().cpu()))
            scores = output["logits"].detach().cpu().tolist()
            raw_local = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[: args.output_top_k]
            constrained_local, selector_debug = constrained_topk(
                example,
                scores,
                top_k=args.output_top_k,
                max_tables=args.max_tables,
                min_tables=None if args.min_tables < 0 else args.min_tables,
                connectivity_weight=args.connectivity_weight,
                baseline_retention_weight=args.baseline_retention_weight,
            )
            local_to_schema = {
                int(node["local_id"]): int(node["schema_item_id"])
                for node in example["candidate_nodes"]
            }
            index = int(example["record_index"])
            selected_raw[index] = [local_to_schema[local] for local in raw_local]
            selected_constrained[index] = [local_to_schema[local] for local in constrained_local]
            selected_baseline[index] = list(example.get("baseline_selected_ids", []))[: args.output_top_k]
            predictions.append(
                {
                    "record_index": index,
                    "db_id": example.get("db_id"),
                    "question_id": example.get("question_id"),
                    "question": example.get("question"),
                    "gold_ids": example.get("gold_ids", []),
                    "candidate_oracle_recall": example.get("candidate_oracle_recall"),
                    f"top_{args.output_top_k}": [
                        {
                            **example["candidate_nodes"][local],
                            "score": float(scores[local]),
                        }
                        for local in constrained_local
                    ],
                    "raw_top_ids": selected_raw[index],
                    "baseline_top_ids": selected_baseline[index],
                    "selector_debug": selector_debug,
                }
            )
    metrics = {"split": split, "sample_count": len(examples), "loss": sum(losses) / len(losses) if losses else 0.0}
    metrics.update(selection_metrics(examples, selected_baseline, args.output_top_k, "baseline"))
    metrics.update(selection_metrics(examples, selected_raw, args.output_top_k, "reranker_raw"))
    metrics.update(selection_metrics(examples, selected_constrained, args.output_top_k, "constrained"))
    metrics["candidate_oracle_recall"] = sum(
        float(example.get("candidate_oracle_recall", 0.0)) for example in examples
    ) / len(examples) if examples else 0.0
    metrics["candidate_complete_coverage"] = sum(
        float(example.get("candidate_oracle_recall", 0.0)) >= 1.0 for example in examples
    ) / len(examples) if examples else 0.0
    return metrics, predictions


def train_epoch(model, examples, cache, maps, args, runtime, device, optimizer, epoch):
    model.train()
    shuffled = list(examples)
    random.Random(args.seed + epoch).shuffle(shuffled)
    total = 0.0
    components = {"node_loss": 0.0, "role_loss": 0.0, "pairwise_loss": 0.0}
    for example in shuffled:
        tensors = example_to_tensors(example, cache, maps, runtime, device)
        output = forward_model(model, tensors)
        loss, detail = reranker_loss(output, tensors, args, runtime)
        optimizer.zero_grad()
        loss.backward()
        runtime["torch"].nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        total += float(loss.detach().cpu())
        for key in components:
            components[key] += detail[key]
    count = max(len(shuffled), 1)
    return {"loss": total / count, **{key: value / count for key, value in components.items()}}


def main():
    parser = argparse.ArgumentParser(description="Train Stage 10-A factor-graph reranker.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--model-type", choices=["mlp", "schema_rgta", "factor_rgta"], default="factor_rgta")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.5)
    parser.add_argument("--role-loss-weight", type=float, default=0.3)
    parser.add_argument("--pairwise-margin", type=float, default=0.5)
    parser.add_argument("--hard-negative-k", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output-top-k", type=int, default=30)
    parser.add_argument("--max-tables", type=int, default=8)
    parser.add_argument("--min-tables", type=int, default=-1)
    parser.add_argument("--connectivity-weight", type=float, default=0.10)
    parser.add_argument("--baseline-retention-weight", type=float, default=0.05)
    parser.add_argument("--selection-metric", default="constrained_complete_coverage@30")
    parser.add_argument("--selection-mode", choices=["max", "min"], default="max")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-save-model", action="store_true")
    args = parser.parse_args()

    runtime = import_runtime()
    torch = runtime["torch"]
    random.seed(args.seed)
    runtime["np"].random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_examples = read_jsonl(Path(args.train_file), args.train_limit)
    dev_examples = read_jsonl(Path(args.dev_file), args.dev_limit)
    if not train_examples or not dev_examples:
        raise ValueError("Train/dev factor graph files must be non-empty")
    train_cache = load_cache(args.embedding_cache_dir, "train", runtime)
    dev_cache = load_cache(args.embedding_cache_dir, "dev", runtime)
    schema_relations = collect_schema_relations(train_examples + dev_examples)
    relation_count = len(train_examples[0]["role_labels"][0])
    maps = {
        "schema_relation_to_id": {name: index for index, name in enumerate(schema_relations)},
        "factor_numeric_dim": len(train_examples[0]["factors"][0]["numeric_features"]) if train_examples[0]["factors"] else 3,
    }
    numeric_dim = len(train_examples[0]["candidate_nodes"][0]["numeric_features"])
    model = runtime["model"](
        dense_dim=train_cache["dense_dim"],
        numeric_dim=numeric_dim,
        factor_numeric_dim=maps["factor_numeric_dim"],
        relation_count=relation_count,
        schema_relation_count=max(len(schema_relations), 1),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "dense_dim": train_cache["dense_dim"],
            "numeric_dim": numeric_dim,
            "factor_numeric_dim": maps["factor_numeric_dim"],
            "relation_count": relation_count,
            "schema_relations": schema_relations,
            "innovation": (
                "A query-conditioned heterogeneous factor graph jointly calibrates relation, "
                "database-value, and join-path evidence before typed owner-closed subset decoding."
            ),
        }
    )
    write_json(output_dir / "train_config.json", config)

    best_epoch = None
    best_value = None
    best_state = None
    best_metrics = None
    epochs_without_improvement = 0
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_examples, train_cache, maps, args, runtime, device, optimizer, epoch
        )
        dev_metrics, _ = evaluate(
            model, dev_examples, dev_cache, maps, args, runtime, device, "dev"
        )
        selected_value = get_metric(dev_metrics, args.selection_metric)
        improved = is_better_metric(
            selected_value, best_value, args.selection_mode, args.min_delta
        )
        if improved:
            best_epoch = epoch
            best_value = selected_value
            best_state = clone_state_dict_to_cpu(model)
            best_metrics = dev_metrics
            epochs_without_improvement = 0
            if not args.no_save_model:
                torch.save(best_state, output_dir / "factor_graph_reranker_model.pt")
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"dev_{key}": value for key, value in dev_metrics.items() if key != "split"},
            "selection_value": selected_value,
            "best_epoch": best_epoch,
            "best_selection_value": best_value,
            "is_best": improved,
        }
        log_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            break
    write_jsonl(output_dir / "train_log.jsonl", log_rows)
    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    final_metrics, predictions = evaluate(
        model, dev_examples, dev_cache, maps, args, runtime, device, "dev"
    )
    write_json(output_dir / "dev_metrics.json", final_metrics)
    write_jsonl(output_dir / "dev_predictions.jsonl", predictions)
    summary = {
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "selection_value": best_value,
        "dev_metrics": final_metrics,
        "best_metrics_during_training": best_metrics,
        "last_epoch": log_rows[-1]["epoch"] if log_rows else 0,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
