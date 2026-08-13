"""Train the Stage 11 recurrent partial-SQL schema grounding controller."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.dynamic_grounding_controller import (  # noqa: E402
    OPERATIONS,
    DynamicSchemaGroundingController,
    partial_sql_features,
)
from src.training.stage10_train_factor_graph_reranker import (  # noqa: E402
    collect_schema_relations,
    full_node_embeddings,
    load_cache,
)
from src.training.stage5_train_dsg_grounder import (  # noqa: E402
    clone_state_dict_to_cpu,
    read_jsonl,
    write_json,
    write_jsonl,
)


def import_runtime():
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Stage 11 requires numpy and PyTorch") from exc
    return {"np": np, "torch": torch}


def trajectory_tensors(example, cache, relation_to_id, model, runtime, device):
    torch = runtime["torch"]
    positions = [int(node["schema_position"]) for node in example["candidate_nodes"]]
    dense = torch.tensor(
        full_node_embeddings(cache, example["record_index"])[positions],
        dtype=torch.float32,
        device=device,
    )
    numeric = torch.tensor(
        [node["numeric_features"] for node in example["candidate_nodes"]],
        dtype=torch.float32,
        device=device,
    )
    cache_row = cache["index"][int(example["record_index"])]
    query_index = int(cache_row["query_embedding_index"])
    query = torch.tensor(
        cache["query"][query_index : query_index + 1],
        dtype=torch.float32,
        device=device,
    )
    pairs, types = [], []
    for edge in example.get("schema_edges", []):
        if edge.get("type") not in relation_to_id:
            continue
        pairs.append((int(edge["src"]), int(edge["dst"])))
        types.append(relation_to_id[edge["type"]])
    edge_index = (
        torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()
        if pairs
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    edge_type = torch.tensor(types, dtype=torch.long, device=device)
    steps = []
    for step in example.get("trajectory_steps", []):
        operation_id = model.operation_to_id.get(
            step.get("operation"), model.operation_to_id["UNKNOWN"]
        )
        target = torch.zeros(len(example["candidate_nodes"]), device=device)
        if step.get("target_local_ids"):
            target[step["target_local_ids"]] = 1.0
        observed_mask = torch.zeros(len(example["candidate_nodes"]), device=device)
        if step.get("observed_local_ids"):
            observed_mask[step["observed_local_ids"]] = 1.0
        steps.append(
            {
                **step,
                "operation_id": torch.tensor([operation_id], dtype=torch.long, device=device),
                "sql_features": torch.tensor(
                    partial_sql_features(step.get("partial_sql")),
                    dtype=torch.float32,
                    device=device,
                ),
                "target": target,
                "observed_mask": observed_mask,
            }
        )
    return dense, numeric, query, (edge_index, edge_type), steps


def trajectory_loss(outputs, steps, pos_weight, runtime):
    torch = runtime["torch"]
    losses = []
    for output, step in zip(outputs, steps):
        if not step["target"].sum():
            continue
        weight = torch.tensor(pos_weight, device=output["logits"].device)
        losses.append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                output["logits"], step["target"], pos_weight=weight
            )
        )
    if not losses:
        return outputs[0]["logits"].sum() * 0.0 if outputs else None
    return torch.stack(losses).mean()


def evaluate(model, examples, cache, relation_to_id, args, runtime, device, split):
    torch = runtime["torch"]
    model.eval()
    losses, recalls, reciprocal_ranks, belief_changes = [], [], [], []
    by_operation = defaultdict(lambda: {"recall": [], "mrr": []})
    predictions = []
    with torch.no_grad():
        for example in examples:
            dense, numeric, query, edges, steps = trajectory_tensors(
                example, cache, relation_to_id, model, runtime, device
            )
            if not steps:
                continue
            outputs = model.forward_trajectory(dense, numeric, query, edges, steps)
            loss = trajectory_loss(outputs, steps, args.pos_weight, runtime)
            if loss is not None:
                losses.append(float(loss.cpu()))
            previous_belief = None
            step_rows = []
            for output, step in zip(outputs, steps):
                ranked = output["logits"].argsort(descending=True).cpu().tolist()
                gold = set(int(index) for index in step["target_local_ids"])
                top = ranked[: args.output_top_k]
                if gold:
                    step_recall = len(gold & set(top)) / len(gold)
                    recalls.append(step_recall)
                    ranks = [ranked.index(index) + 1 for index in gold]
                    step_mrr = 1.0 / min(ranks)
                    reciprocal_ranks.append(step_mrr)
                    by_operation[step["operation"]]["recall"].append(step_recall)
                    by_operation[step["operation"]]["mrr"].append(step_mrr)
                if previous_belief is not None:
                    belief_changes.append(
                        float(torch.abs(output["belief"] - previous_belief).sum().cpu()) / 2.0
                    )
                previous_belief = output["belief"]
                step_rows.append(
                    {
                        "step_index": step["step_index"],
                        "clause": step["clause"],
                        "operation": step["operation"],
                        "partial_sql": step["partial_sql"],
                        "gold_local_ids": sorted(gold),
                        f"top_{args.output_top_k}_local_ids": top,
                        "steering_state_norm": float(output["steering_state"].norm().cpu()),
                    }
                )
            predictions.append(
                {
                    "record_index": example["record_index"],
                    "db_id": example.get("db_id"),
                    "question_id": example.get("question_id"),
                    "steps": step_rows,
                }
            )
    mean = lambda values: sum(values) / len(values) if values else 0.0
    metrics = {
        "split": split,
        "example_count": len(predictions),
        "step_count": sum(len(row["steps"]) for row in predictions),
        "loss": mean(losses),
        f"step_schema_recall@{args.output_top_k}": mean(recalls),
        "step_schema_mrr": mean(reciprocal_ranks),
        "mean_belief_total_variation": mean(belief_changes),
    }
    for operation, values in sorted(by_operation.items()):
        prefix = operation.lower()
        metrics[f"{prefix}_target_step_count"] = len(values["recall"])
        metrics[f"{prefix}_schema_recall@{args.output_top_k}"] = mean(values["recall"])
        metrics[f"{prefix}_schema_mrr"] = mean(values["mrr"])
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output-top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--disable-recurrence",
        action="store_true",
        help="Ablation: condition each event independently without belief/state recurrence.",
    )
    args = parser.parse_args()
    runtime = import_runtime()
    torch = runtime["torch"]
    random.seed(args.seed)
    runtime["np"].random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train = read_jsonl(Path(args.train_file), args.train_limit)
    dev = read_jsonl(Path(args.dev_file), args.dev_limit)
    train_cache = load_cache(args.embedding_cache_dir, "train", runtime)
    dev_cache = load_cache(args.embedding_cache_dir, "dev", runtime)
    relations = collect_schema_relations(train + dev)
    relation_to_id = {name: index for index, name in enumerate(relations)}
    numeric_dim = len(train[0]["candidate_nodes"][0]["numeric_features"])
    model = DynamicSchemaGroundingController(
        dense_dim=train_cache["dense_dim"], numeric_dim=numeric_dim,
        hidden_dim=args.hidden_dim, relation_count=max(len(relations), 1),
        num_layers=args.num_layers, dropout=args.dropout,
        recurrent=not args.disable_recurrence,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_value, best_state, best_metrics = None, None, None
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = list(train)
        random.Random(args.seed + epoch).shuffle(shuffled)
        total, count = 0.0, 0
        for example in shuffled:
            dense, numeric, query, edges, steps = trajectory_tensors(
                example, train_cache, relation_to_id, model, runtime, device
            )
            if not steps:
                continue
            optimizer.zero_grad()
            outputs = model.forward_trajectory(dense, numeric, query, edges, steps)
            loss = trajectory_loss(outputs, steps, args.pos_weight, runtime)
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total += float(loss.detach().cpu())
            count += 1
        metrics, _ = evaluate(
            model, dev, dev_cache, relation_to_id, args, runtime, device, "dev"
        )
        row = {"epoch": epoch, "train_loss": total / max(count, 1), **metrics}
        log_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        value = metrics[f"step_schema_recall@{args.output_top_k}"]
        if best_value is None or value > best_value:
            best_value, best_metrics = value, metrics
            best_state = clone_state_dict_to_cpu(model)
            torch.save(best_state, output_dir / "dynamic_grounding_controller.pt")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    final_metrics, predictions = evaluate(
        model, dev, dev_cache, relation_to_id, args, runtime, device, "dev"
    )
    write_jsonl(output_dir / "train_log.jsonl", log_rows)
    write_jsonl(output_dir / "dev_trajectory_predictions.jsonl", predictions)
    summary = {
        "selection_metric": f"step_schema_recall@{args.output_top_k}",
        "selection_value": best_value,
        "best_metrics": best_metrics,
        "dev_metrics": final_metrics,
        "config": vars(args),
        "relations": relations,
        "operations": OPERATIONS,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
