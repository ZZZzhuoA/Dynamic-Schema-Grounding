"""Train the Stage 13-B1 action-synchronous typed RA pointer decoder."""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stage13b_prepare_typed_trajectories import ACTIONS, OPERATORS, VALUE_ROUTES, read_jsonl


def runtime_imports():
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("Stage 13-B1 requires numpy and PyTorch") from exc
    from src.modeling.typed_ra_decoder import TypedRAPointerDecoder, multilabel_bce
    return np, torch, F, TypedRAPointerDecoder, multilabel_bce


def load_cache(cache_dir, split, np):
    root = Path(cache_dir)
    query = np.load(root / f"{split}_query_embeddings.npy", mmap_mode="r")
    node = np.load(root / f"{split}_node_embeddings.npy", mmap_mode="r")
    index = json.loads((root / f"{split}_index.json").read_text(encoding="utf-8"))
    by_index = {}
    for row in index:
        key = row.get("record_index")
        if key is None:
            key = row["example_index"]
        key = int(key)
        if key in by_index:
            raise ValueError(f"Duplicate embedding-cache record_index={key} for split={split}")
        by_index[key] = row
    return {
        "query": query,
        "node": node,
        "by_index": by_index,
        "dense_dim": int(node.shape[1]),
    }


def dense_tensors(example, cache, torch, device):
    row = cache["by_index"][int(example["record_index"])]
    query_index = int(row["query_embedding_index"])
    node_count = int(row["node_count"])
    if "node_embedding_indices" in row:
        node_values = cache["node"][[int(value) for value in row["node_embedding_indices"]]]
    else:
        start = int(row["node_embedding_start"])
        node_values = cache["node"][start : start + node_count]
    if len(node_values) != len(example["inference_inputs"]["schema_items"]):
        raise ValueError(f"Node/cache mismatch at record_index={example['record_index']}")
    return (
        torch.tensor(node_values, dtype=torch.float32, device=device),
        torch.tensor(cache["query"][query_index : query_index + 1], dtype=torch.float32, device=device),
    )


def relation_vocabulary(examples):
    names = sorted({
        edge["type"] for example in examples
        for edge in example["inference_inputs"].get("schema_edges", [])
    })
    return names or ["self_loop"]


def graph_tensors(example, relation_to_id, torch, device):
    edges = example["inference_inputs"].get("schema_edges", [])
    pairs, types = [], []
    join_pairs, join_seen = [], set()
    for edge in edges:
        if edge["type"] not in relation_to_id:
            continue
        src, dst = int(edge["src"]), int(edge["dst"])
        pairs.append((src, dst))
        types.append(relation_to_id[edge["type"]])
        if edge["type"] in {"foreign_key_forward", "foreign_key_backward"}:
            key = tuple(sorted((src, dst)))
            if key not in join_seen:
                join_seen.add(key)
                join_pairs.append(key)
    edge_index = (
        torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()
        if pairs else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    edge_type = torch.tensor(types, dtype=torch.long, device=device)
    join_edge_index = (
        torch.tensor(join_pairs, dtype=torch.long, device=device).t().contiguous()
        if join_pairs else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    node_types = torch.tensor(
        [0 if item["type"] == "table" else 1 for item in example["inference_inputs"]["schema_items"]],
        dtype=torch.long,
        device=device,
    )
    return node_types, edge_index, edge_type, join_edge_index, join_pairs


def join_target_indices(step, join_pairs):
    by_pair = {tuple(pair): index for index, pair in enumerate(join_pairs)}
    result = []
    for edge in step.get("join_edge_targets", []):
        pair = tuple(sorted((int(edge["left_column_id"]), int(edge["right_column_id"]))))
        if pair in by_pair:
            result.append(by_pair[pair])
    return sorted(set(result))


def trajectory_loss(outputs, steps, node_types, join_pairs, args, torch, F, multilabel_bce):
    losses = []
    action_to_id = {name: index for index, name in enumerate(ACTIONS)}
    route_to_id = {name: index for index, name in enumerate(VALUE_ROUTES)}
    operator_to_id = {name: index for index, name in enumerate(OPERATORS)}
    for output, step in zip(outputs, steps):
        target_action = torch.tensor([action_to_id[step["action"]]], device=node_types.device)
        losses.append(args.action_loss_weight * F.cross_entropy(output["action_logits"].unsqueeze(0), target_action))
        table_ids = step.get("table_pointer_ids", [])
        if table_ids:
            loss = multilabel_bce(output["table_logits"], table_ids, node_types == 0, args.pointer_pos_weight)
            losses.append(args.pointer_loss_weight * loss)
        column_ids = step.get("column_pointer_ids", [])
        if column_ids:
            loss = multilabel_bce(output["column_logits"], column_ids, node_types == 1, args.pointer_pos_weight)
            losses.append(args.pointer_loss_weight * loss)
        edge_ids = join_target_indices(step, join_pairs)
        if edge_ids:
            loss = multilabel_bce(output["join_edge_logits"], edge_ids, None, args.pointer_pos_weight)
            losses.append(args.join_loss_weight * loss)
        route_ids = [route_to_id[name] for name in step.get("value_routes", [])]
        if route_ids:
            loss = multilabel_bce(output["value_route_logits"], route_ids, None, args.pointer_pos_weight)
            losses.append(args.value_route_loss_weight * loss)
        operator_ids = [operator_to_id[name] for name in step.get("operator_targets", [])]
        if operator_ids:
            loss = multilabel_bce(output["operator_logits"], operator_ids, None, args.pointer_pos_weight)
            losses.append(args.operator_loss_weight * loss)
    return torch.stack(losses).mean() if losses else None


def topk_hits(logits, positive_ids, valid_mask, k, torch):
    if not positive_ids:
        return 0, 0, True
    valid_ids = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if not len(valid_ids):
        return 0, len(positive_ids), False
    count = min(k, len(valid_ids))
    selected = set(valid_ids[torch.topk(logits[valid_ids], count).indices].tolist())
    positive = set(int(value) for value in positive_ids)
    return len(selected & positive), len(positive), positive.issubset(selected)


def evaluate(model, examples, cache, relation_to_id, args, np, torch, F, multilabel_bce, device):
    model.eval()
    counts = Counter()
    total_loss = 0.0
    predictions = []
    with torch.no_grad():
        for example in examples:
            dense, query = dense_tensors(example, cache, torch, device)
            node_types, edge_index, edge_type, join_index, join_pairs = graph_tensors(
                example, relation_to_id, torch, device
            )
            steps = example["teacher_steps"]
            outputs = model.forward_trajectory(
                dense, query, node_types, edge_index, edge_type, join_index, steps
            )
            loss = trajectory_loss(outputs, steps, node_types, join_pairs, args, torch, F, multilabel_bce)
            total_loss += float(loss.cpu())
            step_rows = []
            for output, step in zip(outputs, steps):
                predicted_action = ACTIONS[int(output["action_logits"].argmax())]
                action_ok = predicted_action == step["action"]
                counts["action_correct"] += int(action_ok)
                counts["action_total"] += 1
                th, tt, table_complete = topk_hits(
                    output["table_logits"], step.get("table_pointer_ids", []), node_types == 0,
                    args.table_top_k, torch,
                )
                ch, ct, column_complete = topk_hits(
                    output["column_logits"], step.get("column_pointer_ids", []), node_types == 1,
                    args.column_top_k, torch,
                )
                edge_ids = join_target_indices(step, join_pairs)
                if edge_ids:
                    mask = torch.ones_like(output["join_edge_logits"], dtype=torch.bool)
                    eh, et, edge_complete = topk_hits(
                        output["join_edge_logits"], edge_ids, mask, args.join_top_k, torch
                    )
                else:
                    eh, et, edge_complete = 0, 0, True
                route_ids = [VALUE_ROUTES.index(name) for name in step.get("value_routes", [])]
                if route_ids:
                    mask = torch.ones_like(output["value_route_logits"], dtype=torch.bool)
                    vh, vt, value_complete = topk_hits(
                        output["value_route_logits"], route_ids, mask, args.value_route_top_k, torch
                    )
                else:
                    vh, vt, value_complete = 0, 0, True
                operator_ids = [OPERATORS.index(name) for name in step.get("operator_targets", [])]
                if operator_ids:
                    mask = torch.ones_like(output["operator_logits"], dtype=torch.bool)
                    oh, ot, operator_complete = topk_hits(
                        output["operator_logits"], operator_ids, mask, args.operator_top_k, torch
                    )
                else:
                    oh, ot, operator_complete = 0, 0, True
                counts.update({"table_hit": th, "table_total": tt, "column_hit": ch, "column_total": ct,
                               "edge_hit": eh, "edge_total": et, "value_hit": vh, "value_total": vt,
                               "operator_hit": oh, "operator_total": ot})
                complete = (action_ok and table_complete and column_complete and edge_complete
                            and value_complete and operator_complete)
                counts["step_complete"] += int(complete)
                counts["step_total"] += 1
                step_rows.append({"gold_action": step["action"], "predicted_action": predicted_action,
                                  "target_complete": complete})
            predictions.append({"record_index": example["record_index"], "db_id": example["db_id"],
                                "steps": step_rows})
    def ratio(hit, total):
        return counts[hit] / counts[total] if counts[total] else 1.0
    metrics = {
        "loss": total_loss / max(len(examples), 1),
        "example_count": len(examples),
        "action_accuracy": ratio("action_correct", "action_total"),
        f"table_recall@{args.table_top_k}": ratio("table_hit", "table_total"),
        f"column_recall@{args.column_top_k}": ratio("column_hit", "column_total"),
        f"join_edge_recall@{args.join_top_k}": ratio("edge_hit", "edge_total"),
        f"value_route_recall@{args.value_route_top_k}": ratio("value_hit", "value_total"),
        f"operator_recall@{args.operator_top_k}": ratio("operator_hit", "operator_total"),
        "step_complete_rate": ratio("step_complete", "step_total"),
    }
    recalls = [value for key, value in metrics.items() if "recall@" in key]
    metrics["mean_target_recall"] = sum(recalls) / len(recalls)
    return metrics, predictions


def clone_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="experiments/stage13b_clean_typed_trajectories/train_trajectories.jsonl")
    parser.add_argument("--dev-file", default="experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl")
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--output-dir", default="experiments/stage13b_typed_ra_decoder")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--pointer-pos-weight", type=float, default=5.0)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--pointer-loss-weight", type=float, default=1.0)
    parser.add_argument("--join-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-route-loss-weight", type=float, default=0.5)
    parser.add_argument("--operator-loss-weight", type=float, default=0.75)
    parser.add_argument("--table-top-k", type=int, default=3)
    parser.add_argument("--column-top-k", type=int, default=5)
    parser.add_argument("--join-top-k", type=int, default=3)
    parser.add_argument("--value-route-top-k", type=int, default=2)
    parser.add_argument("--operator-top-k", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np, torch, F, Model, multilabel_bce = runtime_imports()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train = read_jsonl(args.train_file)[: args.train_limit]
    dev = read_jsonl(args.dev_file)[: args.dev_limit]
    train_cache = load_cache(args.embedding_cache_dir, "train", np)
    dev_cache = load_cache(args.embedding_cache_dir, "dev", np)
    relations = relation_vocabulary(train + dev)
    relation_to_id = {name: index for index, name in enumerate(relations)}
    model = Model(dense_dim=train_cache["dense_dim"], hidden_dim=args.hidden_dim,
                  relation_count=len(relations), num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    best_value, best_epoch, best_state, best_metrics = -1.0, 0, clone_cpu(model), None
    logs = []
    for epoch in range(1, args.epochs + 1):
        model.train(); shuffled = list(train); random.Random(args.seed + epoch).shuffle(shuffled)
        loss_sum, updates = 0.0, 0
        for example in shuffled:
            dense, query = dense_tensors(example, train_cache, torch, device)
            node_types, edge_index, edge_type, join_index, join_pairs = graph_tensors(
                example, relation_to_id, torch, device
            )
            outputs = model.forward_trajectory(
                dense, query, node_types, edge_index, edge_type, join_index, example["teacher_steps"]
            )
            loss = trajectory_loss(outputs, example["teacher_steps"], node_types, join_pairs,
                                   args, torch, F, multilabel_bce)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); loss_sum += float(loss.detach().cpu()); updates += 1
        metrics, _ = evaluate(model, dev, dev_cache, relation_to_id, args,
                              np, torch, F, multilabel_bce, device)
        row = {"epoch": epoch, "train_loss": loss_sum / max(updates, 1), **metrics}
        logs.append(row); print(json.dumps(row, ensure_ascii=False))
        if metrics["mean_target_recall"] > best_value:
            best_value, best_epoch, best_metrics = metrics["mean_target_recall"], epoch, metrics
            best_state = clone_cpu(model); torch.save(best_state, output_dir / "typed_ra_decoder.pt")
    model.load_state_dict({name: value.to(device) for name, value in best_state.items()})
    final_metrics, predictions = evaluate(model, dev, dev_cache, relation_to_id, args,
                                          np, torch, F, multilabel_bce, device)
    write_jsonl(output_dir / "train_log.jsonl", logs)
    write_jsonl(output_dir / "dev_predictions.jsonl", predictions)
    summary = {"best_epoch": best_epoch, "selection_metric": "mean_target_recall",
               "selection_value": best_value, "best_metrics": best_metrics,
               "dev_metrics": final_metrics, "relations": relations,
               "actions": ACTIONS, "operators": OPERATORS,
               "value_routes": VALUE_ROUTES, "config": vars(args)}
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
