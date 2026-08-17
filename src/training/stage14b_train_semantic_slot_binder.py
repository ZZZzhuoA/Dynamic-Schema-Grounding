"""Train the Stage 14B semantic-slot-conditioned RGTA binder."""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stage13b_prepare_typed_trajectories import OPERATORS, VALUE_ROUTES  # noqa: E402
from src.modeling.stage13c_static_runtime import graph_tensors  # noqa: E402
from src.training.stage13b_train_typed_ra_decoder import load_cache  # noqa: E402


def read_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_slot_cache(root, split, np):
    root = Path(root)
    focus_path = root / f"{split}_focus_embeddings.npy"
    if not focus_path.exists():
        focus_path = root / f"{split}_slot_embeddings.npy"
    focus_embeddings = np.load(focus_path, mmap_mode="r")
    value_path = root / f"{split}_value_embeddings.npy"
    value_embeddings = (
        np.load(value_path, mmap_mode="r")
        if value_path.exists() else np.zeros_like(focus_embeddings)
    )
    index = json.loads((root / f"{split}_slot_index.json").read_text(encoding="utf-8"))
    by_key = {
        (int(row["record_index"]), int(row["step_index"])): row
        for row in index
    }
    by_action = defaultdict(list)
    for row in index:
        by_action[row["action"]].append((int(row["record_index"]), int(row["step_index"])))
    return {
        "focus": focus_embeddings,
        "value": value_embeddings,
        "by_key": by_key,
        "by_action": dict(by_action),
        "dense_dim": int(focus_embeddings.shape[1]),
    }


def relation_vocabulary(graphs, init_summary=None):
    if init_summary:
        values = json.loads(Path(init_summary).read_text(encoding="utf-8")).get("relations", [])
        if values:
            return values
    return sorted({
        edge["type"] for row in graphs
        for edge in row["inference_inputs"].get("schema_edges", [])
    }) or ["self_loop"]


def join_tensors(inputs, device, torch):
    pairs, seen = [], set()
    for edge in inputs.get("schema_edges", []):
        if edge.get("type") not in {"foreign_key_forward", "foreign_key_backward"}:
            continue
        pair = tuple(sorted((int(edge["src"]), int(edge["dst"]))))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    tensor = (
        torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()
        if pairs else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    return tensor, pairs


def target_join_indices(target, pairs):
    lookup = {pair: index for index, pair in enumerate(pairs)}
    result = []
    for edge in target.get("join_edge_targets", []):
        pair = tuple(sorted((int(edge["left_column_id"]), int(edge["right_column_id"]))))
        if pair in lookup:
            result.append(lookup[pair])
    return sorted(set(result))


def multilabel_bce(logits, positives, valid_mask, pos_weight, torch, F):
    if valid_mask is not None:
        valid_ids = torch.nonzero(valid_mask, as_tuple=False).flatten()
        logits = logits[valid_ids]
        remap = {int(original): index for index, original in enumerate(valid_ids.tolist())}
        positives = [remap[value] for value in positives if value in remap]
    if not logits.numel() or not positives:
        return None
    target = torch.zeros_like(logits)
    target[torch.tensor(positives, dtype=torch.long, device=logits.device)] = 1.0
    weights = torch.where(target > 0, torch.full_like(target, pos_weight), torch.ones_like(target))
    return F.binary_cross_entropy_with_logits(logits, target, weight=weights)


def listwise_coverage_loss(logits, positives, valid_mask, torch):
    positives = [int(value) for value in positives]
    if not positives:
        return None
    valid = torch.nonzero(valid_mask, as_tuple=False).flatten()
    positive = torch.tensor(positives, dtype=torch.long, device=logits.device)
    return torch.logsumexp(logits[valid], dim=0) - torch.logsumexp(logits[positive], dim=0)


def slot_schema_contrastive_loss(output, positives, valid_mask, temperature, torch, F):
    if not positives:
        return None
    nodes = F.normalize(output["schema_states"].float(), dim=-1)
    slot = F.normalize(output["slot_state"].float(), dim=-1)
    logits = (nodes @ slot) / max(float(temperature), 1e-4)
    return listwise_coverage_loss(logits, positives, valid_mask, torch)


def same_table_hard_negative_loss(logits, positives, nodes, margin, torch):
    losses = []
    positive_set = {int(value) for value in positives}
    for positive in positive_set:
        table = nodes[positive].get("table")
        negatives = [
            index for index, node in enumerate(nodes)
            if node.get("type") == "column" and node.get("table") == table and index not in positive_set
        ]
        if negatives:
            hardest = logits[torch.tensor(negatives, dtype=torch.long, device=logits.device)].max()
            losses.append(torch.relu(logits.new_tensor(margin) - logits[positive] + hardest))
    return torch.stack(losses).mean() if losses else None


def slot_loss(output, target, node_types, owner_indices, nodes, join_pairs, args, torch, F):
    losses = []
    for key, logits, positives, mask in (
        ("table", output["table_logits"], target.get("table_pointer_ids", []), node_types.eq(0)),
        ("column", output["column_logits"], target.get("column_pointer_ids", []), node_types.eq(1)),
    ):
        bce = multilabel_bce(logits, positives, mask, args.pointer_pos_weight, torch, F)
        if bce is not None:
            losses.append(args.pointer_loss_weight * bce)
        coverage = listwise_coverage_loss(logits, positives, mask, torch)
        if coverage is not None:
            losses.append(args.listwise_weight * coverage)
        contrastive = slot_schema_contrastive_loss(
            output, positives, mask, args.contrastive_temperature, torch, F
        )
        if contrastive is not None:
            losses.append(args.contrastive_weight * contrastive)
        if key == "column" and positives:
            hard = same_table_hard_negative_loss(
                logits, positives, nodes, args.hard_negative_margin, torch
            )
            if hard is not None:
                losses.append(args.hard_negative_weight * hard)
            owner_targets = sorted({int(owner_indices[value]) for value in positives if int(owner_indices[value]) >= 0})
            owner = listwise_coverage_loss(output["table_logits"], owner_targets, node_types.eq(0), torch)
            if owner is not None:
                losses.append(args.owner_consistency_weight * owner)

    edge_ids = target_join_indices(target, join_pairs)
    edge_loss = multilabel_bce(
        output["join_edge_logits"], edge_ids, None, args.pointer_pos_weight, torch, F
    )
    if edge_loss is not None:
        losses.append(args.join_loss_weight * edge_loss)
    route_ids = [VALUE_ROUTES.index(name) for name in target.get("value_routes", [])]
    route_loss = multilabel_bce(
        output["value_route_logits"], route_ids, None, args.pointer_pos_weight, torch, F
    )
    if route_loss is not None:
        losses.append(args.value_route_loss_weight * route_loss)
    operator_ids = [OPERATORS.index(name) for name in target.get("operator_targets", [])]
    operator_loss = multilabel_bce(
        output["operator_logits"], operator_ids, None, args.pointer_pos_weight, torch, F
    )
    if operator_loss is not None:
        losses.append(args.operator_loss_weight * operator_loss)
    return torch.stack(losses).mean() if losses else None


def topk_stats(logits, positives, valid_mask, k, torch):
    positives = {int(value) for value in positives}
    if not positives:
        return 0, 0, True
    valid = torch.nonzero(valid_mask, as_tuple=False).flatten()
    selected = set(valid[torch.topk(logits[valid], min(k, len(valid))).indices].tolist())
    return len(selected & positives), len(positives), positives.issubset(selected)


def aligned_records(graph_rows, slot_rows):
    slots = {int(row["record_index"]): row for row in slot_rows}
    result = []
    for graph in graph_rows:
        index = int(graph["record_index"])
        if index not in slots:
            raise KeyError(f"Missing semantic slots for record_index={index}")
        result.append((graph, slots[index]))
    return result


def slot_tensor(cache, record_index, step_index, device, torch, field="focus"):
    key = (int(record_index), int(step_index))
    if key not in cache["by_key"]:
        raise KeyError(f"Missing slot embedding for record/step={key}")
    row = cache["by_key"][key]
    index_name = "focus_embedding_index" if field == "focus" else "value_embedding_index"
    index = int(row.get(index_name, row["embedding_index"]))
    values = cache[field][index:index + 1]
    if field == "value" and not row.get("has_value", True):
        values = values * 0.0
    return torch.tensor(values, dtype=torch.float32, device=device)


def same_action_donor_key(cache, record_index, step_index, action):
    current = (int(record_index), int(step_index))
    candidates = cache["by_action"].get(action, [])
    if len(candidates) <= 1:
        return current
    position = candidates.index(current)
    for offset in range(1, len(candidates)):
        candidate = candidates[(position + offset) % len(candidates)]
        if candidate[0] != current[0]:
            return candidate
    return candidates[(position + 1) % len(candidates)]


def run_split(model, records, base_cache, slot_cache, relation_to_id, args, runtime, device, optimizer=None):
    torch, F = runtime["torch"], runtime["F"]
    training = optimizer is not None
    model.train(training)
    counts, action_counts = Counter(), defaultdict(Counter)
    total_loss, update_count = 0.0, 0
    prediction_rows = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for graph, slot_row in records:
            dense, query, node_types, edge_index, edge_type, nodes = graph_tensors(
                graph, base_cache, relation_to_id, device
            )
            join_index, join_pairs = join_tensors(graph["inference_inputs"], device, torch)
            from src.modeling.semantic_slot_binder import owner_table_indices
            owners = owner_table_indices(nodes, device)
            requests = slot_row["inference_inputs"]["requests"]
            targets = slot_row["training_targets"]["slot_targets"]
            example_losses, step_rows = [], []
            focus_embeddings = [
                slot_tensor(slot_cache, graph["record_index"], request["step_index"], device, torch, "focus")
                for request in requests
            ]
            value_embeddings = [
                slot_tensor(slot_cache, graph["record_index"], request["step_index"], device, torch, "value")
                for request in requests
            ]
            for step_position, (request, target, focus, value) in enumerate(
                zip(requests, targets, focus_embeddings, value_embeddings)
            ):
                model_focus, model_value = focus, value
                if training and random.random() < args.semantic_dropout:
                    model_focus, model_value = torch.zeros_like(focus), torch.zeros_like(value)
                output = model.forward_slot(
                    dense, query, model_focus, node_types, edge_index, edge_type, join_index,
                    request["action"], owners, model_value,
                    request.get("expected_value_type_id", 0),
                )
                loss = slot_loss(output, target, node_types, owners, nodes, join_pairs, args, torch, F)
                if loss is not None:
                    example_losses.append(loss)
                th, tt, tc = topk_stats(output["table_logits"], target.get("table_pointer_ids", []), node_types.eq(0), args.table_top_k, torch)
                ch, ct, cc = topk_stats(output["column_logits"], target.get("column_pointer_ids", []), node_types.eq(1), args.column_top_k, torch)
                edge_ids = target_join_indices(target, join_pairs)
                edge_mask = torch.ones_like(output["join_edge_logits"], dtype=torch.bool)
                eh, et, ec = topk_stats(output["join_edge_logits"], edge_ids, edge_mask, args.join_top_k, torch)
                counts.update(table_hit=th, table_total=tt, column_hit=ch, column_total=ct, edge_hit=eh, edge_total=et)
                bucket = action_counts[request["action"]]
                bucket.update(column_hit=ch, column_total=ct, complete=int(tc and cc and ec), total=1)

                target_ids = list(target.get("table_pointer_ids", [])) + list(target.get("column_pointer_ids", []))
                target_score = torch.stack([
                    output["table_logits"][idx] if nodes[idx].get("type") == "table" else output["column_logits"][idx]
                    for idx in target_ids
                ]).mean() if target_ids else output["table_logits"].new_zeros(())
                if target_ids and not training:
                    zero_output = model.forward_slot(
                        dense, query, torch.zeros_like(focus), node_types, edge_index, edge_type,
                        join_index, request["action"], owners, torch.zeros_like(value),
                        request.get("expected_value_type_id", 0),
                    )
                    zero_score = torch.stack([
                        zero_output["table_logits"][idx] if nodes[idx].get("type") == "table" else zero_output["column_logits"][idx]
                        for idx in target_ids
                    ]).mean()
                    donor_key = same_action_donor_key(
                        slot_cache, graph["record_index"], request["step_index"], request["action"]
                    )
                    shuffled_focus = slot_tensor(
                        slot_cache, donor_key[0], donor_key[1], device, torch, "focus"
                    )
                    shuffled_value = slot_tensor(
                        slot_cache, donor_key[0], donor_key[1], device, torch, "value"
                    )
                    shuffled_output = model.forward_slot(
                        dense, query, shuffled_focus, node_types, edge_index, edge_type,
                        join_index, request["action"], owners, shuffled_value,
                        request.get("expected_value_type_id", 0),
                    )
                    shuffled_score = torch.stack([
                        shuffled_output["table_logits"][idx] if nodes[idx].get("type") == "table" else shuffled_output["column_logits"][idx]
                        for idx in target_ids
                    ]).mean()
                    counts["causal_slot_count"] += 1
                    counts["correct_over_zero"] += int(target_score > zero_score)
                    counts["correct_over_shuffled"] += int(target_score > shuffled_score)
                    counts["gain_over_zero_sum"] += float((target_score - zero_score).detach().cpu())
                    counts["gain_over_shuffled_sum"] += float((target_score - shuffled_score).detach().cpu())
                step_rows.append({
                    "step_index": request["step_index"], "action": request["action"],
                    "table_complete": tc, "column_complete": cc, "join_complete": ec,
                    "slot_gate_mean": float(output["slot_gate"].mean().detach().cpu()),
                })
            if example_losses:
                example_loss = torch.stack(example_losses).mean()
                if training:
                    optimizer.zero_grad()
                    example_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                total_loss += float(example_loss.detach().cpu())
                update_count += 1
            prediction_rows.append({
                "record_index": int(graph["record_index"]), "db_id": graph.get("db_id"),
                "steps": step_rows,
            })
    def ratio(hit, total):
        return counts[hit] / counts[total] if counts[total] else 1.0
    action_metrics = {
        action: {
            f"column_recall@{args.column_top_k}": bucket["column_hit"] / bucket["column_total"] if bucket["column_total"] else None,
            "slot_complete_rate": bucket["complete"] / bucket["total"] if bucket["total"] else None,
        }
        for action, bucket in sorted(action_counts.items())
    }
    causal_count = counts["causal_slot_count"]
    metrics = {
        "loss": total_loss / max(update_count, 1),
        "record_count": len(records),
        f"table_recall@{args.table_top_k}": ratio("table_hit", "table_total"),
        f"column_recall@{args.column_top_k}": ratio("column_hit", "column_total"),
        f"join_edge_recall@{args.join_top_k}": ratio("edge_hit", "edge_total"),
        "correct_over_zero_rate": counts["correct_over_zero"] / causal_count if causal_count else 0.0,
        "correct_over_shuffled_rate": counts["correct_over_shuffled"] / causal_count if causal_count else 0.0,
        "mean_target_gain_over_zero": counts["gain_over_zero_sum"] / causal_count if causal_count else 0.0,
        "mean_target_gain_over_shuffled": counts["gain_over_shuffled_sum"] / causal_count if causal_count else 0.0,
        "action_metrics": action_metrics,
    }
    return metrics, prediction_rows


def runtime_imports():
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("Stage 14B training requires numpy and PyTorch") from exc
    from src.modeling.semantic_slot_binder import SemanticSlotGraphBinder
    return {"np": np, "torch": torch, "F": F, "Model": SemanticSlotGraphBinder}


def clone_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def freeze_pretrained_backbone(model):
    trainable_prefixes = (
        "slot_input.", "value_input.", "expected_type_embedding.",
        "slot_fusion.", "slot_gate.", "slot_norm.",
    )
    trainable_scalars = {"semantic_scale", "owner_scale"}
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(trainable_prefixes) or name in trainable_scalars
        if parameter.requires_grad:
            trainable.append(name)
    return trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-graph-file", required=True)
    parser.add_argument("--dev-graph-file", required=True)
    parser.add_argument("--train-slot-file", required=True)
    parser.add_argument("--dev-slot-file", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--slot-embedding-cache-dir", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--init-summary")
    parser.add_argument("--output-dir", default="experiments/stage14b_semantic_slot_binder")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--pointer-pos-weight", type=float, default=5.0)
    parser.add_argument("--pointer-loss-weight", type=float, default=1.0)
    parser.add_argument("--listwise-weight", type=float, default=0.3)
    parser.add_argument("--owner-consistency-weight", type=float, default=0.2)
    parser.add_argument("--hard-negative-weight", type=float, default=0.2)
    parser.add_argument("--hard-negative-margin", type=float, default=0.5)
    parser.add_argument("--contrastive-weight", type=float, default=0.3)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--semantic-dropout", type=float, default=0.2)
    parser.add_argument("--join-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-route-loss-weight", type=float, default=0.5)
    parser.add_argument("--operator-loss-weight", type=float, default=0.75)
    parser.add_argument("--table-top-k", type=int, default=3)
    parser.add_argument("--column-top-k", type=int, default=5)
    parser.add_argument("--join-top-k", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-pretrained-backbone", action="store_true", default=True)
    parser.add_argument("--train-full-model", dest="freeze_pretrained_backbone", action="store_false")
    args = parser.parse_args()

    runtime = runtime_imports()
    np, torch, Model = runtime["np"], runtime["torch"], runtime["Model"]
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_graphs = read_jsonl(args.train_graph_file, args.train_limit)
    dev_graphs = read_jsonl(args.dev_graph_file, args.dev_limit)
    train_slots = read_jsonl(args.train_slot_file, args.train_limit)
    dev_slots = read_jsonl(args.dev_slot_file, args.dev_limit)
    train = aligned_records(train_graphs, train_slots)
    dev = aligned_records(dev_graphs, dev_slots)
    base_train = load_cache(args.embedding_cache_dir, "train", np)
    base_dev = load_cache(args.embedding_cache_dir, "dev", np)
    slot_train = load_slot_cache(args.slot_embedding_cache_dir, "train", np)
    slot_dev = load_slot_cache(args.slot_embedding_cache_dir, "dev", np)
    relations = relation_vocabulary(train_graphs + dev_graphs, args.init_summary)
    relation_to_id = {name: index for index, name in enumerate(relations)}
    model = Model(
        dense_dim=base_train["dense_dim"], slot_dim=slot_train["dense_dim"],
        hidden_dim=args.hidden_dim, relation_count=len(relations),
        num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)
    warm_start = None
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        warm_start = model.load_stage13b_state(state)
    if args.freeze_pretrained_backbone and not args.init_checkpoint:
        raise ValueError("--freeze-pretrained-backbone requires --init-checkpoint")
    trainable_parameters = (
        freeze_pretrained_backbone(model)
        if args.freeze_pretrained_backbone else [name for name, _ in model.named_parameters()]
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    best_value, best_epoch, best_state, best_metrics = -1.0, 0, clone_cpu(model), None
    logs = []
    for epoch in range(1, args.epochs + 1):
        shuffled = list(train); random.Random(args.seed + epoch).shuffle(shuffled)
        train_metrics, _ = run_split(model, shuffled, base_train, slot_train, relation_to_id, args, runtime, device, optimizer)
        dev_metrics, _ = run_split(model, dev, base_dev, slot_dev, relation_to_id, args, runtime, device)
        value = dev_metrics[f"column_recall@{args.column_top_k}"]
        row = {"epoch": epoch, "train": train_metrics, "dev": dev_metrics,
               "selection_metric": f"column_recall@{args.column_top_k}", "selection_value": value}
        logs.append(row); print(json.dumps(row, ensure_ascii=False))
        if value > best_value:
            best_value, best_epoch, best_metrics = value, epoch, dev_metrics
            best_state = clone_cpu(model)
            torch.save({"model_state_dict": best_state, "relations": relations, "config": vars(args)}, output_dir / "semantic_slot_binder.pt")
    model.load_state_dict({name: value.to(device) for name, value in best_state.items()})
    final_metrics, predictions = run_split(model, dev, base_dev, slot_dev, relation_to_id, args, runtime, device)
    write_jsonl(output_dir / "train_log.jsonl", logs)
    write_jsonl(output_dir / "dev_predictions.jsonl", predictions)
    summary = {
        "best_epoch": best_epoch,
        "selection_metric": f"column_recall@{args.column_top_k}",
        "selection_value": best_value,
        "best_metrics": best_metrics,
        "dev_metrics": final_metrics,
        "relations": relations,
        "warm_start": warm_start,
        "trainable_parameters": trainable_parameters,
        "config": vars(args),
        "innovation": "Each semantic slot conditions RGTA propagation, hierarchical owner-table priors, and schema pointers; repeated actions no longer share one action-only query.",
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
