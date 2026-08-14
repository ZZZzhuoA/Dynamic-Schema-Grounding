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
    topk_coverage_loss,
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


HISTORY_ADAPTER_PREFIXES = ("history_delta.", "history_relevance.", "residual_norm.")


def configure_safe_history_tuning(model, freeze_base_controller=False):
    """Freeze the validated independent path while training history adapters."""
    trainable = []
    for name, parameter in model.named_parameters():
        if freeze_base_controller:
            parameter.requires_grad = name.startswith(HISTORY_ADAPTER_PREFIXES)
        if parameter.requires_grad:
            trainable.append(name)
    if freeze_base_controller and not trainable:
        raise ValueError("Safe history tuning left no trainable parameters")
    return trainable


def initialize_rejecting_gate(model, bias=-4.0):
    """Initialize the safe expert router to reject history before learning utility."""
    import torch

    torch.nn.init.zeros_(model.history_relevance[-1].weight)
    torch.nn.init.constant_(model.history_relevance[-1].bias, float(bias))


def validate_and_filter_trajectories(examples, split, expected_numeric_dim=None):
    """Filter only empty, target-free graphs and reject malformed supervision."""
    usable = []
    skipped = []
    numeric_dim = expected_numeric_dim
    for example in examples:
        candidates = example.get("candidate_nodes", [])
        steps = example.get("trajectory_steps", [])
        target_ids = {
            int(item_id)
            for step in steps
            for item_id in step.get("target_schema_ids", [])
        }
        diagnostic = {
            "split": split,
            "record_index": example.get("record_index"),
            "db_id": example.get("db_id"),
            "question_id": example.get("question_id"),
        }
        if not candidates:
            if target_ids:
                raise ValueError(
                    "Empty candidate graph has trajectory supervision: "
                    + json.dumps({**diagnostic, "target_schema_ids": sorted(target_ids)}, ensure_ascii=False)
                )
            skipped.append({**diagnostic, "reason": "empty_candidate_graph_without_targets"})
            continue
        current_dim = len(candidates[0].get("numeric_features", []))
        if current_dim <= 0:
            raise ValueError(
                "Candidate graph has empty numeric features: "
                + json.dumps(diagnostic, ensure_ascii=False)
            )
        if any(len(node.get("numeric_features", [])) != current_dim for node in candidates):
            raise ValueError(
                "Candidate graph has inconsistent numeric feature dimensions: "
                + json.dumps(diagnostic, ensure_ascii=False)
            )
        if numeric_dim is None:
            numeric_dim = current_dim
        elif current_dim != numeric_dim:
            raise ValueError(
                f"Numeric feature dimension mismatch in {split}: expected={numeric_dim} "
                f"actual={current_dim} record_index={example.get('record_index')}"
            )
        node_count = len(candidates)
        for step in steps:
            for field in ("target_local_ids", "observed_local_ids"):
                invalid = [
                    int(index)
                    for index in step.get(field, [])
                    if int(index) < 0 or int(index) >= node_count
                ]
                if invalid:
                    raise ValueError(
                        f"Out-of-range {field} in {split} record_index="
                        f"{example.get('record_index')}: {invalid}"
                    )
        usable.append(example)
    if not usable:
        raise ValueError(f"No usable Stage 11 trajectories remain for split={split}")
    return usable, {
        "split": split,
        "input_count": len(examples),
        "usable_count": len(usable),
        "skipped_count": len(skipped),
        "skipped_examples": skipped,
        "numeric_dim": numeric_dim,
    }


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


def counterfactual_utility(base_loss, candidate_loss, temperature=0.05, margin=0.0):
    """Positive value only when the history candidate beats the base path."""
    improvement = (base_loss - candidate_loss - margin).clamp_min(0.0)
    return 1.0 - (-improvement / max(temperature, 1e-6)).exp()


def discrete_selection_quality(logits, labels, top_k=10, mrr_weight=0.1):
    """Detached Top-K coverage plus reciprocal-rank quality for gate supervision."""
    gold = set((labels > 0.5).nonzero(as_tuple=False).flatten().cpu().tolist())
    if not gold:
        return logits.new_zeros(())
    ranked = logits.detach().argsort(descending=True).cpu().tolist()
    selected = set(ranked[: min(int(top_k), len(ranked))])
    recall = len(gold & selected) / len(gold)
    reciprocal_rank = 1.0 / min(ranked.index(index) + 1 for index in gold)
    return logits.new_tensor(recall + float(mrr_weight) * reciprocal_rank)


def selection_regret_utility(
    base_logits, candidate_logits, labels, top_k=10, mrr_weight=0.1,
    temperature=0.05, margin=0.0,
):
    """Positive utility only for candidate improvements in the selection metric."""
    base_quality = discrete_selection_quality(base_logits, labels, top_k, mrr_weight)
    candidate_quality = discrete_selection_quality(
        candidate_logits, labels, top_k, mrr_weight
    )
    # counterfactual_utility expects its first argument to be the better-is-larger
    # quantity; selection quality is maximized, unlike BCE loss which is minimized.
    utility = counterfactual_utility(
        candidate_quality, base_quality, temperature, margin
    )
    return utility, base_quality, candidate_quality


def trajectory_loss(
    outputs, steps, pos_weight, runtime,
    provisional_loss_weight=0.0, history_gate_penalty=0.0,
    history_candidate_loss_weight=0.0, history_gate_loss_weight=0.0,
    history_utility_temperature=0.05, history_utility_margin=0.0,
    history_utility_objective="bce", output_top_k=10,
    history_mrr_weight=0.1, history_selection_loss_weight=0.0,
    coverage_margin=0.1, coverage_temperature=0.2,
):
    torch = runtime["torch"]
    losses = []
    for output, step in zip(outputs, steps):
        if not step["target"].sum():
            continue
        weight = torch.tensor(pos_weight, device=output["logits"].device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output["logits"], step["target"], pos_weight=weight
        )
        residual_step = (
            output.get("residual_history_enabled") and output.get("history_available")
        )
        if residual_step:
            base_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["provisional_logits"], step["target"], pos_weight=weight
            )
            candidate_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["history_candidate_logits"], step["target"], pos_weight=weight
            )
            loss = loss + provisional_loss_weight * base_loss
            loss = loss + history_candidate_loss_weight * candidate_loss
            if history_utility_objective == "selection_regret":
                utility_target, _, _ = selection_regret_utility(
                    output["provisional_logits"], output["history_candidate_logits"],
                    step["target"], output_top_k, history_mrr_weight,
                    history_utility_temperature, history_utility_margin,
                )
                if history_selection_loss_weight:
                    final_coverage, _ = topk_coverage_loss(
                        output["logits"], step["target"], output_top_k,
                        coverage_margin, coverage_temperature,
                    )
                    candidate_coverage, _ = topk_coverage_loss(
                        output["history_candidate_logits"], step["target"], output_top_k,
                        coverage_margin, coverage_temperature,
                    )
                    loss = loss + history_selection_loss_weight * (
                        final_coverage + candidate_coverage
                    ) / 2.0
            elif history_utility_objective == "bce":
                utility_target = counterfactual_utility(
                    base_loss.detach(), candidate_loss.detach(),
                    history_utility_temperature, history_utility_margin,
                )
            else:
                raise ValueError(
                    f"Unsupported history_utility_objective: {history_utility_objective}"
                )
            if history_gate_loss_weight:
                loss = loss + history_gate_loss_weight * torch.nn.functional.binary_cross_entropy(
                    output["history_gate_probability"].clamp(1e-6, 1.0 - 1e-6),
                    utility_target,
                )
            # Retained only for backwards-compatible ablations. New Stage 11-B-fix1
            # runs should leave this at zero and use utility supervision instead.
            if history_gate_penalty:
                loss = loss + history_gate_penalty * output["history_gate"]
        losses.append(loss)
    if not losses:
        return outputs[0]["logits"].sum() * 0.0 if outputs else None
    return torch.stack(losses).mean()


def evaluate(model, examples, cache, relation_to_id, args, runtime, device, split):
    torch = runtime["torch"]
    model.eval()
    losses, recalls, reciprocal_ranks, belief_changes = [], [], [], []
    gates, gate_probabilities, entropies, margins, residual_norms = [], [], [], [], []
    utility_targets, gate_utility_errors, candidate_wins = [], [], []
    by_operation = defaultdict(
        lambda: {
            "recall": [], "mrr": [], "gate": [], "gate_probability": [],
            "utility": [], "candidate_win": [],
        }
    )
    predictions = []
    with torch.no_grad():
        for example in examples:
            dense, numeric, query, edges, steps = trajectory_tensors(
                example, cache, relation_to_id, model, runtime, device
            )
            if not steps:
                continue
            outputs = model.forward_trajectory(dense, numeric, query, edges, steps)
            loss = trajectory_loss(
                outputs, steps, args.pos_weight, runtime,
                args.provisional_loss_weight, args.history_gate_penalty,
                args.history_candidate_loss_weight, args.history_gate_loss_weight,
                args.history_utility_temperature, args.history_utility_margin,
                args.history_utility_objective, args.output_top_k,
                args.history_mrr_weight, args.history_selection_loss_weight,
                args.coverage_margin, args.coverage_temperature,
            )
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
                entropies.append(float(output["provisional_entropy"].cpu()))
                margins.append(float(output["provisional_margin"].cpu()))
                if output["history_available"]:
                    gate_value = float(output["history_gate"].cpu())
                    gate_probability = float(output["history_gate_probability"].cpu())
                    weight = torch.tensor(args.pos_weight, device=output["logits"].device)
                    base_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        output["provisional_logits"], step["target"], pos_weight=weight
                    )
                    candidate_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        output["history_candidate_logits"], step["target"], pos_weight=weight
                    )
                    if args.history_utility_objective == "selection_regret":
                        utility_tensor, base_quality, candidate_quality = selection_regret_utility(
                            output["provisional_logits"],
                            output["history_candidate_logits"], step["target"],
                            args.output_top_k, args.history_mrr_weight,
                            args.history_utility_temperature, args.history_utility_margin,
                        )
                        utility = float(utility_tensor.cpu())
                        candidate_win = float(candidate_quality > base_quality)
                    else:
                        utility = float(counterfactual_utility(
                            base_loss, candidate_loss,
                            args.history_utility_temperature, args.history_utility_margin,
                        ).cpu())
                        candidate_win = float(candidate_loss < base_loss)
                    gates.append(gate_value)
                    gate_probabilities.append(gate_probability)
                    residual_norms.append(float(output["history_delta_norm"].cpu()))
                    utility_targets.append(utility)
                    gate_utility_errors.append(abs(gate_probability - utility))
                    candidate_wins.append(candidate_win)
                    by_operation[step["operation"]]["gate"].append(gate_value)
                    by_operation[step["operation"]]["gate_probability"].append(
                        gate_probability
                    )
                    by_operation[step["operation"]]["utility"].append(utility)
                    by_operation[step["operation"]]["candidate_win"].append(candidate_win)
                step_rows.append(
                    {
                        "step_index": step["step_index"],
                        "clause": step["clause"],
                        "operation": step["operation"],
                        "partial_sql": step["partial_sql"],
                        "gold_local_ids": sorted(gold),
                        f"top_{args.output_top_k}_local_ids": top,
                        "steering_state_norm": float(output["steering_state"].norm().cpu()),
                        "provisional_entropy": float(output["provisional_entropy"].cpu()),
                        "provisional_margin": float(output["provisional_margin"].cpu()),
                        "history_gate": float(output["history_gate"].cpu()),
                        "history_gate_probability": float(
                            output["history_gate_probability"].cpu()
                        ),
                        "history_delta_norm": float(output["history_delta_norm"].cpu()),
                        "counterfactual_utility_target": utility if output["history_available"] else 0.0,
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
        "step_selection_quality": (
            mean(recalls) + args.history_mrr_weight * mean(reciprocal_ranks)
        ),
        "mean_belief_total_variation": mean(belief_changes),
        "mean_provisional_entropy": mean(entropies),
        "mean_provisional_margin": mean(margins),
        "mean_history_gate": mean(gates),
        "mean_history_gate_probability": mean(gate_probabilities),
        "history_gate_activation_rate@0.1": mean([float(value > 0.1) for value in gates]),
        "history_gate_acceptance_rate": mean([float(value >= 0.5) for value in gates]),
        "mean_history_delta_norm": mean(residual_norms),
        "mean_counterfactual_utility_target": mean(utility_targets),
        "history_candidate_win_rate": mean(candidate_wins),
        "mean_gate_utility_absolute_error": mean(gate_utility_errors),
    }
    for operation, values in sorted(by_operation.items()):
        prefix = operation.lower()
        metrics[f"{prefix}_target_step_count"] = len(values["recall"])
        metrics[f"{prefix}_schema_recall@{args.output_top_k}"] = mean(values["recall"])
        metrics[f"{prefix}_schema_mrr"] = mean(values["mrr"])
        metrics[f"{prefix}_mean_history_gate"] = mean(values["gate"])
        metrics[f"{prefix}_mean_history_gate_probability"] = mean(
            values["gate_probability"]
        )
        metrics[f"{prefix}_mean_counterfactual_utility"] = mean(values["utility"])
        metrics[f"{prefix}_history_candidate_win_rate"] = mean(values["candidate_win"])
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
        "--base-checkpoint", default=None,
        help="Optional independent-controller checkpoint used to initialize the protected base path.",
    )
    parser.add_argument(
        "--freeze-base-controller", action="store_true",
        help="Train only history residual/gate modules; requires --base-checkpoint.",
    )
    parser.add_argument(
        "--history-mode",
        choices=("legacy_recurrent", "independent", "uncertainty_residual"),
        default="legacy_recurrent",
        help="How latent history participates in each clause-level grounding step.",
    )
    parser.add_argument("--provisional-loss-weight", type=float, default=0.3)
    parser.add_argument("--history-gate-penalty", type=float, default=0.0)
    parser.add_argument("--history-candidate-loss-weight", type=float, default=0.3)
    parser.add_argument("--history-gate-loss-weight", type=float, default=0.1)
    parser.add_argument("--history-utility-temperature", type=float, default=0.05)
    parser.add_argument("--history-utility-margin", type=float, default=0.0)
    parser.add_argument(
        "--history-utility-objective",
        choices=("bce", "selection_regret"), default="bce",
    )
    parser.add_argument("--history-mrr-weight", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("recall", "selection_quality"), default="recall",
    )
    parser.add_argument("--history-selection-loss-weight", type=float, default=0.0)
    parser.add_argument("--coverage-margin", type=float, default=0.1)
    parser.add_argument("--coverage-temperature", type=float, default=0.2)
    parser.add_argument(
        "--history-gate-policy", choices=("soft", "straight_through"), default="soft"
    )
    parser.add_argument("--history-gate-threshold", type=float, default=0.5)
    parser.add_argument(
        "--allow-history-gradients", action="store_true",
        help="Allow backpropagation through prior predicted states/beliefs; detached by default.",
    )
    parser.add_argument(
        "--disable-recurrence",
        action="store_true",
        help="Ablation: condition each event independently without belief/state recurrence.",
    )
    args = parser.parse_args()
    if args.disable_recurrence:
        if args.history_mode != "legacy_recurrent":
            parser.error("--disable-recurrence cannot be combined with an explicit --history-mode")
        args.history_mode = "independent"
    if args.freeze_base_controller and not args.base_checkpoint:
        parser.error("--freeze-base-controller requires --base-checkpoint")
    runtime = import_runtime()
    torch = runtime["torch"]
    random.seed(args.seed)
    runtime["np"].random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    raw_train = read_jsonl(Path(args.train_file), args.train_limit)
    raw_dev = read_jsonl(Path(args.dev_file), args.dev_limit)
    train, train_validation = validate_and_filter_trajectories(raw_train, "train")
    dev, dev_validation = validate_and_filter_trajectories(
        raw_dev, "dev", expected_numeric_dim=train_validation["numeric_dim"]
    )
    train_cache = load_cache(args.embedding_cache_dir, "train", runtime)
    dev_cache = load_cache(args.embedding_cache_dir, "dev", runtime)
    relations = collect_schema_relations(train + dev)
    relation_to_id = {name: index for index, name in enumerate(relations)}
    numeric_dim = int(train_validation["numeric_dim"])
    model = DynamicSchemaGroundingController(
        dense_dim=train_cache["dense_dim"], numeric_dim=numeric_dim,
        hidden_dim=args.hidden_dim, relation_count=max(len(relations), 1),
        num_layers=args.num_layers, dropout=args.dropout,
        recurrent=args.history_mode == "legacy_recurrent",
        history_mode=args.history_mode,
        detach_history=not args.allow_history_gradients,
        history_gate_policy=args.history_gate_policy,
        history_gate_threshold=args.history_gate_threshold,
    ).to(device)
    checkpoint_load = None
    if args.base_checkpoint:
        checkpoint = torch.load(args.base_checkpoint, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        incompatible = model.load_state_dict(checkpoint, strict=False)
        missing_base_keys = [
            key for key in incompatible.missing_keys
            if not key.startswith(HISTORY_ADAPTER_PREFIXES)
        ]
        if args.freeze_base_controller and missing_base_keys:
            raise ValueError(
                "Base checkpoint is missing protected controller parameters: "
                + ", ".join(missing_base_keys[:20])
            )
        checkpoint_load = {
            "path": args.base_checkpoint,
            "missing_keys": list(incompatible.missing_keys),
            "missing_base_keys": missing_base_keys,
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
    if args.freeze_base_controller:
        initialize_rejecting_gate(model)
    trainable_parameters = configure_safe_history_tuning(
        model, args.freeze_base_controller
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "data_validation.json",
        {"train": train_validation, "dev": dev_validation},
    )
    initial_metrics, _ = evaluate(
        model, dev, dev_cache, relation_to_id, args, runtime, device, "dev_initial"
    )
    selection_metric = (
        "step_selection_quality"
        if args.checkpoint_selection == "selection_quality"
        else f"step_schema_recall@{args.output_top_k}"
    )
    best_value = initial_metrics[selection_metric]
    best_state = clone_state_dict_to_cpu(model)
    best_metrics = initial_metrics
    best_epoch = 0
    torch.save(best_state, output_dir / "dynamic_grounding_controller.pt")
    log_rows = [{"epoch": 0, "train_loss": None, **initial_metrics}]
    print(json.dumps(log_rows[0], ensure_ascii=False))
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
            loss = trajectory_loss(
                outputs, steps, args.pos_weight, runtime,
                args.provisional_loss_weight, args.history_gate_penalty,
                args.history_candidate_loss_weight, args.history_gate_loss_weight,
                args.history_utility_temperature, args.history_utility_margin,
                args.history_utility_objective, args.output_top_k,
                args.history_mrr_weight, args.history_selection_loss_weight,
                args.coverage_margin, args.coverage_temperature,
            )
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
        value = metrics[selection_metric]
        if best_value is None or value > best_value:
            best_value, best_metrics = value, metrics
            best_epoch = epoch
            best_state = clone_state_dict_to_cpu(model)
            torch.save(best_state, output_dir / "dynamic_grounding_controller.pt")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    final_metrics, predictions = evaluate(
        model, dev, dev_cache, relation_to_id, args, runtime, device, "dev"
    )
    write_jsonl(output_dir / "train_log.jsonl", log_rows)
    write_jsonl(output_dir / "dev_trajectory_predictions.jsonl", predictions)
    summary = {
        "selection_metric": selection_metric,
        "selection_value": best_value,
        "best_epoch": best_epoch,
        "initial_metrics": initial_metrics,
        "best_metrics": best_metrics,
        "dev_metrics": final_metrics,
        "config": vars(args),
        "relations": relations,
        "operations": OPERATIONS,
        "data_validation": {"train": train_validation, "dev": dev_validation},
        "checkpoint_load": checkpoint_load,
        "trainable_parameters": trainable_parameters,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
