import argparse
import json
import random
import sys
from collections import defaultdict
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


def validate_and_filter_examples(examples, split):
    """Remove only label-empty/candidate-empty records and reject real failures."""
    usable = []
    skipped = []
    numeric_dim = None
    role_dim = None
    for example in examples:
        candidates = example.get("candidate_nodes", [])
        gold_ids = example.get("gold_ids", [])
        if not candidates:
            diagnostic = {
                "split": split,
                "record_index": example.get("record_index"),
                "db_id": example.get("db_id"),
                "question_id": example.get("question_id"),
                "gold_count": len(gold_ids),
                "reason": "empty_candidate_graph",
            }
            if gold_ids:
                raise ValueError(
                    "Candidate generation lost every node for a gold-bearing sample: "
                    + json.dumps(diagnostic, ensure_ascii=False)
                )
            skipped.append(diagnostic)
            continue
        current_numeric_dim = len(candidates[0].get("numeric_features", []))
        if current_numeric_dim <= 0:
            raise ValueError(
                f"Empty numeric features at {split} record_index={example.get('record_index')}"
            )
        if any(
            len(node.get("numeric_features", [])) != current_numeric_dim
            for node in candidates
        ):
            raise ValueError(
                f"Inconsistent node feature dimensions at {split} "
                f"record_index={example.get('record_index')}"
            )
        if numeric_dim is None:
            numeric_dim = current_numeric_dim
        elif current_numeric_dim != numeric_dim:
            raise ValueError(
                f"Numeric feature dimension changed in {split}: "
                f"expected={numeric_dim} actual={current_numeric_dim} "
                f"record_index={example.get('record_index')}"
            )
        current_role_dim = len(example.get("role_labels", [[]])[0])
        if role_dim is None:
            role_dim = current_role_dim
        elif current_role_dim != role_dim:
            raise ValueError(
                f"Role label dimension changed in {split}: "
                f"expected={role_dim} actual={current_role_dim}"
            )
        usable.append(example)
    return usable, {
        "split": split,
        "input_count": len(examples),
        "usable_count": len(usable),
        "skipped_empty_unlabeled_count": len(skipped),
        "skipped_examples": skipped,
        "numeric_dim": numeric_dim,
        "role_dim": role_dim,
    }


def parse_named_files(specs):
    """Parse repeated NAME=PATH arguments while preserving user-facing names."""
    result = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(
                f"Control file must use NAME=PATH syntax, received: {spec}"
            )
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid control file specification: {spec}")
        if not all(character.isalnum() or character in {"_", "-"} for character in name):
            raise ValueError(
                f"Control name may contain only letters, digits, '_' and '-': {name}"
            )
        if name in result:
            raise ValueError(f"Duplicate control file name: {name}")
        result[name] = Path(path)
    return result


def validate_control_alignment(reference, control, name, expected_numeric_dim):
    if len(reference) != len(control):
        raise ValueError(
            f"Control '{name}' length mismatch: reference={len(reference)} control={len(control)}"
        )
    for ref, candidate in zip(reference, control):
        ref_index = int(ref["record_index"])
        candidate_index = int(candidate["record_index"])
        if ref_index != candidate_index:
            raise ValueError(
                f"Control '{name}' record order mismatch: {ref_index} != {candidate_index}"
            )
        ref_ids = [int(node["schema_item_id"]) for node in ref["candidate_nodes"]]
        candidate_ids = [
            int(node["schema_item_id"]) for node in candidate["candidate_nodes"]
        ]
        if ref_ids != candidate_ids:
            raise ValueError(
                f"Control '{name}' candidate identity mismatch at record_index={ref_index}"
            )
        dimensions = {
            len(node.get("numeric_features", []))
            for node in candidate["candidate_nodes"]
        }
        if dimensions != {expected_numeric_dim}:
            raise ValueError(
                f"Control '{name}' numeric dimension mismatch at record_index={ref_index}: "
                f"expected={expected_numeric_dim} actual={sorted(dimensions)}"
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
        "coverage_eligible": bool(
            float(example.get("candidate_oracle_recall", 0.0)) >= 1.0 - 1e-9
        ),
        "selector_example": example,
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


def topk_coverage_loss(
    logits,
    labels,
    top_k,
    margin,
    temperature,
    eligible=True,
):
    """Surrogate for keeping every positive node inside a fixed Top-K budget.

    If P positives must fit in K positions, at most K-P negatives may outrank the
    weakest positive. The (K-P+1)-th highest negative is therefore the first
    boundary violation. A smooth minimum focuses the loss on the weakest positive.
    """
    import torch

    if temperature <= 0:
        raise ValueError("coverage_temperature must be positive")
    zero = logits.sum() * 0.0
    positive = logits[labels > 0.5]
    negative = logits[labels <= 0.5]
    positive_count = int(positive.numel())
    negative_count = int(negative.numel())
    boundary_rank = int(top_k) - positive_count + 1
    active = bool(
        eligible
        and positive_count > 0
        and positive_count <= int(top_k)
        and boundary_rank > 0
        and negative_count >= boundary_rank
    )
    if not active:
        return zero, {
            "coverage_active": 0.0,
            "coverage_positive_count": float(positive_count),
            "coverage_boundary_rank": float(max(boundary_rank, 0)),
            "coverage_violation": 0.0,
        }

    # This lower-bound soft minimum becomes the exact minimum as temperature -> 0
    # and sends gradients to every positive, with the weakest positives dominating.
    weakest_positive = -temperature * torch.logsumexp(
        -positive / temperature, dim=0
    )
    boundary_negative = negative.topk(boundary_rank).values[-1]
    violation = margin + boundary_negative - weakest_positive
    loss = temperature * torch.nn.functional.softplus(violation / temperature)
    return loss, {
        "coverage_active": 1.0,
        "coverage_positive_count": float(positive_count),
        "coverage_boundary_rank": float(boundary_rank),
        "coverage_violation": float(violation.detach().cpu()),
        "coverage_weakest_positive": float(weakest_positive.detach().cpu()),
        "coverage_boundary_negative": float(boundary_negative.detach().cpu()),
    }


def constrained_structured_coverage_loss(logits, labels, example, args):
    """Structured hinge between decoded and gold-complete feasible schema sets.

    Discrete set construction is intentionally detached. Gradients flow through the
    scores of the two selected sets, as in a structured perceptron/hinge objective.
    """
    zero = logits.sum() * 0.0
    eligible = bool(float(example.get("candidate_oracle_recall", 0.0)) >= 1.0 - 1e-9)
    positive_locals = {
        index for index, value in enumerate(labels.detach().cpu().tolist()) if value > 0.5
    }
    base_detail = {
        "structured_active": 0.0,
        "structured_feasible": 0.0,
        "structured_missing_gold": 0.0,
        "structured_swap_count": 0.0,
        "structured_violation": 0.0,
    }
    if not eligible or not positive_locals:
        return zero, base_detail

    scores = logits.detach().cpu().tolist()
    selector_kwargs = {
        "top_k": args.output_top_k,
        "max_tables": args.max_tables,
        "min_tables": None if args.min_tables < 0 else args.min_tables,
        "connectivity_weight": args.connectivity_weight,
        "baseline_retention_weight": args.baseline_retention_weight,
    }
    predicted, _ = constrained_topk(example, scores, **selector_kwargs)
    predicted_set = set(predicted)
    missing_gold = positive_locals - predicted_set
    if not missing_gold:
        detail = dict(base_detail)
        detail["structured_feasible"] = 1.0
        return zero, detail

    gold_feasible, gold_debug = constrained_topk(
        example,
        scores,
        required_local_ids=positive_locals,
        **selector_kwargs,
    )
    gold_set = set(gold_feasible)
    if not gold_debug.get("required_feasible") or not positive_locals.issubset(gold_set):
        return zero, base_detail

    predicted_only = sorted(predicted_set - gold_set)
    gold_only = sorted(gold_set - predicted_set)
    if not predicted_only or not gold_only:
        detail = dict(base_detail)
        detail["structured_feasible"] = 1.0
        return zero, detail

    predicted_score = logits[predicted_only].sum()
    gold_score = logits[gold_only].sum()
    missing_count = len(missing_gold)
    margin = float(getattr(args, "structured_coverage_margin", 0.1)) * missing_count
    violation = margin + predicted_score - gold_score
    # Normalize by the number of missing gold nodes so query complexity does not
    # silently change the effective structured-loss weight.
    loss = violation.clamp_min(0.0) / missing_count
    return loss, {
        "structured_active": 1.0,
        "structured_feasible": 1.0,
        "structured_missing_gold": float(missing_count),
        "structured_swap_count": float(max(len(predicted_only), len(gold_only))),
        "structured_violation": float(violation.detach().cpu()),
    }


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
    coverage_loss, coverage_detail = topk_coverage_loss(
        logits,
        labels,
        top_k=args.output_top_k,
        margin=float(getattr(args, "coverage_margin", 0.1)),
        temperature=float(getattr(args, "coverage_temperature", 0.2)),
        eligible=bool(tensors.get("coverage_eligible", True)),
    )
    coverage_weight = float(getattr(args, "coverage_loss_weight", 0.0))
    structured_weight = float(
        getattr(args, "structured_coverage_loss_weight", 0.0)
    )
    if structured_weight > 0.0:
        structured_loss, structured_detail = constrained_structured_coverage_loss(
            logits,
            labels,
            tensors["selector_example"],
            args,
        )
    else:
        structured_loss = logits.sum() * 0.0
        structured_detail = {
            "structured_active": 0.0,
            "structured_feasible": 0.0,
            "structured_missing_gold": 0.0,
            "structured_swap_count": 0.0,
            "structured_violation": 0.0,
        }
    total = (
        node_loss
        + args.role_loss_weight * role_loss
        + args.pairwise_loss_weight * pair_loss
        + coverage_weight * coverage_loss
        + structured_weight * structured_loss
    )
    return total, {
        "node_loss": float(node_loss.detach().cpu()),
        "role_loss": float(role_loss.detach().cpu()),
        "pairwise_loss": float(pair_loss.detach().cpu()),
        "coverage_loss": float(coverage_loss.detach().cpu()),
        "structured_coverage_loss": float(structured_loss.detach().cpu()),
        **coverage_detail,
        **structured_detail,
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
    loss_components = defaultdict(float)
    selected_raw = {}
    selected_constrained = {}
    selected_baseline = {}
    predictions = []
    with torch.no_grad():
        for example in examples:
            tensors = example_to_tensors(example, cache, maps, runtime, device)
            output = forward_model(model, tensors)
            loss, detail = reranker_loss(output, tensors, args, runtime)
            losses.append(float(loss.detach().cpu()))
            for key, value in detail.items():
                loss_components[key] += float(value)
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
    metrics.update(
        {
            f"loss_{key}": value / len(examples) if examples else 0.0
            for key, value in sorted(loss_components.items())
        }
    )
    active_count = loss_components.get("coverage_active", 0.0)
    metrics["loss_coverage_active_count"] = active_count
    metrics["loss_coverage_loss_per_active"] = (
        loss_components.get("coverage_loss", 0.0) / active_count
        if active_count
        else 0.0
    )
    metrics["loss_coverage_violation_per_active"] = (
        loss_components.get("coverage_violation", 0.0) / active_count
        if active_count
        else 0.0
    )
    structured_active = loss_components.get("structured_active", 0.0)
    metrics["loss_structured_active_count"] = structured_active
    metrics["loss_structured_loss_per_active"] = (
        loss_components.get("structured_coverage_loss", 0.0) / structured_active
        if structured_active
        else 0.0
    )
    metrics["loss_structured_missing_gold_per_active"] = (
        loss_components.get("structured_missing_gold", 0.0) / structured_active
        if structured_active
        else 0.0
    )
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


def train_epoch(
    model,
    examples,
    cache,
    maps,
    args,
    runtime,
    device,
    optimizer,
    epoch,
    progress_callback=None,
):
    model.train()
    shuffled = list(examples)
    random.Random(args.seed + epoch).shuffle(shuffled)
    total = 0.0
    components = {
        "node_loss": 0.0,
        "role_loss": 0.0,
        "pairwise_loss": 0.0,
        "coverage_loss": 0.0,
        "coverage_active": 0.0,
        "coverage_violation": 0.0,
        "structured_coverage_loss": 0.0,
        "structured_active": 0.0,
        "structured_feasible": 0.0,
        "structured_missing_gold": 0.0,
        "structured_swap_count": 0.0,
        "structured_violation": 0.0,
    }
    accumulation_steps = int(getattr(args, "gradient_accumulation_steps", 1))
    eval_every_examples = int(getattr(args, "eval_every_examples", 0))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if eval_every_examples < 0:
        raise ValueError("eval_every_examples must be non-negative")
    processed = 0
    optimizer_steps = 0
    next_evaluation = eval_every_examples if eval_every_examples else None
    for batch_start in range(0, len(shuffled), accumulation_steps):
        micro_batch = shuffled[batch_start : batch_start + accumulation_steps]
        optimizer.zero_grad()
        for example in micro_batch:
            tensors = example_to_tensors(example, cache, maps, runtime, device)
            output = forward_model(model, tensors)
            loss, detail = reranker_loss(output, tensors, args, runtime)
            # Graphs have different node counts and are currently materialized one at
            # a time. Averaging gradients over micro-batches gives an exact effective
            # batch without padding heterogeneous graphs.
            (loss / len(micro_batch)).backward()
            total += float(loss.detach().cpu())
            for key in components:
                components[key] += detail[key]
        runtime["torch"].nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer_steps += 1
        processed += len(micro_batch)
        if (
            progress_callback is not None
            and next_evaluation is not None
            and processed >= next_evaluation
            and processed < len(shuffled)
        ):
            partial = {
                "example_count": processed,
                "optimizer_steps": optimizer_steps,
                "loss": total / processed,
                **{
                    key: value / processed
                    for key, value in components.items()
                },
            }
            active_count = components["coverage_active"]
            partial["coverage_active_count"] = active_count
            partial["coverage_loss_per_active"] = (
                components["coverage_loss"] / active_count if active_count else 0.0
            )
            partial["coverage_violation_per_active"] = (
                components["coverage_violation"] / active_count
                if active_count
                else 0.0
            )
            structured_active = components["structured_active"]
            partial["structured_active_count"] = structured_active
            partial["structured_loss_per_active"] = (
                components["structured_coverage_loss"] / structured_active
                if structured_active
                else 0.0
            )
            partial["structured_missing_gold_per_active"] = (
                components["structured_missing_gold"] / structured_active
                if structured_active
                else 0.0
            )
            progress_callback(partial)
            model.train()
            while next_evaluation <= processed:
                next_evaluation += eval_every_examples
    count = max(len(shuffled), 1)
    result = {
        "loss": total / count,
        **{key: value / count for key, value in components.items()},
        "example_count": len(shuffled),
        "optimizer_steps": optimizer_steps,
        "effective_batch_size": accumulation_steps,
    }
    active_count = components["coverage_active"]
    result["coverage_active_count"] = active_count
    result["coverage_loss_per_active"] = (
        components["coverage_loss"] / active_count if active_count else 0.0
    )
    result["coverage_violation_per_active"] = (
        components["coverage_violation"] / active_count if active_count else 0.0
    )
    structured_active = components["structured_active"]
    result["structured_active_count"] = structured_active
    result["structured_loss_per_active"] = (
        components["structured_coverage_loss"] / structured_active
        if structured_active
        else 0.0
    )
    result["structured_missing_gold_per_active"] = (
        components["structured_missing_gold"] / structured_active
        if structured_active
        else 0.0
    )
    return result


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
    parser.add_argument(
        "--coverage-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the query-level budget-aware Top-K coverage surrogate.",
    )
    parser.add_argument(
        "--coverage-margin",
        type=float,
        default=0.1,
        help="Required score gap between the weakest positive and Top-K negative boundary.",
    )
    parser.add_argument(
        "--coverage-temperature",
        type=float,
        default=0.2,
        help="Temperature for smooth weakest-positive and softplus boundary loss.",
    )
    parser.add_argument(
        "--structured-coverage-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the constrained selection-aware structured coverage hinge."
        ),
    )
    parser.add_argument(
        "--structured-coverage-margin",
        type=float,
        default=0.1,
        help="Per-missing-gold margin for the constrained structured hinge.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of heterogeneous query graphs averaged per optimizer update.",
    )
    parser.add_argument(
        "--eval-every-examples",
        type=int,
        default=0,
        help=(
            "Evaluate after approximately this many training examples within each "
            "epoch; 0 preserves epoch-only evaluation."
        ),
    )
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
    parser.add_argument(
        "--dev-control-file",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Evaluate the best checkpoint on an aligned dev feature control without "
            "using it for checkpoint selection. May be repeated."
        ),
    )
    args = parser.parse_args()
    if args.coverage_loss_weight < 0:
        raise ValueError("coverage_loss_weight must be non-negative")
    if args.coverage_temperature <= 0:
        raise ValueError("coverage_temperature must be positive")
    if args.structured_coverage_loss_weight < 0:
        raise ValueError("structured_coverage_loss_weight must be non-negative")
    if args.structured_coverage_margin < 0:
        raise ValueError("structured_coverage_margin must be non-negative")

    runtime = import_runtime()
    torch = runtime["torch"]
    random.seed(args.seed)
    runtime["np"].random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    raw_train_examples = read_jsonl(Path(args.train_file), args.train_limit)
    raw_dev_examples = read_jsonl(Path(args.dev_file), args.dev_limit)
    train_examples, train_validation = validate_and_filter_examples(
        raw_train_examples, "train"
    )
    dev_examples, dev_validation = validate_and_filter_examples(
        raw_dev_examples, "dev"
    )
    if not train_examples or not dev_examples:
        raise ValueError("Train/dev factor graph files must be non-empty")
    numeric_dim = len(train_examples[0]["candidate_nodes"][0]["numeric_features"])
    if len(dev_examples[0]["candidate_nodes"][0]["numeric_features"]) != numeric_dim:
        raise ValueError("Train/dev numeric feature dimensions do not match")
    control_paths = parse_named_files(args.dev_control_file)
    dev_controls = {}
    control_validation = {}
    for name, path in control_paths.items():
        raw_control = read_jsonl(path, args.dev_limit)
        control_examples, report = validate_and_filter_examples(
            raw_control, f"dev_control::{name}"
        )
        validate_control_alignment(dev_examples, control_examples, name, numeric_dim)
        dev_controls[name] = control_examples
        control_validation[name] = report
    train_cache = load_cache(args.embedding_cache_dir, "train", runtime)
    dev_cache = load_cache(args.embedding_cache_dir, "dev", runtime)
    schema_relations = collect_schema_relations(train_examples + dev_examples)
    relation_count = len(train_examples[0]["role_labels"][0])
    maps = {
        "schema_relation_to_id": {name: index for index, name in enumerate(schema_relations)},
        "factor_numeric_dim": len(train_examples[0]["factors"][0]["numeric_features"]) if train_examples[0]["factors"] else 3,
    }
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
    write_json(
        output_dir / "data_validation.json",
        {
            "train": train_validation,
            "dev": dev_validation,
            "dev_controls": control_validation,
        },
    )
    config = vars(args).copy()
    config.update(
        {
            "dense_dim": train_cache["dense_dim"],
            "numeric_dim": numeric_dim,
            "factor_numeric_dim": maps["factor_numeric_dim"],
            "relation_count": relation_count,
            "schema_relations": schema_relations,
            "architecture": {
                "model_type": args.model_type,
                "schema_graph_enabled": args.model_type in {"schema_rgta", "factor_rgta"},
                "factor_graph_enabled": args.model_type == "factor_rgta",
            },
            "training_protocol": {
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "eval_every_examples": args.eval_every_examples,
                "checkpoint_selection": args.selection_metric,
                "early_stopping_unit": "epoch",
                "coverage_loss": {
                    "weight": args.coverage_loss_weight,
                    "top_k": args.output_top_k,
                    "margin": args.coverage_margin,
                    "temperature": args.coverage_temperature,
                    "eligibility": "candidate_oracle_recall == 1",
                    "negative_boundary_rank": "K - positive_count + 1",
                },
                "structured_coverage_loss": {
                    "weight": args.structured_coverage_loss_weight,
                    "margin_per_missing_gold": args.structured_coverage_margin,
                    "eligibility": "candidate_oracle_recall == 1",
                    "prediction": "constrained_topk(model_scores)",
                    "target": "constrained_topk(model_scores, required=gold)",
                    "normalization": "missing_gold_count",
                },
            },
            "innovation": (
                "A query-conditioned candidate schema subgraph is globally reranked before "
                "typed owner-closed subset decoding. Stage 10-B-fix1 separates graph-structure "
                "gain from optimization artifacts through OOF inputs, effective multi-graph "
                "batches, and within-epoch checkpoint observation. Stage 10-D aligns "
                "training with the typed owner-closed decoder through a structured "
                "gold-feasible versus predicted-set margin."
            ),
        }
    )
    write_json(output_dir / "train_config.json", config)

    best_epoch = None
    best_value = None
    best_state = None
    best_metrics = None
    best_checkpoint = None
    epochs_without_improvement = 0
    log_rows = []
    global_examples_seen = 0
    global_optimizer_steps = 0

    def evaluate_checkpoint(
        epoch,
        examples_seen_in_epoch,
        optimizer_steps_in_epoch,
        train_metrics,
        checkpoint_type,
    ):
        nonlocal best_epoch, best_value, best_state, best_metrics, best_checkpoint
        dev_metrics, _ = evaluate(
            model, dev_examples, dev_cache, maps, args, runtime, device, "dev"
        )
        selected_value = get_metric(dev_metrics, args.selection_metric)
        improved = is_better_metric(
            selected_value, best_value, args.selection_mode, args.min_delta
        )
        checkpoint = {
            "epoch": epoch,
            "examples_seen_in_epoch": examples_seen_in_epoch,
            "epoch_progress": examples_seen_in_epoch / max(len(train_examples), 1),
            "global_examples_seen": global_examples_seen + examples_seen_in_epoch,
            "optimizer_steps_in_epoch": optimizer_steps_in_epoch,
            "global_optimizer_steps": global_optimizer_steps + optimizer_steps_in_epoch,
            "checkpoint_type": checkpoint_type,
        }
        if improved:
            best_epoch = epoch
            best_value = selected_value
            best_state = clone_state_dict_to_cpu(model)
            best_metrics = dev_metrics
            best_checkpoint = checkpoint
            if not args.no_save_model:
                torch.save(best_state, output_dir / "factor_graph_reranker_model.pt")
        row = {
            **checkpoint,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{
                f"dev_{key}": value
                for key, value in dev_metrics.items()
                if key != "split"
            },
            "selection_value": selected_value,
            "best_epoch": best_epoch,
            "best_selection_value": best_value,
            "best_checkpoint": best_checkpoint,
            "is_best": improved,
        }
        log_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        return improved

    for epoch in range(1, args.epochs + 1):
        epoch_improved = False

        def progress_callback(partial_train_metrics):
            nonlocal epoch_improved
            improved = evaluate_checkpoint(
                epoch,
                int(partial_train_metrics["example_count"]),
                int(partial_train_metrics["optimizer_steps"]),
                partial_train_metrics,
                "periodic",
            )
            epoch_improved = epoch_improved or improved

        train_metrics = train_epoch(
            model,
            train_examples,
            train_cache,
            maps,
            args,
            runtime,
            device,
            optimizer,
            epoch,
            progress_callback=progress_callback if args.eval_every_examples else None,
        )
        improved = evaluate_checkpoint(
            epoch,
            int(train_metrics["example_count"]),
            int(train_metrics["optimizer_steps"]),
            train_metrics,
            "epoch_end",
        )
        epoch_improved = epoch_improved or improved
        global_examples_seen += int(train_metrics["example_count"])
        global_optimizer_steps += int(train_metrics["optimizer_steps"])
        if epoch_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
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
    control_metrics = {}
    for name, examples in dev_controls.items():
        metrics, control_predictions = evaluate(
            model,
            examples,
            dev_cache,
            maps,
            args,
            runtime,
            device,
            f"dev_control::{name}",
        )
        control_metrics[name] = metrics
        write_json(output_dir / f"dev_control_{name}_metrics.json", metrics)
        write_jsonl(
            output_dir / f"dev_control_{name}_predictions.jsonl",
            control_predictions,
        )
    summary = {
        "best_epoch": best_epoch,
        "best_checkpoint": best_checkpoint,
        "selection_metric": args.selection_metric,
        "selection_value": best_value,
        "dev_metrics": final_metrics,
        "best_metrics_during_training": best_metrics,
        "dev_control_metrics": control_metrics,
        "last_epoch": log_rows[-1]["epoch"] if log_rows else 0,
        "evaluation_count": len(log_rows),
        "global_examples_seen": global_examples_seen,
        "global_optimizer_steps": global_optimizer_steps,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
