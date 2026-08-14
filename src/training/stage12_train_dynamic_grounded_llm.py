"""Train decoder-layer adapters that connect dynamic RGTA grounding to a frozen LLM."""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stage12_llm_grounding_data import (  # noqa: E402
    TOKEN_ROLE_BASE,
    TOKEN_ROLE_NAMES,
    build_chat_prompt,
    build_full_schema_prompt,
    sql_token_roles,
    teacher_forcing_token_steps,
)
from src.modeling.dynamic_grounded_llm import DynamicGroundedCausalLM  # noqa: E402
from src.modeling.stage12_grounding_runtime import (  # noqa: E402
    load_controller,
    trajectory_grounding_context,
)
from src.training.stage10_train_factor_graph_reranker import load_cache  # noqa: E402
from src.training.stage5_train_dsg_grounder import read_jsonl, write_json  # noqa: E402


def graph_by_record_index(path, limit=None):
    rows = read_jsonl(Path(path), limit)
    result = {}
    for index, row in enumerate(rows):
        record_index = int(row.get("metadata", {}).get("record_index", index))
        result[record_index] = row
    return result


def encode_teacher_forcing(tokenizer, prompt, sql, steps, graph, max_length, device, torch):
    chat = build_chat_prompt(tokenizer, prompt)
    prompt_ids = tokenizer(chat, add_special_tokens=False)["input_ids"]
    sql_ids, sql_roles = sql_token_roles(
        tokenizer, sql, graph, append_eos=True
    )
    if len(prompt_ids) + len(sql_ids) > max_length:
        return None
    input_ids = torch.tensor([prompt_ids + sql_ids], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    token_steps = torch.tensor(
        teacher_forcing_token_steps(tokenizer, prompt_ids, sql_ids, steps),
        dtype=torch.long,
    )
    token_roles = torch.tensor(
        [[TOKEN_ROLE_BASE] * len(prompt_ids) + sql_roles],
        dtype=torch.long,
    )
    return input_ids, labels, token_steps, token_roles


def causal_token_nll(logits, labels, torch):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].to(shift_logits.device).contiguous()
    valid = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    nll = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        safe_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    return nll, valid


def semantic_utility_objective(
    normal_nll, zero_nll, valid, token_roles, role_weights,
    counterfactual_weight, counterfactual_margin, preservation_weight, torch,
):
    roles = token_roles[:, 1:].to(normal_nll.device)
    weights = torch.ones_like(normal_nll)
    for role_id, weight in role_weights.items():
        weights = torch.where(roles.eq(int(role_id)), float(weight), weights)
    valid_float = valid.to(normal_nll.dtype)
    weighted_denominator = (weights * valid_float).sum().clamp_min(1.0)
    weighted_ce = (normal_nll * weights * valid_float).sum() / weighted_denominator
    standard_ce = (normal_nll * valid_float).sum() / valid_float.sum().clamp_min(1.0)

    key_mask = valid & roles.ne(TOKEN_ROLE_BASE)
    base_mask = valid & roles.eq(TOKEN_ROLE_BASE)
    zero_nll = zero_nll.to(normal_nll.device, normal_nll.dtype)
    if key_mask.any():
        key_gain = (zero_nll - normal_nll)[key_mask]
        utility_loss = torch.relu(float(counterfactual_margin) - key_gain).mean()
        mean_key_gain = key_gain.mean()
        key_improvement_rate = key_gain.gt(0).to(normal_nll.dtype).mean()
    else:
        utility_loss = normal_nll.sum() * 0.0
        mean_key_gain = normal_nll.sum() * 0.0
        key_improvement_rate = normal_nll.sum() * 0.0
    if base_mask.any():
        preservation_loss = ((normal_nll - zero_nll)[base_mask] ** 2).mean()
    else:
        preservation_loss = normal_nll.sum() * 0.0
    objective = (
        weighted_ce
        + float(counterfactual_weight) * utility_loss
        + float(preservation_weight) * preservation_loss
    )
    return objective, {
        "standard_ce": standard_ce,
        "weighted_ce": weighted_ce,
        "counterfactual_loss": utility_loss,
        "preservation_loss": preservation_loss,
        "mean_key_logprob_gain": mean_key_gain,
        "key_improvement_rate": key_improvement_rate,
        "key_token_count": key_mask.sum(),
        "base_token_count": base_mask.sum(),
    }


def run_split(
    wrapper, controller, trajectories, full_graphs, cache, relation_to_id,
    tokenizer, args, runtime, controller_device, optimizer=None,
):
    torch = runtime["torch"]
    training = optimizer is not None
    wrapper.base_model.eval()
    wrapper.adapters.train(training)
    metric_values = {
        "objective_loss": [], "standard_ce": [], "weighted_ce": [],
        "counterfactual_loss": [], "preservation_loss": [],
        "mean_key_logprob_gain": [], "key_improvement_rate": [],
    }
    role_counts = {name: 0 for name in TOKEN_ROLE_NAMES.values()}
    used, skipped_missing, skipped_length = 0, 0, 0
    optimizer_steps = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for row_index, trajectory in enumerate(trajectories):
        record_index = int(trajectory["record_index"])
        graph = full_graphs.get(record_index)
        if graph is None or not trajectory.get("trajectory_steps"):
            skipped_missing += 1
            continue
        prompt = build_full_schema_prompt(trajectory, graph)
        encoded = encode_teacher_forcing(
            tokenizer, prompt, trajectory.get("gold_sql", ""),
            trajectory["trajectory_steps"], graph, args.max_length,
            wrapper.base_model.get_input_embeddings().weight.device, torch,
        )
        if encoded is None:
            skipped_length += 1
            continue
        input_ids, labels, token_steps, token_roles = encoded
        grounding, steering, _ = trajectory_grounding_context(
            controller, trajectory, cache, relation_to_id, runtime, controller_device
        )
        zero_nll = None
        if args.training_objective == "semantic_utility":
            wrapper.clear_grounding_context()
            with torch.no_grad():
                zero_output = wrapper(input_ids=input_ids, use_cache=False)
                zero_nll, _ = causal_token_nll(zero_output.logits, labels, torch)
                zero_nll = zero_nll.detach()
            del zero_output
        wrapper.set_grounding_context(grounding, steering, token_steps)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            output = wrapper(input_ids=input_ids, use_cache=False)
            normal_nll, valid = causal_token_nll(output.logits, labels, torch)
            if args.training_objective == "semantic_utility":
                role_weights = {
                    0: args.base_token_weight,
                    1: args.schema_token_weight,
                    2: args.operator_token_weight,
                    3: args.value_token_weight,
                }
                loss, metrics = semantic_utility_objective(
                    normal_nll, zero_nll, valid, token_roles, role_weights,
                    args.counterfactual_loss_weight,
                    args.counterfactual_margin,
                    args.preservation_loss_weight,
                    torch,
                )
            else:
                valid_float = valid.to(normal_nll.dtype)
                loss = (normal_nll * valid_float).sum() / valid_float.sum().clamp_min(1.0)
                zero = loss.detach() * 0.0
                metrics = {
                    "standard_ce": loss,
                    "weighted_ce": loss,
                    "counterfactual_loss": zero,
                    "preservation_loss": zero,
                    "mean_key_logprob_gain": zero,
                    "key_improvement_rate": zero,
                }
            if training:
                (loss / args.gradient_accumulation_steps).backward()
        metric_values["objective_loss"].append(float(loss.detach().cpu()))
        for name, value in metrics.items():
            if name in metric_values:
                metric_values[name].append(float(value.detach().cpu()))
        valid_roles = token_roles[:, 1:].flatten()[valid.detach().cpu().flatten()]
        for role_id, role_name in TOKEN_ROLE_NAMES.items():
            role_counts[role_name] += int(valid_roles.eq(role_id).sum())
        used += 1
        if training and used % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(wrapper.adapters.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        if training and args.log_every and used % args.log_every == 0:
            print(json.dumps({
                "used": used,
                "mean_objective_loss": sum(metric_values["objective_loss"]) / used,
                "mean_key_logprob_gain": sum(metric_values["mean_key_logprob_gain"]) / used,
            }))
    if training and used % args.gradient_accumulation_steps:
        torch.nn.utils.clip_grad_norm_(wrapper.adapters.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    wrapper.clear_grounding_context()
    return {
        "example_count": used,
        **{
            f"mean_{name}": sum(values) / len(values) if values else 0.0
            for name, values in metric_values.items()
        },
        "mean_token_loss": (
            sum(metric_values["standard_ce"]) / len(metric_values["standard_ce"])
            if metric_values["standard_ce"] else 0.0
        ),
        "token_role_counts": role_counts,
        "skipped_missing": skipped_missing,
        "skipped_over_max_length": skipped_length,
        "optimizer_steps": optimizer_steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--controller-checkpoint", required=True)
    parser.add_argument("--controller-summary", required=True)
    parser.add_argument("--train-trajectory-file", required=True)
    parser.add_argument("--dev-trajectory-file", required=True)
    parser.add_argument("--train-full-graph-file", required=True)
    parser.add_argument("--dev-full-graph-file", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--adapter-layer-fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--adapter-heads", type=int, default=8)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument(
        "--training-objective", choices=("ce", "semantic_utility"), default="ce"
    )
    parser.add_argument("--base-token-weight", type=float, default=0.5)
    parser.add_argument("--schema-token-weight", type=float, default=4.0)
    parser.add_argument("--operator-token-weight", type=float, default=2.5)
    parser.add_argument("--value-token-weight", type=float, default=3.0)
    parser.add_argument("--counterfactual-loss-weight", type=float, default=0.5)
    parser.add_argument("--counterfactual-margin", type=float, default=0.05)
    parser.add_argument("--preservation-loss-weight", type=float, default=0.1)
    parser.add_argument("--controller-device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1 because epoch 0 is diagnostic only")

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    runtime = {"np": np, "torch": torch}
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    train_trajectories = read_jsonl(Path(args.train_trajectory_file), args.train_limit)
    dev_trajectories = read_jsonl(Path(args.dev_trajectory_file), args.dev_limit)
    train_graphs = graph_by_record_index(args.train_full_graph_file, args.train_limit)
    dev_graphs = graph_by_record_index(args.dev_full_graph_file, args.dev_limit)
    train_cache = load_cache(args.embedding_cache_dir, "train", runtime)
    dev_cache = load_cache(args.embedding_cache_dir, "dev", runtime)
    sample = next(row for row in train_trajectories if row.get("candidate_nodes"))
    numeric_dim = len(sample["candidate_nodes"][0]["numeric_features"])
    controller_device = torch.device(args.controller_device)
    controller, relation_to_id, controller_summary = load_controller(
        args.controller_summary, args.controller_checkpoint,
        train_cache["dense_dim"], numeric_dim, controller_device,
    )
    fractions = [float(value) for value in args.adapter_layer_fractions.split(",") if value.strip()]
    grounding_dim = int(controller_summary.get("config", {}).get("hidden_dim", 256))
    wrapper = DynamicGroundedCausalLM(
        base_model, grounding_dim=grounding_dim, layer_fractions=fractions,
        num_heads=args.adapter_heads, dropout=args.adapter_dropout,
    )
    wrapper.freeze_base_model()
    optimizer = torch.optim.AdamW(
        wrapper.adapters.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_dev = run_split(
        wrapper, controller, dev_trajectories, dev_graphs, dev_cache,
        relation_to_id, tokenizer, args, runtime, controller_device,
    )
    # Epoch 0 is an exact identity adapter. Keep it as a diagnostic baseline,
    # but never let it win selection for the trained adapter used at inference.
    torch.save(wrapper.adapter_state_dict(), output_dir / "identity_dynamic_llm_adapters.pt")
    best_loss, best_epoch, best_scales = None, None, None
    selection_metric = (
        "mean_objective_loss"
        if args.training_objective == "semantic_utility"
        else "mean_token_loss"
    )
    history = [{
        "epoch": 0,
        "train": None,
        "dev": initial_dev,
        "adapter_scales": wrapper.adapter_scale_summary(),
        "checkpoint_role": "identity_baseline_only",
    }]
    print(json.dumps(history[0], ensure_ascii=False))
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(train_trajectories)
        train_metrics = run_split(
            wrapper, controller, train_trajectories, train_graphs, train_cache,
            relation_to_id, tokenizer, args, runtime, controller_device, optimizer,
        )
        dev_metrics = run_split(
            wrapper, controller, dev_trajectories, dev_graphs, dev_cache,
            relation_to_id, tokenizer, args, runtime, controller_device,
        )
        scales = wrapper.adapter_scale_summary()
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "dev": dev_metrics,
            "adapter_scales": scales,
            "checkpoint_role": "trained_candidate",
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        torch.save(wrapper.adapter_state_dict(), output_dir / "last_dynamic_llm_adapters.pt")
        selection_value = dev_metrics[selection_metric]
        if best_loss is None or selection_value < best_loss:
            best_loss, best_epoch = selection_value, epoch
            best_scales = scales
            torch.save(wrapper.adapter_state_dict(), output_dir / "dynamic_llm_adapters.pt")
    config = {
        **vars(args),
        "layer_indices": wrapper.layer_indices,
        "layer_path": wrapper.layer_path,
        "grounding_dim": grounding_dim,
        "llm_hidden_dim": int(base_model.config.hidden_size),
    }
    write_json(output_dir / "adapter_config.json", config)
    write_json(
        output_dir / "training_summary.json",
        {
            "best_epoch": best_epoch,
            "selection_metric": selection_metric,
            "selection_value": best_loss,
            "best_dev_token_loss": history[best_epoch]["dev"]["mean_token_loss"],
            "best_adapter_scales": best_scales,
            "identity_dev_token_loss": initial_dev["mean_token_loss"],
            "identity_dev_objective_loss": initial_dev["mean_objective_loss"],
            "selection_policy": (
                f"minimum {selection_metric} among trained epochs only; epoch 0 is excluded"
            ),
            "history": history,
            "config": config,
        },
    )
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
