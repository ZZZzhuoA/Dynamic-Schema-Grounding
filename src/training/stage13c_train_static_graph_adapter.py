"""Align a frozen query-conditioned schema GNN with a frozen code LLM."""

import argparse
import json
import random
import re
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
)
from src.modeling.stage13c_static_runtime import (  # noqa: E402
    frozen_graph_memory,
    graph_tensors,
    load_frozen_typed_graph_encoder,
)
from src.modeling.static_graph_adapter import (  # noqa: E402
    GraphMemoryProjector,
    StaticGraphConditionedCausalLM,
)
from src.training.stage13b_train_typed_ra_decoder import load_cache  # noqa: E402
from src.training.stage5_train_dsg_grounder import read_jsonl, write_json  # noqa: E402


def record_index(example, fallback=0):
    return int(example.get("metadata", {}).get("record_index", example.get("record_index", fallback)))


def gold_sql(example):
    targets = example.get("training_targets", {})
    return str(targets.get("gold_sql") or targets.get("sql") or example.get("gold_sql") or "").strip()


def has_structural_sql(sql):
    return bool(re.search(r"(?i)\bjoin\b|\bfrom\b[^;]+,", str(sql or "")))


def encode_example(tokenizer, example, max_length, input_device, torch, include_foreign_keys):
    inputs = example.get("inference_inputs", example)
    sql = gold_sql(example)
    prompt = build_full_schema_prompt(
        inputs, example, include_foreign_keys=include_foreign_keys
    )
    chat = build_chat_prompt(tokenizer, prompt)
    prompt_ids = tokenizer(chat, add_special_tokens=False)["input_ids"]
    sql_ids, roles = sql_token_roles(tokenizer, sql, example, append_eos=True)
    if not sql or len(prompt_ids) + len(sql_ids) > max_length:
        return None
    input_ids = torch.tensor([prompt_ids + sql_ids], dtype=torch.long, device=input_device)
    labels = input_ids.clone(); labels[:, : len(prompt_ids)] = -100
    token_roles = torch.tensor(
        [[TOKEN_ROLE_BASE] * len(prompt_ids) + roles], dtype=torch.long
    )
    return input_ids, labels, token_roles, prompt


def causal_nll(logits, labels, torch):
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].to(logits.device).contiguous()
    valid = labels.ne(-100)
    safe = labels.masked_fill(~valid, 0)
    nll = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]), safe.view(-1), reduction="none"
    ).view_as(labels)
    return nll, valid


def node_target_embeddings(tokenizer, embedding_layer, nodes, device, torch):
    targets = []
    embedding_device = embedding_layer.weight.device
    with torch.no_grad():
        for node in nodes:
            name = str(node.get("name") or "")
            ids = tokenizer(name, add_special_tokens=False)["input_ids"]
            if not ids:
                targets.append(torch.zeros(embedding_layer.weight.shape[1], device=device))
                continue
            token_ids = torch.tensor(ids, dtype=torch.long, device=embedding_device)
            targets.append(embedding_layer(token_ids).float().mean(0).to(device))
    return torch.stack(targets)


def contrastive_alignment(projected, targets, temperature, torch):
    left = torch.nn.functional.normalize(projected.float(), dim=-1)
    right = torch.nn.functional.normalize(targets.float(), dim=-1)
    logits = left @ right.t() / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.t(), labels)
    )
    recall = (logits.argmax(-1) == labels).float().mean()
    return loss, recall


def generation_objective(
    normal_nll, negative_nll, valid, token_roles, args, torch,
    use_topology_counterfactual=True,
):
    roles = token_roles[:, 1:].to(normal_nll.device)
    weighted_ce = weighted_token_ce(normal_nll, valid, roles, args, torch)
    schema_mask = valid & roles.eq(1)
    if use_topology_counterfactual and schema_mask.any():
        gain = (negative_nll.to(normal_nll.device) - normal_nll)[schema_mask]
        counterfactual = torch.relu(float(args.counterfactual_margin) - gain).mean()
        mean_gain = gain.mean()
        improvement = gain.gt(0).float().mean()
    else:
        counterfactual = normal_nll.sum() * 0.0
        mean_gain = normal_nll.sum() * 0.0
        improvement = normal_nll.sum() * 0.0
    return weighted_ce + args.counterfactual_weight * counterfactual, {
        "weighted_ce": weighted_ce,
        "counterfactual_loss": counterfactual,
        "schema_logprob_gain": mean_gain,
        "schema_improvement_rate": improvement,
    }


def weighted_token_ce(nll, valid, roles, args, torch):
    weights = torch.ones_like(nll)
    role_weights = {
        0: args.base_token_weight,
        1: args.schema_token_weight,
        2: args.operator_token_weight,
        3: args.value_token_weight,
    }
    for role_id, weight in role_weights.items():
        weights = torch.where(roles.eq(role_id), float(weight), weights)
    active = valid.to(nll.dtype)
    return (nll * weights * active).sum() / (weights * active).sum().clamp_min(1.0)


def evaluate_identity(examples, tokenizer, wrapper, args, torch):
    """Measure the untouched frozen LLM before any graph adapter is active."""
    wrapper.clear_graph_memory()
    input_device = wrapper.base_model.get_input_embeddings().weight.device
    losses, used, skipped = [], 0, 0
    with torch.no_grad():
        for example in examples:
            encoded = encode_example(
                tokenizer, example, args.max_length, input_device, torch,
                args.prompt_includes_foreign_keys,
            )
            if encoded is None:
                skipped += 1
                continue
            input_ids, labels, token_roles, _ = encoded
            output = wrapper(input_ids=input_ids, use_cache=False)
            nll, valid = causal_nll(output.logits, labels, torch)
            roles = token_roles[:, 1:].to(nll.device)
            losses.append(float(weighted_token_ce(nll, valid, roles, args, torch).cpu()))
            used += 1
    return {
        "example_count": used,
        "skipped_count": skipped,
        "mean_weighted_ce": sum(losses) / len(losses) if losses else 0.0,
    }


def run_split(
    examples, cache, graph_encoder, relation_to_id, projector, wrapper,
    tokenizer, args, torch, graph_device, optimizer=None,
):
    training = optimizer is not None
    graph_encoder.eval(); wrapper.base_model.eval()
    projector.train(training); wrapper.adapters.train(training)
    values = {name: [] for name in (
        "loss", "weighted_ce", "alignment_loss", "alignment_recall@1",
        "counterfactual_loss", "schema_logprob_gain", "schema_improvement_rate",
    )}
    used = skipped = steps = counterfactual_examples = 0
    diagnostic_values = {}
    if training:
        optimizer.zero_grad(set_to_none=True)
    input_device = wrapper.base_model.get_input_embeddings().weight.device
    for example in examples:
        encoded = encode_example(
            tokenizer, example, args.max_length, input_device, torch,
            args.prompt_includes_foreign_keys,
        )
        if encoded is None:
            skipped += 1; continue
        input_ids, labels, token_roles, _ = encoded
        tensors = graph_tensors(example, cache, relation_to_id, graph_device)
        raw_memory = frozen_graph_memory(
            graph_encoder, tensors, corrupt=False, relation_to_id=relation_to_id
        )
        negative_raw = frozen_graph_memory(
            graph_encoder, tensors, corrupt=True, relation_to_id=relation_to_id
        )
        dense_semantics, query_embedding = tensors[0], tensors[1]
        with torch.no_grad():
            negative_memory = projector(
                negative_raw, dense_semantics, query_embedding
            ).detach()
            wrapper.set_graph_memory(negative_memory)
            negative_output = wrapper(input_ids=input_ids, use_cache=False)
            negative_nll, _ = causal_nll(negative_output.logits, labels, torch)
            negative_nll = negative_nll.detach()
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            projected = projector(
                raw_memory, dense_semantics, query_embedding,
                return_components=True,
            )
            memory = projected["memory"]
            targets = node_target_embeddings(
                tokenizer, wrapper.base_model.get_input_embeddings(), tensors[-1],
                memory.device, torch,
            )
            alignment_loss, alignment_recall = contrastive_alignment(
                projected["semantic_memory"], targets, args.alignment_temperature, torch
            )
            wrapper.set_graph_memory(memory)
            output = wrapper(input_ids=input_ids, use_cache=False)
            normal_nll, valid = causal_nll(output.logits, labels, torch)
            counterfactual_eligible = has_structural_sql(gold_sql(example))
            generation_loss, metrics = generation_objective(
                normal_nll, negative_nll, valid, token_roles, args, torch,
                use_topology_counterfactual=counterfactual_eligible,
            )
            loss = generation_loss + args.alignment_weight * alignment_loss
            if training:
                (loss / args.gradient_accumulation_steps).backward()
        values["loss"].append(float(loss.detach().cpu()))
        values["alignment_loss"].append(float(alignment_loss.detach().cpu()))
        values["alignment_recall@1"].append(float(alignment_recall.detach().cpu()))
        for name, metric in metrics.items():
            if name in {
                "counterfactual_loss", "schema_logprob_gain",
                "schema_improvement_rate",
            } and not counterfactual_eligible:
                continue
            values[name].append(float(metric.detach().cpu()))
        counterfactual_examples += int(counterfactual_eligible)
        for layer, diagnostics in wrapper.diagnostics().items():
            bucket = diagnostic_values.setdefault(layer, {})
            for name, value in diagnostics.items():
                bucket.setdefault(name, []).append(float(value))
        used += 1
        if training and used % args.gradient_accumulation_steps == 0:
            parameters = list(projector.parameters()) + list(wrapper.adapters.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); steps += 1
        if training and args.log_every and used % args.log_every == 0:
            print(json.dumps({
                "used": used,
                "mean_loss": sum(values["loss"]) / used,
                "mean_schema_logprob_gain": sum(values["schema_logprob_gain"]) / used,
            }))
    if training and used % args.gradient_accumulation_steps:
        parameters = list(projector.parameters()) + list(wrapper.adapters.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        optimizer.step(); optimizer.zero_grad(set_to_none=True); steps += 1
    wrapper.clear_graph_memory()
    return {
        "example_count": used,
        "skipped_count": skipped,
        "optimizer_steps": steps,
        "counterfactual_example_count": counterfactual_examples,
        **{f"mean_{name}": sum(rows) / len(rows) if rows else 0.0 for name, rows in values.items()},
        "adapter_diagnostics": {
            layer: {
                name: sum(rows) / len(rows) if rows else 0.0
                for name, rows in values_by_name.items()
            }
            for layer, values_by_name in diagnostic_values.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--graph-checkpoint", required=True)
    parser.add_argument("--graph-summary", required=True)
    parser.add_argument("--train-graph-file", required=True)
    parser.add_argument("--dev-graph-file", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--adapter-layer-fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--adapter-heads", type=int, default=8)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--residual-scale-init", type=float, default=0.02)
    parser.add_argument("--structure-scale-init", type=float, default=0.1)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--alignment-temperature", type=float, default=0.07)
    parser.add_argument("--counterfactual-weight", type=float, default=0.5)
    parser.add_argument("--counterfactual-margin", type=float, default=0.05)
    parser.add_argument("--min-dev-schema-improvement-rate", type=float, default=0.5)
    parser.add_argument("--max-dev-ce-degradation", type=float, default=0.05)
    parser.add_argument("--base-token-weight", type=float, default=0.5)
    parser.add_argument("--schema-token-weight", type=float, default=4.0)
    parser.add_argument("--operator-token-weight", type=float, default=2.5)
    parser.add_argument("--value-token-weight", type=float, default=3.0)
    parser.add_argument("--prompt-includes-foreign-keys", action="store_true")
    parser.add_argument("--graph-device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    train = read_jsonl(Path(args.train_graph_file), args.train_limit)
    dev = read_jsonl(Path(args.dev_graph_file), args.dev_limit)
    train_cache = load_cache(args.embedding_cache_dir, "train", np)
    dev_cache = load_cache(args.embedding_cache_dir, "dev", np)
    graph_device = torch.device(args.graph_device)
    graph_encoder, relation_to_id, graph_summary = load_frozen_typed_graph_encoder(
        args.graph_summary, args.graph_checkpoint, train_cache["dense_dim"], graph_device
    )
    fractions = [float(value) for value in args.adapter_layer_fractions.split(",") if value.strip()]
    graph_dim = int(graph_summary.get("config", {}).get("hidden_dim", 256))
    llm_dim = int(base_model.config.hidden_size)
    # Keep the small trainable graph-to-LLM projector in FP32. Decoder-layer
    # adapters cast its output to each frozen LLM layer's dtype on consumption.
    projector = GraphMemoryProjector(
        graph_dim=graph_dim,
        llm_dim=llm_dim,
        semantic_dim=train_cache["dense_dim"],
        query_dim=train_cache["dense_dim"],
        dropout=args.adapter_dropout,
        structure_scale_init=args.structure_scale_init,
    ).to(graph_device)
    wrapper = StaticGraphConditionedCausalLM(
        base_model, graph_dim=llm_dim, layer_fractions=fractions,
        num_heads=args.adapter_heads, dropout=args.adapter_dropout,
        residual_scale_init=args.residual_scale_init,
    )
    wrapper.freeze_base_model()
    parameters = list(projector.parameters()) + list(wrapper.adapters.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    valid_checkpoint_path = output_dir / "static_graph_adapter.pt"
    valid_checkpoint_path.unlink(missing_ok=True)
    identity_dev = evaluate_identity(dev, tokenizer, wrapper, args, torch)
    print(json.dumps({"epoch": 0, "identity_dev": identity_dev}, ensure_ascii=False))
    history, best_value, best_epoch = [], None, None
    fallback_value, fallback_epoch = None, None
    for epoch in range(1, args.epochs + 1):
        shuffled = list(train); random.Random(args.seed + epoch).shuffle(shuffled)
        train_metrics = run_split(
            shuffled, train_cache, graph_encoder, relation_to_id, projector,
            wrapper, tokenizer, args, torch, graph_device, optimizer,
        )
        dev_metrics = run_split(
            dev, dev_cache, graph_encoder, relation_to_id, projector,
            wrapper, tokenizer, args, torch, graph_device,
        )
        row = {"epoch": epoch, "train": train_metrics, "dev": dev_metrics}
        history.append(row); print(json.dumps(row, ensure_ascii=False))
        value = dev_metrics["mean_loss"]
        checkpoint = {
            "projector_state_dict": {k: v.detach().cpu() for k, v in projector.state_dict().items()},
            "adapter_state_dict": wrapper.adapter_state_dict(),
        }
        torch.save(checkpoint, output_dir / "last_static_graph_adapter.pt")
        if fallback_value is None or value < fallback_value:
            fallback_value, fallback_epoch = value, epoch
            torch.save(checkpoint, output_dir / "lowest_loss_static_graph_adapter.pt")
        causally_valid = (
            dev_metrics["mean_schema_logprob_gain"] > 0.0
            and dev_metrics["mean_schema_improvement_rate"]
            >= args.min_dev_schema_improvement_rate
            and dev_metrics["mean_weighted_ce"]
            <= identity_dev["mean_weighted_ce"] * (1.0 + args.max_dev_ce_degradation)
        )
        if causally_valid and (best_value is None or value < best_value):
            best_value, best_epoch = value, epoch
            torch.save(checkpoint, valid_checkpoint_path)
    config = {
        **vars(args), "layer_indices": wrapper.layer_indices,
        "layer_path": wrapper.layer_path, "graph_dim": graph_dim,
        "llm_hidden_dim": llm_dim, "relations": list(relation_to_id),
        "frozen_modules": ["typed_rgta_graph_encoder", "code_llm"],
        "trainable_modules": ["graph_memory_projector", "cross_attention", "gate"],
    }
    write_json(output_dir / "adapter_config.json", config)
    write_json(output_dir / "training_summary.json", {
        "best_epoch": best_epoch,
        "selection_metric": "mean_loss",
        "selection_value": best_value,
        "valid_alignment_checkpoint_found": best_epoch is not None,
        "fallback_epoch": fallback_epoch,
        "fallback_selection_value": fallback_value,
        "identity_dev": identity_dev,
        "selection_constraint": {
            "mean_schema_logprob_gain": "> 0",
            "mean_schema_improvement_rate": f">= {args.min_dev_schema_improvement_rate}",
            "mean_weighted_ce": (
                f"<= identity * (1 + {args.max_dev_ce_degradation})"
            ),
        },
        "history": history,
        "config": config,
    })
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
