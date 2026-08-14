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
    build_chat_prompt,
    build_full_schema_prompt,
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


def encode_teacher_forcing(tokenizer, prompt, sql, steps, max_length, device, torch):
    chat = build_chat_prompt(tokenizer, prompt)
    prompt_ids = tokenizer(chat, add_special_tokens=False)["input_ids"]
    sql_ids = tokenizer(str(sql).strip(), add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        sql_ids = sql_ids + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(sql_ids) > max_length:
        return None
    input_ids = torch.tensor([prompt_ids + sql_ids], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    token_steps = torch.tensor(
        teacher_forcing_token_steps(tokenizer, prompt_ids, sql_ids, steps),
        dtype=torch.long,
    )
    return input_ids, labels, token_steps


def run_split(
    wrapper, controller, trajectories, full_graphs, cache, relation_to_id,
    tokenizer, args, runtime, controller_device, optimizer=None,
):
    torch = runtime["torch"]
    training = optimizer is not None
    wrapper.base_model.eval()
    wrapper.adapters.train(training)
    losses, used, skipped_missing, skipped_length = [], 0, 0, 0
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
            trajectory["trajectory_steps"], args.max_length,
            wrapper.base_model.get_input_embeddings().weight.device, torch,
        )
        if encoded is None:
            skipped_length += 1
            continue
        input_ids, labels, token_steps = encoded
        grounding, steering, _ = trajectory_grounding_context(
            controller, trajectory, cache, relation_to_id, runtime, controller_device
        )
        wrapper.set_grounding_context(grounding, steering, token_steps)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            output = wrapper(input_ids=input_ids, labels=labels, use_cache=False)
            loss = output.loss
            if training:
                (loss / args.gradient_accumulation_steps).backward()
        losses.append(float(loss.detach().cpu()))
        used += 1
        if training and used % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(wrapper.adapters.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        if training and args.log_every and used % args.log_every == 0:
            print(json.dumps({"used": used, "mean_loss": sum(losses) / len(losses)}))
    if training and used % args.gradient_accumulation_steps:
        torch.nn.utils.clip_grad_norm_(wrapper.adapters.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    wrapper.clear_grounding_context()
    return {
        "example_count": used,
        "mean_token_loss": sum(losses) / len(losses) if losses else 0.0,
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
    parser.add_argument("--controller-device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    best_loss, best_epoch = initial_dev["mean_token_loss"], 0
    torch.save(wrapper.adapter_state_dict(), output_dir / "dynamic_llm_adapters.pt")
    history = [{"epoch": 0, "train": None, "dev": initial_dev}]
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
        row = {"epoch": epoch, "train": train_metrics, "dev": dev_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if best_loss is None or dev_metrics["mean_token_loss"] < best_loss:
            best_loss, best_epoch = dev_metrics["mean_token_loss"], epoch
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
        {"best_epoch": best_epoch, "best_dev_token_loss": best_loss, "history": history, "config": config},
    )
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
