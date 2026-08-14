"""Autoregressive SQL generation with dynamic RGTA decoder-layer adapters."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stage12_llm_grounding_data import build_chat_prompt, build_full_schema_prompt  # noqa: E402
from src.modeling.dynamic_grounded_llm import DynamicGroundedCausalLM  # noqa: E402
from src.modeling.stage12_grounding_runtime import (  # noqa: E402
    load_controller,
    partial_sql_grounding_context,
)
from src.training.stage10_train_factor_graph_reranker import load_cache  # noqa: E402
from src.training.stage11_train_dynamic_grounding_controller import trajectory_tensors  # noqa: E402
from src.training.stage12_train_dynamic_grounded_llm import graph_by_record_index  # noqa: E402
from src.training.stage5_train_dsg_grounder import read_jsonl  # noqa: E402


def clean_sql(text):
    text = str(text or "").strip()
    if "```" in text:
        pieces = [piece.strip() for piece in text.split("```")]
        candidates = [piece[3:].strip() if piece.lower().startswith("sql") else piece for piece in pieces]
        text = next((piece for piece in candidates if "select" in piece.lower() or "with" in piece.lower()), text)
    lower = text.lower()
    starts = [position for token in ("select", "with") if (position := lower.find(token)) >= 0]
    if starts:
        text = text[min(starts):]
    return text.rstrip().rstrip(";") + ";" if text else ""


def intervene(tokens, steering, mode, torch):
    if mode == "none":
        return tokens, steering
    if mode == "zero":
        return torch.zeros_like(tokens), torch.zeros_like(steering)
    if mode == "random":
        return torch.randn_like(tokens), torch.randn_like(steering)
    if mode == "negate":
        return -tokens, -steering
    raise ValueError(f"Unknown intervention: {mode}")


def sample_next(logits, temperature, top_p, torch):
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_ids = probabilities.sort(descending=True)
        cumulative = sorted_probs.cumsum(-1)
        remove = cumulative - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True)
        sampled = torch.multinomial(sorted_probs, 1)
        return sorted_ids.gather(-1, sampled)
    return torch.multinomial(probabilities, 1)


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--controller-checkpoint", required=True)
    parser.add_argument("--controller-summary", required=True)
    parser.add_argument("--trajectory-file", required=True)
    parser.add_argument("--full-graph-file", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--refresh-interval", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--intervention", choices=("none", "zero", "random", "negate"), default="none")
    parser.add_argument("--controller-device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    runtime = {"np": np, "torch": torch}
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    adapter_dir = Path(args.adapter_dir)
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    wrapper = DynamicGroundedCausalLM(
        base_model,
        grounding_dim=int(adapter_config["grounding_dim"]),
        layer_indices=adapter_config["layer_indices"],
        num_heads=int(adapter_config["adapter_heads"]),
        dropout=0.0,
    )
    adapter_state = torch.load(adapter_dir / "dynamic_llm_adapters.pt", map_location="cpu")
    wrapper.adapters.load_state_dict(adapter_state)
    wrapper.base_model.eval()
    wrapper.adapters.eval()

    all_trajectories = read_jsonl(Path(args.trajectory_file))
    trajectories = all_trajectories[args.offset :]
    if args.limit is not None:
        trajectories = trajectories[: args.limit]
    full_graphs = graph_by_record_index(args.full_graph_file)
    cache = load_cache(args.embedding_cache_dir, args.split, runtime)
    sample = next(row for row in all_trajectories if row.get("candidate_nodes"))
    numeric_dim = len(sample["candidate_nodes"][0]["numeric_features"])
    controller_device = torch.device(args.controller_device)
    controller, relation_to_id, _ = load_controller(
        args.controller_summary, args.controller_checkpoint,
        cache["dense_dim"], numeric_dim, controller_device,
    )
    input_device = base_model.get_input_embeddings().weight.device
    output_path = Path(args.output_file)
    if output_path.exists():
        output_path.unlink()

    for local_index, trajectory in enumerate(trajectories):
        record_index = int(trajectory["record_index"])
        graph = full_graphs[record_index]
        prompt = build_full_schema_prompt(trajectory, graph)
        chat = build_chat_prompt(tokenizer, prompt)
        prompt_ids = tokenizer(chat, return_tensors="pt", add_special_tokens=False)["input_ids"].to(input_device)
        dense, numeric, query, edges, _ = trajectory_tensors(
            trajectory, cache, relation_to_id, controller, runtime, controller_device
        )
        tensors = (dense, numeric, query, edges)
        generated = []
        past = None
        cached_context = None
        operation_trace = []
        for generation_step in range(args.max_new_tokens):
            partial_sql = tokenizer.decode(generated, skip_special_tokens=True)
            if cached_context is None or generation_step % max(args.refresh_interval, 1) == 0:
                grounding, steering, operation = partial_sql_grounding_context(
                    controller, trajectory, tensors, partial_sql
                )
                grounding, steering = intervene(grounding, steering, args.intervention, torch)
                cached_context = (grounding, steering)
                operation_trace.append({"token": generation_step, "operation": operation})
            grounding, steering = cached_context
            if past is None:
                current_ids = prompt_ids
                token_steps = torch.full((prompt_ids.shape[1],), -1, dtype=torch.long)
                token_steps[-1] = 0
                attention_mask = torch.ones_like(prompt_ids)
            else:
                current_ids = torch.tensor([[generated[-1]]], dtype=torch.long, device=input_device)
                token_steps = torch.tensor([0], dtype=torch.long)
                attention_mask = torch.ones(
                    (1, prompt_ids.shape[1] + len(generated)), dtype=torch.long, device=input_device
                )
            wrapper.set_grounding_context([grounding], [steering], token_steps)
            with torch.no_grad():
                output = wrapper(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
            past = output.past_key_values
            next_id = int(sample_next(output.logits[:, -1, :], args.temperature, args.top_p, torch).item())
            generated.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
        raw_output = tokenizer.decode(generated, skip_special_tokens=True)
        row = {
            "question_id": trajectory.get("question_id"),
            "record_index": record_index,
            "db_id": trajectory.get("db_id"),
            "question": trajectory.get("question"),
            "evidence": trajectory.get("evidence"),
            "prompt": prompt,
            "raw_output": raw_output,
            "generated_sql": clean_sql(raw_output),
            "gold_sql": trajectory.get("gold_sql"),
            "error": None,
            "generation_config": vars(args),
            "grounding_control": {
                "intervention": args.intervention,
                "refresh_interval": args.refresh_interval,
                "adapter_scales": wrapper.adapter_scale_summary(),
                "operation_trace": operation_trace,
                "adapter_diagnostics": wrapper.last_diagnostics,
            },
        }
        append_jsonl(output_path, row)
        print(f"generated {local_index + 1}/{len(trajectories)} db={row['db_id']}")
    wrapper.clear_grounding_context()
    print(f"Outputs written to: {output_path}")


if __name__ == "__main__":
    main()
