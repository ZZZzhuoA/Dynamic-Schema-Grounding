"""Run the Stage 14B semantic-slot-conditioned RGTA tool."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stage13b_prepare_typed_trajectories import OPERATORS, VALUE_ROUTES  # noqa: E402
from src.grounding.stage14_typed_schema_tool import (  # noqa: E402
    assemble_tool_result,
    fk_pairs,
    ranked_join_edges,
    ranked_schema_candidates,
    ranked_vocabulary,
    write_jsonl,
)
from src.modeling.stage13c_static_runtime import graph_tensors  # noqa: E402
from src.training.stage13b_train_typed_ra_decoder import load_cache  # noqa: E402
from src.training.stage14b_train_semantic_slot_binder import (  # noqa: E402
    load_slot_cache,
    same_action_donor_key,
    slot_tensor,
)


def read_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def normalize_request(request):
    values = request.get("value_surfaces", [])
    return {
        **request,
        "value_surface": values[0] if len(values) == 1 else (values or None),
    }


def load_model(checkpoint_path, base_dim, slot_dim, device, torch):
    from src.modeling.semantic_slot_binder import SemanticSlotGraphBinder

    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    config = payload.get("config", {})
    relations = payload.get("relations", [])
    model = SemanticSlotGraphBinder(
        dense_dim=base_dim,
        slot_dim=slot_dim,
        hidden_dim=int(config.get("hidden_dim", 256)),
        relation_count=max(len(relations), 1),
        num_layers=int(config.get("num_layers", 2)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, {name: index for index, name in enumerate(relations)}, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graph-file", required=True)
    parser.add_argument("--slot-file", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--slot-embedding-cache-dir", required=True)
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument(
        "--slot-mode",
        choices=["correct", "action_only", "shuffled", "same_action_shuffled"],
        default="correct",
    )
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--table-top-k", type=int, default=3)
    parser.add_argument("--column-top-k", type=int, default=5)
    parser.add_argument("--join-top-k", type=int, default=3)
    parser.add_argument("--operator-top-k", type=int, default=3)
    parser.add_argument("--value-route-top-k", type=int, default=2)
    parser.add_argument("--max-schema-items", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Stage 14B inference requires numpy and PyTorch") from exc
    from src.modeling.semantic_slot_binder import owner_table_indices

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    graphs = read_jsonl(args.graph_file)
    graph_by_index = {int(row["record_index"]): row for row in graphs}
    slot_rows = read_jsonl(args.slot_file, args.limit)
    base_cache = load_cache(args.embedding_cache_dir, args.split, np)
    slot_cache = load_slot_cache(args.slot_embedding_cache_dir, args.split, np)
    model, relation_to_id, config = load_model(
        args.checkpoint, base_cache["dense_dim"], slot_cache["dense_dim"], device, torch
    )
    outputs = []
    with torch.no_grad():
        for slot_row in slot_rows:
            index = int(slot_row["record_index"])
            graph = graph_by_index[index]
            dense, query, node_types, edge_index, edge_type, nodes = graph_tensors(
                graph, base_cache, relation_to_id, device
            )
            pairs = fk_pairs(graph["inference_inputs"])
            join_index = (
                torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()
                if pairs else torch.empty((2, 0), dtype=torch.long, device=device)
            )
            nodes_by_id = {int(node.get("id", position)): node for position, node in enumerate(nodes)}
            owners = owner_table_indices(nodes, device)
            requests = slot_row["inference_inputs"]["requests"]
            focus_embeddings = [
                slot_tensor(slot_cache, index, request["step_index"], device, torch, "focus")
                for request in requests
            ]
            value_embeddings = [
                slot_tensor(slot_cache, index, request["step_index"], device, torch, "value")
                for request in requests
            ]
            steps = []
            for position, raw_request in enumerate(requests):
                if args.slot_mode == "correct":
                    focus, value = focus_embeddings[position], value_embeddings[position]
                elif args.slot_mode == "action_only":
                    focus = torch.zeros_like(focus_embeddings[position])
                    value = torch.zeros_like(value_embeddings[position])
                elif args.slot_mode == "shuffled":
                    donor = (position + 1) % len(focus_embeddings)
                    focus, value = focus_embeddings[donor], value_embeddings[donor]
                else:
                    donor_key = same_action_donor_key(
                        slot_cache, index, raw_request["step_index"], raw_request["action"]
                    )
                    focus = slot_tensor(slot_cache, donor_key[0], donor_key[1], device, torch, "focus")
                    value = slot_tensor(slot_cache, donor_key[0], donor_key[1], device, torch, "value")
                request = normalize_request(raw_request)
                prediction = model.forward_slot(
                    dense, query, focus, node_types, edge_index, edge_type, join_index,
                    request["action"], owners, value,
                    request.get("expected_value_type_id", 0),
                )
                steps.append(
                    {
                        "request": request,
                        "slot_mode": args.slot_mode,
                        "slot_gate_mean": float(prediction["slot_gate"].mean().cpu()),
                        "table_candidates": ranked_schema_candidates(
                            prediction["table_logits"], nodes, "table", args.table_top_k, torch
                        ),
                        "column_candidates": ranked_schema_candidates(
                            prediction["column_logits"], nodes, "column", args.column_top_k, torch
                        ),
                        "join_edge_candidates": ranked_join_edges(
                            prediction["join_edge_logits"], pairs, nodes_by_id, args.join_top_k, torch
                        ),
                        "operator_candidates": ranked_vocabulary(
                            prediction["operator_logits"], OPERATORS, args.operator_top_k, torch
                        ),
                        "value_route_candidates": ranked_vocabulary(
                            prediction["value_route_logits"], VALUE_ROUTES, args.value_route_top_k, torch
                        ),
                    }
                )
            assembly = assemble_tool_result(
                nodes, graph["inference_inputs"].get("schema_edges", []), steps, args.max_schema_items
            )
            assembly["literal_surfaces"] = [
                value for request in requests for value in request.get("value_surfaces", [])
            ]
            outputs.append(
                {
                    "tool_schema_version": "semantic_slot_rgta_tool_v1",
                    "record_index": index,
                    "question_id": slot_row.get("question_id"),
                    "db_id": slot_row.get("db_id"),
                    "question": slot_row["inference_inputs"].get("question"),
                    "evidence": slot_row["inference_inputs"].get("evidence"),
                    "slot_mode": args.slot_mode,
                    "tool_steps": steps,
                    "assembly": assembly,
                    "llm_tool_payload": {
                        "schema": [
                            {"id": item["schema_id"], "type": item["type"], "name": item["name"], "confidence": item["confidence"]}
                            for item in assembly["selected_schema"]
                        ],
                        "join_paths": assembly["join_paths"],
                        "literal_surfaces": assembly["literal_surfaces"],
                    },
                }
            )
    write_jsonl(args.output_file, outputs)
    summary = {
        "sample_count": len(outputs), "slot_mode": args.slot_mode,
        "config": vars(args), "training_config": config,
    }
    Path(args.output_file).with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {args.output_file}")


if __name__ == "__main__":
    main()
