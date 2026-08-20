"""Apply the Stage 15A-fix1 verifier to real LLM SQL hypotheses."""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def read_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def shuffled_fk_row(row, seed):
    """Corrupt FK destinations while preserving all non-FK graph evidence."""
    result = copy.deepcopy(row)
    edges = result["inference_inputs"].get("schema_edges", [])
    indices = [
        index for index, edge in enumerate(edges)
        if edge.get("type") in {"foreign_key_forward", "foreign_key_backward"}
    ]
    destinations = [int(edges[index]["dst"]) for index in indices]
    if len(destinations) > 1:
        rng = random.Random(seed)
        shift = rng.randrange(1, len(destinations))
        destinations = destinations[shift:] + destinations[:shift]
        for index, destination in zip(indices, destinations):
            edges[index]["dst"] = destination
    elif destinations:
        node_count = len(result["inference_inputs"].get("schema_items", []))
        edges[indices[0]]["dst"] = (destinations[0] + 1) % max(node_count, 1)
    return result


def shuffled_identity_dense(dense, node_types, seed, torch):
    """Shuffle semantic node embeddings within table/column types, keeping topology fixed."""
    permutation = list(range(int(dense.shape[0])))
    rng = random.Random(seed)
    for node_type in (0, 1):
        positions = [
            index for index, value in enumerate(node_types.detach().cpu().tolist())
            if int(value) == node_type
        ]
        shuffled = list(positions)
        rng.shuffle(shuffled)
        for target, source in zip(positions, shuffled):
            permutation[target] = source
    return dense[torch.tensor(permutation, dtype=torch.long, device=dense.device)]


def model_scores(model, tensors, schema_edges, candidates):
    dense, query, node_types, edge_index, edge_type, schema_items = tensors
    return model(
        dense, query, node_types, edge_index, edge_type,
        schema_items, schema_edges, candidates,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--flat-only", action="store_true", help="Do not score partial nested/set-query parses")
    parser.add_argument(
        "--control-modes",
        default="shuffled_fk,shuffled_node_identity",
        help="Comma-separated causal controls; use an empty string to disable.",
    )
    parser.add_argument("--control-seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Stage 15B scoring requires numpy and PyTorch") from exc
    from src.modeling.sql_hypothesis_verifier import SQLHypothesisGraphVerifier
    from src.modeling.stage13c_static_runtime import graph_tensors
    from src.training.stage13b_train_typed_ra_decoder import load_cache

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    relations = list(checkpoint.get("relations", [])) or ["self_loop"]
    relation_to_id = {name: index for index, name in enumerate(relations)}
    cache = load_cache(args.embedding_cache_dir, args.split, np)
    dense_dim = int(checkpoint.get("dense_dim", cache["dense_dim"]))
    if dense_dim != cache["dense_dim"]:
        raise ValueError(f"Checkpoint/cache dense_dim mismatch: {dense_dim} != {cache['dense_dim']}")
    model = SQLHypothesisGraphVerifier(
        dense_dim=dense_dim,
        hidden_dim=int(config.get("hidden_dim", 256)),
        schema_relation_count=len(relations),
        num_schema_layers=int(config.get("num_schema_layers", 2)),
        num_plan_layers=int(config.get("num_plan_layers", 2)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    control_modes = [value.strip() for value in args.control_modes.split(",") if value.strip()]
    unknown = set(control_modes) - {"shuffled_fk", "shuffled_node_identity"}
    if unknown:
        raise ValueError(f"Unknown control modes: {sorted(unknown)}")
    output, scored_count = [], 0
    control_scored_counts = {mode: 0 for mode in control_modes}
    with torch.no_grad():
        for row in read_jsonl(args.candidate_file, args.limit):
            eligible_indices = [
                index for index, candidate in enumerate(row.get("candidates", []))
                if candidate.get("parse_ok")
                and (not args.flat_only or candidate.get("parse_status") == "supported_flat")
            ]
            result = json.loads(json.dumps(row))
            if eligible_indices:
                dense, query, node_types, edge_index, edge_type, schema_items = graph_tensors(
                    row, cache, relation_to_id, device
                )
                candidates = [row["candidates"][index] for index in eligible_indices]
                tensors = (dense, query, node_types, edge_index, edge_type, schema_items)
                scores = model_scores(
                    model, tensors, row["inference_inputs"].get("schema_edges", []), candidates
                )
                for source_index, score, detail in zip(
                    eligible_indices, scores["scores"], scores["candidate_outputs"]
                ):
                    target = result["candidates"][source_index]
                    target["verifier_score"] = float(score.detach().float().cpu())
                    target["verifier_detail"] = {
                        "step_energy": float(detail["step_energy"].detach().float().cpu()),
                        "join_energy": float(detail["join_energy"].detach().float().cpu()),
                        "consistency_energy": float(detail["consistency_energy"].detach().float().cpu()),
                        "pointer_validity": float(detail["pointer_validity"]),
                        "consistency_features": detail["consistency_features"],
                    }
                    scored_count += 1
                control_outputs = {}
                control_seed = args.control_seed + int(row["record_index"]) * 1009
                if "shuffled_fk" in control_modes:
                    control_row = shuffled_fk_row(row, control_seed)
                    control_tensors = graph_tensors(control_row, cache, relation_to_id, device)
                    control_outputs["shuffled_fk"] = model_scores(
                        model,
                        control_tensors,
                        control_row["inference_inputs"].get("schema_edges", []),
                        candidates,
                    )
                if "shuffled_node_identity" in control_modes:
                    shuffled_dense = shuffled_identity_dense(
                        dense, node_types, control_seed + 17, torch
                    )
                    identity_tensors = (
                        shuffled_dense, query, node_types, edge_index, edge_type, schema_items
                    )
                    control_outputs["shuffled_node_identity"] = model_scores(
                        model,
                        identity_tensors,
                        row["inference_inputs"].get("schema_edges", []),
                        candidates,
                    )
                for mode, control in control_outputs.items():
                    for source_index, score in zip(eligible_indices, control["scores"]):
                        result["candidates"][source_index].setdefault("control_scores", {})[
                            mode
                        ] = float(score.detach().float().cpu())
                        control_scored_counts[mode] += 1
            for index, candidate in enumerate(result.get("candidates", [])):
                candidate.setdefault("verifier_score", None)
                candidate.setdefault("verifier_detail", None)
                candidate.setdefault("control_scores", {})
            output.append(result)

    write_jsonl(args.output_file, output)
    summary = {
        "candidate_file": args.candidate_file,
        "checkpoint": args.checkpoint,
        "group_count": len(output),
        "candidate_count": sum(len(row.get("candidates", [])) for row in output),
        "scored_candidate_count": scored_count,
        "control_scored_candidate_counts": control_scored_counts,
        "control_modes": control_modes,
        "flat_only": args.flat_only,
    }
    Path(args.output_file).with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
