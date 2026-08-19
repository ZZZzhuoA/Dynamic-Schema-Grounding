"""Apply the Stage 15A-fix1 verifier to real LLM SQL hypotheses."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.stage13c_static_runtime import graph_tensors
from src.training.stage13b_train_typed_ra_decoder import load_cache


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
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Stage 15B scoring requires numpy and PyTorch") from exc
    from src.modeling.sql_hypothesis_verifier import SQLHypothesisGraphVerifier

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

    output, scored_count = [], 0
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
                scores = model(
                    dense,
                    query,
                    node_types,
                    edge_index,
                    edge_type,
                    schema_items,
                    row["inference_inputs"].get("schema_edges", []),
                    candidates,
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
            for index, candidate in enumerate(result.get("candidates", [])):
                candidate.setdefault("verifier_score", None)
                candidate.setdefault("verifier_detail", None)
            output.append(result)

    write_jsonl(args.output_file, output)
    summary = {
        "candidate_file": args.candidate_file,
        "checkpoint": args.checkpoint,
        "group_count": len(output),
        "candidate_count": sum(len(row.get("candidates", [])) for row in output),
        "scored_candidate_count": scored_count,
        "flat_only": args.flat_only,
    }
    Path(args.output_file).with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
