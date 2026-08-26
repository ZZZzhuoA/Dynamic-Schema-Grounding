"""Run Stage 17-A1 causal controls against one frozen normal checkpoint."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage17a_train_full_schema_qrgta import (  # noqa: E402
    align_graphs_and_labels,
    evaluate,
    import_runtime,
    load_embedding_cache,
    read_jsonl,
    write_json,
    write_jsonl,
)


CONTROL_MODES = (
    "normal",
    "zero_query_edges",
    "shuffled_schema_edges",
    "shuffled_node_identity",
)
PRIMARY_METRICS = (
    "complete_coverage@30",
    "schema_recall@30",
    "table_recall@30",
    "column_recall@30",
    "complete_coverage@50",
    "mrr",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def ranked_id_rows(records):
    return {
        int(record["record_index"]): [
            int(row["schema_item_id"]) for row in record.get("ranked_schema", [])
        ]
        for record in records
    }


def assert_reference_normal_matches(actual, reference_path):
    expected = read_jsonl(reference_path)
    actual_ids = ranked_id_rows(actual)
    expected_ids = ranked_id_rows(expected)
    if actual_ids.keys() != expected_ids.keys():
        missing = sorted(expected_ids.keys() - actual_ids.keys())
        foreign = sorted(actual_ids.keys() - expected_ids.keys())
        raise ValueError(
            f"Normal prediction identity mismatch: missing={missing[:10]} foreign={foreign[:10]}"
        )
    mismatched = [key for key in actual_ids if actual_ids[key] != expected_ids[key]]
    if mismatched:
        raise ValueError(
            "Normal checkpoint intervention run does not reproduce the reference "
            f"ranking for record_index values {mismatched[:10]}"
        )


def load_checkpoint_model(checkpoint_path, runtime, device):
    torch = runtime["torch"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain model_config")
    if config.get("model_type") != "qrgta":
        raise ValueError(
            "Checkpoint intervention requires a normal qrgta checkpoint, got "
            f"{config.get('model_type')!r}"
        )
    if config.get("control_mode", "normal") != "normal":
        raise ValueError(
            "Checkpoint intervention requires control_mode='normal', got "
            f"{config.get('control_mode')!r}"
        )
    model = runtime["model"](
        dense_dim=int(config["dense_dim"]),
        relation_count=int(config["relation_count"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dropout=float(config["dropout"]),
        model_type=str(config["model_type"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, config, checkpoint.get("epoch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dev-graph-file", required=True)
    parser.add_argument("--dev-label-file", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-modes", default=",".join(CONTROL_MODES))
    parser.add_argument("--reference-normal-predictions", default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    modes = [value.strip() for value in args.control_modes.split(",") if value.strip()]
    unknown = sorted(set(modes) - set(CONTROL_MODES))
    if unknown or len(modes) != len(set(modes)):
        raise ValueError(f"Invalid or duplicate control modes: unknown={unknown}, modes={modes}")
    if "normal" not in modes:
        raise ValueError("--control-modes must include normal for paired deltas")

    runtime = import_runtime()
    torch = runtime["torch"]
    device = torch.device(args.device)
    graphs = read_jsonl(args.dev_graph_file, args.dev_limit)
    labels = read_jsonl(args.dev_label_file)
    examples, alignment = align_graphs_and_labels(graphs, labels, "dev")
    cache = load_embedding_cache(args.embedding_cache_dir, "dev", runtime)
    model, model_config, checkpoint_epoch = load_checkpoint_model(
        args.checkpoint, runtime, device
    )
    if cache["dense_dim"] != int(model_config["dense_dim"]):
        raise ValueError(
            f"Embedding dimension mismatch: cache={cache['dense_dim']} "
            f"checkpoint={model_config['dense_dim']}"
        )
    relations = {str(key): int(value) for key, value in model_config["relations"].items()}
    graph_relations = {
        str(edge["type"]) for example in examples for edge in example.get("schema_edges", [])
    }
    missing_relations = sorted(graph_relations - set(relations))
    if missing_relations:
        raise ValueError(f"Checkpoint relation map misses graph relations: {missing_relations}")

    checkpoint_sha = file_sha256(args.checkpoint)
    state_sha_before = state_dict_sha256(model)
    output_dir = Path(args.output_dir)
    results = {}
    prediction_rows = {}
    for mode in modes:
        control_args = SimpleNamespace(control_mode=mode, seed=args.seed)
        metrics, predictions = evaluate(
            model,
            examples,
            cache,
            relations,
            control_args,
            runtime,
            device,
            f"dev::{mode}",
            predictions=True,
        )
        mode_dir = output_dir / mode
        write_jsonl(mode_dir / "dev_predictions.jsonl", predictions)
        write_json(
            mode_dir / "metrics.json",
            {
                **metrics,
                "control_mode": mode,
                "control_seed": args.seed,
                "checkpoint": args.checkpoint,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_epoch": checkpoint_epoch,
            },
        )
        results[mode] = metrics
        prediction_rows[mode] = predictions

    if args.reference_normal_predictions:
        assert_reference_normal_matches(
            prediction_rows["normal"], args.reference_normal_predictions
        )
    state_sha_after = state_dict_sha256(model)
    if state_sha_before != state_sha_after:
        raise RuntimeError("Model parameters changed during checkpoint interventions")

    normal = results["normal"]
    deltas = {
        mode: {
            metric: float(normal[metric]) - float(results[mode][metric])
            for metric in PRIMARY_METRICS
        }
        for mode in modes
        if mode != "normal"
    }
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": checkpoint_epoch,
        "model_state_sha256_before": state_sha_before,
        "model_state_sha256_after": state_sha_after,
        "parameters_unchanged": state_sha_before == state_sha_after,
        "model_config": model_config,
        "alignment": alignment,
        "control_seed": args.seed,
        "control_modes": modes,
        "metrics": results,
        "normal_minus_control": deltas,
        "reference_normal_predictions": args.reference_normal_predictions,
        "reference_normal_reproduced": bool(args.reference_normal_predictions),
        "data_config": {
            "dev_graph_file": args.dev_graph_file,
            "dev_label_file": args.dev_label_file,
            "embedding_cache_dir": args.embedding_cache_dir,
            "dev_limit": args.dev_limit,
        },
        "control_semantics": {
            "zero_query_edges": "Remove Query-to-Schema graph messages; retain the final query-conditioned scorer.",
            "shuffled_schema_edges": "Permute non-self edge destinations within relation type while preserving relation/source/destination marginals.",
            "shuffled_node_identity": "Permute dense semantic embeddings within table/column type while preserving node IDs, types, graph edges, and labels.",
        },
    }
    write_json(output_dir / "intervention_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
