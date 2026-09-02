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
PATH_CONTROL_MODES = (
    "shuffled_distance_buckets",
    "shuffled_path_signatures",
    "zero_path_features",
)
PERSISTENT_CONTROL_MODES = (
    "zero_update_gates",
)
COMPETITION_CONTROL_MODES = (
    "zero_table_competition",
    "shuffle_column_parent_table",
    "zero_competition_gates",
)
PRIMARY_KEY_CONTROL_MODES = (
    "downgrade_primary_key_edges",
)
PRIMARY_KEY_MODIFIER_CONTROL_MODES = (
    "zero_pk_modifier",
)
ALL_CONTROL_MODES = (
    CONTROL_MODES
    + PATH_CONTROL_MODES
    + PERSISTENT_CONTROL_MODES
    + COMPETITION_CONTROL_MODES
    + PRIMARY_KEY_CONTROL_MODES
    + PRIMARY_KEY_MODIFIER_CONTROL_MODES
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


def prediction_rows(records):
    return {int(record["record_index"]): record for record in records}


def _ranked_ids(record):
    return [int(row["schema_item_id"]) for row in record.get("ranked_schema", [])]


def _first_gold_rank(ranked_ids, gold_ids):
    gold = set(gold_ids)
    return next(
        (rank for rank, schema_id in enumerate(ranked_ids, start=1) if schema_id in gold),
        None,
    )


def assert_reference_normal_matches(
    actual, reference_path, examples, score_atol=1e-5, top_ks=(10, 20, 30, 50)
):
    """Accept metric-equivalent CUDA tie drift while preserving a strict audit trail.

    Sparse CUDA index aggregation can move nearly tied logits by a few ulps. Requiring
    every one of N schema nodes to retain the exact total order rejects otherwise
    identical checkpoints even when all reported Top-K sets and MRR are unchanged.
    """
    expected = read_jsonl(reference_path)
    actual_by_id = prediction_rows(actual)
    expected_by_id = prediction_rows(expected)
    if actual_by_id.keys() != expected_by_id.keys():
        missing = sorted(expected_by_id.keys() - actual_by_id.keys())
        foreign = sorted(actual_by_id.keys() - expected_by_id.keys())
        raise ValueError(
            f"Normal prediction identity mismatch: missing={missing[:10]} foreign={foreign[:10]}"
        )
    gold_by_id = {int(example["record_index"]): example["gold_ids"] for example in examples}
    diagnostics = []
    invalid = []
    max_abs_logit_delta = 0.0
    for key in actual_by_id:
        actual_record = actual_by_id[key]
        expected_record = expected_by_id[key]
        actual_ids = _ranked_ids(actual_record)
        expected_ids = _ranked_ids(expected_record)
        if set(actual_ids) != set(expected_ids) or len(actual_ids) != len(expected_ids):
            invalid.append({"record_index": key, "reason": "schema_identity_set_changed"})
            continue
        actual_scores = {
            int(row["schema_item_id"]): float(row["logit"])
            for row in actual_record.get("ranked_schema", [])
        }
        expected_scores = {
            int(row["schema_item_id"]): float(row["logit"])
            for row in expected_record.get("ranked_schema", [])
        }
        row_max_delta = max(
            (abs(actual_scores[schema_id] - expected_scores[schema_id]) for schema_id in actual_ids),
            default=0.0,
        )
        max_abs_logit_delta = max(max_abs_logit_delta, row_max_delta)
        if actual_ids == expected_ids:
            continue
        first_difference = next(
            index for index, pair in enumerate(zip(actual_ids, expected_ids)) if pair[0] != pair[1]
        )
        top_k_set_equal = {
            str(k): set(actual_ids[:k]) == set(expected_ids[:k]) for k in top_ks
        }
        first_gold_equal = _first_gold_rank(
            actual_ids, gold_by_id.get(key, [])
        ) == _first_gold_rank(expected_ids, gold_by_id.get(key, []))
        diagnostic = {
            "record_index": key,
            "first_different_rank": first_difference + 1,
            "top_k_set_equal": top_k_set_equal,
            "first_gold_rank_equal": first_gold_equal,
            "max_abs_logit_delta": row_max_delta,
        }
        diagnostics.append(diagnostic)
        if not all(top_k_set_equal.values()) or not first_gold_equal or row_max_delta > score_atol:
            invalid.append(diagnostic)
    if invalid:
        raise ValueError(
            "Normal checkpoint intervention run is not metric-equivalent to the reference: "
            f"{invalid[:10]}"
        )
    return {
        "metric_equivalent": True,
        "exact_full_ranking": not diagnostics,
        "numerical_rank_drift_count": len(diagnostics),
        "numerical_rank_drift_record_indices": [row["record_index"] for row in diagnostics],
        "max_abs_logit_delta": max_abs_logit_delta,
        "score_atol": score_atol,
        "top_ks_checked": list(top_ks),
        "drift_details": diagnostics,
    }


def load_checkpoint_model(checkpoint_path, runtime, device):
    torch = runtime["torch"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain model_config")
    if config.get("model_type") not in {
        "qrgta",
        "path_qrgta",
        "persistent_path_qrgta",
        "table_competitive_path_qrgta",
        "enhanced_table_competitive_path_qrgta",
        "pk_residual_table_competitive_path_qrgta",
    }:
        raise ValueError(
            "Checkpoint intervention requires a normal graph QRGTA checkpoint, got "
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
        distance_bucket_count=int(config.get("distance_bucket_count", 1)),
        path_signature_count=int(config.get("path_signature_count", 1)),
        role_count=int(config.get("role_count", 0)),
        competition_hidden_dim=int(config.get("competition_hidden_dim", 128)),
        competition_dropout=float(config.get("competition_dropout", config.get("dropout", 0.1))),
        competition_temperature=float(config.get("competition_temperature", 1.5)),
        competition_residual_scale=float(config.get("competition_residual_scale", 0.5)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, config, checkpoint.get("epoch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dev-graph-file", required=True)
    parser.add_argument("--dev-label-file", required=True)
    parser.add_argument("--dev-role-label-file", default=None)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-modes", default=None)
    parser.add_argument("--reference-normal-predictions", default=None)
    parser.add_argument("--reference-logit-atol", type=float, default=1e-5)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runtime = import_runtime()
    torch = runtime["torch"]
    device = torch.device(args.device)
    graphs = read_jsonl(args.dev_graph_file, args.dev_limit)
    labels = read_jsonl(args.dev_label_file)
    role_labels = read_jsonl(args.dev_role_label_file) if args.dev_role_label_file else None
    examples, alignment = align_graphs_and_labels(graphs, labels, "dev", role_labels)
    cache = load_embedding_cache(args.embedding_cache_dir, "dev", runtime)
    model, model_config, checkpoint_epoch = load_checkpoint_model(
        args.checkpoint, runtime, device
    )
    model_type = str(model_config.get("model_type", "qrgta"))
    default_modes = list(CONTROL_MODES)
    if model_type in {
        "path_qrgta",
        "persistent_path_qrgta",
        "table_competitive_path_qrgta",
        "enhanced_table_competitive_path_qrgta",
        "pk_residual_table_competitive_path_qrgta",
    }:
        default_modes.extend(PATH_CONTROL_MODES)
    if model_type == "persistent_path_qrgta":
        default_modes.extend(PERSISTENT_CONTROL_MODES)
    if model_type in {
        "table_competitive_path_qrgta",
        "enhanced_table_competitive_path_qrgta",
        "pk_residual_table_competitive_path_qrgta",
    }:
        default_modes.extend(COMPETITION_CONTROL_MODES)
    graph_relations = {
        str(edge["type"]) for example in examples for edge in example.get("schema_edges", [])
    }
    has_primary_key_relations = {
        "table_to_primary_key",
        "primary_key_to_table",
    }.issubset(graph_relations)
    if has_primary_key_relations:
        default_modes.extend(PRIMARY_KEY_CONTROL_MODES)
    has_primary_key_attributes = any(
        edge.get("is_primary_key_edge") is True
        for example in examples
        for edge in example.get("schema_edges", [])
    )
    if model_type == "pk_residual_table_competitive_path_qrgta":
        if not has_primary_key_attributes:
            raise ValueError(
                "PK-residual checkpoint evaluation requires is_primary_key_edge graph attributes"
            )
        default_modes.extend(PRIMARY_KEY_MODIFIER_CONTROL_MODES)
    modes_text = args.control_modes or ",".join(default_modes)
    modes = [value.strip() for value in modes_text.split(",") if value.strip()]
    unknown = sorted(set(modes) - set(ALL_CONTROL_MODES))
    if unknown or len(modes) != len(set(modes)):
        raise ValueError(f"Invalid or duplicate control modes: unknown={unknown}, modes={modes}")
    if "normal" not in modes:
        raise ValueError("--control-modes must include normal for paired deltas")
    if model_type != "persistent_path_qrgta" and "zero_update_gates" in modes:
        raise ValueError("zero_update_gates requires a persistent_path_qrgta checkpoint")
    if model_type not in {
        "table_competitive_path_qrgta",
        "enhanced_table_competitive_path_qrgta",
        "pk_residual_table_competitive_path_qrgta",
    } and (
        set(modes) & set(COMPETITION_CONTROL_MODES)
    ):
        raise ValueError(
            "table competition controls require a table-competitive checkpoint"
        )
    if cache["dense_dim"] != int(model_config["dense_dim"]):
        raise ValueError(
            f"Embedding dimension mismatch: cache={cache['dense_dim']} "
            f"checkpoint={model_config['dense_dim']}"
        )
    relations = {str(key): int(value) for key, value in model_config["relations"].items()}
    if set(modes) & set(PRIMARY_KEY_CONTROL_MODES) and not has_primary_key_relations:
        raise ValueError(
            "downgrade_primary_key_edges requires a graph with primary-key relations"
        )
    if set(modes) & set(PRIMARY_KEY_MODIFIER_CONTROL_MODES) and model_type != (
        "pk_residual_table_competitive_path_qrgta"
    ):
        raise ValueError("zero_pk_modifier requires a PK-residual checkpoint")
    missing_relations = sorted(graph_relations - set(relations))
    if missing_relations:
        raise ValueError(f"Checkpoint relation map misses graph relations: {missing_relations}")

    checkpoint_sha = file_sha256(args.checkpoint)
    state_sha_before = state_dict_sha256(model)
    output_dir = Path(args.output_dir)
    results = {}
    prediction_rows = {}
    for mode in modes:
        control_args = SimpleNamespace(
            control_mode=mode,
            seed=args.seed,
            model_type=model_config.get("model_type", "qrgta"),
            distance_buckets={
                str(key): int(value)
                for key, value in model_config.get("distance_bucket_mapping", {}).items()
            },
            path_signatures={
                str(key): int(value)
                for key, value in model_config.get("path_type_mapping", {}).items()
            },
            max_path_distance=int(model_config.get("max_path_distance", 3)),
            max_path_edges_per_destination=model_config.get(
                "max_path_edges_per_destination", 32
            ),
            coverage_surrogate_weight=float(
                model_config.get("coverage_surrogate_weight", 0.0)
            ),
            coverage_margin=float(model_config.get("coverage_margin", 0.1)),
            coverage_target_k=int(model_config.get("coverage_target_k", 30)),
            role_mapping={
                str(key): int(value)
                for key, value in model_config.get("role_mapping", {}).items()
            },
            role_loss_weight=float(model_config.get("role_loss_weight", 0.0)),
            record_persistent_diagnostics=bool(
                model_config.get("record_persistent_diagnostics", True)
            ),
            competition_hidden_dim=int(model_config.get("competition_hidden_dim", 128)),
            competition_dropout=float(
                model_config.get("competition_dropout", model_config.get("dropout", 0.1))
            ),
            competition_temperature=float(model_config.get("competition_temperature", 1.5)),
            competition_residual_scale=float(
                model_config.get("competition_residual_scale", 0.5)
            ),
        )
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

    reference_check = None
    if args.reference_normal_predictions:
        reference_check = assert_reference_normal_matches(
            prediction_rows["normal"],
            args.reference_normal_predictions,
            examples,
            score_atol=args.reference_logit_atol,
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
        "reference_normal_reproduced": bool(
            reference_check and reference_check["metric_equivalent"]
        ),
        "reference_normal_check": reference_check,
        "data_config": {
            "dev_graph_file": args.dev_graph_file,
            "dev_label_file": args.dev_label_file,
            "dev_role_label_file": args.dev_role_label_file,
            "embedding_cache_dir": args.embedding_cache_dir,
            "dev_limit": args.dev_limit,
        },
        "control_semantics": {
            "zero_query_edges": "Remove Query-to-Schema graph messages; retain the final query-conditioned scorer.",
            "shuffled_schema_edges": "Permute non-self edge destinations within relation type while preserving relation/source/destination marginals.",
            "shuffled_node_identity": "Permute dense semantic embeddings within table/column type while preserving node IDs, types, graph edges, and labels.",
            "shuffled_distance_buckets": "Shuffle non-query schema edge distance buckets while preserving edge index, relation type, and path signature.",
            "shuffled_path_signatures": "Shuffle non-query schema edge path signatures while preserving edge index, relation type, and distance bucket.",
            "zero_path_features": "Keep path-augmented schema edges but neutralize non-query path and distance features.",
            "zero_update_gates": "Keep path-augmented edges but set persistent message/FFN update gates to zero.",
            "zero_table_competition": "Keep path-aware graph propagation but skip table-scoped column competition.",
            "shuffle_column_parent_table": "Shuffle column-to-parent-table assignments while preserving node identity and graph/path edges.",
            "zero_competition_gates": "Compute table-scoped competition features but set competition write gates to zero.",
            "zero_pk_modifier": "Keep generic ownership relations and PK edge markers but disable the learned PK key/value/bias residual.",
        },
    }
    write_json(output_dir / "intervention_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
