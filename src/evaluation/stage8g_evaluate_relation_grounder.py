import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5_train_dsg_grounder import write_json, write_jsonl  # noqa: E402
from src.training.stage5g_train_clause_grounder import (  # noqa: E402
    assemble_predictions,
    evaluate_assembled,
    parse_budget_text,
)
from src.training.stage5j_train_relation_grounder import (  # noqa: E402
    load_aligned_records,
    make_relation_examples,
)
from src.training.stage8g_train_dense_relation_grounder import (  # noqa: E402
    build_grounder,
    configure_reproducibility,
    evaluate_relation_examples,
    import_runtime,
    infer_input_dim,
    load_cache,
)


COMPATIBILITY_DEFAULTS = {
    "feature_mode": "dense",
    "hash_dim": 256,
    "hidden_dim": 96,
    "num_layers": 2,
    "dropout": 0.1,
    "encoder_type": "rgcn",
    "use_lexical_features": False,
    "dense_l2_normalize": True,
    "pos_weight": 3.0,
    "output_top_k": 30,
    "include_empty_relation_examples": False,
    "use_relation_conditioned_prior": False,
    "use_relation_conditioning": False,
    "use_similarity_prior": False,
    "similarity_prior_init": 1.0,
    "join_prior_init": 0.1,
    "similarity_prior_normalization": "zscore",
    "similarity_prior_clip": 4.0,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_checkpoint(path, torch, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def restore_training_args(config):
    restored = dict(COMPATIBILITY_DEFAULTS)
    restored.update(config)
    if restored.get("use_relation_conditioned_prior"):
        restored["use_relation_conditioning"] = True
        restored["use_similarity_prior"] = True
    relation_types = restored.get("relation_types")
    if not isinstance(relation_types, list) or not relation_types:
        raise ValueError("train_config.json does not contain a non-empty relation_types list")
    restored["relation_to_id"] = {
        relation: index for index, relation in enumerate(relation_types)
    }
    return SimpleNamespace(**restored)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved Stage 8G relation grounder checkpoint without training or "
            "checkpoint selection on the evaluation split."
        )
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-file", default=None)
    parser.add_argument("--relation-file", required=True)
    parser.add_argument("--graph-file", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many base records before evaluation (use 100 for the untouched dev remainder).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-top-k", type=int, default=None)
    parser.add_argument("--assembly-budgets", type=parse_budget_text, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    cli = parser.parse_args()

    checkpoint_dir = Path(cli.checkpoint_dir)
    config_path = checkpoint_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing training config: {config_path}")
    train_config = read_json(config_path)
    args = restore_training_args(train_config)
    if cli.output_top_k is not None:
        args.output_top_k = cli.output_top_k
    if cli.assembly_budgets is not None:
        args.assembly_budgets = cli.assembly_budgets
    if not hasattr(args, "assembly_budgets"):
        raise ValueError("No assembly budgets found in train_config.json or CLI")

    checkpoint_file = (
        Path(cli.checkpoint_file)
        if cli.checkpoint_file
        else checkpoint_dir / "dense_relation_grounder_model.pt"
    )
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_file}")
    output_dir = (
        Path(cli.output_dir)
        if cli.output_dir
        else checkpoint_dir
        / f"evaluation_{cli.split}_offset{cli.offset}_{cli.limit or 'remainder'}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = import_runtime()
    configure_reproducibility(cli.seed, runtime, deterministic=cli.deterministic)
    torch = runtime["torch"]
    device = torch.device(cli.device)

    if cli.offset < 0:
        raise ValueError("--offset must be non-negative")
    all_aligned = load_aligned_records(Path(cli.relation_file), Path(cli.graph_file), None)
    stop = cli.offset + cli.limit if cli.limit is not None else None
    aligned = all_aligned[cli.offset:stop]
    if not aligned:
        raise ValueError(
            f"Evaluation slice is empty: total={len(all_aligned)}, offset={cli.offset}, limit={cli.limit}"
        )
    examples = make_relation_examples(
        aligned,
        args.relation_types,
        include_empty_relation_examples=args.include_empty_relation_examples,
    )
    cache = load_cache(Path(cli.embedding_cache_dir), cli.split, runtime)
    input_dim = infer_input_dim(args, cache)
    trained_input_dim = train_config.get("input_dim")
    if trained_input_dim is not None and int(trained_input_dim) != input_dim:
        raise ValueError(
            f"Checkpoint/cache input dimension mismatch: checkpoint={trained_input_dim}, "
            f"evaluation_cache={input_dim}"
        )
    graph_relations = train_config.get("graph_relations") or list(runtime["DEFAULT_RELATIONS"])
    model = build_grounder(args, runtime, input_dim, graph_relations, device)
    state = load_checkpoint(checkpoint_file, torch, device)
    model.load_state_dict(state, strict=True)

    relation_metrics, relation_predictions = evaluate_relation_examples(
        model,
        examples,
        cache,
        runtime,
        args,
        device,
        graph_relations,
        output_dir=output_dir,
        split=cli.split,
    )
    assembled = assemble_predictions(
        relation_predictions,
        aligned,
        output_top_k=args.output_top_k,
        budgets=args.assembly_budgets,
    )
    assembled_metrics = evaluate_assembled(assembled)
    write_jsonl(output_dir / f"{cli.split}_assembled_predictions.jsonl", assembled)

    metrics = {**relation_metrics, **assembled_metrics}
    if args.use_similarity_prior:
        metrics["learned_similarity_prior_scales"] = {
            relation: float(scale)
            for relation, scale in zip(
                args.relation_types,
                model.prior_scales().detach().cpu().tolist(),
            )
        }
    summary = {
        "evaluation_only": True,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file": str(checkpoint_file),
        "checkpoint_best_epoch": (
            read_json(checkpoint_dir / "training_summary.json").get("best_epoch")
            if (checkpoint_dir / "training_summary.json").exists()
            else None
        ),
        "relation_file": cli.relation_file,
        "graph_file": cli.graph_file,
        "embedding_cache_dir": cli.embedding_cache_dir,
        "split": cli.split,
        "base_sample_count": len(aligned),
        "source_base_sample_count": len(all_aligned),
        "relation_example_count": len(examples),
        "offset": cli.offset,
        "limit": cli.limit,
        "seed": cli.seed,
        "deterministic": cli.deterministic,
        "metrics": metrics,
        "note": (
            "The checkpoint is loaded once and evaluated directly. This script performs no "
            "training, early stopping, or checkpoint selection on the evaluation records."
        ),
    }
    write_json(output_dir / "evaluation_summary.json", summary)
    write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
