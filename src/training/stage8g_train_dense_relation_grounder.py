import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5_train_dsg_grounder import (  # noqa: E402
    clone_state_dict_to_cpu,
    get_metric,
    is_better_metric,
    mean,
    precision_at_k,
    ranked_prediction_rows,
    read_jsonl,
    recall_at_k,
    reciprocal_rank,
    write_json,
    write_jsonl,
)
from src.training.stage5g_train_clause_grounder import (  # noqa: E402
    assemble_predictions,
    evaluate_assembled,
    parse_budget_text,
    parse_clause_list,
)
from src.training.stage5j_train_relation_grounder import (  # noqa: E402
    DEFAULT_RELATIONS as DEFAULT_OPERATION_RELATIONS,
    load_aligned_records,
    make_relation_examples,
)


def import_runtime():
    try:
        import numpy as np
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("Stage 8G-B requires numpy and PyTorch.") from exc
    from src.modeling.dsg_grounder import (
        DEFAULT_RELATIONS,
        DSGGrounder,
        RelationConditionedDSGGrounder,
        lexical_features,
        make_edge_tensors,
        make_node_features,
        make_query_features,
    )

    return {
        "np": np,
        "torch": torch,
        "nn": nn,
        "DEFAULT_RELATIONS": DEFAULT_RELATIONS,
        "DSGGrounder": DSGGrounder,
        "RelationConditionedDSGGrounder": RelationConditionedDSGGrounder,
        "lexical_features": lexical_features,
        "make_edge_tensors": make_edge_tensors,
        "make_node_features": make_node_features,
        "make_query_features": make_query_features,
    }


def load_cache(prefix: Path, split: str, runtime):
    np = runtime["np"]
    query = np.load(prefix / f"{split}_query_embeddings.npy", mmap_mode="r")
    node = np.load(prefix / f"{split}_node_embeddings.npy", mmap_mode="r")
    index_rows = json.loads((prefix / f"{split}_index.json").read_text(encoding="utf-8"))
    by_example_index = {int(row["example_index"]): row for row in index_rows}
    return {
        "query": query,
        "node": node,
        "index_rows": index_rows,
        "by_example_index": by_example_index,
        "query_dim": int(query.shape[1]) if len(query.shape) == 2 and query.shape[0] else int(query.shape[1]),
        "node_dim": int(node.shape[1]) if len(node.shape) == 2 and node.shape[0] else int(node.shape[1]),
    }


def dense_features_for_example(example, cache, runtime, device):
    torch = runtime["torch"]
    record_index = int(example["record_index"])
    index_row = cache["by_example_index"].get(record_index)
    if index_row is None:
        raise KeyError(f"No embedding cache index for record_index={record_index}")
    node_count = int(index_row["node_count"])
    query_index = int(index_row["query_embedding_index"])
    if "node_embedding_indices" in index_row:
        node_indices = [int(index) for index in index_row["node_embedding_indices"]]
        if len(node_indices) != node_count:
            raise ValueError(
                f"Deduplicated cache node-count mismatch for record_index={record_index}: "
                f"indices={len(node_indices)} expected={node_count}"
            )
        node_embeddings = cache["node"][node_indices]
    else:
        node_start = int(index_row["node_embedding_start"])
        node_embeddings = cache["node"][node_start : node_start + node_count]
    query_embedding = cache["query"][query_index : query_index + 1]
    return (
        torch.tensor(query_embedding, dtype=torch.float32, device=device),
        torch.tensor(node_embeddings, dtype=torch.float32, device=device),
    )


def build_feature_tensors(example, cache, runtime, args, graph_relations, device):
    torch = runtime["torch"]
    inputs = example["inference_inputs"]
    query_parts = []
    node_parts = []
    dense_query = None
    dense_nodes = None
    if args.feature_mode in {"hash", "hash_dense"}:
        query_parts.append(runtime["make_query_features"](inputs, args.hash_dim).to(device))
        node_parts.append(runtime["make_node_features"](inputs, args.hash_dim).to(device))
    if args.feature_mode in {"dense", "hash_dense"}:
        dense_query, dense_nodes = dense_features_for_example(example, cache, runtime, device)
        if args.dense_l2_normalize:
            dense_query = torch.nn.functional.normalize(dense_query, p=2, dim=-1)
            dense_nodes = torch.nn.functional.normalize(dense_nodes, p=2, dim=-1)
        query_parts.append(dense_query)
        node_parts.append(dense_nodes)
    if args.use_similarity_prior and dense_query is None:
        dense_query, dense_nodes = dense_features_for_example(example, cache, runtime, device)
        dense_query = torch.nn.functional.normalize(dense_query, p=2, dim=-1)
        dense_nodes = torch.nn.functional.normalize(dense_nodes, p=2, dim=-1)
    if not query_parts or not node_parts:
        raise ValueError(f"Unsupported feature_mode: {args.feature_mode}")
    query_features = torch.cat(query_parts, dim=-1)
    node_features = torch.cat(node_parts, dim=-1)
    if query_features.shape[1] != node_features.shape[1]:
        raise ValueError(
            f"Query/node feature dim mismatch: query={query_features.shape}, node={node_features.shape}"
        )
    edge_tensors = runtime["make_edge_tensors"](inputs, graph_relations, device)
    lex = runtime["lexical_features"](inputs).to(device) if args.use_lexical_features else None
    labels = torch.tensor(
        example.get("training_targets", {}).get("grounding_label_vector", [0] * node_features.shape[0]),
        dtype=torch.float32,
        device=device,
    )
    similarity_prior = None
    if args.use_similarity_prior:
        normalized_query = torch.nn.functional.normalize(dense_query, p=2, dim=-1)
        normalized_nodes = torch.nn.functional.normalize(dense_nodes, p=2, dim=-1)
        similarity_prior = (normalized_nodes * normalized_query).sum(dim=-1)
        if args.similarity_prior_normalization == "zscore":
            similarity_prior = (similarity_prior - similarity_prior.mean()) / similarity_prior.std(
                unbiased=False
            ).clamp_min(1e-6)
        elif args.similarity_prior_normalization == "center":
            similarity_prior = similarity_prior - similarity_prior.mean()
        similarity_prior = similarity_prior.clamp(
            min=-args.similarity_prior_clip,
            max=args.similarity_prior_clip,
        )
    return {
        "query_features": query_features,
        "node_features": node_features,
        "edge_tensors": edge_tensors,
        "lexical_features": lex,
        "labels": labels,
        "similarity_prior": similarity_prior,
    }


def forward_model(model, example, tensors, args):
    if args.use_relation_conditioning or args.use_similarity_prior:
        relation_id = args.relation_to_id[example["relation_type"]]
        return model(
            tensors["query_features"],
            tensors["node_features"],
            tensors["edge_tensors"],
            tensors["lexical_features"],
            relation_id=relation_id,
            similarity_prior=tensors["similarity_prior"],
        )
    return model(
        tensors["query_features"],
        tensors["node_features"],
        tensors["edge_tensors"],
        tensors["lexical_features"],
    )


def split_gold_ids(example):
    nodes = {node["id"]: node for node in example["inference_inputs"]["schema_nodes"]}
    gold_ids = example["training_targets"].get("grounding_label_ids", [])
    column_ids = [item_id for item_id in gold_ids if nodes.get(item_id, {}).get("type") == "column"]
    return gold_ids, column_ids


def train_one_epoch(
    model,
    examples,
    cache,
    runtime,
    args,
    optimizer,
    criterion,
    device,
    graph_relations,
    epoch_index=0,
):
    model.train()
    total_loss = 0.0
    epoch_examples = list(examples)
    if args.shuffle_train_examples:
        random.Random(args.seed + int(epoch_index)).shuffle(epoch_examples)
    for example in epoch_examples:
        tensors = build_feature_tensors(example, cache, runtime, args, graph_relations, device)
        output = forward_model(model, example, tensors, args)
        loss = criterion(output["logits"], tensors["labels"])
        optimizer.zero_grad()
        loss.backward()
        runtime["torch"].nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(examples), 1)


def evaluate_relation_examples(model, examples, cache, runtime, args, device, graph_relations, output_dir=None, split="dev"):
    torch = runtime["torch"]
    model.eval()
    criterion = runtime["nn"].BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, dtype=torch.float32, device=device)
    )
    losses = []
    by_relation = {
        relation: {
            "recall@5": [],
            "recall@10": [],
            "recall@20": [],
            "precision@5": [],
            "mrr": [],
            "column_recall@10": [],
            "column_recall@20": [],
        }
        for relation in args.relation_types
    }
    predictions = []
    with torch.no_grad():
        for example in examples:
            tensors = build_feature_tensors(example, cache, runtime, args, graph_relations, device)
            output = forward_model(model, example, tensors, args)
            loss = criterion(output["logits"], tensors["labels"])
            losses.append(float(loss.detach().cpu()))
            scores = output["logits"].detach().cpu().tolist()
            top_rows, ranked = ranked_prediction_rows(example, scores, args.output_top_k)
            gold_ids, gold_column_ids = split_gold_ids(example)
            ranked_column_ids = [
                item_id
                for item_id in ranked
                if example["inference_inputs"]["schema_nodes"][item_id]["type"] == "column"
            ]
            relation = example["relation_type"]
            by_relation[relation]["recall@5"].append(recall_at_k(gold_ids, ranked, 5))
            by_relation[relation]["recall@10"].append(recall_at_k(gold_ids, ranked, 10))
            by_relation[relation]["recall@20"].append(recall_at_k(gold_ids, ranked, 20))
            by_relation[relation]["precision@5"].append(precision_at_k(gold_ids, ranked, 5))
            by_relation[relation]["mrr"].append(reciprocal_rank(gold_ids, ranked))
            by_relation[relation]["column_recall@10"].append(recall_at_k(gold_column_ids, ranked_column_ids, 10))
            by_relation[relation]["column_recall@20"].append(recall_at_k(gold_column_ids, ranked_column_ids, 20))
            predictions.append(
                {
                    "example_id": example["example_id"],
                    "base_example_id": example["base_example_id"],
                    "record_index": example["record_index"],
                    "db_id": example["metadata"].get("db_id"),
                    "question_id": example["metadata"].get("question_id"),
                    "clause": relation,
                    "relation_type": relation,
                    "gold_label_ids": gold_ids,
                    "gold_label_names": example["training_targets"].get("grounding_label_names", []),
                    "similarity_prior_scale": (
                        float(output["prior_scale"].detach().cpu())
                        if args.use_similarity_prior
                        else None
                    ),
                    f"top_{args.output_top_k}": top_rows,
                }
            )

    metrics = {
        "split": split,
        "relation_example_count": len(examples),
        "loss": mean(losses),
    }
    for relation, metric_lists in by_relation.items():
        prefix = f"{relation}_"
        metrics[prefix + "example_count"] = len([ex for ex in examples if ex["relation_type"] == relation])
        for name, values in metric_lists.items():
            metrics[prefix + name] = mean(values)
    if output_dir is not None:
        write_json(output_dir / f"{split}_relation_metrics.json", metrics)
        write_jsonl(output_dir / f"{split}_relation_predictions.jsonl", predictions)
    return metrics, predictions


def infer_input_dim(args, cache):
    dim = 0
    if args.feature_mode in {"hash", "hash_dense"}:
        dim += args.hash_dim
    if args.feature_mode in {"dense", "hash_dense"}:
        if cache["query_dim"] != cache["node_dim"]:
            raise ValueError(f"Dense query/node dim mismatch: {cache['query_dim']} vs {cache['node_dim']}")
        dim += cache["query_dim"]
    return dim


def configure_reproducibility(seed, runtime, deterministic=False):
    np = runtime["np"]
    torch = runtime["torch"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def build_grounder(args, runtime, input_dim, graph_relations, device):
    model_cls = (
        runtime["RelationConditionedDSGGrounder"]
        if args.use_relation_conditioning or args.use_similarity_prior
        else runtime["DSGGrounder"]
    )
    model_kwargs = dict(
        hash_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        relations=graph_relations,
        encoder_type=args.encoder_type,
        lexical_dim=6 if args.use_lexical_features else 0,
    )
    if args.use_relation_conditioning or args.use_similarity_prior:
        model_kwargs.update(
            operation_relations=args.relation_types,
            use_relation_embedding=args.use_relation_conditioning,
            prior_init=args.similarity_prior_init,
            join_prior_init=args.join_prior_init,
        )
    return model_cls(**model_kwargs).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-relation-file", default="experiments/stage5j_relation_labels_v1/train_relation_labels.jsonl")
    parser.add_argument("--dev-relation-file", default="experiments/stage5j_relation_labels_v1/dev_relation_labels.jsonl")
    parser.add_argument("--train-graph-file", default="experiments/stage8f_dsg_data_llm_cards_qwen25_train1000_dev100/train_examples.jsonl")
    parser.add_argument("--dev-graph-file", default="experiments/stage8f_dsg_data_llm_cards_qwen25_train1000_dev100/dev_examples.jsonl")
    parser.add_argument("--embedding-cache-dir", default="experiments/stage8g_embedding_cache_qwen3_06b_train1000_dev100")
    parser.add_argument("--output-dir", default="experiments/stage8g_dense_relation_grounder_smoke")
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--dev-limit", type=int, default=100)
    parser.add_argument("--relation-types", type=parse_clause_list, default=",".join(DEFAULT_OPERATION_RELATIONS))
    parser.add_argument("--include-empty-relation-examples", action="store_true")
    parser.add_argument("--feature-mode", choices=["hash", "dense", "hash_dense"], default="dense")
    parser.add_argument("--dense-l2-normalize", action="store_true", default=True)
    parser.add_argument("--no-dense-l2-normalize", dest="dense_l2_normalize", action="store_false")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-train-examples",
        action="store_true",
        default=True,
        help="Deterministically reshuffle relation examples at each epoch (enabled by default).",
    )
    parser.add_argument(
        "--no-shuffle-train-examples",
        dest="shuffle_train_examples",
        action="store_false",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic PyTorch kernels when available (may reduce throughput).",
    )
    parser.add_argument("--encoder-type", choices=["rgcn", "rgta"], default="rgcn")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-top-k", type=int, default=30)
    parser.add_argument("--use-lexical-features", action="store_true")
    parser.add_argument(
        "--use-relation-conditioned-prior",
        action="store_true",
        help=(
            "Convenience switch enabling both --use-relation-conditioning and "
            "--use-similarity-prior."
        ),
    )
    parser.add_argument(
        "--use-relation-conditioning",
        action="store_true",
        help="Condition the query state on a trainable operation-relation embedding.",
    )
    parser.add_argument(
        "--use-similarity-prior",
        action="store_true",
        help="Fuse cosine similarity through a trainable relation-specific non-negative scale.",
    )
    parser.add_argument("--similarity-prior-init", type=float, default=1.0)
    parser.add_argument("--join-prior-init", type=float, default=0.1)
    parser.add_argument(
        "--similarity-prior-normalization",
        choices=["none", "center", "zscore"],
        default="zscore",
    )
    parser.add_argument("--similarity-prior-clip", type=float, default=4.0)
    parser.add_argument("--selection-metric", default="assembled_schema_recall@30")
    parser.add_argument("--selection-mode", choices=["max", "min"], default="max")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--assembly-budgets",
        type=parse_budget_text,
        default=(
            "OUTPUT_TARGET:7,JOIN_BRIDGE:8,PREDICATE_COLUMN:7,"
            "METRIC_TARGET:4,VALUE_ANCHOR:2,TEMPORAL_FILTER:1,ORDER_KEY:1"
        ),
    )
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--dry-run-data-check", action="store_true")
    args = parser.parse_args()
    if args.use_relation_conditioned_prior:
        args.use_relation_conditioning = True
        args.use_similarity_prior = True
    args.relation_to_id = {relation: index for index, relation in enumerate(args.relation_types)}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = import_runtime()
    configure_reproducibility(args.seed, runtime, deterministic=args.deterministic)
    train_aligned = load_aligned_records(Path(args.train_relation_file), Path(args.train_graph_file), args.train_limit)
    dev_aligned = load_aligned_records(Path(args.dev_relation_file), Path(args.dev_graph_file), args.dev_limit)
    train_examples = make_relation_examples(
        train_aligned,
        args.relation_types,
        include_empty_relation_examples=args.include_empty_relation_examples,
    )
    dev_examples = make_relation_examples(
        dev_aligned,
        args.relation_types,
        include_empty_relation_examples=args.include_empty_relation_examples,
    )
    cache_dir = Path(args.embedding_cache_dir)
    train_cache = load_cache(cache_dir, "train", runtime)
    dev_cache = load_cache(cache_dir, "dev", runtime)
    input_dim = infer_input_dim(args, train_cache)
    data_report = {
        "train_base_count": len(train_aligned),
        "dev_base_count": len(dev_aligned),
        "train_relation_example_count": len(train_examples),
        "dev_relation_example_count": len(dev_examples),
        "relation_types": args.relation_types,
        "feature_mode": args.feature_mode,
        "input_dim": input_dim,
        "embedding_cache_dir": str(cache_dir),
        "train_query_dim": train_cache["query_dim"],
        "train_node_dim": train_cache["node_dim"],
        "dev_query_dim": dev_cache["query_dim"],
        "dev_node_dim": dev_cache["node_dim"],
        "innovation": (
            "Dense semantic-card embeddings are injected into schema/query features before graph propagation. "
            "Relation conditioning makes the operation an explicit latent input; similarity-prior mode "
            "preserves direct semantic evidence through a learned relation-specific residual score."
        ),
    }
    write_json(output_dir / "data_report.json", data_report)
    if args.dry_run_data_check:
        print(json.dumps(data_report, ensure_ascii=False, indent=2))
        return

    torch = runtime["torch"]
    device = torch.device(args.device)
    graph_relations = list(runtime["DEFAULT_RELATIONS"])
    model = build_grounder(args, runtime, input_dim, graph_relations, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = runtime["nn"].BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, dtype=torch.float32, device=device)
    )
    config = vars(args).copy()
    config.pop("relation_to_id", None)
    config["torch_version"] = torch.__version__
    config["graph_relations"] = graph_relations
    config["input_dim"] = input_dim
    write_json(output_dir / "train_config.json", config)

    best_epoch = None
    best_value = None
    best_state = None
    epochs_without_improvement = 0
    stopped_early = False
    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_examples,
                train_cache,
                runtime,
                args,
                optimizer,
                criterion,
                device,
                graph_relations,
                epoch_index=epoch,
            )
            dev_relation_metrics, dev_relation_predictions = evaluate_relation_examples(
                model, dev_examples, dev_cache, runtime, args, device, graph_relations
            )
            assembled_rows = assemble_predictions(
                dev_relation_predictions,
                dev_aligned,
                output_top_k=args.output_top_k,
                budgets=args.assembly_budgets,
            )
            assembled_metrics = evaluate_assembled(assembled_rows)
            dev_metrics = {**dev_relation_metrics, **assembled_metrics}
            if args.use_similarity_prior:
                dev_metrics["learned_similarity_prior_scales"] = {
                    relation: float(scale)
                    for relation, scale in zip(
                        args.relation_types,
                        model.prior_scales().detach().cpu().tolist(),
                    )
                }
            selected_value = get_metric(dev_metrics, args.selection_metric)
            improved = is_better_metric(selected_value, best_value, args.selection_mode, args.min_delta)
            if improved:
                best_epoch = epoch
                best_value = selected_value
                best_state = clone_state_dict_to_cpu(model)
                epochs_without_improvement = 0
                best_metrics = dev_metrics.copy()
                best_metrics.update(
                    {
                        "best_epoch": best_epoch,
                        "selection_metric": args.selection_metric,
                        "selection_value": best_value,
                        "selection_mode": args.selection_mode,
                    }
                )
                write_json(output_dir / "best_metrics.json", best_metrics)
                if not args.no_save_model:
                    torch.save(best_state, output_dir / "dense_relation_grounder_model.pt")
            else:
                epochs_without_improvement += 1
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "selection_metric": args.selection_metric,
                "selection_value": selected_value,
                "best_epoch": best_epoch,
                "best_selection_value": best_value,
                "is_best": improved,
                "epochs_without_improvement": epochs_without_improvement,
            }
            row.update({f"dev_{key}": value for key, value in dev_metrics.items() if key != "split"})
            log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False))
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                stopped_early = True
                break

    last_relation_metrics, last_relation_predictions = evaluate_relation_examples(
        model, dev_examples, dev_cache, runtime, args, device, graph_relations, output_dir=output_dir, split="dev_last"
    )
    last_assembled = assemble_predictions(
        last_relation_predictions,
        dev_aligned,
        output_top_k=args.output_top_k,
        budgets=args.assembly_budgets,
    )
    last_assembled_metrics = evaluate_assembled(last_assembled)
    write_jsonl(output_dir / "dev_last_assembled_predictions.jsonl", last_assembled)
    if not args.no_save_model:
        torch.save(model.state_dict(), output_dir / "dense_relation_grounder_last_model.pt")

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
        final_relation_metrics, final_relation_predictions = evaluate_relation_examples(
            model, dev_examples, dev_cache, runtime, args, device, graph_relations, output_dir=output_dir, split="dev"
        )
        final_assembled = assemble_predictions(
            final_relation_predictions,
            dev_aligned,
            output_top_k=args.output_top_k,
            budgets=args.assembly_budgets,
        )
        final_assembled_metrics = evaluate_assembled(final_assembled)
        write_jsonl(output_dir / "dev_assembled_predictions.jsonl", final_assembled)
    else:
        final_relation_metrics = last_relation_metrics
        final_assembled_metrics = last_assembled_metrics

    final_metrics = {**final_relation_metrics, **final_assembled_metrics}
    if args.use_similarity_prior:
        final_metrics["learned_similarity_prior_scales"] = {
            relation: float(scale)
            for relation, scale in zip(
                args.relation_types,
                model.prior_scales().detach().cpu().tolist(),
            )
        }
    write_json(output_dir / "dev_metrics.json", final_metrics)
    summary = {
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "selection_value": best_value,
        "selection_mode": args.selection_mode,
        "stopped_early": stopped_early,
        "last_epoch": epoch if "epoch" in locals() else 0,
        "dev_metrics_are_from": "best_checkpoint" if best_state is not None else "last_checkpoint",
        "dev_metrics": final_metrics,
        "dev_last_metrics": {**last_relation_metrics, **last_assembled_metrics},
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
