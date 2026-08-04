import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.operation_grounder import (  # noqa: E402
    DEFAULT_OPERATIONS,
    OperationConditionedGrounder,
)
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
    load_aligned_records,
    make_relation_examples,
)
from src.assembly.stage8b_calibrated_assembly import (  # noqa: E402
    DEFAULT_OPERATION_BUDGETS,
    DEFAULT_OPERATION_WEIGHTS,
    assemble_calibrated,
    evaluate_assembled_rows,
    parse_float_map,
)


def import_torch_and_features():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Stage 8-A training.") from exc
    from src.modeling.dsg_grounder import (
        DEFAULT_RELATIONS,
        lexical_features,
        make_edge_tensors,
        make_node_features,
        make_query_features,
    )

    return {
        "torch": torch,
        "nn": nn,
        "DEFAULT_RELATIONS": DEFAULT_RELATIONS,
        "lexical_features": lexical_features,
        "make_edge_tensors": make_edge_tensors,
        "make_node_features": make_node_features,
        "make_query_features": make_query_features,
    }


def operation_id_for(example, operations):
    relation_type = example.get("relation_type") or example.get("clause")
    if relation_type not in operations:
        raise ValueError(f"Unknown operation/relation type {relation_type!r}; available={operations}")
    return operations.index(relation_type)


def example_to_tensors(example, helpers, args, relations, operations, device):
    torch = helpers["torch"]
    inputs = example["inference_inputs"]
    targets = example.get("training_targets", {})
    node_features = helpers["make_node_features"](inputs, args.hash_dim).to(device)
    query_features = helpers["make_query_features"](inputs, args.hash_dim).to(device)
    edge_tensors = helpers["make_edge_tensors"](inputs, relations, device)
    lex = helpers["lexical_features"](inputs).to(device) if args.use_lexical_features else None
    labels = torch.tensor(
        targets.get("grounding_label_vector", [0] * node_features.shape[0]),
        dtype=torch.float32,
        device=device,
    )
    operation_id = torch.tensor([operation_id_for(example, operations)], dtype=torch.long, device=device)
    return {
        "node_features": node_features,
        "query_features": query_features,
        "edge_tensors": edge_tensors,
        "lexical_features": lex,
        "labels": labels,
        "operation_id": operation_id,
    }


def split_gold_ids(example):
    nodes = {node["id"]: node for node in example["inference_inputs"]["schema_nodes"]}
    gold_ids = example["training_targets"].get("grounding_label_ids", [])
    column_ids = [item_id for item_id in gold_ids if nodes.get(item_id, {}).get("type") == "column"]
    return gold_ids, column_ids


def assemble_operation_predictions(predictions, aligned_records, args):
    if args.assembly_method == "legacy":
        return assemble_predictions(
            predictions,
            aligned_records,
            output_top_k=args.output_top_k,
            budgets=args.assembly_budgets,
        )
    return assemble_calibrated(
        operation_predictions=predictions,
        aligned_records=aligned_records,
        output_top_k=args.output_top_k,
        operation_budgets=args.operation_budgets,
        operation_weights=args.operation_weights,
        rank_weight=args.calibration_rank_weight,
        z_weight=args.calibration_z_weight,
        table_column_bonus=args.table_column_bonus,
        fk_bonus=args.fk_bonus,
        max_tables=args.max_tables,
        include_tables=not args.exclude_tables,
    )


def evaluate_assembled_predictions(rows, args):
    if args.assembly_method == "legacy":
        return evaluate_assembled(rows)
    return evaluate_assembled_rows(rows)


def train_one_epoch(model, examples, helpers, args, optimizer, criterion, device, relations, operations):
    model.train()
    total_loss = 0.0
    for example in examples:
        tensors = example_to_tensors(example, helpers, args, relations, operations, device)
        output = model(
            tensors["query_features"],
            tensors["node_features"],
            tensors["edge_tensors"],
            tensors["operation_id"],
            tensors["lexical_features"],
        )
        loss = criterion(output["logits"], tensors["labels"])
        optimizer.zero_grad()
        loss.backward()
        helpers["torch"].nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(examples), 1)


def evaluate_operation_examples(model, examples, helpers, args, device, relations, operations, output_dir=None, split="dev"):
    torch = helpers["torch"]
    model.eval()
    criterion = helpers["nn"].BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, dtype=torch.float32, device=device)
    )
    losses = []
    by_operation = {
        operation: {
            "recall@5": [],
            "recall@10": [],
            "recall@20": [],
            "precision@5": [],
            "mrr": [],
            "column_recall@10": [],
            "column_recall@20": [],
        }
        for operation in operations
    }
    predictions = []

    with torch.no_grad():
        for example in examples:
            tensors = example_to_tensors(example, helpers, args, relations, operations, device)
            output = model(
                tensors["query_features"],
                tensors["node_features"],
                tensors["edge_tensors"],
                tensors["operation_id"],
                tensors["lexical_features"],
            )
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
            operation = example["relation_type"]
            by_operation[operation]["recall@5"].append(recall_at_k(gold_ids, ranked, 5))
            by_operation[operation]["recall@10"].append(recall_at_k(gold_ids, ranked, 10))
            by_operation[operation]["recall@20"].append(recall_at_k(gold_ids, ranked, 20))
            by_operation[operation]["precision@5"].append(precision_at_k(gold_ids, ranked, 5))
            by_operation[operation]["mrr"].append(reciprocal_rank(gold_ids, ranked))
            by_operation[operation]["column_recall@10"].append(recall_at_k(gold_column_ids, ranked_column_ids, 10))
            by_operation[operation]["column_recall@20"].append(recall_at_k(gold_column_ids, ranked_column_ids, 20))
            predictions.append(
                {
                    "example_id": example["example_id"],
                    "base_example_id": example["base_example_id"],
                    "record_index": example["record_index"],
                    "db_id": example["metadata"].get("db_id"),
                    "question_id": example["metadata"].get("question_id"),
                    "clause": operation,
                    "relation_type": operation,
                    "gold_label_ids": gold_ids,
                    "gold_label_names": example["training_targets"].get("grounding_label_names", []),
                    f"top_{args.output_top_k}": top_rows,
                }
            )

    metrics = {
        "split": split,
        "operation_example_count": len(examples),
        "loss": mean(losses),
    }
    for operation, metric_lists in by_operation.items():
        prefix = f"{operation}_"
        metrics[prefix + "example_count"] = len([ex for ex in examples if ex["relation_type"] == operation])
        for name, values in metric_lists.items():
            metrics[prefix + name] = mean(values)

    if output_dir is not None:
        write_json(output_dir / f"{split}_operation_metrics.json", metrics)
        write_jsonl(output_dir / f"{split}_operation_predictions.jsonl", predictions)
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-relation-file", default="experiments/stage5j_relation_labels_v1/train_relation_labels.jsonl")
    parser.add_argument("--dev-relation-file", default="experiments/stage5j_relation_labels_v1/dev_relation_labels.jsonl")
    parser.add_argument("--train-graph-file", default="experiments/stage5i_dsg_data_semantic_v1/train_examples.jsonl")
    parser.add_argument("--dev-graph-file", default="experiments/stage5i_dsg_data_semantic_v1/dev_examples.jsonl")
    parser.add_argument("--output-dir", default="experiments/stage8_operation_grounder_rgta_smoke")
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--dev-limit", type=int, default=100)
    parser.add_argument("--operation-types", type=parse_clause_list, default=",".join(DEFAULT_OPERATIONS))
    parser.add_argument("--include-empty-operation-examples", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-top-k", type=int, default=30)
    parser.add_argument("--use-lexical-features", action="store_true")
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
    parser.add_argument("--assembly-method", choices=["legacy", "calibrated"], default="calibrated")
    parser.add_argument(
        "--operation-budgets",
        type=parse_budget_text,
        default=",".join(f"{key}:{value}" for key, value in DEFAULT_OPERATION_BUDGETS.items()),
    )
    parser.add_argument("--operation-weights", type=lambda text: parse_float_map(text, DEFAULT_OPERATION_WEIGHTS), default=None)
    parser.add_argument("--calibration-rank-weight", type=float, default=0.7)
    parser.add_argument("--calibration-z-weight", type=float, default=0.3)
    parser.add_argument("--table-column-bonus", type=float, default=0.2)
    parser.add_argument("--fk-bonus", type=float, default=0.1)
    parser.add_argument("--max-tables", type=int, default=4)
    parser.add_argument("--exclude-tables", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--dry-run-data-check", action="store_true")
    args = parser.parse_args()
    if args.operation_weights is None:
        args.operation_weights = dict(DEFAULT_OPERATION_WEIGHTS)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_aligned = load_aligned_records(Path(args.train_relation_file), Path(args.train_graph_file), args.train_limit)
    dev_aligned = load_aligned_records(Path(args.dev_relation_file), Path(args.dev_graph_file), args.dev_limit)
    train_examples = make_relation_examples(
        train_aligned,
        args.operation_types,
        include_empty_relation_examples=args.include_empty_operation_examples,
    )
    dev_examples = make_relation_examples(
        dev_aligned,
        args.operation_types,
        include_empty_relation_examples=args.include_empty_operation_examples,
    )

    data_report = {
        "train_base_count": len(train_aligned),
        "dev_base_count": len(dev_aligned),
        "train_operation_example_count": len(train_examples),
        "dev_operation_example_count": len(dev_examples),
        "operation_types": args.operation_types,
        "include_empty_operation_examples": args.include_empty_operation_examples,
        "assembly_budgets": args.assembly_budgets,
        "assembly_method": args.assembly_method,
        "operation_budgets": args.operation_budgets,
        "operation_weights": args.operation_weights,
        "calibration_rank_weight": args.calibration_rank_weight,
        "calibration_z_weight": args.calibration_z_weight,
        "table_column_bonus": args.table_column_bonus,
        "fk_bonus": args.fk_bonus,
        "max_tables": args.max_tables,
        "include_tables": not args.exclude_tables,
        "innovation": (
            "Operation embeddings condition RGTA message passing, so schema graph propagation "
            "changes for different SQL relational operations instead of using static R-GCN/RGTA encodings. "
            "Stage 8-B additionally calibrates operation-specific belief distributions before schema assembly."
        ),
    }
    write_json(output_dir / "data_report.json", data_report)
    if args.dry_run_data_check:
        print(json.dumps(data_report, ensure_ascii=False, indent=2))
        return

    helpers = import_torch_and_features()
    torch = helpers["torch"]
    device = torch.device(args.device)
    graph_relations = list(helpers["DEFAULT_RELATIONS"])
    operations = list(args.operation_types)
    model = OperationConditionedGrounder(
        hash_dim=args.hash_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        relations=graph_relations,
        operations=operations,
        lexical_dim=6 if args.use_lexical_features else 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = helpers["nn"].BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, dtype=torch.float32, device=device)
    )

    config = vars(args).copy()
    config["torch_version"] = torch.__version__
    config["graph_relations"] = graph_relations
    write_json(output_dir / "train_config.json", config)

    best_epoch = None
    best_value = None
    best_state = None
    epochs_without_improvement = 0
    stopped_early = False
    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_examples, helpers, args, optimizer, criterion, device, graph_relations, operations
            )
            dev_operation_metrics, dev_operation_predictions = evaluate_operation_examples(
                model, dev_examples, helpers, args, device, graph_relations, operations
            )
            assembled_rows = assemble_operation_predictions(dev_operation_predictions, dev_aligned, args)
            assembled_metrics = evaluate_assembled_predictions(assembled_rows, args)
            dev_metrics = {**dev_operation_metrics, **assembled_metrics}
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
                    torch.save(best_state, output_dir / "operation_grounder_model.pt")
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

    last_operation_metrics, last_operation_predictions = evaluate_operation_examples(
        model, dev_examples, helpers, args, device, graph_relations, operations, output_dir=output_dir, split="dev_last"
    )
    last_assembled = assemble_operation_predictions(last_operation_predictions, dev_aligned, args)
    last_assembled_metrics = evaluate_assembled_predictions(last_assembled, args)
    write_jsonl(output_dir / "dev_last_assembled_predictions.jsonl", last_assembled)
    if not args.no_save_model:
        torch.save(model.state_dict(), output_dir / "operation_grounder_last_model.pt")

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
        final_operation_metrics, final_operation_predictions = evaluate_operation_examples(
            model, dev_examples, helpers, args, device, graph_relations, operations, output_dir=output_dir, split="dev"
        )
        final_assembled = assemble_operation_predictions(final_operation_predictions, dev_aligned, args)
        final_assembled_metrics = evaluate_assembled_predictions(final_assembled, args)
        write_jsonl(output_dir / "dev_assembled_predictions.jsonl", final_assembled)
    else:
        final_operation_metrics = last_operation_metrics
        final_assembled_metrics = last_assembled_metrics

    final_metrics = {**final_operation_metrics, **final_assembled_metrics}
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
        "dev_last_metrics": {**last_operation_metrics, **last_assembled_metrics},
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
