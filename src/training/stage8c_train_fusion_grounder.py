import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.operation_grounder import (  # noqa: E402
    DEFAULT_OPERATIONS,
    OperationFusionGrounder,
)
from src.training.stage5_train_dsg_grounder import (  # noqa: E402
    clone_state_dict_to_cpu,
    get_metric,
    is_better_metric,
    mean,
    precision_at_k,
    ranked_prediction_rows,
    recall_at_k,
    reciprocal_rank,
    write_json,
    write_jsonl,
)
from src.training.stage5g_train_clause_grounder import parse_clause_list  # noqa: E402
from src.training.stage5j_train_relation_grounder import load_aligned_records  # noqa: E402


def import_torch_and_features():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Stage 8-C training.") from exc
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


def label_vector(node_count, ids):
    labels = [0.0] * node_count
    for item_id in ids or []:
        item_id = int(item_id)
        if 0 <= item_id < node_count:
            labels[item_id] = 1.0
    return labels


def item_to_tensors(item, helpers, args, relations, operations, device):
    torch = helpers["torch"]
    relation_record = item["clause_record"]
    graph_example = item["graph_example"]
    inputs = graph_example["inference_inputs"]
    node_count = len(inputs.get("schema_nodes", []))
    operation_labels = []
    for operation in operations:
        operation_labels.append(label_vector(node_count, relation_record.get("relation_labels", {}).get(operation, [])))
    whole_labels = label_vector(node_count, relation_record.get("whole_sql_labels", []))
    lex = helpers["lexical_features"](inputs).to(device) if args.use_lexical_features else None
    return {
        "node_features": helpers["make_node_features"](inputs, args.hash_dim).to(device),
        "query_features": helpers["make_query_features"](inputs, args.hash_dim).to(device),
        "edge_tensors": helpers["make_edge_tensors"](inputs, relations, device),
        "lexical_features": lex,
        "operation_labels": torch.tensor(operation_labels, dtype=torch.float32, device=device).t().contiguous(),
        "whole_labels": torch.tensor(whole_labels, dtype=torch.float32, device=device),
    }


def split_gold_ids_from_graph(graph_example, gold_ids):
    nodes = {int(node["id"]): node for node in graph_example["inference_inputs"]["schema_nodes"]}
    column_ids = [int(item_id) for item_id in gold_ids if nodes.get(int(item_id), {}).get("type") == "column"]
    return [int(item_id) for item_id in gold_ids], column_ids


def operation_metrics_for_scores(graph_example, scores, gold_ids, top_k):
    top_rows, ranked = ranked_prediction_rows(graph_example, scores, top_k)
    _, gold_column_ids = split_gold_ids_from_graph(graph_example, gold_ids)
    ranked_column_ids = [
        item_id
        for item_id in ranked
        if graph_example["inference_inputs"]["schema_nodes"][item_id]["type"] == "column"
    ]
    return top_rows, ranked, {
        "recall@5": recall_at_k(gold_ids, ranked, 5),
        "recall@10": recall_at_k(gold_ids, ranked, 10),
        "recall@20": recall_at_k(gold_ids, ranked, 20),
        "precision@5": precision_at_k(gold_ids, ranked, 5),
        "mrr": reciprocal_rank(gold_ids, ranked),
        "column_recall@10": recall_at_k(gold_column_ids, ranked_column_ids, 10),
        "column_recall@20": recall_at_k(gold_column_ids, ranked_column_ids, 20),
    }


def whole_prediction_row(item, scores, output_top_k):
    relation_record = item["clause_record"]
    graph_example = item["graph_example"]
    gold_ids, gold_column_ids = split_gold_ids_from_graph(graph_example, relation_record.get("whole_sql_labels", []))
    top_rows, ranked = ranked_prediction_rows(graph_example, scores, output_top_k)
    ranked_column_ids = [
        item_id
        for item_id in ranked
        if graph_example["inference_inputs"]["schema_nodes"][item_id]["type"] == "column"
    ]
    return {
        "example_id": graph_example.get("example_id"),
        "record_index": item["record_index"],
        "db_id": relation_record.get("db_id"),
        "question_id": relation_record.get("question_id"),
        "question": relation_record.get("question"),
        "evidence": relation_record.get("evidence"),
        "gold_label_ids": sorted(gold_ids),
        "gold_label_names": relation_record.get("whole_sql_label_names", []),
        f"top_{output_top_k}": top_rows,
        "whole_recall@30": recall_at_k(gold_ids, ranked, 30),
        "whole_precision@30": precision_at_k(gold_ids, ranked, min(30, len(ranked))),
        "whole_column_recall@30": recall_at_k(gold_column_ids, ranked_column_ids, 30),
        # Alias for downstream Stage 7 prompt/decoding tools that expect assembled files.
        "assembled_recall@30": recall_at_k(gold_ids, ranked, 30),
        "assembled_precision@30": precision_at_k(gold_ids, ranked, min(30, len(ranked))),
    }


def train_one_epoch(model, aligned_records, helpers, args, optimizer, op_criterion, whole_criterion, device, relations, operations):
    model.train()
    losses = []
    op_losses = []
    whole_losses = []
    for item in aligned_records:
        tensors = item_to_tensors(item, helpers, args, relations, operations, device)
        output = model(
            tensors["query_features"],
            tensors["node_features"],
            tensors["edge_tensors"],
            tensors["lexical_features"],
        )
        op_loss = op_criterion(output["operation_logits"], tensors["operation_labels"])
        whole_loss = whole_criterion(output["whole_logits"], tensors["whole_labels"])
        loss = args.operation_loss_weight * op_loss + args.whole_loss_weight * whole_loss
        optimizer.zero_grad()
        loss.backward()
        helpers["torch"].nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        op_losses.append(float(op_loss.detach().cpu()))
        whole_losses.append(float(whole_loss.detach().cpu()))
    return {
        "loss": mean(losses),
        "operation_loss": mean(op_losses),
        "whole_loss": mean(whole_losses),
    }


def evaluate(model, aligned_records, helpers, args, device, relations, operations, output_dir=None, split="dev"):
    torch = helpers["torch"]
    model.eval()
    op_criterion = helpers["nn"].BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.operation_pos_weight, dtype=torch.float32, device=device)
    )
    whole_criterion = helpers["nn"].BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.whole_pos_weight, dtype=torch.float32, device=device)
    )
    losses = []
    op_losses = []
    whole_losses = []
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
    whole_recalls = []
    whole_precisions = []
    whole_column_recalls = []
    whole_predictions = []
    operation_predictions = []

    with torch.no_grad():
        for item in aligned_records:
            tensors = item_to_tensors(item, helpers, args, relations, operations, device)
            output = model(
                tensors["query_features"],
                tensors["node_features"],
                tensors["edge_tensors"],
                tensors["lexical_features"],
            )
            op_loss = op_criterion(output["operation_logits"], tensors["operation_labels"])
            whole_loss = whole_criterion(output["whole_logits"], tensors["whole_labels"])
            loss = args.operation_loss_weight * op_loss + args.whole_loss_weight * whole_loss
            losses.append(float(loss.detach().cpu()))
            op_losses.append(float(op_loss.detach().cpu()))
            whole_losses.append(float(whole_loss.detach().cpu()))

            graph_example = item["graph_example"]
            relation_record = item["clause_record"]
            op_scores = output["operation_logits"].detach().cpu()
            for op_idx, operation in enumerate(operations):
                gold_ids = relation_record.get("relation_labels", {}).get(operation, [])
                top_rows, _ranked, metrics = operation_metrics_for_scores(
                    graph_example, op_scores[:, op_idx].tolist(), gold_ids, args.output_top_k
                )
                for metric_name, value in metrics.items():
                    by_operation[operation][metric_name].append(value)
                operation_predictions.append(
                    {
                        "example_id": f"{graph_example.get('example_id')}::{operation}",
                        "base_example_id": graph_example.get("example_id"),
                        "record_index": item["record_index"],
                        "db_id": relation_record.get("db_id"),
                        "question_id": relation_record.get("question_id"),
                        "clause": operation,
                        "relation_type": operation,
                        "gold_label_ids": [int(x) for x in gold_ids],
                        "gold_label_names": relation_record.get("relation_label_names", {}).get(operation, []),
                        f"top_{args.output_top_k}": top_rows,
                    }
                )

            whole_row = whole_prediction_row(item, output["whole_logits"].detach().cpu().tolist(), args.output_top_k)
            whole_predictions.append(whole_row)
            whole_recalls.append(whole_row["whole_recall@30"])
            whole_precisions.append(whole_row["whole_precision@30"])
            whole_column_recalls.append(whole_row["whole_column_recall@30"])

    metrics = {
        "split": split,
        "sample_count": len(aligned_records),
        "loss": mean(losses),
        "operation_loss": mean(op_losses),
        "whole_loss": mean(whole_losses),
        "whole_schema_recall@30": mean(whole_recalls),
        "whole_schema_precision@30": mean(whole_precisions),
        "whole_column_recall@30": mean(whole_column_recalls),
        "whole_missing_samples@30": sum(1 for row in whole_predictions if (row["whole_recall@30"] or 0) < 1.0),
        # Compatibility aliases.
        "assembled_schema_recall@30": mean(whole_recalls),
        "assembled_schema_precision@30": mean(whole_precisions),
        "assembled_missing_samples@30": sum(1 for row in whole_predictions if (row["whole_recall@30"] or 0) < 1.0),
    }
    for operation, metric_lists in by_operation.items():
        prefix = f"{operation}_"
        metrics[prefix + "example_count"] = len(aligned_records)
        for name, values in metric_lists.items():
            metrics[prefix + name] = mean(values)

    if output_dir is not None:
        write_json(output_dir / f"{split}_metrics.json", metrics)
        write_jsonl(output_dir / f"{split}_operation_predictions.jsonl", operation_predictions)
        write_jsonl(output_dir / f"{split}_whole_predictions.jsonl", whole_predictions)
        write_jsonl(output_dir / f"{split}_assembled_predictions.jsonl", whole_predictions)
    return metrics, operation_predictions, whole_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-relation-file", default="experiments/stage5j_relation_labels_v1/train_relation_labels.jsonl")
    parser.add_argument("--dev-relation-file", default="experiments/stage5j_relation_labels_v1/dev_relation_labels.jsonl")
    parser.add_argument("--train-graph-file", default="experiments/stage5i_dsg_data_semantic_v1/train_examples.jsonl")
    parser.add_argument("--dev-graph-file", default="experiments/stage5i_dsg_data_semantic_v1/dev_examples.jsonl")
    parser.add_argument("--output-dir", default="experiments/stage8c_fusion_grounder_smoke")
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--dev-limit", type=int, default=100)
    parser.add_argument("--operation-types", type=parse_clause_list, default=",".join(DEFAULT_OPERATIONS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--operation-pos-weight", type=float, default=3.0)
    parser.add_argument("--whole-pos-weight", type=float, default=3.0)
    parser.add_argument("--operation-loss-weight", type=float, default=1.0)
    parser.add_argument("--whole-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-top-k", type=int, default=30)
    parser.add_argument("--use-lexical-features", action="store_true")
    parser.add_argument("--selection-metric", default="whole_schema_recall@30")
    parser.add_argument("--selection-mode", choices=["max", "min"], default="max")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--dry-run-data-check", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_aligned = load_aligned_records(Path(args.train_relation_file), Path(args.train_graph_file), args.train_limit)
    dev_aligned = load_aligned_records(Path(args.dev_relation_file), Path(args.dev_graph_file), args.dev_limit)
    data_report = {
        "train_base_count": len(train_aligned),
        "dev_base_count": len(dev_aligned),
        "operation_types": args.operation_types,
        "selection_metric": args.selection_metric,
        "innovation": (
            "Stage 8-C learns whole-schema fusion from operation-conditioned graph beliefs. "
            "Operation logits are latent variables supervised by relation labels, while whole logits "
            "are directly supervised by whole-SQL schema labels."
        ),
    }
    write_json(output_dir / "data_report.json", data_report)
    if args.dry_run_data_check:
        print(json.dumps(data_report, ensure_ascii=False, indent=2))
        return

    helpers = import_torch_and_features()
    torch = helpers["torch"]
    device = torch.device(args.device)
    relations = list(helpers["DEFAULT_RELATIONS"])
    operations = list(args.operation_types)
    model = OperationFusionGrounder(
        hash_dim=args.hash_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        relations=relations,
        operations=operations,
        lexical_dim=6 if args.use_lexical_features else 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    op_criterion = helpers["nn"].BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.operation_pos_weight, dtype=torch.float32, device=device)
    )
    whole_criterion = helpers["nn"].BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.whole_pos_weight, dtype=torch.float32, device=device)
    )
    config = vars(args).copy()
    config["torch_version"] = torch.__version__
    config["graph_relations"] = relations
    write_json(output_dir / "train_config.json", config)

    best_epoch = None
    best_value = None
    best_state = None
    best_metrics = None
    epochs_without_improvement = 0
    stopped_early = False
    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model, train_aligned, helpers, args, optimizer, op_criterion, whole_criterion, device, relations, operations
            )
            dev_metrics, _op_predictions, _whole_predictions = evaluate(
                model, dev_aligned, helpers, args, device, relations, operations
            )
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
                    torch.save(best_state, output_dir / "fusion_grounder_model.pt")
            else:
                epochs_without_improvement += 1

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_operation_loss": train_metrics["operation_loss"],
                "train_whole_loss": train_metrics["whole_loss"],
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

    last_metrics, _last_op, _last_whole = evaluate(
        model, dev_aligned, helpers, args, device, relations, operations, output_dir=output_dir, split="dev_last"
    )
    if not args.no_save_model:
        torch.save(model.state_dict(), output_dir / "fusion_grounder_last_model.pt")

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
        final_metrics, _final_op, _final_whole = evaluate(
            model, dev_aligned, helpers, args, device, relations, operations, output_dir=output_dir, split="dev"
        )
    else:
        final_metrics = last_metrics

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
        "dev_last_metrics": last_metrics,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
