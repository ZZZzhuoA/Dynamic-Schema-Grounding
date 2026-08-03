import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5_train_dsg_grounder import (  # noqa: E402
    clone_state_dict_to_cpu,
    get_metric,
    import_torch_and_model,
    is_better_metric,
    read_jsonl,
    write_json,
    write_jsonl,
)
from src.training.stage5g_train_clause_grounder import (  # noqa: E402
    assemble_predictions,
    evaluate_assembled,
    evaluate_clause_examples,
    label_vector,
    parse_budget_text,
    parse_clause_list,
    train_one_epoch,
)


DEFAULT_RELATIONS = [
    "OUTPUT_TARGET",
    "ENTITY_NAME",
    "METRIC_TARGET",
    "PREDICATE_COLUMN",
    "VALUE_ANCHOR",
    "TEMPORAL_FILTER",
    "ORDER_KEY",
    "GROUP_KEY",
    "JOIN_BRIDGE",
    "FORMULA_COMPONENT",
]


def load_aligned_records(relation_file: Path, graph_file: Path, limit=None):
    relation_records = read_jsonl(relation_file, limit=limit)
    graph_examples = read_jsonl(graph_file, limit=limit)
    if len(relation_records) != len(graph_examples):
        raise ValueError(
            f"Relation/graph length mismatch: {relation_file} has {len(relation_records)}, "
            f"{graph_file} has {len(graph_examples)}"
        )
    aligned = []
    for index, (relation_record, graph_example) in enumerate(zip(relation_records, graph_examples)):
        graph_inputs = graph_example.get("inference_inputs", {})
        if relation_record.get("db_id") != graph_inputs.get("db_id"):
            raise ValueError(
                f"db_id mismatch at index {index}: relation={relation_record.get('db_id')} "
                f"graph={graph_inputs.get('db_id')}"
            )
        aligned.append({"clause_record": relation_record, "graph_example": graph_example, "record_index": index})
    return aligned


def make_relation_inference_inputs(graph_inputs, relation_type):
    inputs = deepcopy(graph_inputs)
    question = inputs.get("question") or ""
    evidence = inputs.get("evidence") or ""
    relation_text = relation_type.upper().replace("_", " ")
    inputs["question"] = f"[{relation_text} question-schema relation grounding] {question}"
    inputs["evidence"] = f"{evidence}\nTarget relation type: {relation_text}".strip()
    inputs["target_relation_type"] = relation_type
    # Keep this alias so reused evaluation/assembly helpers can treat relation types as slots.
    inputs["target_clause"] = relation_type
    return inputs


def make_relation_examples(aligned_records, relation_types, include_empty_relation_examples=False):
    examples = []
    for item in aligned_records:
        relation_record = item["clause_record"]
        graph_example = item["graph_example"]
        graph_inputs = graph_example["inference_inputs"]
        node_count = len(graph_inputs.get("schema_nodes", []))
        for relation_type in relation_types:
            labels = relation_record.get("relation_labels", {}).get(relation_type, [])
            if not labels and not include_empty_relation_examples:
                continue
            inputs = make_relation_inference_inputs(graph_inputs, relation_type)
            names = relation_record.get("relation_label_names", {}).get(relation_type, [])
            examples.append(
                {
                    "example_id": f"{graph_example.get('example_id')}::{relation_type}",
                    "base_example_id": graph_example.get("example_id"),
                    "record_index": item["record_index"],
                    # Reuse clause-grounder internals: relation_type occupies the clause slot.
                    "clause": relation_type,
                    "relation_type": relation_type,
                    "inference_inputs": inputs,
                    "training_targets": {
                        "grounding_label_ids": labels,
                        "grounding_label_names": names,
                        "grounding_label_vector": label_vector(node_count, labels),
                    },
                    "metadata": {
                        "db_id": relation_record.get("db_id"),
                        "question_id": relation_record.get("question_id"),
                        "difficulty": relation_record.get("difficulty"),
                        "sql": relation_record.get("sql"),
                    },
                }
            )
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-relation-file", default="experiments/stage5j_relation_labels_v1/train_relation_labels.jsonl")
    parser.add_argument("--dev-relation-file", default="experiments/stage5j_relation_labels_v1/dev_relation_labels.jsonl")
    parser.add_argument("--train-graph-file", default="experiments/stage5i_dsg_data_semantic_v1/train_examples.jsonl")
    parser.add_argument("--dev-graph-file", default="experiments/stage5i_dsg_data_semantic_v1/dev_examples.jsonl")
    parser.add_argument("--output-dir", default="experiments/stage5j_relation_grounder_smoke")
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--dev-limit", type=int, default=100)
    parser.add_argument("--relation-types", type=parse_clause_list, default=",".join(DEFAULT_RELATIONS))
    parser.add_argument("--include-empty-relation-examples", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--encoder-type", choices=["rgcn", "rgta"], default="rgcn")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-top-k", type=int, default=30)
    parser.add_argument("--use-lexical-features", action="store_true")
    parser.add_argument("--selection-metric", default="assembled_schema_recall@30")
    parser.add_argument("--selection-mode", choices=["max", "min"], default="max")
    parser.add_argument("--patience", type=int, default=2)
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

    # Reused evaluator expects args.clauses.
    args.clauses = args.relation_types

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    data_report = {
        "train_base_count": len(train_aligned),
        "dev_base_count": len(dev_aligned),
        "train_relation_example_count": len(train_examples),
        "dev_relation_example_count": len(dev_examples),
        "relation_types": args.relation_types,
        "include_empty_relation_examples": args.include_empty_relation_examples,
        "assembly_budgets": args.assembly_budgets,
        "generalization_boundary": (
            "Relation labels are training targets only. Inference features use question, evidence, "
            "semantic schema graph, and a test-time relation token."
        ),
    }
    write_json(output_dir / "data_report.json", data_report)
    if args.dry_run_data_check:
        print(json.dumps(data_report, ensure_ascii=False, indent=2))
        return

    helpers = import_torch_and_model()
    torch = helpers["torch"]
    device = torch.device(args.device)
    graph_relations = list(helpers["DEFAULT_RELATIONS"])
    model = helpers["DSGGrounder"](
        hash_dim=args.hash_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        relations=graph_relations,
        encoder_type=args.encoder_type,
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
    best_metrics = None
    epochs_without_improvement = 0
    stopped_early = False
    log_path = output_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_examples, helpers, args, optimizer, criterion, device, graph_relations)
            dev_relation_metrics, dev_relation_predictions = evaluate_clause_examples(
                model, dev_examples, helpers, args, device, graph_relations
            )
            assembled_rows = assemble_predictions(
                dev_relation_predictions,
                dev_aligned,
                output_top_k=args.output_top_k,
                budgets=args.assembly_budgets,
            )
            assembled_metrics = evaluate_assembled(assembled_rows)
            dev_metrics = {**dev_relation_metrics, **assembled_metrics}
            selected_value = get_metric(dev_metrics, args.selection_metric)
            improved = is_better_metric(selected_value, best_value, args.selection_mode, args.min_delta)
            if improved:
                best_epoch = epoch
                best_value = selected_value
                best_metrics = dev_metrics.copy()
                best_metrics.update(
                    {
                        "best_epoch": best_epoch,
                        "selection_metric": args.selection_metric,
                        "selection_value": best_value,
                        "selection_mode": args.selection_mode,
                    }
                )
                best_state = clone_state_dict_to_cpu(model)
                epochs_without_improvement = 0
                write_json(output_dir / "best_metrics.json", best_metrics)
                if not args.no_save_model:
                    torch.save(best_state, output_dir / "relation_grounder_model.pt")
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

    last_relation_metrics, last_relation_predictions = evaluate_clause_examples(
        model, dev_examples, helpers, args, device, graph_relations, output_dir=output_dir, split="dev_last"
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
        torch.save(model.state_dict(), output_dir / "relation_grounder_last_model.pt")

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
        final_relation_metrics, final_relation_predictions = evaluate_clause_examples(
            model, dev_examples, helpers, args, device, graph_relations, output_dir=output_dir, split="dev"
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
