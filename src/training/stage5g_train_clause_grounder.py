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
    mean,
    precision_at_k,
    ranked_prediction_rows,
    read_jsonl,
    recall_at_k,
    reciprocal_rank,
    write_json,
    write_jsonl,
)


DEFAULT_CLAUSES = ["select", "join", "where", "order_by"]


def load_aligned_records(clause_file: Path, graph_file: Path, limit=None):
    clause_records = read_jsonl(clause_file, limit=limit)
    graph_examples = read_jsonl(graph_file, limit=limit)
    if len(clause_records) != len(graph_examples):
        raise ValueError(
            f"Clause/graph length mismatch: {clause_file} has {len(clause_records)}, "
            f"{graph_file} has {len(graph_examples)}"
        )
    aligned = []
    for index, (clause_record, graph_example) in enumerate(zip(clause_records, graph_examples)):
        graph_inputs = graph_example.get("inference_inputs", {})
        if clause_record.get("db_id") != graph_inputs.get("db_id"):
            raise ValueError(
                f"db_id mismatch at index {index}: clause={clause_record.get('db_id')} "
                f"graph={graph_inputs.get('db_id')}"
            )
        aligned.append({"clause_record": clause_record, "graph_example": graph_example, "record_index": index})
    return aligned


def make_clause_inference_inputs(graph_inputs, clause):
    inputs = deepcopy(graph_inputs)
    question = inputs.get("question") or ""
    evidence = inputs.get("evidence") or ""
    clause_text = clause.upper().replace("_", " ")
    inputs["question"] = f"[{clause_text} schema grounding] {question}"
    inputs["evidence"] = f"{evidence}\nTarget SQL clause: {clause_text}".strip()
    inputs["target_clause"] = clause
    return inputs


def label_vector(node_count, label_ids):
    vector = [0] * node_count
    for item_id in label_ids:
        if 0 <= int(item_id) < node_count:
            vector[int(item_id)] = 1
    return vector


def make_clause_examples(aligned_records, clauses, include_empty_clause_examples=False):
    examples = []
    for item in aligned_records:
        clause_record = item["clause_record"]
        graph_example = item["graph_example"]
        graph_inputs = graph_example["inference_inputs"]
        node_count = len(graph_inputs.get("schema_nodes", []))
        for clause in clauses:
            labels = clause_record.get("clause_labels", {}).get(clause, [])
            if not labels and not include_empty_clause_examples:
                continue
            inputs = make_clause_inference_inputs(graph_inputs, clause)
            names = clause_record.get("clause_label_names", {}).get(clause, [])
            examples.append(
                {
                    "example_id": f"{graph_example.get('example_id')}::{clause}",
                    "base_example_id": graph_example.get("example_id"),
                    "record_index": item["record_index"],
                    "clause": clause,
                    "inference_inputs": inputs,
                    "training_targets": {
                        "grounding_label_ids": labels,
                        "grounding_label_names": names,
                        "grounding_label_vector": label_vector(node_count, labels),
                    },
                    "metadata": {
                        "db_id": clause_record.get("db_id"),
                        "question_id": clause_record.get("question_id"),
                        "difficulty": clause_record.get("difficulty"),
                        "sql": clause_record.get("sql"),
                    },
                }
            )
    return examples


def example_to_tensors(example, helpers, hash_dim, relations, device, use_lexical_features):
    torch = helpers["torch"]
    inputs = example["inference_inputs"]
    targets = example.get("training_targets", {})
    node_features = helpers["make_node_features"](inputs, hash_dim).to(device)
    query_features = helpers["make_query_features"](inputs, hash_dim).to(device)
    edge_tensors = helpers["make_edge_tensors"](inputs, relations, device)
    lex = helpers["lexical_features"](inputs).to(device) if use_lexical_features else None
    labels = torch.tensor(
        targets.get("grounding_label_vector", [0] * node_features.shape[0]),
        dtype=torch.float32,
        device=device,
    )
    return {
        "node_features": node_features,
        "query_features": query_features,
        "edge_tensors": edge_tensors,
        "lexical_features": lex,
        "labels": labels,
    }


def split_gold_ids(example):
    nodes = {node["id"]: node for node in example["inference_inputs"]["schema_nodes"]}
    gold_ids = example["training_targets"].get("grounding_label_ids", [])
    table_ids = [item_id for item_id in gold_ids if nodes.get(item_id, {}).get("type") == "table"]
    column_ids = [item_id for item_id in gold_ids if nodes.get(item_id, {}).get("type") == "column"]
    return gold_ids, table_ids, column_ids


def train_one_epoch(model, examples, helpers, args, optimizer, criterion, device, relations):
    model.train()
    total_loss = 0.0
    for example in examples:
        tensors = example_to_tensors(
            example,
            helpers,
            args.hash_dim,
            relations,
            device,
            use_lexical_features=args.use_lexical_features,
        )
        output = model(
            tensors["query_features"],
            tensors["node_features"],
            tensors["edge_tensors"],
            tensors["lexical_features"],
        )
        loss = criterion(output["logits"], tensors["labels"])
        optimizer.zero_grad()
        loss.backward()
        helpers["torch"].nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(examples), 1)


def evaluate_clause_examples(model, examples, helpers, args, device, relations, output_dir=None, split="dev"):
    torch = helpers["torch"]
    model.eval()
    criterion = helpers["nn"].BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, dtype=torch.float32, device=device)
    )
    losses = []
    by_clause = {
        clause: {
            "recall@5": [],
            "recall@10": [],
            "recall@20": [],
            "precision@5": [],
            "mrr": [],
            "column_recall@10": [],
            "column_recall@20": [],
        }
        for clause in args.clauses
    }
    predictions = []

    with torch.no_grad():
        for example in examples:
            tensors = example_to_tensors(
                example,
                helpers,
                args.hash_dim,
                relations,
                device,
                use_lexical_features=args.use_lexical_features,
            )
            output = model(
                tensors["query_features"],
                tensors["node_features"],
                tensors["edge_tensors"],
                tensors["lexical_features"],
            )
            loss = criterion(output["logits"], tensors["labels"])
            losses.append(float(loss.detach().cpu()))
            scores = output["logits"].detach().cpu().tolist()
            top_rows, ranked = ranked_prediction_rows(example, scores, args.output_top_k)
            gold_ids, _, gold_column_ids = split_gold_ids(example)
            ranked_column_ids = [
                item_id
                for item_id in ranked
                if example["inference_inputs"]["schema_nodes"][item_id]["type"] == "column"
            ]
            clause = example["clause"]
            by_clause[clause]["recall@5"].append(recall_at_k(gold_ids, ranked, 5))
            by_clause[clause]["recall@10"].append(recall_at_k(gold_ids, ranked, 10))
            by_clause[clause]["recall@20"].append(recall_at_k(gold_ids, ranked, 20))
            by_clause[clause]["precision@5"].append(precision_at_k(gold_ids, ranked, 5))
            by_clause[clause]["mrr"].append(reciprocal_rank(gold_ids, ranked))
            by_clause[clause]["column_recall@10"].append(recall_at_k(gold_column_ids, ranked_column_ids, 10))
            by_clause[clause]["column_recall@20"].append(recall_at_k(gold_column_ids, ranked_column_ids, 20))
            predictions.append(
                {
                    "example_id": example["example_id"],
                    "base_example_id": example["base_example_id"],
                    "record_index": example["record_index"],
                    "db_id": example["metadata"].get("db_id"),
                    "question_id": example["metadata"].get("question_id"),
                    "clause": clause,
                    "gold_label_ids": gold_ids,
                    "gold_label_names": example["training_targets"].get("grounding_label_names", []),
                    f"top_{args.output_top_k}": top_rows,
                }
            )

    metrics = {
        "split": split,
        "clause_example_count": len(examples),
        "loss": mean(losses),
    }
    for clause, metric_lists in by_clause.items():
        prefix = f"{clause}_"
        metrics[prefix + "example_count"] = len([ex for ex in examples if ex["clause"] == clause])
        for name, values in metric_lists.items():
            metrics[prefix + name] = mean(values)

    if output_dir is not None:
        write_json(output_dir / f"{split}_clause_metrics.json", metrics)
        write_jsonl(output_dir / f"{split}_clause_predictions.jsonl", predictions)
    return metrics, predictions


def assemble_predictions(clause_predictions, aligned_records, output_top_k=30, budgets=None):
    budgets = budgets or {"select": 8, "join": 10, "where": 8, "order_by": 4}
    grouped = {}
    for row in clause_predictions:
        grouped.setdefault(row["record_index"], {})[row["clause"]] = row

    assembled = []
    for item in aligned_records:
        record_index = item["record_index"]
        clause_record = item["clause_record"]
        graph_example = item["graph_example"]
        selected = []
        seen = set()
        clause_rows = grouped.get(record_index, {})
        for clause, budget in budgets.items():
            row = clause_rows.get(clause)
            if not row:
                continue
            for node in row.get(f"top_{output_top_k}", [])[:budget]:
                item_id = int(node["id"])
                if item_id in seen:
                    continue
                seen.add(item_id)
                selected.append({**node, "source_clause": clause})
        if len(selected) < output_top_k:
            leftovers = []
            for clause, row in clause_rows.items():
                for node in row.get(f"top_{output_top_k}", []):
                    item_id = int(node["id"])
                    if item_id in seen:
                        continue
                    leftovers.append({**node, "source_clause": clause})
            leftovers.sort(key=lambda item: item["score"], reverse=True)
            for node in leftovers:
                if len(selected) >= output_top_k:
                    break
                item_id = int(node["id"])
                if item_id in seen:
                    continue
                seen.add(item_id)
                selected.append(node)

        gold_ids = set(clause_record.get("whole_sql_labels", []))
        selected_ids = [int(node["id"]) for node in selected]
        assembled.append(
            {
                "example_id": graph_example.get("example_id"),
                "record_index": record_index,
                "db_id": clause_record.get("db_id"),
                "question_id": clause_record.get("question_id"),
                "question": clause_record.get("question"),
                "evidence": clause_record.get("evidence"),
                "gold_label_ids": sorted(gold_ids),
                "gold_label_names": clause_record.get("whole_sql_label_names", []),
                f"top_{output_top_k}": selected[:output_top_k],
                "assembled_recall@30": recall_at_k(gold_ids, selected_ids, 30),
                "assembled_precision@30": precision_at_k(gold_ids, selected_ids, min(30, len(selected_ids))),
            }
        )
    return assembled


def evaluate_assembled(assembled_rows):
    recalls = [row["assembled_recall@30"] for row in assembled_rows]
    precisions = [row["assembled_precision@30"] for row in assembled_rows]
    return {
        "assembled_sample_count": len(assembled_rows),
        "assembled_schema_recall@30": mean(recalls),
        "assembled_schema_precision@30": mean(precisions),
        "assembled_missing_samples@30": sum(1 for row in assembled_rows if (row["assembled_recall@30"] or 0) < 1.0),
    }


def parse_clause_list(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_budget_text(text):
    budgets = {}
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid budget entry {part!r}; expected clause:k")
        clause, value = part.split(":", 1)
        budgets[clause.strip()] = int(value)
    return budgets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-clause-file", default="experiments/stage5g_clause_labels/train_clause_labels.jsonl")
    parser.add_argument("--dev-clause-file", default="experiments/stage5g_clause_labels/dev_clause_labels.jsonl")
    parser.add_argument("--train-graph-file", default="experiments/stage5_dsg_data_v2/train_examples.jsonl")
    parser.add_argument("--dev-graph-file", default="experiments/stage5_dsg_data_v2/dev_examples.jsonl")
    parser.add_argument("--output-dir", default="experiments/stage5g_clause_grounder_smoke")
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--dev-limit", type=int, default=100)
    parser.add_argument("--clauses", type=parse_clause_list, default="select,join,where,order_by")
    parser.add_argument("--include-empty-clause-examples", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--encoder-type", choices=["rgcn", "rgta"], default="rgta")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-top-k", type=int, default=30)
    parser.add_argument("--use-lexical-features", action="store_true")
    parser.add_argument("--selection-metric", default="assembled_schema_recall@30")
    parser.add_argument("--selection-mode", choices=["max", "min"], default="max")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--assembly-budgets", type=parse_budget_text, default="select:8,join:10,where:8,order_by:4")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--dry-run-data-check", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_aligned = load_aligned_records(
        Path(args.train_clause_file), Path(args.train_graph_file), limit=args.train_limit
    )
    dev_aligned = load_aligned_records(Path(args.dev_clause_file), Path(args.dev_graph_file), limit=args.dev_limit)
    train_examples = make_clause_examples(
        train_aligned, args.clauses, include_empty_clause_examples=args.include_empty_clause_examples
    )
    dev_examples = make_clause_examples(
        dev_aligned, args.clauses, include_empty_clause_examples=args.include_empty_clause_examples
    )

    data_report = {
        "train_base_count": len(train_aligned),
        "dev_base_count": len(dev_aligned),
        "train_clause_example_count": len(train_examples),
        "dev_clause_example_count": len(dev_examples),
        "clauses": args.clauses,
        "include_empty_clause_examples": args.include_empty_clause_examples,
        "generalization_boundary": (
            "Clause labels are training targets only. Inference features use question, evidence, "
            "schema graph, and a test-time clause token."
        ),
    }
    write_json(output_dir / "data_report.json", data_report)
    if args.dry_run_data_check:
        print(json.dumps(data_report, ensure_ascii=False, indent=2))
        return

    helpers = import_torch_and_model()
    torch = helpers["torch"]
    device = torch.device(args.device)
    relations = list(helpers["DEFAULT_RELATIONS"])
    model = helpers["DSGGrounder"](
        hash_dim=args.hash_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        relations=relations,
        encoder_type=args.encoder_type,
        lexical_dim=6 if args.use_lexical_features else 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = helpers["nn"].BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, dtype=torch.float32, device=device)
    )

    config = vars(args).copy()
    config["torch_version"] = torch.__version__
    config["relations"] = relations
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
            train_loss = train_one_epoch(model, train_examples, helpers, args, optimizer, criterion, device, relations)
            dev_clause_metrics, dev_clause_predictions = evaluate_clause_examples(
                model, dev_examples, helpers, args, device, relations
            )
            assembled_rows = assemble_predictions(
                dev_clause_predictions,
                dev_aligned,
                output_top_k=args.output_top_k,
                budgets=args.assembly_budgets,
            )
            assembled_metrics = evaluate_assembled(assembled_rows)
            dev_metrics = {**dev_clause_metrics, **assembled_metrics}
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
                    torch.save(best_state, output_dir / "clause_grounder_model.pt")
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

    last_clause_metrics, last_clause_predictions = evaluate_clause_examples(
        model, dev_examples, helpers, args, device, relations, output_dir=output_dir, split="dev_last"
    )
    last_assembled = assemble_predictions(
        last_clause_predictions,
        dev_aligned,
        output_top_k=args.output_top_k,
        budgets=args.assembly_budgets,
    )
    last_assembled_metrics = evaluate_assembled(last_assembled)
    write_jsonl(output_dir / "dev_last_assembled_predictions.jsonl", last_assembled)
    if not args.no_save_model:
        torch.save(model.state_dict(), output_dir / "clause_grounder_last_model.pt")

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
        final_clause_metrics, final_clause_predictions = evaluate_clause_examples(
            model, dev_examples, helpers, args, device, relations, output_dir=output_dir, split="dev"
        )
        final_assembled = assemble_predictions(
            final_clause_predictions,
            dev_aligned,
            output_top_k=args.output_top_k,
            budgets=args.assembly_budgets,
        )
        final_assembled_metrics = evaluate_assembled(final_assembled)
        write_jsonl(output_dir / "dev_assembled_predictions.jsonl", final_assembled)
    else:
        final_clause_metrics = last_clause_metrics
        final_assembled_metrics = last_assembled_metrics

    final_metrics = {**final_clause_metrics, **final_assembled_metrics}
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
        "dev_last_metrics": {**last_clause_metrics, **last_assembled_metrics},
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
