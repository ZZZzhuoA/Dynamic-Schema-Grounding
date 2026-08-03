import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path, limit=None):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def import_torch_and_model():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Stage 5-B1 training.") from exc
    from src.modeling.dsg_grounder import (
        DEFAULT_RELATIONS,
        DSGGrounder,
        lexical_features,
        make_edge_tensors,
        make_node_features,
        make_query_features,
    )

    return {
        "torch": torch,
        "nn": nn,
        "DEFAULT_RELATIONS": DEFAULT_RELATIONS,
        "DSGGrounder": DSGGrounder,
        "lexical_features": lexical_features,
        "make_edge_tensors": make_edge_tensors,
        "make_node_features": make_node_features,
        "make_query_features": make_query_features,
    }


def example_to_tensors(example, helpers, hash_dim, relations, device):
    torch = helpers["torch"]
    inputs = example["inference_inputs"]
    targets = example.get("training_targets", {})
    node_features = helpers["make_node_features"](inputs, hash_dim).to(device)
    query_features = helpers["make_query_features"](inputs, hash_dim).to(device)
    edge_tensors = helpers["make_edge_tensors"](inputs, relations, device)
    lex = helpers["lexical_features"](inputs).to(device)
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


def recall_at_k(gold_ids, ranked_ids, k):
    if not gold_ids:
        return None
    return len(set(gold_ids) & set(ranked_ids[:k])) / len(set(gold_ids))


def precision_at_k(gold_ids, ranked_ids, k):
    if k <= 0:
        return None
    return len(set(gold_ids) & set(ranked_ids[:k])) / k


def reciprocal_rank(gold_ids, ranked_ids):
    gold = set(gold_ids)
    if not gold:
        return None
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in gold:
            return 1.0 / rank
    return 0.0


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def get_metric(metrics, metric_name):
    if metric_name not in metrics:
        available = sorted(key for key in metrics.keys() if key not in {"split"})
        raise ValueError(f"Metric {metric_name!r} is not available. Available metrics: {available}")
    return float(metrics[metric_name])


def is_better_metric(value, best_value, mode, min_delta):
    if best_value is None:
        return True
    if mode == "max":
        return value > best_value + min_delta
    if mode == "min":
        return value < best_value - min_delta
    raise ValueError(f"Unsupported selection mode: {mode}")


def clone_state_dict_to_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def normalize_identifier(text):
    return "".join(ch.lower() if ch.isalnum() else " " for ch in str(text or "")).split()


def token_overlap(left_tokens, right_tokens):
    left = set(left_tokens)
    right = set(right_tokens)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def gold_context(example):
    inputs = example["inference_inputs"]
    nodes = inputs["schema_nodes"]
    gold_ids = set(example.get("training_targets", {}).get("grounding_label_ids", []))
    gold_tables = set()
    gold_column_tokens = []
    for item_id in gold_ids:
        if item_id < 0 or item_id >= len(nodes):
            continue
        node = nodes[item_id]
        if node.get("type") == "table":
            gold_tables.add(node.get("name"))
        else:
            gold_tables.add(node.get("table"))
            gold_column_tokens.append(normalize_identifier(node.get("column") or node.get("name")))
    fk_neighbors = set()
    for edge in inputs.get("schema_edges", []):
        if edge.get("type") not in {"foreign_key_forward", "foreign_key_backward"}:
            continue
        src = edge.get("src")
        dst = edge.get("dst")
        if src in gold_ids and dst not in gold_ids:
            fk_neighbors.add(dst)
        if dst in gold_ids and src not in gold_ids:
            fk_neighbors.add(src)
    return {
        "gold_ids": gold_ids,
        "gold_tables": gold_tables,
        "gold_column_tokens": gold_column_tokens,
        "fk_neighbors": fk_neighbors,
    }


def hard_negative_weights(example, helpers, args, device):
    torch = helpers["torch"]
    inputs = example["inference_inputs"]
    labels = example.get("training_targets", {}).get("grounding_label_vector", [])
    context = gold_context(example)
    weights = torch.ones((len(inputs["schema_nodes"]),), dtype=torch.float32, device=device)

    for node in inputs["schema_nodes"]:
        item_id = node["id"]
        if item_id >= len(labels) or labels[item_id] > 0:
            continue

        node_weight = 1.0
        if item_id in context["fk_neighbors"]:
            node_weight = max(node_weight, args.fk_hard_negative_weight)

        if node.get("type") == "column":
            if node.get("table") in context["gold_tables"]:
                node_weight = max(node_weight, args.same_table_hard_negative_weight)
            node_tokens = normalize_identifier(node.get("column") or node.get("name"))
            max_overlap = max(
                [token_overlap(node_tokens, gold_tokens) for gold_tokens in context["gold_column_tokens"]],
                default=0.0,
            )
            if max_overlap >= args.lexical_hard_negative_overlap:
                node_weight = max(node_weight, args.lexical_hard_negative_weight)

        weights[item_id] = node_weight

    return weights


def compute_training_loss(logits, labels, example, helpers, args, criterion, device):
    if not args.use_hard_negatives:
        return criterion(logits, labels)
    torch = helpers["torch"]
    per_node_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.tensor(args.pos_weight, dtype=torch.float32, device=device),
        reduction="none",
    )
    weights = hard_negative_weights(example, helpers, args, device)
    return (per_node_loss * weights).sum() / weights.sum().clamp_min(1.0)


def split_gold_ids(example):
    nodes = {node["id"]: node for node in example["inference_inputs"]["schema_nodes"]}
    gold_ids = example["training_targets"].get("grounding_label_ids", [])
    table_ids = [item_id for item_id in gold_ids if nodes.get(item_id, {}).get("type") == "table"]
    column_ids = [item_id for item_id in gold_ids if nodes.get(item_id, {}).get("type") == "column"]
    return gold_ids, table_ids, column_ids


def ranked_prediction_rows(example, scores, top_k):
    nodes = {node["id"]: node for node in example["inference_inputs"]["schema_nodes"]}
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    rows = []
    for item_id in ranked[:top_k]:
        node = nodes[item_id]
        rows.append(
            {
                "id": int(item_id),
                "type": node["type"],
                "name": node["name"],
                "score": float(scores[item_id]),
            }
        )
    return rows, ranked


def train_one_epoch(model, examples, helpers, args, optimizer, criterion, device, relations):
    model.train()
    total_loss = 0.0
    for example in examples:
        tensors = example_to_tensors(example, helpers, args.hash_dim, relations, device)
        output = model(
            tensors["query_features"],
            tensors["node_features"],
            tensors["edge_tensors"],
            tensors["lexical_features"] if args.use_lexical_features else None,
        )
        loss = compute_training_loss(
            output["logits"], tensors["labels"], example, helpers, args, criterion, device
        )
        optimizer.zero_grad()
        loss.backward()
        helpers["torch"].nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(examples), 1)


def evaluate(model, examples, helpers, args, device, relations, output_dir=None, split="dev"):
    torch = helpers["torch"]
    model.eval()
    losses = []
    criterion = helpers["nn"].BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, dtype=torch.float32, device=device)
    )
    schema_recall_10 = []
    schema_recall_20 = []
    schema_recall_30 = []
    schema_precision_10 = []
    schema_mrr = []
    table_recall_5 = []
    column_recall_20 = []
    column_recall_30 = []
    predictions = []

    with torch.no_grad():
        for example in examples:
            tensors = example_to_tensors(example, helpers, args.hash_dim, relations, device)
            output = model(
                tensors["query_features"],
                tensors["node_features"],
                tensors["edge_tensors"],
                tensors["lexical_features"] if args.use_lexical_features else None,
            )
            loss = criterion(output["logits"], tensors["labels"])
            losses.append(float(loss.detach().cpu()))
            scores = output["logits"].detach().cpu().tolist()
            top_rows, ranked = ranked_prediction_rows(example, scores, args.output_top_k)
            gold_ids, gold_table_ids, gold_column_ids = split_gold_ids(example)
            ranked_table_ids = [
                item_id
                for item_id in ranked
                if example["inference_inputs"]["schema_nodes"][item_id]["type"] == "table"
            ]
            ranked_column_ids = [
                item_id
                for item_id in ranked
                if example["inference_inputs"]["schema_nodes"][item_id]["type"] == "column"
            ]
            schema_recall_10.append(recall_at_k(gold_ids, ranked, 10))
            schema_recall_20.append(recall_at_k(gold_ids, ranked, 20))
            schema_recall_30.append(recall_at_k(gold_ids, ranked, 30))
            schema_precision_10.append(precision_at_k(gold_ids, ranked, 10))
            schema_mrr.append(reciprocal_rank(gold_ids, ranked))
            table_recall_5.append(recall_at_k(gold_table_ids, ranked_table_ids, 5))
            column_recall_20.append(recall_at_k(gold_column_ids, ranked_column_ids, 20))
            column_recall_30.append(recall_at_k(gold_column_ids, ranked_column_ids, 30))
            predictions.append(
                {
                    "example_id": example["example_id"],
                    "db_id": example["inference_inputs"]["db_id"],
                    "question": example["inference_inputs"].get("question"),
                    "evidence": example["inference_inputs"].get("evidence"),
                    "gold_label_ids": gold_ids,
                    "gold_label_names": example["training_targets"].get("grounding_label_names", []),
                    f"top_{args.output_top_k}": top_rows,
                }
            )

    metrics = {
        "split": split,
        "sample_count": len(examples),
        "loss": mean(losses),
        "schema_recall@10": mean(schema_recall_10),
        "schema_recall@20": mean(schema_recall_20),
        "schema_recall@30": mean(schema_recall_30),
        "schema_precision@10": mean(schema_precision_10),
        "schema_mrr": mean(schema_mrr),
        "table_recall@5": mean(table_recall_5),
        "column_recall@20": mean(column_recall_20),
        "column_recall@30": mean(column_recall_30),
    }
    if output_dir is not None:
        write_json(output_dir / f"{split}_metrics.json", metrics)
        write_jsonl(output_dir / f"{split}_predictions.jsonl", predictions)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="experiments/stage5_dsg_data/train_examples.jsonl")
    parser.add_argument("--dev-file", default="experiments/stage5_dsg_data/dev_examples.jsonl")
    parser.add_argument("--output-dir", default="experiments/stage5_dsg_grounder_smoke")
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--dev-limit", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
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
    parser.add_argument(
        "--use-hard-negatives",
        action="store_true",
        help="Up-weight difficult negative schema nodes during training.",
    )
    parser.add_argument(
        "--same-table-hard-negative-weight",
        type=float,
        default=2.0,
        help="Negative columns from gold tables receive at least this loss weight.",
    )
    parser.add_argument(
        "--fk-hard-negative-weight",
        type=float,
        default=1.5,
        help="Negative FK-neighbor nodes of gold nodes receive at least this loss weight.",
    )
    parser.add_argument(
        "--lexical-hard-negative-weight",
        type=float,
        default=2.0,
        help="Lexically similar negative columns receive at least this loss weight.",
    )
    parser.add_argument(
        "--lexical-hard-negative-overlap",
        type=float,
        default=0.5,
        help="Token-overlap threshold for lexical hard negatives.",
    )
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument(
        "--selection-metric",
        default="schema_recall@30",
        help="Dev metric used to select the best checkpoint.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["max", "min"],
        default="max",
        help="Whether a larger or smaller selection metric is better.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop after this many epochs without dev selection-metric improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum dev selection-metric change required to count as an improvement.",
    )
    parser.add_argument(
        "--dry-run-data-check",
        action="store_true",
        help="Validate Stage 5 data structure without importing PyTorch or training.",
    )
    args = parser.parse_args()

    if args.dry_run_data_check:
        train_examples = read_jsonl(Path(args.train_file), limit=args.train_limit)
        dev_examples = read_jsonl(Path(args.dev_file), limit=args.dev_limit)
        forbidden = {
            "sql",
            "gold_sql",
            "grounding_label_ids",
            "grounding_label_names",
            "grounding_label_vector",
            "whole_sql_labels",
            "label_names",
            "label_sources",
        }
        report = {"train_count": len(train_examples), "dev_count": len(dev_examples), "violations": []}
        for split, examples in [("train", train_examples), ("dev", dev_examples)]:
            for index, example in enumerate(examples):
                inputs = example.get("inference_inputs", {})
                targets = example.get("training_targets", {})
                leaked = sorted(forbidden & set(inputs.keys()))
                if leaked:
                    report["violations"].append({"split": split, "index": index, "leaked_keys": leaked})
                if "grounding_label_vector" not in targets:
                    report["violations"].append(
                        {"split": split, "index": index, "missing": "training_targets.grounding_label_vector"}
                    )
                nodes = inputs.get("schema_nodes", [])
                edges = inputs.get("schema_edges", [])
                node_ids = {node.get("id") for node in nodes}
                bad_edges = [
                    edge for edge in edges if edge.get("src") not in node_ids or edge.get("dst") not in node_ids
                ]
                if bad_edges:
                    report["violations"].append(
                        {"split": split, "index": index, "bad_edge_count": len(bad_edges)}
                    )
        report["ok"] = len(report["violations"]) == 0
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    helpers = import_torch_and_model()
    torch = helpers["torch"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    relations = list(helpers["DEFAULT_RELATIONS"])

    train_examples = read_jsonl(Path(args.train_file), limit=args.train_limit)
    dev_examples = read_jsonl(Path(args.dev_file), limit=args.dev_limit)

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
    config["generalization_boundary"] = (
        "Model features are built from inference_inputs only. "
        "training_targets are consumed only for loss and evaluation metrics."
    )
    write_json(output_dir / "train_config.json", config)

    log_path = output_dir / "train_log.jsonl"
    best_epoch = None
    best_value = None
    best_state = None
    best_metrics = None
    epochs_without_improvement = 0
    stopped_early = False

    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_examples, helpers, args, optimizer, criterion, device, relations
            )
            dev_metrics = evaluate(model, dev_examples, helpers, args, device, relations)
            selected_value = get_metric(dev_metrics, args.selection_metric)
            improved = is_better_metric(selected_value, best_value, args.selection_mode, args.min_delta)
            if improved:
                best_epoch = epoch
                best_value = selected_value
                best_metrics = dev_metrics.copy()
                best_metrics["best_epoch"] = best_epoch
                best_metrics["selection_metric"] = args.selection_metric
                best_metrics["selection_value"] = best_value
                best_metrics["selection_mode"] = args.selection_mode
                best_state = clone_state_dict_to_cpu(model)
                epochs_without_improvement = 0
                write_json(output_dir / "best_metrics.json", best_metrics)
                if not args.no_save_model:
                    torch.save(best_state, output_dir / "dsg_grounder_model.pt")
            else:
                epochs_without_improvement += 1

            row = {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_metrics["loss"]}
            row.update({f"dev_{key}": value for key, value in dev_metrics.items() if key != "split"})
            row.update(
                {
                    "selection_metric": args.selection_metric,
                    "selection_value": selected_value,
                    "best_epoch": best_epoch,
                    "best_selection_value": best_value,
                    "is_best": improved,
                    "epochs_without_improvement": epochs_without_improvement,
                }
            )
            log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False))

            if args.patience > 0 and epochs_without_improvement >= args.patience:
                stopped_early = True
                break

    last_metrics = evaluate(model, dev_examples, helpers, args, device, relations, output_dir, split="dev_last")
    if not args.no_save_model:
        torch.save(model.state_dict(), output_dir / "dsg_grounder_last_model.pt")

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
        final_metrics = evaluate(model, dev_examples, helpers, args, device, relations, output_dir, split="dev")
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
