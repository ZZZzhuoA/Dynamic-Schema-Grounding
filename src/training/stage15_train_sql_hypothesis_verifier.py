"""Train the Stage 15A typed plan–schema graph hypothesis verifier."""

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.stage15_evaluate_sql_hypothesis_verifier import ranking_metrics  # noqa: E402
from src.modeling.stage13c_static_runtime import corrupt_destinations, graph_tensors  # noqa: E402
from src.training.stage13b_train_typed_ra_decoder import load_cache  # noqa: E402


def read_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def relation_vocabulary(rows, init_summary=None):
    if init_summary:
        summary = json.loads(Path(init_summary).read_text(encoding="utf-8"))
        relations = summary.get("relations", [])
        if relations:
            return list(relations)
    return sorted(
        {
            edge["type"]
            for row in rows
            for edge in row["inference_inputs"].get("schema_edges", [])
        }
    ) or ["self_loop"]


def checkpoint_state(path, torch):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    return state


def labels_tensor(candidates, torch, device):
    return torch.tensor(
        [float(candidate.get("label", 0)) for candidate in candidates],
        dtype=torch.float32,
        device=device,
    )


def forward_group(model, row, cache, relation_to_id, device, torch):
    dense, query, node_types, edge_index, edge_type, schema_items = graph_tensors(
        row, cache, relation_to_id, device
    )
    return model(
        dense,
        query,
        node_types,
        edge_index,
        edge_type,
        schema_items,
        row["candidates"],
    )


def forward_schema_control(model, row, cache, relation_to_id, device, torch):
    dense, query, node_types, edge_index, edge_type, schema_items = graph_tensors(
        row, cache, relation_to_id, device
    )
    structural_ids = [
        relation_to_id[name]
        for name in (
            "table_to_column", "column_to_table",
            "foreign_key_forward", "foreign_key_backward",
        )
        if name in relation_to_id
    ]
    corrupted = corrupt_destinations(
        edge_index, len(schema_items), edge_type, structural_ids
    )
    return model(
        dense, query, node_types, corrupted, edge_type, schema_items, row["candidates"]
    )


def prediction_row(row, output):
    scores = output["scores"].detach().float().cpu().tolist()
    candidates = []
    for candidate, score, detail in zip(row["candidates"], scores, output["candidate_outputs"]):
        candidates.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "label": int(candidate.get("label", 0)),
                "corruption_type": candidate.get("corruption_type"),
                "score": float(score),
                "step_energy": float(detail["step_energy"].detach().float().cpu()),
                "join_energy": float(detail["join_energy"].detach().float().cpu()),
                "pointer_validity": float(detail.get("pointer_validity", 0.0)),
            }
        )
    return {
        "record_index": int(row["record_index"]),
        "db_id": row.get("db_id"),
        "question_id": row.get("question_id"),
        "candidates": candidates,
    }


def run_split(
    model,
    rows,
    cache,
    relation_to_id,
    args,
    torch,
    F,
    device,
    optimizer=None,
):
    from src.modeling.sql_hypothesis_verifier import grouped_ranking_loss

    training = optimizer is not None
    model.train(training)
    losses, listwise_losses, pairwise_losses, predictions = [], [], [], []
    schema_control_gains, schema_control_wins = [], 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for index, row in enumerate(rows):
            output = forward_group(model, row, cache, relation_to_id, device, torch)
            labels = labels_tensor(row["candidates"], torch, device)
            loss, components = grouped_ranking_loss(
                output["scores"], labels, args.margin, args.margin_weight
            )
            if training:
                (loss / args.gradient_accumulation_steps).backward()
                if (index + 1) % args.gradient_accumulation_steps == 0 or index + 1 == len(rows):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            listwise_losses.append(float(components["listwise_loss"].detach().cpu()))
            pairwise_losses.append(float(components["pairwise_loss"].detach().cpu()))
            predictions.append(prediction_row(row, output))
            if not training:
                control = forward_schema_control(
                    model, row, cache, relation_to_id, device, torch
                )
                positive_index = next(
                    candidate_index
                    for candidate_index, candidate in enumerate(row["candidates"])
                    if int(candidate.get("label", 0)) == 1
                )
                gain = float(
                    (output["scores"][positive_index] - control["scores"][positive_index])
                    .detach().float().cpu()
                )
                schema_control_gains.append(gain)
                schema_control_wins += int(gain > 0.0)
                predictions[-1]["schema_control"] = {
                    "positive_score": float(output["scores"][positive_index].detach().float().cpu()),
                    "corrupted_schema_positive_score": float(
                        control["scores"][positive_index].detach().float().cpu()
                    ),
                    "positive_gain": gain,
                }
    metrics = ranking_metrics(predictions)
    metrics.update(
        {
            "loss": sum(losses) / len(losses) if losses else 0.0,
            "listwise_loss": sum(listwise_losses) / len(listwise_losses) if listwise_losses else 0.0,
            "pairwise_loss": sum(pairwise_losses) / len(pairwise_losses) if pairwise_losses else 0.0,
            "schema_control_positive_gain": (
                sum(schema_control_gains) / len(schema_control_gains)
                if schema_control_gains else None
            ),
            "schema_control_win_rate": (
                schema_control_wins / len(schema_control_gains)
                if schema_control_gains else None
            ),
        }
    )
    return metrics, predictions


def cpu_state_dict(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="experiments/stage15a_sql_hypothesis_data/train_hypotheses.jsonl")
    parser.add_argument("--dev-file", default="experiments/stage15a_sql_hypothesis_data/dev_hypotheses.jsonl")
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--init-summary")
    parser.add_argument("--output-dir", default="experiments/stage15a_sql_hypothesis_verifier_rgta_seed42")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-schema-layers", type=int, default=2)
    parser.add_argument("--num-plan-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--freeze-schema-encoder", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("Stage 15A training requires numpy and PyTorch") from exc
    from src.modeling.sql_hypothesis_verifier import SQLHypothesisGraphVerifier

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_rows = read_jsonl(args.train_file, args.train_limit)
    dev_rows = read_jsonl(args.dev_file, args.dev_limit)
    if not train_rows or not dev_rows:
        raise ValueError("Stage 15A requires non-empty train and dev candidate groups")
    train_cache = load_cache(args.embedding_cache_dir, "train", np)
    dev_cache = load_cache(args.embedding_cache_dir, "dev", np)
    if train_cache["dense_dim"] != dev_cache["dense_dim"]:
        raise ValueError("Train/dev embedding dimensions differ")
    relations = relation_vocabulary(train_rows + dev_rows, args.init_summary)
    relation_to_id = {name: index for index, name in enumerate(relations)}
    model = SQLHypothesisGraphVerifier(
        dense_dim=train_cache["dense_dim"],
        hidden_dim=args.hidden_dim,
        schema_relation_count=len(relations),
        num_schema_layers=args.num_schema_layers,
        num_plan_layers=args.num_plan_layers,
        dropout=args.dropout,
    ).to(device)
    warm_start = None
    if args.init_checkpoint:
        warm_start = model.load_stage13b_state(checkpoint_state(args.init_checkpoint, torch))
    if args.freeze_schema_encoder:
        prefixes = ("node_input", "query_input", "node_type", "initial_state", "graph_layers")
        for name, parameter in model.named_parameters():
            if name.startswith(prefixes):
                parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        shuffled = list(train_rows)
        random.Random(args.seed + epoch).shuffle(shuffled)
        train_metrics, _ = run_split(
            model, shuffled, train_cache, relation_to_id, args, torch, F, device, optimizer
        )
        dev_metrics, predictions = run_split(
            model, dev_rows, dev_cache, relation_to_id, args, torch, F, device
        )
        row = {"epoch": epoch, "train": train_metrics, "dev": dev_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        selection = float(dev_metrics["mrr"])
        if best is None or selection > best["selection_value"]:
            best = {
                "epoch": epoch,
                "selection_metric": "mrr",
                "selection_value": selection,
                "model_state_dict": cpu_state_dict(model),
                "dev_metrics": dev_metrics,
                "predictions": predictions,
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    torch.save(
        {
            "model_state_dict": best["model_state_dict"],
            "relations": relations,
            "config": vars(args),
            "dense_dim": train_cache["dense_dim"],
        },
        output_dir / "sql_hypothesis_verifier.pt",
    )
    write_jsonl(output_dir / "dev_predictions.jsonl", best["predictions"])
    write_json(output_dir / "best_metrics.json", best["dev_metrics"])
    summary = {
        "best_epoch": best["epoch"],
        "selection_metric": best["selection_metric"],
        "selection_value": best["selection_value"],
        "dev_metrics": best["dev_metrics"],
        "history": history,
        "relations": relations,
        "warm_start": warm_start,
        "config": vars(args),
        "decision_gate": {
            "hits@1>=0.80": best["dev_metrics"]["hits@1"] >= 0.80,
            "pairwise_accuracy>=0.80": best["dev_metrics"]["pairwise_accuracy"] >= 0.80,
            "same_table_column>=0.70": best["dev_metrics"]["by_corruption"].get(
                "same_table_column", {}
            ).get("pairwise_accuracy", 0.0) >= 0.70,
            "join_edge>=0.90": best["dev_metrics"]["by_corruption"].get(
                "join_edge", {}
            ).get("pairwise_accuracy", 0.0) >= 0.90,
            "operator>=0.80": best["dev_metrics"]["by_corruption"].get(
                "operator", {}
            ).get("pairwise_accuracy", 0.0) >= 0.80,
            "value_route>=0.80": best["dev_metrics"]["by_corruption"].get(
                "value_route", {}
            ).get("pairwise_accuracy", 0.0) >= 0.80,
            "schema_control_positive_gain>0": (
                best["dev_metrics"].get("schema_control_positive_gain") is not None
                and best["dev_metrics"]["schema_control_positive_gain"] > 0.0
            ),
        },
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
