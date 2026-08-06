import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5_train_dsg_grounder import (  # noqa: E402
    mean,
    precision_at_k,
    read_jsonl,
    recall_at_k,
    write_json,
    write_jsonl,
)
from src.training.stage5g_train_clause_grounder import parse_budget_text  # noqa: E402
from src.training.stage5j_train_relation_grounder import load_aligned_records  # noqa: E402


DEFAULT_OPERATION_WEIGHTS = {
    "OUTPUT_TARGET": 1.15,
    "ENTITY_NAME": 0.95,
    "METRIC_TARGET": 1.20,
    "PREDICATE_COLUMN": 1.25,
    "VALUE_ANCHOR": 0.85,
    "TEMPORAL_FILTER": 1.10,
    "ORDER_KEY": 1.00,
    "GROUP_KEY": 1.00,
    "JOIN_BRIDGE": 0.65,
    "FORMULA_COMPONENT": 1.20,
}

DEFAULT_OPERATION_BUDGETS = {
    "OUTPUT_TARGET": 5,
    "PREDICATE_COLUMN": 6,
    "METRIC_TARGET": 4,
    "FORMULA_COMPONENT": 4,
    "ENTITY_NAME": 3,
    "JOIN_BRIDGE": 3,
    "VALUE_ANCHOR": 2,
    "ORDER_KEY": 2,
    "GROUP_KEY": 1,
    "TEMPORAL_FILTER": 1,
}


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def parse_float_map(text, defaults):
    if text is None:
        return dict(defaults)
    values = dict(defaults)
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid map entry {part!r}; expected NAME:value")
        key, value = part.split(":", 1)
        values[key.strip()] = float(value)
    return values


def node_table(node):
    if node.get("type") == "table":
        return node.get("name")
    return node.get("table") or str(node.get("name", "")).split(".", 1)[0]


def build_schema_indexes(graph_example):
    inputs = graph_example["inference_inputs"]
    nodes = {int(node["id"]): node for node in inputs.get("schema_nodes", [])}
    table_to_columns = {}
    table_name_to_id = {}
    fk_neighbors = {}
    for node_id, node in nodes.items():
        if node.get("type") == "table":
            table_name_to_id[node.get("name")] = node_id
        elif node.get("type") == "column":
            table_to_columns.setdefault(node_table(node), []).append(node_id)
    for edge in inputs.get("schema_edges", []):
        edge_type = edge.get("type")
        if edge_type not in {"foreign_key_forward", "foreign_key_backward"}:
            continue
        src = int(edge["src"])
        dst = int(edge["dst"])
        fk_neighbors.setdefault(src, set()).add(dst)
        fk_neighbors.setdefault(dst, set()).add(src)
    return {
        "nodes": nodes,
        "table_to_columns": table_to_columns,
        "table_name_to_id": table_name_to_id,
        "fk_neighbors": fk_neighbors,
    }


def calibrated_candidates_for_row(row, top_k, operation_weights, rank_weight, z_weight):
    operation = row.get("relation_type") or row.get("clause")
    candidates = row.get(f"top_{top_k}") or row.get("top_30") or []
    if not candidates:
        return []
    raw_scores = [float(item.get("score", 0.0)) for item in candidates]
    score_mean = sum(raw_scores) / len(raw_scores)
    variance = sum((score - score_mean) ** 2 for score in raw_scores) / max(len(raw_scores), 1)
    score_std = math.sqrt(variance) or 1.0
    op_weight = operation_weights.get(operation, 1.0)
    calibrated = []
    denom = max(len(candidates), 1)
    for rank, item in enumerate(candidates, start=1):
        raw_score = float(item.get("score", 0.0))
        rank_score = (denom - rank + 1) / denom
        z_score = sigmoid((raw_score - score_mean) / score_std)
        calibrated_score = op_weight * (rank_weight * rank_score + z_weight * z_score)
        calibrated.append(
            {
                **item,
                "id": int(item["id"]),
                "raw_score": raw_score,
                "score": calibrated_score,
                "source_operation": operation,
                "source_clause": operation,
                "rank_in_operation": rank,
                "operation_weight": op_weight,
            }
        )
    return calibrated


def add_candidate(selected, seen, candidate, source, max_items):
    item_id = int(candidate["id"])
    if item_id in seen or len(selected) >= max_items:
        return False
    seen.add(item_id)
    selected.append({**candidate, "assembly_source": source})
    return True


def collect_selected_tables(selected, indexes):
    nodes = indexes["nodes"]
    tables = set()
    for item in selected:
        node = nodes.get(int(item["id"]))
        if not node:
            continue
        table = node_table(node)
        if table:
            tables.add(table)
    return tables


def assemble_one_record(
    item,
    operation_rows,
    output_top_k,
    operation_budgets,
    operation_weights,
    rank_weight,
    z_weight,
    table_column_bonus,
    fk_bonus,
    max_tables,
    include_tables,
):
    graph_example = item["graph_example"]
    relation_record = item["clause_record"]
    indexes = build_schema_indexes(graph_example)
    nodes = indexes["nodes"]

    per_operation = {}
    fused = {}
    provenance = {}
    for operation, row in operation_rows.items():
        calibrated = calibrated_candidates_for_row(row, output_top_k, operation_weights, rank_weight, z_weight)
        per_operation[operation] = calibrated
        for cand in calibrated:
            item_id = int(cand["id"])
            if not include_tables and nodes.get(item_id, {}).get("type") == "table":
                continue
            fused[item_id] = fused.get(item_id, 0.0) + float(cand["score"])
            provenance.setdefault(item_id, []).append(
                {
                    "operation": operation,
                    "rank": cand["rank_in_operation"],
                    "score": cand["score"],
                    "raw_score": cand["raw_score"],
                }
            )

    selected = []
    seen = set()

    # Stage 1: reserve minimal operation-specific slots. This prevents easy operations
    # such as ORDER_KEY from consuming all Top-K space and preserves hard WHERE/JOIN evidence.
    for operation, budget in operation_budgets.items():
        for cand in per_operation.get(operation, [])[: max(budget, 0)]:
            if nodes.get(int(cand["id"]), {}).get("type") == "table":
                if not include_tables:
                    continue
                if sum(1 for x in selected if nodes.get(int(x["id"]), {}).get("type") == "table") >= max_tables:
                    continue
            add_candidate(selected, seen, cand, f"reserved:{operation}", output_top_k)

    # Stage 2: if a table/subgraph has high belief, give its columns a calibrated local bonus.
    selected_tables = collect_selected_tables(selected, indexes)
    expansion_candidates = []
    for table in selected_tables:
        for column_id in indexes["table_to_columns"].get(table, []):
            if column_id in seen:
                continue
            if column_id not in fused:
                continue
            base_score = fused[column_id]
            expansion_candidates.append(
                {
                    **nodes[column_id],
                    "id": column_id,
                    "score": base_score + table_column_bonus,
                    "raw_score": base_score,
                    "source_operation": "TABLE_COLUMN_EXPANSION",
                    "source_clause": "TABLE_COLUMN_EXPANSION",
                    "provenance": provenance.get(column_id, []),
                }
            )
    expansion_candidates.sort(key=lambda cand: cand["score"], reverse=True)
    for cand in expansion_candidates:
        if len(selected) >= output_top_k:
            break
        add_candidate(selected, seen, cand, "table_column_expansion", output_top_k)

    # Stage 3: add FK endpoint closure softly. JOIN_BRIDGE should guide connectivity
    # but not dominate the semantic column budget.
    fk_candidates = []
    for selected_item in list(selected):
        item_id = int(selected_item["id"])
        for neighbor_id in indexes["fk_neighbors"].get(item_id, set()):
            if neighbor_id in seen or neighbor_id not in nodes:
                continue
            base_score = fused.get(neighbor_id, 0.0)
            fk_candidates.append(
                {
                    **nodes[neighbor_id],
                    "id": neighbor_id,
                    "score": base_score + fk_bonus,
                    "raw_score": base_score,
                    "source_operation": "FK_ENDPOINT_CLOSURE",
                    "source_clause": "FK_ENDPOINT_CLOSURE",
                    "provenance": provenance.get(neighbor_id, []),
                }
            )
    fk_candidates.sort(key=lambda cand: cand["score"], reverse=True)
    for cand in fk_candidates:
        if len(selected) >= output_top_k:
            break
        add_candidate(selected, seen, cand, "fk_endpoint_closure", output_top_k)

    # Stage 4: fill remaining slots by fused calibrated score.
    leftovers = []
    for item_id, score in fused.items():
        if item_id in seen or item_id not in nodes:
            continue
        node = nodes[item_id]
        if node.get("type") == "table":
            if not include_tables:
                continue
            if sum(1 for x in selected if nodes.get(int(x["id"]), {}).get("type") == "table") >= max_tables:
                continue
        leftovers.append(
            {
                **node,
                "id": item_id,
                "score": score,
                "raw_score": score,
                "source_operation": "FUSED_FILL",
                "source_clause": "FUSED_FILL",
                "provenance": provenance.get(item_id, []),
            }
        )
    leftovers.sort(key=lambda cand: cand["score"], reverse=True)
    for cand in leftovers:
        if len(selected) >= output_top_k:
            break
        add_candidate(selected, seen, cand, "fused_fill", output_top_k)

    gold_ids = set(relation_record.get("whole_sql_labels", []))
    selected_ids = [int(node["id"]) for node in selected]
    return {
        "example_id": graph_example.get("example_id"),
        "record_index": item["record_index"],
        "db_id": relation_record.get("db_id"),
        "question_id": relation_record.get("question_id"),
        "question": relation_record.get("question"),
        "evidence": relation_record.get("evidence"),
        "gold_label_ids": sorted(gold_ids),
        "gold_label_names": relation_record.get("whole_sql_label_names", []),
        f"top_{output_top_k}": selected[:output_top_k],
        "assembled_recall@30": recall_at_k(gold_ids, selected_ids, 30),
        "assembled_precision@30": precision_at_k(gold_ids, selected_ids, min(30, len(selected_ids))),
        "assembly_config": {
            "method": "operation_calibrated_fusion",
            "operation_budgets": operation_budgets,
            "operation_weights": operation_weights,
            "rank_weight": rank_weight,
            "z_weight": z_weight,
            "table_column_bonus": table_column_bonus,
            "fk_bonus": fk_bonus,
            "max_tables": max_tables,
            "include_tables": include_tables,
        },
    }


def assemble_calibrated(
    operation_predictions,
    aligned_records,
    output_top_k=30,
    operation_budgets=None,
    operation_weights=None,
    rank_weight=0.7,
    z_weight=0.3,
    table_column_bonus=0.2,
    fk_bonus=0.1,
    max_tables=4,
    include_tables=True,
):
    operation_budgets = operation_budgets or dict(DEFAULT_OPERATION_BUDGETS)
    operation_weights = operation_weights or dict(DEFAULT_OPERATION_WEIGHTS)
    grouped = {}
    for row in operation_predictions:
        operation = row.get("relation_type") or row.get("clause")
        grouped.setdefault(int(row["record_index"]), {})[operation] = row

    assembled = []
    for item in aligned_records:
        assembled.append(
            assemble_one_record(
                item=item,
                operation_rows=grouped.get(int(item["record_index"]), {}),
                output_top_k=output_top_k,
                operation_budgets=operation_budgets,
                operation_weights=operation_weights,
                rank_weight=rank_weight,
                z_weight=z_weight,
                table_column_bonus=table_column_bonus,
                fk_bonus=fk_bonus,
                max_tables=max_tables,
                include_tables=include_tables,
            )
        )
    return assembled


def evaluate_assembled_rows(assembled_rows):
    recalls = [row["assembled_recall@30"] for row in assembled_rows]
    precisions = [row["assembled_precision@30"] for row in assembled_rows]
    sample_count = len(assembled_rows)
    missing_count = sum(
        1 for row in assembled_rows if (row["assembled_recall@30"] or 0) < 1.0
    )
    complete_count = sample_count - missing_count
    return {
        "assembled_sample_count": sample_count,
        "assembled_schema_recall@30": mean(recalls),
        "assembled_schema_precision@30": mean(precisions),
        "assembled_complete_samples@30": complete_count,
        "assembled_complete_coverage@30": complete_count / sample_count if sample_count else 0.0,
        "assembled_missing_samples@30": missing_count,
        "assembled_missing_rate@30": missing_count / sample_count if sample_count else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation-predictions", required=True)
    parser.add_argument("--relation-file", default="experiments/stage5j_relation_labels_v1/dev_relation_labels.jsonl")
    parser.add_argument("--graph-file", default="experiments/stage5i_dsg_data_semantic_v1/dev_examples.jsonl")
    parser.add_argument("--output-dir", default="experiments/stage8b_calibrated_assembly")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-top-k", type=int, default=30)
    parser.add_argument("--operation-budgets", type=parse_budget_text, default=",".join(f"{k}:{v}" for k, v in DEFAULT_OPERATION_BUDGETS.items()))
    parser.add_argument("--operation-weights", default=None)
    parser.add_argument("--rank-weight", type=float, default=0.7)
    parser.add_argument("--z-weight", type=float, default=0.3)
    parser.add_argument("--table-column-bonus", type=float, default=0.2)
    parser.add_argument("--fk-bonus", type=float, default=0.1)
    parser.add_argument("--max-tables", type=int, default=4)
    parser.add_argument("--exclude-tables", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    operation_predictions = read_jsonl(Path(args.operation_predictions))
    aligned_records = load_aligned_records(Path(args.relation_file), Path(args.graph_file), args.limit)
    operation_weights = parse_float_map(args.operation_weights, DEFAULT_OPERATION_WEIGHTS)
    assembled = assemble_calibrated(
        operation_predictions=operation_predictions,
        aligned_records=aligned_records,
        output_top_k=args.output_top_k,
        operation_budgets=args.operation_budgets,
        operation_weights=operation_weights,
        rank_weight=args.rank_weight,
        z_weight=args.z_weight,
        table_column_bonus=args.table_column_bonus,
        fk_bonus=args.fk_bonus,
        max_tables=args.max_tables,
        include_tables=not args.exclude_tables,
    )
    metrics = evaluate_assembled_rows(assembled)
    report = {
        **metrics,
        "operation_prediction_file": args.operation_predictions,
        "relation_file": args.relation_file,
        "graph_file": args.graph_file,
        "output_top_k": args.output_top_k,
        "operation_budgets": args.operation_budgets,
        "operation_weights": operation_weights,
        "rank_weight": args.rank_weight,
        "z_weight": args.z_weight,
        "table_column_bonus": args.table_column_bonus,
        "fk_bonus": args.fk_bonus,
        "max_tables": args.max_tables,
        "include_tables": not args.exclude_tables,
        "innovation": (
            "Calibrate operation-conditioned beliefs before assembling the final schema set. "
            "This treats each SQL operation as a separate latent belief distribution and fuses "
            "them with operation budgets plus graph-aware table/FK closure."
        ),
    }
    write_jsonl(output_dir / "dev_assembled_predictions.jsonl", assembled)
    write_json(output_dir / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
