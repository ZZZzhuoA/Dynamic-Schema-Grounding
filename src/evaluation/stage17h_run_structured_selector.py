"""Run the frozen Stage 17-H fixed-budget structured schema selector.

The selector consumes leakage-free Stage 17 full-schema logits and applies the
existing Stage 10 owner-closed greedy decoder. Gold labels are read only after
selection for paired evaluation and are never written to prediction artifacts.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grounding.stage10_constrained_selector import constrained_topk  # noqa: E402
from src.training.stage17a_train_full_schema_qrgta import (  # noqa: E402
    align_graphs_and_labels,
    read_jsonl,
)


DEFAULT_DIRECT_EDGE_TYPES = (
    "column_to_table",
    "table_to_column",
    "foreign_key_forward",
    "foreign_key_backward",
    "table_to_primary_key",
    "primary_key_to_table",
)
FORBIDDEN_PREDICTION_KEYS = {
    "gold",
    "gold_ids",
    "gold_schema_ids",
    "labels",
    "whole_sql_labels",
}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assignment(value, option):
    if "=" not in value:
        raise ValueError(f"{option} expects NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"Invalid run name for {option}: {name!r}")
    return name, Path(path)


def assert_leakage_free(value, location="prediction"):
    if isinstance(value, dict):
        forbidden = sorted(FORBIDDEN_PREDICTION_KEYS & set(value))
        if forbidden:
            raise ValueError(f"Gold/label fields in {location}: {forbidden}")
        for key, child in value.items():
            assert_leakage_free(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_leakage_free(child, f"{location}[{index}]")


def prediction_rows_by_record(rows, name):
    by_record = {}
    for row_index, row in enumerate(rows):
        assert_leakage_free(row, f"{name}[{row_index}]")
        if "record_index" not in row:
            raise ValueError(f"Missing record_index in {name}[{row_index}]")
        record_index = int(row["record_index"])
        if record_index in by_record:
            raise ValueError(f"Duplicate record_index={record_index} in {name}")
        by_record[record_index] = row
    return by_record


def validate_ranked_schema(example, prediction):
    ranked = prediction.get("ranked_schema")
    if not isinstance(ranked, list):
        raise ValueError(
            f"Missing ranked_schema at record_index={example['record_index']}"
        )
    nodes = example["nodes"]
    if len(ranked) != len(nodes):
        raise ValueError(
            f"Ranked schema length mismatch at record_index={example['record_index']}: "
            f"prediction={len(ranked)} graph={len(nodes)}"
        )
    graph_by_id = {int(node["id"]): node for node in nodes}
    if len(graph_by_id) != len(nodes):
        raise ValueError(
            f"Duplicate graph node ids at record_index={example['record_index']}"
        )
    seen = set()
    for position, item in enumerate(ranked):
        item_id = int(item["schema_item_id"])
        if item_id in seen:
            raise ValueError(
                f"Duplicate ranked schema id={item_id} at "
                f"record_index={example['record_index']}"
            )
        seen.add(item_id)
        node = graph_by_id.get(item_id)
        if node is None:
            raise ValueError(
                f"Unknown ranked schema id={item_id} at "
                f"record_index={example['record_index']}"
            )
        if str(item.get("name")) != str(node.get("name")) or str(
            item.get("type")
        ) != str(node.get("type")):
            raise ValueError(
                f"Ranked schema identity mismatch at "
                f"record_index={example['record_index']}, position={position}"
            )
        logit = float(item["logit"])
        if not math.isfinite(logit):
            raise ValueError(
                f"Non-finite logit at record_index={example['record_index']}, "
                f"schema_item_id={item_id}"
            )
    if seen != set(graph_by_id):
        raise ValueError(
            f"Ranked schema coverage mismatch at record_index={example['record_index']}"
        )
    if str(prediction.get("db_id")) != str(example.get("db_id")):
        raise ValueError(
            f"Prediction database mismatch at record_index={example['record_index']}"
        )
    prediction_question_id = prediction.get("question_id")
    example_question_id = example.get("question_id")
    if prediction_question_id is not None or example_question_id is not None:
        if str(prediction_question_id) != str(example_question_id):
            raise ValueError(
                f"Prediction question mismatch at record_index={example['record_index']}"
            )
    return ranked


def stage17_selector_example(example, raw_top_ids, direct_edge_types):
    nodes = example["nodes"]
    id_to_local = {int(node["id"]): index for index, node in enumerate(nodes)}
    table_name_to_id = {}
    for node in nodes:
        if node.get("type") != "table":
            continue
        table_name = str(node.get("name"))
        if table_name in table_name_to_id:
            raise ValueError(
                f"Duplicate table name={table_name!r} at "
                f"record_index={example['record_index']}"
            )
        table_name_to_id[table_name] = int(node["id"])

    candidate_nodes = []
    for local_id, node in enumerate(nodes):
        node_type = str(node.get("type"))
        item_id = int(node["id"])
        owner_id = item_id
        if node_type == "column":
            table_name = str(node.get("table"))
            owner_id = table_name_to_id.get(table_name)
            if owner_id is None:
                raise ValueError(
                    f"Column owner table is absent at record_index={example['record_index']}: "
                    f"column={node.get('name')!r}, table={table_name!r}"
                )
        elif node_type != "table":
            raise ValueError(
                f"Unsupported schema node type={node_type!r} at "
                f"record_index={example['record_index']}"
            )
        candidate_nodes.append(
            {
                "local_id": local_id,
                "schema_item_id": item_id,
                "name": node.get("name"),
                "type": node_type,
                "owner_table_id": owner_id,
                "owner_local_id": id_to_local[owner_id],
            }
        )

    allowed = set(direct_edge_types)
    schema_edges = []
    seen_edges = set()
    for edge in example.get("schema_edges", []):
        edge_type = str(edge.get("type"))
        if edge_type not in allowed:
            continue
        src_id = int(edge["src"])
        dst_id = int(edge["dst"])
        if src_id not in id_to_local or dst_id not in id_to_local:
            raise ValueError(
                f"Direct schema edge references an unknown node at "
                f"record_index={example['record_index']}"
            )
        key = (id_to_local[src_id], id_to_local[dst_id], edge_type)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        schema_edges.append({"src": key[0], "dst": key[1], "type": edge_type})

    return {
        "candidate_nodes": candidate_nodes,
        "schema_edges": schema_edges,
        "baseline_selected_ids": [int(item_id) for item_id in raw_top_ids],
    }


def owner_closed(local_ids, selector_example):
    selected = set(int(local_id) for local_id in local_ids)
    for local_id in selected:
        node = selector_example["candidate_nodes"][local_id]
        if node["type"] == "column" and int(node["owner_local_id"]) not in selected:
            return False
    return True


def select_one(example, prediction, args):
    ranked = validate_ranked_schema(example, prediction)
    item_by_id = {int(item["schema_item_id"]): item for item in ranked}
    local_by_id = {int(node["id"]): index for index, node in enumerate(example["nodes"])}
    logits = [0.0] * len(example["nodes"])
    for item_id, item in item_by_id.items():
        logits[local_by_id[item_id]] = float(item["logit"])
    raw_local = sorted(range(len(logits)), key=lambda index: (-logits[index], index))
    target_k = min(int(args.top_k), len(raw_local))
    raw_top_local = raw_local[:target_k]
    raw_top_ids = [int(example["nodes"][index]["id"]) for index in raw_top_local]
    source_top_ids = prediction.get(f"top_{int(args.top_k)}_ids")
    if source_top_ids is not None and [int(value) for value in source_top_ids] != raw_top_ids:
        raise ValueError(
            f"Source Top-{args.top_k} does not match deterministic logit ranking at "
            f"record_index={example['record_index']}"
        )
    selector_example = stage17_selector_example(
        example, raw_top_ids, args.direct_edge_types
    )
    table_count = sum(
        node["type"] == "table" for node in selector_example["candidate_nodes"]
    )
    max_tables = table_count if int(args.max_tables) < 0 else int(args.max_tables)
    min_tables = None if int(args.min_tables) < 0 else int(args.min_tables)
    structured_local, debug = constrained_topk(
        selector_example,
        logits,
        top_k=target_k,
        max_tables=max_tables,
        min_tables=min_tables,
        connectivity_weight=float(args.connectivity_weight),
        baseline_retention_weight=float(args.baseline_retention_weight),
    )
    if len(structured_local) != target_k:
        raise ValueError(
            f"Structured selector did not fill its budget at "
            f"record_index={example['record_index']}: "
            f"selected={len(structured_local)} target={target_k}"
        )
    if not owner_closed(structured_local, selector_example):
        raise ValueError(
            f"Structured selector violated owner closure at "
            f"record_index={example['record_index']}"
        )
    structured_set = set(structured_local)
    full_local_order = structured_local + [
        local_id for local_id in raw_local if local_id not in structured_set
    ]
    structured_ids = [
        int(example["nodes"][local_id]["id"]) for local_id in structured_local
    ]
    raw_set = set(raw_top_local)
    changed_count = len(raw_set.symmetric_difference(structured_set)) // 2
    debug = {
        **debug,
        "target_k": target_k,
        "changed_slot_count": changed_count,
        "raw_structured_overlap": len(raw_set & structured_set),
        "raw_structured_jaccard": (
            len(raw_set & structured_set) / len(raw_set | structured_set)
            if raw_set or structured_set
            else 1.0
        ),
        "owner_closed": True,
        "direct_edge_count": len(selector_example["schema_edges"]),
    }

    raw_rank_by_local = {
        local_id: rank for rank, local_id in enumerate(raw_local, start=1)
    }
    output_ranked = []
    for rank, local_id in enumerate(full_local_order, start=1):
        item_id = int(example["nodes"][local_id]["id"])
        source = dict(item_by_id[item_id])
        source["raw_rank"] = int(source.get("rank", raw_rank_by_local[local_id]))
        source["rank"] = rank
        output_ranked.append(source)
    output_ids = [int(item["schema_item_id"]) for item in output_ranked]
    output_row = {
        "record_index": int(example["record_index"]),
        "db_id": example.get("db_id"),
        "question_id": example.get("question_id"),
        "schema_node_count": len(example["nodes"]),
        "selector": "stage17h_frozen_fixed_budget_structured_selector",
        "raw_top_k_ids": raw_top_ids,
        "structured_top_k_ids": structured_ids,
        "semantic_core_ids": structured_ids,
        "ranked_schema": output_ranked,
        "selector_debug": debug,
    }
    if int(args.top_k) == 30:
        output_row["raw_top_30_ids"] = raw_top_ids
    for k in (10, 20, 30, 50):
        output_row[f"top_{k}_ids"] = output_ids[:k]
    assert_leakage_free(output_row, "structured_prediction")
    return output_row, raw_top_ids, structured_ids, debug


def mean(values):
    return sum(values) / len(values) if values else 0.0


def selection_metrics(examples, selected_by_record, top_k):
    accumulated = defaultdict(list)
    by_database = defaultdict(list)
    for example in examples:
        selected_ids = list(selected_by_record[int(example["record_index"])])
        selected = set(selected_ids[:top_k])
        gold = set(int(item_id) for item_id in example["gold_ids"])
        first_gold_rank = next(
            (
                rank
                for rank, item_id in enumerate(selected_ids, start=1)
                if item_id in gold
            ),
            None,
        )
        node_by_id = {int(node["id"]): node for node in example["nodes"]}
        gold_tables = {
            item_id for item_id in gold if node_by_id[item_id].get("type") == "table"
        }
        gold_columns = gold - gold_tables
        schema_recall = len(selected & gold) / len(gold)
        complete = float(gold.issubset(selected))
        precision = len(selected & gold) / max(min(top_k, len(example["nodes"])), 1)
        table_recall = (
            len(selected & gold_tables) / len(gold_tables) if gold_tables else 1.0
        )
        column_recall = (
            len(selected & gold_columns) / len(gold_columns) if gold_columns else 1.0
        )
        row = {
            "schema_recall@k": schema_recall,
            "schema_precision@k": precision,
            "complete_coverage@k": complete,
            "mrr": 1.0 / first_gold_rank if first_gold_rank else 0.0,
        }
        if gold_tables:
            row["table_recall@k"] = table_recall
        if gold_columns:
            row["column_recall@k"] = column_recall
        for key, value in row.items():
            accumulated[key].append(value)
        by_database[str(example["db_id"])].append(row)
    suffix = str(top_k)
    metrics = {"sample_count": len(examples)}
    metrics.update(
        {
            key.replace("@k", f"@{suffix}"): mean(values)
            for key, values in sorted(accumulated.items())
        }
    )
    metrics["by_database"] = {
        db_id: {
            "sample_count": len(rows),
            **{
                key.replace("@k", f"@{suffix}"): mean(
                    [row[key] for row in rows if key in row]
                )
                for key in sorted({key for row in rows for key in row})
            },
        }
        for db_id, rows in sorted(by_database.items())
    }
    return metrics


def paired_evaluation(examples, raw_by_record, structured_by_record, top_k):
    recovered = []
    regressed = []
    unchanged_complete = 0
    unchanged_incomplete = 0
    evaluation_cases = []
    for example in examples:
        record_index = int(example["record_index"])
        gold = set(int(item_id) for item_id in example["gold_ids"])
        raw = set(raw_by_record[record_index][:top_k])
        structured = set(structured_by_record[record_index][:top_k])
        raw_complete = gold.issubset(raw)
        structured_complete = gold.issubset(structured)
        if not raw_complete and structured_complete:
            outcome = "recovered"
            recovered.append(record_index)
        elif raw_complete and not structured_complete:
            outcome = "regressed"
            regressed.append(record_index)
        elif raw_complete:
            outcome = "unchanged_complete"
            unchanged_complete += 1
        else:
            outcome = "unchanged_incomplete"
            unchanged_incomplete += 1
        if raw != structured:
            evaluation_cases.append(
                {
                    "artifact_type": "posthoc_evaluation",
                    "record_index": record_index,
                    "db_id": example.get("db_id"),
                    "question_id": example.get("question_id"),
                    "outcome": outcome,
                    "raw_missing_gold_ids": sorted(gold - raw),
                    "structured_missing_gold_ids": sorted(gold - structured),
                    "added_ids": sorted(structured - raw),
                    "removed_ids": sorted(raw - structured),
                }
            )
    return {
        "sample_count": len(examples),
        "recovered_complete_count": len(recovered),
        "regressed_complete_count": len(regressed),
        "net_recovered_complete_count": len(recovered) - len(regressed),
        "unchanged_complete_count": unchanged_complete,
        "unchanged_incomplete_count": unchanged_incomplete,
        "recovered_record_indices": recovered,
        "regressed_record_indices": regressed,
    }, evaluation_cases


def metric_deltas(raw_metrics, structured_metrics, top_k):
    keys = (
        f"schema_recall@{top_k}",
        f"schema_precision@{top_k}",
        f"complete_coverage@{top_k}",
        f"table_recall@{top_k}",
        f"column_recall@{top_k}",
        "mrr",
    )
    return {
        key: float(structured_metrics[key]) - float(raw_metrics[key]) for key in keys
    }


def run_selector(name, prediction_file, examples, args, output_dir):
    predictions = read_jsonl(prediction_file)
    by_record = prediction_rows_by_record(predictions, str(prediction_file))
    expected = {int(example["record_index"]) for example in examples}
    if set(by_record) != expected:
        raise ValueError(
            f"Prediction/example record mismatch for run={name}: "
            f"missing={sorted(expected - set(by_record))[:5]} "
            f"unexpected={sorted(set(by_record) - expected)[:5]}"
        )

    output_rows = []
    raw_by_record = {}
    structured_by_record = {}
    debug_rows = []
    for example in examples:
        record_index = int(example["record_index"])
        row, raw_ids, structured_ids, debug = select_one(
            example, by_record[record_index], args
        )
        output_rows.append(row)
        raw_by_record[record_index] = [
            int(item["schema_item_id"])
            for item in sorted(row["ranked_schema"], key=lambda item: item["raw_rank"])
        ]
        structured_by_record[record_index] = [
            int(item["schema_item_id"]) for item in row["ranked_schema"]
        ]
        debug_rows.append(debug)

    raw_metrics = selection_metrics(examples, raw_by_record, int(args.top_k))
    structured_metrics = selection_metrics(
        examples, structured_by_record, int(args.top_k)
    )
    paired, evaluation_cases = paired_evaluation(
        examples, raw_by_record, structured_by_record, int(args.top_k)
    )
    run_dir = output_dir / name
    prediction_output = run_dir / "structured_predictions.jsonl"
    evaluation_output = run_dir / "evaluation_cases.jsonl"
    run_summary_output = run_dir / "summary.json"
    existing_outputs = [
        path
        for path in (prediction_output, evaluation_output, run_summary_output)
        if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            f"Refusing to overwrite Stage 17-H outputs: {existing_outputs}"
        )
    write_jsonl(prediction_output, output_rows)
    write_jsonl(evaluation_output, evaluation_cases)
    run_summary = {
        "run": name,
        "source_prediction_file": str(prediction_file),
        "source_prediction_sha256": file_sha256(prediction_file),
        "structured_prediction_file": str(prediction_output),
        "evaluation_cases_file": str(evaluation_output),
        "raw_metrics": raw_metrics,
        "structured_metrics": structured_metrics,
        "structured_minus_raw": metric_deltas(
            raw_metrics, structured_metrics, int(args.top_k)
        ),
        "paired_outcomes": paired,
        "selector_statistics": {
            "avg_selected_table_count": mean(
                [float(row["selected_table_count"]) for row in debug_rows]
            ),
            "avg_selected_column_count": mean(
                [float(row["selected_column_count"]) for row in debug_rows]
            ),
            "avg_owner_closure_additions": mean(
                [float(row["owner_closure_additions"]) for row in debug_rows]
            ),
            "avg_changed_slot_count": mean(
                [float(row["changed_slot_count"]) for row in debug_rows]
            ),
            "avg_raw_structured_jaccard": mean(
                [float(row["raw_structured_jaccard"]) for row in debug_rows]
            ),
            "unchanged_selection_count": sum(
                int(row["changed_slot_count"] == 0) for row in debug_rows
            ),
            "all_owner_closed": all(bool(row["owner_closed"]) for row in debug_rows),
            "all_budgets_exact": all(
                int(row["selected_count"]) == int(row["target_k"])
                for row in debug_rows
            ),
        },
        "prediction_leakage_free": True,
    }
    write_json(run_summary_output, run_summary)
    return run_summary


def aggregate_runs(runs, top_k):
    metric_keys = (
        f"schema_recall@{top_k}",
        f"schema_precision@{top_k}",
        f"complete_coverage@{top_k}",
        f"table_recall@{top_k}",
        f"column_recall@{top_k}",
        "mrr",
    )
    raw = {
        key: mean([float(run["raw_metrics"][key]) for run in runs.values()])
        for key in metric_keys
    }
    structured = {
        key: mean(
            [float(run["structured_metrics"][key]) for run in runs.values()]
        )
        for key in metric_keys
    }
    deltas = {key: structured[key] - raw[key] for key in metric_keys}
    recovered = sum(
        int(run["paired_outcomes"]["recovered_complete_count"])
        for run in runs.values()
    )
    regressed = sum(
        int(run["paired_outcomes"]["regressed_complete_count"])
        for run in runs.values()
    )
    checks = {
        "mean_complete_coverage_improves": deltas[f"complete_coverage@{top_k}"] > 0,
        "total_recovered_exceeds_regressed": recovered > regressed,
        "mean_column_recall_does_not_decrease": deltas[f"column_recall@{top_k}"] >= 0,
        "mean_mrr_drop_within_0.005": deltas["mrr"] >= -0.005,
        "all_predictions_leakage_free": all(
            bool(run["prediction_leakage_free"]) for run in runs.values()
        ),
        "all_selections_owner_closed": all(
            bool(run["selector_statistics"]["all_owner_closed"])
            for run in runs.values()
        ),
        "all_selection_budgets_exact": all(
            bool(run["selector_statistics"]["all_budgets_exact"])
            for run in runs.values()
        ),
    }
    return {
        "raw_metrics_mean": raw,
        "structured_metrics_mean": structured,
        "structured_minus_raw_mean": deltas,
        "total_recovered_complete_count": recovered,
        "total_regressed_complete_count": regressed,
        "net_recovered_complete_count": recovered - regressed,
        "decision_checks": checks,
        "decision_passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="NAME=experiment_dir containing dev_predictions.jsonl",
    )
    parser.add_argument(
        "--prediction-file",
        action="append",
        default=[],
        help="NAME=leakage-free Stage 17 prediction JSONL",
    )
    parser.add_argument("--dev-graph-file", required=True)
    parser.add_argument("--dev-label-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--max-tables",
        type=int,
        default=8,
        help="Maximum selected tables; negative means all schema tables.",
    )
    parser.add_argument(
        "--min-tables",
        type=int,
        default=-1,
        help="Minimum selected tables; negative preserves the raw Top-K table count.",
    )
    parser.add_argument("--connectivity-weight", type=float, default=0.10)
    parser.add_argument("--baseline-retention-weight", type=float, default=0.05)
    parser.add_argument(
        "--direct-edge-types",
        default=",".join(DEFAULT_DIRECT_EDGE_TYPES),
        help="Comma-separated direct graph relations eligible for connectivity reward.",
    )
    parser.add_argument("--dev-limit", type=int, default=None)
    args = parser.parse_args()
    if not args.run and not args.prediction_file:
        raise ValueError("Provide at least one --run or --prediction-file")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.max_tables == 0:
        raise ValueError("--max-tables must be positive or negative for unlimited")
    if args.min_tables < -1:
        raise ValueError("--min-tables must be -1 or non-negative")
    if (
        not math.isfinite(args.connectivity_weight)
        or not math.isfinite(args.baseline_retention_weight)
        or args.connectivity_weight < 0
        or args.baseline_retention_weight < 0
    ):
        raise ValueError("Selector weights must be non-negative")
    args.direct_edge_types = tuple(
        value.strip() for value in args.direct_edge_types.split(",") if value.strip()
    )
    if not args.direct_edge_types:
        raise ValueError("--direct-edge-types must not be empty")

    specs = []
    seen_names = set()
    for value in args.run:
        name, run_dir = assignment(value, "--run")
        if name in seen_names:
            raise ValueError(f"Duplicate run name: {name}")
        specs.append((name, run_dir / "dev_predictions.jsonl"))
        seen_names.add(name)
    for value in args.prediction_file:
        name, path = assignment(value, "--prediction-file")
        if name in seen_names:
            raise ValueError(f"Duplicate run name: {name}")
        specs.append((name, path))
        seen_names.add(name)

    graphs = read_jsonl(args.dev_graph_file, args.dev_limit)
    labels = read_jsonl(args.dev_label_file)
    examples, alignment = align_graphs_and_labels(graphs, labels, "dev")
    output_dir = Path(args.output_dir)
    root_summary = output_dir / "summary.json"
    if root_summary.exists():
        raise FileExistsError(
            f"Refusing to overwrite Stage 17-H summary: {root_summary}"
        )
    runs = {
        name: run_selector(name, path, examples, args, output_dir)
        for name, path in specs
    }
    summary = {
        "stage": "17-H",
        "method": "frozen_fixed_budget_structured_selector",
        "config": {
            "dev_graph_file": args.dev_graph_file,
            "dev_graph_sha256": file_sha256(args.dev_graph_file),
            "dev_label_file": args.dev_label_file,
            "dev_label_sha256": file_sha256(args.dev_label_file),
            "top_k": args.top_k,
            "max_tables": args.max_tables,
            "min_tables": "raw_top_k_table_count"
            if args.min_tables < 0
            else args.min_tables,
            "connectivity_weight": args.connectivity_weight,
            "baseline_retention_weight": args.baseline_retention_weight,
            "direct_edge_types": list(args.direct_edge_types),
            "dev_limit": args.dev_limit,
        },
        "alignment": alignment,
        "runs": runs,
        "aggregate": aggregate_runs(runs, int(args.top_k)),
        "leakage_note": (
            "Gold labels are used only for post-hoc evaluation. Structured prediction "
            "files contain no gold labels or gold-derived diagnostics."
        ),
    }
    write_json(root_summary, summary)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
