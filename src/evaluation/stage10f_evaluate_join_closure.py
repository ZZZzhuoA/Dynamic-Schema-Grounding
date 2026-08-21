"""Evaluate semantic-core-preserving Stage 10-F join closure."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stage13_build_typed_ra_data import explicit_sql_join_edges  # noqa: E402
from src.diagnosis.stage10_complete_coverage_diagnosis import (  # noqa: E402
    index_rows,
    read_jsonl,
    semantic_and_join_targets,
    write_json,
    write_jsonl,
)


def graph_inputs(row):
    return row.get("inference_inputs", row)


def full_graph_connected(graph, selected, required_tables, explicit_edges=None):
    required_tables = set(required_tables)
    selected = set(selected)
    if not required_tables:
        return True
    if not required_tables.issubset(selected):
        return False
    if len(required_tables) == 1:
        return True
    adjacency = defaultdict(set)

    def add(left, right):
        left, right = int(left), int(right)
        if left in selected and right in selected:
            adjacency[left].add(right)
            adjacency[right].add(left)

    for edge in graph_inputs(graph).get("schema_edges", []):
        add(edge["src"], edge["dst"])
    for edge in explicit_edges or []:
        add(edge["left_column_id"], edge["right_column_id"])
    start = next(iter(required_tables))
    visited, frontier = set(), [start]
    while frontier:
        node = frontier.pop()
        if node in visited:
            continue
        visited.add(node)
        frontier.extend(adjacency[node] - visited)
    return required_tables.issubset(visited)


def evaluate_rows(full_graphs, closures, labels):
    metrics = Counter()
    failures = []
    closure_sizes = []
    path_hops = []
    common = sorted(set(full_graphs) & set(closures) & set(labels))
    for index in common:
        graph = full_graphs[index]
        closure = closures[index]
        label = labels[index]
        targets = semantic_and_join_targets(label)
        semantic = set(targets["semantic"])
        required_tables = set(targets["required_tables"])
        reference_join = set(targets["reference_join"])
        core = set(int(value) for value in closure.get("semantic_core_ids", []))
        grounded = set(int(value) for value in closure.get("grounded_schema_ids", []))
        all_ids = {
            int(node["id"]) for node in graph_inputs(graph).get("schema_nodes", [])
        }
        explicit_edges = explicit_sql_join_edges(
            label.get("sql") or "", label.get("schema_items", [])
        )
        semantic_complete = semantic.issubset(core)
        before_connected = full_graph_connected(
            graph, core, required_tables, explicit_edges
        )
        after_connected = full_graph_connected(
            graph, grounded, required_tables, explicit_edges
        )
        declared_path_exists = full_graph_connected(
            graph, all_ids, required_tables, explicit_edges
        )
        before_complete = semantic_complete and before_connected
        after_complete = semantic_complete and after_connected
        metrics["semantic_complete"] += int(semantic_complete)
        metrics["before_join_connected"] += int(before_connected)
        metrics["after_join_connected"] += int(after_connected)
        metrics["declared_or_explicit_path_exists"] += int(declared_path_exists)
        metrics["before_grounding_complete"] += int(before_complete)
        metrics["after_grounding_complete"] += int(after_complete)
        metrics["recovered"] += int(after_complete and not before_complete)
        metrics["regressed"] += int(before_complete and not after_complete)
        metrics["reference_join_complete_before"] += int(reference_join.issubset(core))
        metrics["reference_join_complete_after"] += int(reference_join.issubset(grounded))
        closure_sizes.append(len(grounded - core))
        path_hops.extend(
            int(path.get("hop_count", 0)) for path in closure.get("paths", [])
        )

        if after_complete:
            continue
        if not semantic_complete:
            reason = "semantic_core_missing"
        elif not declared_path_exists:
            reason = "no_declared_or_explicit_path"
        elif not required_tables.issubset(
            set(int(value) for value in closure.get("terminal_table_ids", []))
        ):
            reason = "terminal_detection_missing"
        else:
            reason = "closure_failed_despite_available_path"
        metrics[f"failure::{reason}"] += 1
        failures.append(
            {
                "record_index": index,
                "db_id": closure.get("db_id"),
                "question": closure.get("question"),
                "reason": reason,
                "closure_status": closure.get("status"),
                "semantic_missing_ids": sorted(semantic - core),
                "required_table_ids": sorted(required_tables),
                "terminal_table_ids": closure.get("terminal_table_ids", []),
                "structural_closure_ids": closure.get("structural_closure_ids", []),
                "terminal_components": closure.get("terminal_components", []),
            }
        )
    count = len(common)
    summary = {
        "sample_count": count,
        "semantic_complete_samples": metrics["semantic_complete"],
        "semantic_complete_coverage": metrics["semantic_complete"] / count if count else 0.0,
        "before_join_connected_samples": metrics["before_join_connected"],
        "before_join_connected_coverage": metrics["before_join_connected"] / count if count else 0.0,
        "after_join_connected_samples": metrics["after_join_connected"],
        "after_join_connected_coverage": metrics["after_join_connected"] / count if count else 0.0,
        "declared_or_explicit_path_exists_samples": metrics["declared_or_explicit_path_exists"],
        "declared_or_explicit_path_exists_coverage": metrics["declared_or_explicit_path_exists"] / count if count else 0.0,
        "before_grounding_complete_samples": metrics["before_grounding_complete"],
        "before_grounding_complete_coverage": metrics["before_grounding_complete"] / count if count else 0.0,
        "after_grounding_complete_samples": metrics["after_grounding_complete"],
        "after_grounding_complete_coverage": metrics["after_grounding_complete"] / count if count else 0.0,
        "recovered_samples": metrics["recovered"],
        "regressed_samples": metrics["regressed"],
        "reference_join_complete_before_coverage": metrics["reference_join_complete_before"] / count if count else 0.0,
        "reference_join_complete_after_coverage": metrics["reference_join_complete_after"] / count if count else 0.0,
        "avg_added_closure_nodes": sum(closure_sizes) / len(closure_sizes) if closure_sizes else 0.0,
        "max_added_closure_nodes": max(closure_sizes, default=0),
        "avg_path_hops": sum(path_hops) / len(path_hops) if path_hops else 0.0,
        "failure_reasons": {
            key.split("::", 1)[1]: value
            for key, value in metrics.items()
            if key.startswith("failure::")
        },
        "metric_policy": (
            "Semantic completeness is measured on the immutable Top-K core; added closure nodes "
            "can improve connectivity but cannot receive semantic-recall credit."
        ),
    }
    return summary, failures


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 10-F full-schema join closure.")
    parser.add_argument("--full-graph-file", required=True)
    parser.add_argument("--closure-file", required=True)
    parser.add_argument("--exact-label-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    graphs = index_rows(read_jsonl(args.full_graph_file)[: args.limit])
    closures = index_rows(read_jsonl(args.closure_file)[: args.limit])
    labels = index_rows(read_jsonl(args.exact_label_file)[: args.limit])
    summary, failures = evaluate_rows(graphs, closures, labels)
    summary["config"] = vars(args)
    output_dir = Path(args.output_dir)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "failures.jsonl", failures)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
