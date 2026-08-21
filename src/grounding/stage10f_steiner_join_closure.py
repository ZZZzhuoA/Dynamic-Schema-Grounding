"""Complete Stage 10-E semantic cores with full-schema FK Steiner paths.

This is an inference-only structural operator.  It never changes or truncates the
semantic Top-K selected by the grounder.  Instead it returns a separate closure
containing intermediate tables and FK endpoint columns from the full schema graph.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grounding.stage9_value_join_completion import (  # noqa: E402
    build_table_fk_graph,
    metric_closure_mst_paths,
    node_indexes,
    owner_table,
)


SEMANTIC_ROLES = [
    "OUTPUT_TARGET",
    "ENTITY_NAME",
    "METRIC_TARGET",
    "PREDICATE_COLUMN",
    "VALUE_ANCHOR",
    "TEMPORAL_FILTER",
    "ORDER_KEY",
    "GROUP_KEY",
    "FORMULA_COMPONENT",
]


def read_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def index_rows(rows):
    result = {}
    for fallback, row in enumerate(rows):
        index = int(row.get("record_index", row.get("question_id", fallback)))
        if index in result:
            raise ValueError(f"Duplicate record_index={index}")
        result[index] = row
    return result


def prediction_top_key(row):
    keys = [
        key for key, value in row.items()
        if key.startswith("top_") and isinstance(value, list)
    ]
    return max(keys, key=lambda key: int(key.split("_", 1)[1])) if keys else None


def schema_id(item):
    if isinstance(item, int):
        return int(item)
    for key in ["schema_item_id", "id"]:
        if item.get(key) is not None:
            return int(item[key])
    raise ValueError(f"Prediction item lacks schema identity: {item}")


def prior_by_schema(prior):
    return {
        int(item["schema_item_id"]): item.get("role_scores", {})
        for item in prior.get("node_priors", [])
    }


def selected_core(prediction):
    top_key = prediction_top_key(prediction)
    items = prediction.get(top_key, []) if top_key else []
    return top_key, items, [schema_id(item) for item in items]


def terminal_table_scores(
    graph_example,
    semantic_core_ids,
    prior,
    minimum_score,
):
    _, by_id, _ = node_indexes(graph_example)
    priors = prior_by_schema(prior)
    scores = defaultdict(float)
    support = defaultdict(list)
    for item_id in semantic_core_ids:
        node = by_id.get(int(item_id))
        table = owner_table(node)
        if not node or not table:
            continue
        role_scores = priors.get(int(item_id), {})
        best_role = None
        best_score = 0.0
        for role in SEMANTIC_ROLES:
            score = float(role_scores.get(role, 0.0))
            if score > best_score:
                best_role, best_score = role, score
        if best_score < minimum_score:
            continue
        scores[table] = max(scores[table], best_score)
        support[table].append(
            {"schema_item_id": int(item_id), "role": best_role, "score": best_score}
        )
    return scores, support


def join_endpoint_support(prediction_items, prior):
    priors = prior_by_schema(prior)
    support = {}
    for rank, item in enumerate(prediction_items, start=1):
        item_id = schema_id(item)
        rank_score = 1.0 / rank
        join_score = float(priors.get(item_id, {}).get("JOIN_BRIDGE", 0.0))
        support[item_id] = max(rank_score, join_score)
    for item_id, role_scores in priors.items():
        join_score = float(role_scores.get("JOIN_BRIDGE", 0.0))
        if join_score > 0:
            support[item_id] = max(support.get(item_id, 0.0), join_score)
    return support


def connected_terminal_components(terminals, paths):
    adjacency = defaultdict(set)
    for path in paths:
        for edge in path.get("edges", []):
            left = edge["left_table"]
            right = edge["right_table"]
            adjacency[left].add(right)
            adjacency[right].add(left)
    components = []
    remaining = set(terminals)
    while remaining:
        start = next(iter(remaining))
        visited, frontier = set(), [start]
        while frontier:
            table = frontier.pop()
            if table in visited:
                continue
            visited.add(table)
            frontier.extend(adjacency[table] - visited)
        component = sorted(visited & set(terminals))
        components.append(component)
        remaining -= set(component)
    return components


def complete_one(
    graph_example,
    prediction,
    prior,
    minimum_terminal_score=0.5,
    max_terminal_tables=6,
    support_weight=0.25,
    max_path_hops=6,
):
    inputs = graph_example.get("inference_inputs", graph_example)
    wrapped_graph = graph_example if "inference_inputs" in graph_example else {
        "inference_inputs": inputs
    }
    _, by_id, table_to_id = node_indexes(wrapped_graph)
    top_key, prediction_items, core_ids = selected_core(prediction)
    invalid_core_ids = [item_id for item_id in core_ids if item_id not in by_id]
    if invalid_core_ids:
        raise ValueError(
            f"Prediction contains IDs absent from full graph: {invalid_core_ids[:10]}"
        )
    scores, terminal_support = terminal_table_scores(
        wrapped_graph, core_ids, prior, minimum_terminal_score
    )
    ranked_terminals = sorted(scores, key=lambda table: (-scores[table], table))
    dropped_terminals = ranked_terminals[max_terminal_tables:]
    terminals = ranked_terminals[:max_terminal_tables]
    adjacency, _, _ = build_table_fk_graph(wrapped_graph)
    endpoint_support = join_endpoint_support(prediction_items, prior)

    paths = []
    if len(terminals) >= 2:
        paths = metric_closure_mst_paths(
            adjacency, terminals, endpoint_support, support_weight
        )
        paths = [path for path in paths if len(path.get("edges", [])) <= max_path_hops]

    closure_ids = set()
    path_rows = []
    for path in paths:
        edge_rows = []
        for edge in path.get("edges", []):
            left_table_id = table_to_id.get(edge["left_table"])
            right_table_id = table_to_id.get(edge["right_table"])
            for item_id in [
                left_table_id,
                right_table_id,
                int(edge["left_endpoint"]),
                int(edge["right_endpoint"]),
            ]:
                if item_id is not None:
                    closure_ids.add(int(item_id))
            edge_rows.append(
                {
                    **edge,
                    "left_table_id": left_table_id,
                    "right_table_id": right_table_id,
                }
            )
        path_rows.append(
            {
                "terminals": path.get("terminals", []),
                "cost": float(path.get("cost", 0.0)),
                "hop_count": len(edge_rows),
                "confidence": 1.0 / (1.0 + float(path.get("cost", 0.0))),
                "edges": edge_rows,
            }
        )

    core_set = set(core_ids)
    added_ids = sorted(closure_ids - core_set)
    grounded_ids = sorted(core_set | closure_ids)
    components = connected_terminal_components(terminals, path_rows)
    if not terminals:
        status = "insufficient_semantic_terminals"
    elif len(terminals) == 1:
        status = "no_closure_needed"
    elif len(components) == 1:
        status = "connected"
    else:
        status = "declared_fk_disconnected"
    return {
        "record_index": int(prediction["record_index"]),
        "db_id": prediction.get("db_id") or inputs.get("db_id"),
        "question_id": prediction.get("question_id"),
        "question": prediction.get("question") or inputs.get("question"),
        "semantic_top_key": top_key,
        "semantic_core_ids": core_ids,
        "semantic_core_count": len(core_ids),
        "terminal_table_ids": [table_to_id[table] for table in terminals if table in table_to_id],
        "terminal_tables": terminals,
        "terminal_scores": {table: scores[table] for table in terminals},
        "terminal_support": {table: terminal_support[table] for table in terminals},
        "dropped_terminal_tables": dropped_terminals,
        "structural_closure_ids": added_ids,
        "structural_closure_count": len(added_ids),
        "grounded_schema_ids": grounded_ids,
        "grounded_schema_count": len(grounded_ids),
        "paths": path_rows,
        "terminal_components": components,
        "status": status,
        "inference_boundary": (
            "Uses only frozen LLM priors, Stage 10-E predictions, and the full declared schema graph."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Add full-schema Steiner FK closure after Stage 10-E semantic grounding."
    )
    parser.add_argument("--full-graph-file", required=True)
    parser.add_argument("--prediction-file", required=True)
    parser.add_argument("--prior-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", default=None)
    parser.add_argument("--minimum-terminal-score", type=float, default=0.5)
    parser.add_argument("--max-terminal-tables", type=int, default=6)
    parser.add_argument("--support-weight", type=float, default=0.25)
    parser.add_argument("--max-path-hops", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_terminal_score <= 1.0:
        parser.error("--minimum-terminal-score must be in [0,1]")
    if args.max_terminal_tables < 2:
        parser.error("--max-terminal-tables must be at least 2")
    if not 0.0 <= args.support_weight <= 1.0:
        parser.error("--support-weight must be in [0,1]")
    if args.max_path_hops < 1:
        parser.error("--max-path-hops must be positive")

    graph_rows = read_jsonl(args.full_graph_file, args.limit)
    prediction_rows = read_jsonl(args.prediction_file, args.limit)
    prior_rows = read_jsonl(args.prior_file)
    graphs = index_rows(graph_rows)
    predictions = index_rows(prediction_rows)
    priors = index_rows(prior_rows)
    common = sorted(set(graphs) & set(predictions) & set(priors))
    if len(common) != len(predictions):
        missing = sorted(set(predictions) - set(graphs) | set(predictions) - set(priors))
        raise ValueError(
            f"Full graph/prior alignment lost prediction rows: count={len(missing)} first={missing[:10]}"
        )
    rows = [
        complete_one(
            graphs[index],
            predictions[index],
            priors[index],
            args.minimum_terminal_score,
            args.max_terminal_tables,
            args.support_weight,
            args.max_path_hops,
        )
        for index in common
    ]
    write_jsonl(args.output_file, rows)
    counts = defaultdict(int)
    for row in rows:
        counts[row["status"]] += 1
    summary = {
        "config": vars(args),
        "sample_count": len(rows),
        "status_counts": dict(counts),
        "path_sample_count": sum(bool(row["paths"]) for row in rows),
        "avg_terminal_count": sum(len(row["terminal_tables"]) for row in rows) / len(rows) if rows else 0.0,
        "avg_closure_node_count": sum(row["structural_closure_count"] for row in rows) / len(rows) if rows else 0.0,
        "max_closure_node_count": max((row["structural_closure_count"] for row in rows), default=0),
        "semantic_budget_policy": "The original semantic Top-K is immutable; closure nodes are outside that budget.",
        "gold_leakage": False,
    }
    summary_file = args.summary_file or str(
        Path(args.output_file).with_name(Path(args.output_file).stem + "_summary.json")
    )
    write_json(summary_file, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {args.output_file}")


if __name__ == "__main__":
    main()
