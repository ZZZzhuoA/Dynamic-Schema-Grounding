"""Expose the frozen Stage 13-B RGTA as an explicit typed schema tool.

The tool never modifies LLM hidden states.  A caller supplies typed relational
requests (normally produced by an LLM planner), and RGTA returns ranked schema
IDs, FK edges, operators, and value routes.  A deterministic assembler then
adds owner tables and FK paths while preserving literal surfaces verbatim.
"""

import argparse
import heapq
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stage13b_prepare_typed_trajectories import (  # noqa: E402
    ACTIONS,
    OPERATORS,
    VALUE_ROUTES,
)
from src.modeling.stage13c_static_runtime import (  # noqa: E402
    graph_tensors,
    load_frozen_typed_graph_encoder,
)
from src.training.stage13b_train_typed_ra_decoder import load_cache  # noqa: E402


TOOL_SCHEMA_VERSION = "typed_rgta_schema_tool_v1"


def read_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
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


def record_index(row, fallback=0):
    return int(row.get("metadata", {}).get("record_index", row.get("record_index", fallback)))


def normalize_requests(row):
    """Accept native requests or strip an oracle trajectory to action names only."""
    if "requests" in row:
        source = row.get("plan_source", "external_typed_plan")
        requests = row["requests"]
    elif "teacher_steps" in row:
        source = "oracle_action_skeleton_diagnostic"
        requests = [
            {
                "request_id": f"step_{index}",
                "action": step["action"],
                # Arity belongs to the relational plan, not schema grounding.
                # Keep only target counts; all schema IDs, edge IDs, operators,
                # routes, and literal values remain hidden from the tool.
                "table_cardinality": len(step.get("table_pointer_ids", [])),
                "column_cardinality": len(step.get("column_pointer_ids", [])),
                "join_edge_cardinality": len(step.get("join_edge_targets", [])),
                "operator_cardinality": len(step.get("operator_targets", [])),
                "value_route_cardinality": len(step.get("value_routes", [])),
            }
            for index, step in enumerate(row["teacher_steps"])
        ]
    else:
        raise ValueError("Plan row must contain requests or teacher_steps")
    normalized = []
    for index, request in enumerate(requests):
        action = str(request.get("action") or "").upper()
        if action not in ACTIONS:
            raise ValueError(f"Unknown typed action at request {index}: {action}")
        legacy_cardinality = max(1, int(request.get("cardinality", 1)))
        default_table = legacy_cardinality if action == "SCAN" else 0
        default_column = (
            legacy_cardinality
            if action in {"JOIN", "FILTER", "AGGREGATE", "HAVING_FILTER", "SORT", "PROJECT"}
            else 0
        )
        normalized.append(
            {
                "request_id": str(request.get("request_id", f"step_{index}")),
                "action": action,
                "table_cardinality": max(
                    0, int(request.get("table_cardinality", default_table))
                ),
                "column_cardinality": max(
                    0, int(request.get("column_cardinality", default_column))
                ),
                "join_edge_cardinality": max(
                    0, int(request.get("join_edge_cardinality", 1 if action == "JOIN" else 0))
                ),
                "operator_cardinality": max(
                    0, int(request.get("operator_cardinality", 1 if action in {
                        "JOIN", "FILTER", "AGGREGATE", "HAVING_FILTER",
                        "SORT", "LIMIT", "PROJECT",
                    } else 0))
                ),
                "value_route_cardinality": max(
                    0, int(request.get("value_route_cardinality", 1 if action in {
                        "FILTER", "HAVING_FILTER", "LIMIT"
                    } else 0))
                ),
                "value_surface": request.get("value_surface"),
                "requested_operator": request.get("requested_operator"),
                "role": request.get("role"),
            }
        )
    return source, normalized


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def ranked_schema_candidates(logits, nodes, expected_type, top_k, torch):
    eligible = [
        index for index, node in enumerate(nodes)
        if node.get("type") == expected_type
    ]
    if not eligible or top_k <= 0:
        return []
    count = min(int(top_k), len(eligible))
    index_tensor = torch.tensor(eligible, dtype=torch.long, device=logits.device)
    selected = index_tensor[torch.topk(logits[index_tensor], count).indices].tolist()
    return [
        {
            "schema_id": int(nodes[index].get("id", index)),
            "node_index": int(index),
            "type": nodes[index].get("type"),
            "name": nodes[index].get("name"),
            "table": nodes[index].get("table"),
            "rank": rank,
            "logit": float(logits[index].detach().float().cpu()),
            "confidence": sigmoid(float(logits[index].detach().float().cpu())),
            "confidence_status": "uncalibrated_sigmoid",
        }
        for rank, index in enumerate(selected, start=1)
    ]


def ranked_vocabulary(logits, vocabulary, top_k, torch):
    if logits.numel() == 0 or top_k <= 0:
        return []
    count = min(int(top_k), logits.numel())
    selected = torch.topk(logits, count).indices.tolist()
    return [
        {
            "name": vocabulary[index],
            "rank": rank,
            "logit": float(logits[index].detach().float().cpu()),
            "confidence": sigmoid(float(logits[index].detach().float().cpu())),
            "confidence_status": "uncalibrated_sigmoid",
        }
        for rank, index in enumerate(selected, start=1)
    ]


def fk_pairs(inputs):
    pairs, seen = [], set()
    for edge in inputs.get("schema_edges", []):
        if edge.get("type") not in {"foreign_key_forward", "foreign_key_backward"}:
            continue
        left, right = int(edge["src"]), int(edge["dst"])
        key = tuple(sorted((left, right)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def ranked_join_edges(logits, pairs, nodes_by_id, top_k, torch):
    if logits.numel() == 0 or top_k <= 0:
        return []
    count = min(int(top_k), logits.numel())
    selected = torch.topk(logits, count).indices.tolist()
    result = []
    for rank, index in enumerate(selected, start=1):
        left, right = pairs[index]
        left_node, right_node = nodes_by_id[left], nodes_by_id[right]
        raw = float(logits[index].detach().float().cpu())
        result.append(
            {
                "edge_index": int(index),
                "left_schema_id": left,
                "right_schema_id": right,
                "left_name": left_node.get("name"),
                "right_name": right_node.get("name"),
                "left_table": left_node.get("table"),
                "right_table": right_node.get("table"),
                "rank": rank,
                "logit": raw,
                "confidence": sigmoid(raw),
                "confidence_status": "uncalibrated_sigmoid",
            }
        )
    return result


def owner_maps(nodes):
    table_id_by_name = {
        node.get("name"): int(node.get("id", index))
        for index, node in enumerate(nodes)
        if node.get("type") == "table"
    }
    owner_by_column = {}
    for index, node in enumerate(nodes):
        if node.get("type") != "column":
            continue
        table_name = node.get("table")
        if table_name in table_id_by_name:
            owner_by_column[int(node.get("id", index))] = table_id_by_name[table_name]
    return table_id_by_name, owner_by_column


def table_fk_graph(nodes_by_id, pairs, edge_scores):
    adjacency = defaultdict(list)
    for left, right in pairs:
        left_node, right_node = nodes_by_id.get(left), nodes_by_id.get(right)
        if not left_node or not right_node:
            continue
        left_table, right_table = left_node.get("table"), right_node.get("table")
        if not left_table or not right_table or left_table == right_table:
            continue
        confidence = max(float(edge_scores.get(tuple(sorted((left, right))), 0.0)), 0.0)
        payload = {
            "left_schema_id": left,
            "right_schema_id": right,
            "left_name": left_node.get("name"),
            "right_name": right_node.get("name"),
            "left_table": left_table,
            "right_table": right_table,
            "confidence": confidence,
        }
        cost = max(0.05, 1.0 - 0.5 * confidence)
        adjacency[left_table].append((right_table, cost, payload))
        adjacency[right_table].append((left_table, cost, payload))
    return adjacency


def shortest_path(adjacency, source, target):
    counter = itertools.count()
    queue = [(0.0, next(counter), source, [])]
    best = {source: 0.0}
    while queue:
        cost, _, table, path = heapq.heappop(queue)
        if table == target:
            return cost, path
        if cost > best.get(table, float("inf")):
            continue
        for neighbor, edge_cost, payload in adjacency.get(table, []):
            new_cost = cost + edge_cost
            if new_cost < best.get(neighbor, float("inf")):
                best[neighbor] = new_cost
                heapq.heappush(
                    queue, (new_cost, next(counter), neighbor, path + [payload])
                )
    return float("inf"), []


def connect_terminals(adjacency, terminals):
    terminals = sorted(set(value for value in terminals if value))
    if len(terminals) < 2:
        return [], []
    closure = []
    for index, left in enumerate(terminals):
        for right in terminals[index + 1 :]:
            cost, path = shortest_path(adjacency, left, right)
            if path:
                closure.append((cost, left, right, path))
    closure.sort(key=lambda item: item[0])
    parent = {item: item for item in terminals}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    paths = []
    for cost, left, right, path in closure:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        parent[right_root] = left_root
        paths.append({"terminals": [left, right], "cost": cost, "edges": path})
    roots = sorted({find(item) for item in terminals})
    return paths, roots


def assemble_tool_result(nodes, schema_edges, steps, max_schema_items=30):
    """Choose one typed assignment per request, then add owner/FK closure."""
    nodes_by_id = {
        int(node.get("id", index)): node for index, node in enumerate(nodes)
    }
    table_id_by_name, owner_by_column = owner_maps(nodes)
    selected = {}
    terminals = set()
    edge_scores = {}

    def keep(candidate, source):
        if not candidate:
            return
        schema_id = int(candidate["schema_id"])
        current = selected.get(schema_id)
        row = {**candidate, "sources": sorted(set((current or {}).get("sources", []) + [source]))}
        if current is None or float(candidate["confidence"]) > float(current["confidence"]):
            selected[schema_id] = row
        else:
            current["sources"] = row["sources"]

    for step in steps:
        request = step["request"]
        if request["table_cardinality"]:
            for candidate in step["table_candidates"][: request["table_cardinality"]]:
                keep(candidate, request["request_id"])
                terminals.add(candidate.get("name"))
        if request["column_cardinality"]:
            for candidate in step["column_candidates"][: request["column_cardinality"]]:
                keep(candidate, request["request_id"])
                terminals.add(candidate.get("table"))
        for edge in step.get("join_edge_candidates", []):
            key = tuple(sorted((int(edge["left_schema_id"]), int(edge["right_schema_id"]))))
            edge_scores[key] = max(edge_scores.get(key, 0.0), float(edge["confidence"]))

    for schema_id in list(selected):
        owner_id = owner_by_column.get(schema_id)
        if owner_id is None or owner_id in selected:
            continue
        owner = nodes_by_id[owner_id]
        selected[owner_id] = {
            "schema_id": owner_id,
            "type": "table",
            "name": owner.get("name"),
            "table": None,
            "rank": None,
            "logit": None,
            "confidence": float(selected[schema_id]["confidence"]),
            "confidence_status": "owner_closure",
            "sources": ["owner_closure"],
        }
        terminals.add(owner.get("name"))

    pairs = []
    seen = set()
    for edge in schema_edges:
        if edge.get("type") not in {"foreign_key_forward", "foreign_key_backward"}:
            continue
        key = tuple(sorted((int(edge["src"]), int(edge["dst"]))))
        if key not in seen:
            seen.add(key); pairs.append(key)
    adjacency = table_fk_graph(nodes_by_id, pairs, edge_scores)
    paths, disconnected_components = connect_terminals(adjacency, terminals)
    for path in paths:
        for edge in path["edges"]:
            for table_name in (edge["left_table"], edge["right_table"]):
                table_id = table_id_by_name.get(table_name)
                if table_id is not None and table_id not in selected:
                    node = nodes_by_id[table_id]
                    selected[table_id] = {
                        "schema_id": table_id, "type": "table", "name": node.get("name"),
                        "table": None, "rank": None, "logit": None,
                        "confidence": float(edge["confidence"]),
                        "confidence_status": "join_path_closure", "sources": ["join_path"],
                    }
            for endpoint in (edge["left_schema_id"], edge["right_schema_id"]):
                if endpoint not in selected:
                    node = nodes_by_id[endpoint]
                    selected[endpoint] = {
                        "schema_id": endpoint, "type": "column", "name": node.get("name"),
                        "table": node.get("table"), "rank": None, "logit": None,
                        "confidence": float(edge["confidence"]),
                        "confidence_status": "join_path_closure", "sources": ["join_path"],
                    }

    ranked = sorted(
        selected.values(),
        key=lambda item: (
            item.get("confidence_status") in {"owner_closure", "join_path_closure"},
            float(item.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    return {
        "selected_schema": ranked,
        "join_paths": paths,
        "terminal_tables": sorted(terminals),
        "connected": len(disconnected_components) <= 1,
        "disconnected_components": disconnected_components,
        "selected_count": len(ranked),
        "max_schema_items": int(max_schema_items),
        "budget_feasible": len(ranked) <= int(max_schema_items),
        "literal_surfaces": [
            step["request"]["value_surface"]
            for step in steps if step["request"].get("value_surface") is not None
        ],
    }


def infer_record(model, tensors, requests, args, torch):
    dense, query, node_types, edge_index, edge_type, nodes = tensors
    inputs = args["inputs"]
    pairs = fk_pairs(inputs)
    join_index = (
        torch.tensor(pairs, dtype=torch.long, device=dense.device).t().contiguous()
        if pairs else torch.empty((2, 0), dtype=torch.long, device=dense.device)
    )
    nodes_by_id = {
        int(node.get("id", index)): node for index, node in enumerate(nodes)
    }
    base_nodes, query_state, state = model.initialize(dense, query, node_types)
    steps = []
    with torch.no_grad():
        for request in requests:
            output = model.step(
                base_nodes, query_state, state, node_types, edge_index, edge_type,
                join_index, forced_action=request["action"],
                inference_table_count=request["table_cardinality"],
                inference_column_count=request["column_cardinality"],
                inference_value_route_count=request["value_route_cardinality"],
                inference_operator_count=request["operator_cardinality"],
            )
            action_probs = torch.softmax(output["action_logits"].float(), dim=-1)
            forced_id = ACTIONS.index(request["action"])
            step = {
                "request": request,
                "model_action": ACTIONS[int(action_probs.argmax())],
                "forced_action_probability": float(action_probs[forced_id].cpu()),
                "table_candidates": ranked_schema_candidates(
                    output["table_logits"], nodes, "table", args["table_top_k"], torch
                ),
                "column_candidates": ranked_schema_candidates(
                    output["column_logits"], nodes, "column", args["column_top_k"], torch
                ),
                "join_edge_candidates": ranked_join_edges(
                    output["join_edge_logits"], pairs, nodes_by_id, args["join_top_k"], torch
                ),
                "operator_candidates": ranked_vocabulary(
                    output["operator_logits"], OPERATORS, args["operator_top_k"], torch
                ),
                "value_route_candidates": ranked_vocabulary(
                    output["value_route_logits"], VALUE_ROUTES, args["value_route_top_k"], torch
                ),
            }
            steps.append(step)
            state = output["controller_state"]
    assembly = assemble_tool_result(
        nodes, inputs.get("schema_edges", []), steps, args["max_schema_items"]
    )
    return steps, assembly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-checkpoint", required=True)
    parser.add_argument("--graph-summary", required=True)
    parser.add_argument("--graph-file", required=True)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--table-top-k", type=int, default=3)
    parser.add_argument("--column-top-k", type=int, default=5)
    parser.add_argument("--join-top-k", type=int, default=3)
    parser.add_argument("--operator-top-k", type=int, default=3)
    parser.add_argument("--value-route-top-k", type=int, default=2)
    parser.add_argument("--max-schema-items", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    cli = parser.parse_args()

    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Stage 14 requires numpy and PyTorch") from exc

    device = torch.device(cli.device if cli.device != "cuda" or torch.cuda.is_available() else "cpu")
    graphs = read_jsonl(cli.graph_file)
    graph_by_index = {record_index(row, index): row for index, row in enumerate(graphs)}
    plans = read_jsonl(cli.plan_file, cli.limit)
    cache = load_cache(cli.embedding_cache_dir, cli.split, np)
    model, relation_to_id, summary = load_frozen_typed_graph_encoder(
        cli.graph_summary, cli.graph_checkpoint, cache["dense_dim"], device
    )
    outputs = []
    for fallback, plan in enumerate(plans):
        index = record_index(plan, fallback)
        if index not in graph_by_index:
            raise KeyError(f"No graph record for plan record_index={index}")
        graph = graph_by_index[index]
        source, requests = normalize_requests(plan)
        inputs = graph.get("inference_inputs", graph)
        tensors = graph_tensors(graph, cache, relation_to_id, device)
        runtime = {
            "inputs": inputs,
            "table_top_k": cli.table_top_k,
            "column_top_k": cli.column_top_k,
            "join_top_k": cli.join_top_k,
            "operator_top_k": cli.operator_top_k,
            "value_route_top_k": cli.value_route_top_k,
            "max_schema_items": cli.max_schema_items,
        }
        steps, assembly = infer_record(model, tensors, requests, runtime, torch)
        outputs.append(
            {
                "tool_schema_version": TOOL_SCHEMA_VERSION,
                "record_index": index,
                "question_id": inputs.get("question_id", graph.get("question_id")),
                "db_id": inputs.get("db_id", graph.get("db_id")),
                "question": inputs.get("question"),
                "evidence": inputs.get("evidence"),
                "plan_source": source,
                "requests": requests,
                "tool_steps": steps,
                "assembly": assembly,
                "llm_tool_payload": {
                    "schema": [
                        {
                            "id": item["schema_id"],
                            "type": item["type"],
                            "name": item["name"],
                            "confidence": item["confidence"],
                        }
                        for item in assembly["selected_schema"]
                    ],
                    "join_paths": assembly["join_paths"],
                    "literal_surfaces": assembly["literal_surfaces"],
                },
            }
        )
    write_jsonl(cli.output_file, outputs)
    summary_output = {
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "sample_count": len(outputs),
        "plan_source_counts": dict(
            (source, sum(row["plan_source"] == source for row in outputs))
            for source in sorted({row["plan_source"] for row in outputs})
        ),
        "connected_rate": (
            sum(row["assembly"]["connected"] for row in outputs) / len(outputs)
            if outputs else 0.0
        ),
        "budget_feasible_rate": (
            sum(row["assembly"]["budget_feasible"] for row in outputs) / len(outputs)
            if outputs else 0.0
        ),
        "config": vars(cli),
        "graph_model_best_epoch": summary.get("best_epoch"),
    }
    summary_path = Path(cli.output_file).with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary_output, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {cli.output_file}")


if __name__ == "__main__":
    main()
