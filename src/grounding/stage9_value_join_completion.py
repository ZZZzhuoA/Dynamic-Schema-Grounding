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

from src.grounding.value_index import ValueIndex  # noqa: E402
from src.training.stage5_train_dsg_grounder import write_json, write_jsonl  # noqa: E402
from src.training.stage5g_train_clause_grounder import (  # noqa: E402
    assemble_predictions,
    evaluate_assembled,
    parse_budget_text,
)
from src.training.stage5j_train_relation_grounder import load_aligned_records  # noqa: E402


TERMINAL_RELATION_WEIGHTS = {
    "OUTPUT_TARGET": 1.0,
    "ENTITY_NAME": 0.9,
    "METRIC_TARGET": 1.0,
    "PREDICATE_COLUMN": 0.9,
    "VALUE_ANCHOR": 0.8,
    "TEMPORAL_FILTER": 0.7,
    "ORDER_KEY": 0.7,
    "GROUP_KEY": 0.7,
    "FORMULA_COMPONENT": 0.8,
}


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


def group_relation_predictions(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[int(row["record_index"])][row.get("relation_type") or row.get("clause")] = row
    return grouped


def node_indexes(graph_example):
    nodes = graph_example["inference_inputs"].get("schema_nodes", [])
    by_id = {int(node["id"]): node for node in nodes}
    table_to_id = {
        node["name"]: int(node["id"]) for node in nodes if node.get("type") == "table"
    }
    return nodes, by_id, table_to_id


def owner_table(node):
    if not node:
        return None
    return node.get("name") if node.get("type") == "table" else node.get("table")


def relation_rank_support(relation_rows, by_id, top_per_relation):
    table_scores = defaultdict(float)
    table_relations = defaultdict(set)
    for relation, weight in TERMINAL_RELATION_WEIGHTS.items():
        row = relation_rows.get(relation)
        if not row:
            continue
        top_key = next((key for key in row if key.startswith("top_")), None)
        if not top_key:
            continue
        for rank, candidate in enumerate(row.get(top_key, [])[:top_per_relation], start=1):
            table = owner_table(by_id.get(int(candidate["id"])))
            if not table:
                continue
            table_scores[table] += weight / rank
            table_relations[table].add(relation)
    return table_scores, table_relations


def select_terminal_tables(
    relation_rows,
    by_id,
    value_matches,
    top_per_relation,
    max_terminals,
    min_ratio,
    value_terminal_weight,
):
    scores, relations = relation_rank_support(relation_rows, by_id, top_per_relation)
    for match in value_matches:
        table = owner_table(by_id.get(int(match["schema_item_id"])))
        if table:
            scores[table] += value_terminal_weight * float(match["score"])
            relations[table].add("VALUE_INDEX")
    ranked = sorted(scores, key=lambda table: scores[table], reverse=True)
    if not ranked:
        return [], scores, relations
    threshold = scores[ranked[0]] * min_ratio
    terminals = [table for table in ranked if scores[table] >= threshold][:max_terminals]
    return terminals, scores, relations


def contextualize_value_matches(value_matches, relation_rows, by_id, args):
    """Calibrate value evidence with ambiguity, table, and relation support.

    A raw value match is not itself a schema decision.  The same value may occur in
    several columns, so each candidate receives a confidence derived from its value
    likelihood, cross-column ambiguity, table prior, relation prior, and its margin
    over competing columns that contain the same value.  Downstream code uses the
    resulting three-way gate instead of injecting every value hit.
    """
    if not value_matches:
        return []
    table_scores, _ = relation_rank_support(
        relation_rows, by_id, args.terminal_top_per_relation
    )
    max_table_score = max(table_scores.values(), default=1.0)
    column_support = defaultdict(float)
    for relation in ["PREDICATE_COLUMN", "VALUE_ANCHOR"]:
        row = relation_rows.get(relation)
        if not row:
            continue
        top_key = prediction_top_key(row)
        if not top_key:
            continue
        for rank, candidate in enumerate(row.get(top_key, []), start=1):
            item_id = int(candidate["id"])
            column_support[item_id] = max(column_support[item_id], 1.0 / rank)
    contextualized = []
    for match in value_matches:
        match = dict(match)
        item_id = int(match["schema_item_id"])
        table = owner_table(by_id.get(item_id))
        table_context = table_scores.get(table, 0.0) / max(max_table_score, 1e-8)
        relation_context = column_support.get(item_id, 0.0)
        raw_score = float(match["score"])
        match["raw_value_score"] = raw_score
        match["table_context_score"] = table_context
        match["relation_context_score"] = relation_context
        match["contextual_score"] = (
            raw_score
            + args.value_table_context_weight * table_context
            + args.value_relation_context_weight * relation_context
        )
        match["score"] = match["contextual_score"]
        contextualized.append(match)

    anchor_to_candidates = defaultdict(list)
    for match in contextualized:
        anchors = {
            item.get("normalized_value")
            for item in match.get("matches", [])
            if item.get("normalized_value")
        }
        for anchor in anchors:
            anchor_to_candidates[anchor].append(match)

    ambiguity_weight = getattr(args, "value_ambiguity_weight", 0.55)
    support_weight = getattr(args, "value_support_weight", 0.30)
    margin_weight = getattr(args, "value_margin_weight", 0.15)
    injection_threshold = getattr(args, "value_injection_threshold", 0.55)
    rerank_threshold = getattr(args, "value_rerank_threshold", 0.40)
    terminal_threshold = getattr(args, "value_terminal_threshold", 0.65)
    terminal_min_support = getattr(args, "value_terminal_min_support", 0.20)
    weight_total = ambiguity_weight + support_weight + margin_weight
    if weight_total <= 0:
        raise ValueError("Value confidence weights must sum to a positive value")
    for match in contextualized:
        candidate_anchors = [
            (item.get("normalized_value"), float(item.get("score", 0.0)))
            for item in match.get("matches", [])
            if item.get("normalized_value")
        ]
        best_anchor = None
        best_confidence = -1.0
        best_ambiguity_count = 1
        best_margin = 0.0
        semantic_support = (
            0.6 * float(match["table_context_score"])
            + 0.4 * float(match["relation_context_score"])
        )
        for anchor, anchor_value_score in candidate_anchors:
            competitors = anchor_to_candidates.get(anchor, [match])
            ranked_scores = sorted(
                (float(item["contextual_score"]), int(item["schema_item_id"]))
                for item in competitors
            )[::-1]
            own_score = float(match["contextual_score"])
            best_score = ranked_scores[0][0]
            second_score = ranked_scores[1][0] if len(ranked_scores) > 1 else 0.0
            margin = (
                max(0.0, own_score - second_score) / max(abs(best_score), 1e-8)
                if own_score >= best_score - 1e-12
                else 0.0
            )
            ambiguity_score = 1.0 / math.sqrt(max(len(competitors), 1))
            anchor_confidence = anchor_value_score * min(
                1.0,
                (
                    ambiguity_weight * ambiguity_score
                    + support_weight * semantic_support
                    + margin_weight * margin
                )
                / weight_total,
            )
            if anchor_confidence > best_confidence:
                best_anchor = anchor
                best_confidence = anchor_confidence
                best_ambiguity_count = len(competitors)
                best_margin = margin

        ambiguity_score = 1.0 / math.sqrt(max(best_ambiguity_count, 1))
        confidence = max(0.0, best_confidence)
        if confidence >= injection_threshold:
            gate = "inject"
        elif confidence >= rerank_threshold:
            gate = "rerank"
        else:
            gate = "reject"
        match["value_anchor"] = best_anchor
        match["value_ambiguity_count"] = best_ambiguity_count
        match["value_ambiguity_score"] = ambiguity_score
        match["value_margin_score"] = best_margin
        match["value_semantic_support"] = semantic_support
        match["value_confidence"] = confidence
        match["value_gate"] = gate
        match["eligible_for_injection"] = gate == "inject"
        match["eligible_for_terminal"] = (
            confidence >= terminal_threshold
            and semantic_support >= terminal_min_support
        )
        match["score"] = (
            match["contextual_score"]
            if getattr(args, "value_fusion_mode", "gated") == "direct"
            else confidence
        )
    contextualized.sort(key=lambda item: item["score"], reverse=True)
    return contextualized


def build_table_fk_graph(graph_example):
    _, by_id, table_to_id = node_indexes(graph_example)
    adjacency = defaultdict(list)
    seen = set()
    for edge in graph_example["inference_inputs"].get("schema_edges", []):
        if edge.get("type") not in {"foreign_key_forward", "foreign_key_backward"}:
            continue
        left_id = int(edge["src"])
        right_id = int(edge["dst"])
        left = by_id.get(left_id)
        right = by_id.get(right_id)
        left_table = owner_table(left)
        right_table = owner_table(right)
        if not left_table or not right_table or left_table == right_table:
            continue
        key = (frozenset((left_table, right_table)), frozenset((left_id, right_id)))
        if key in seen:
            continue
        seen.add(key)
        payload = {
            "left_table": left_table,
            "right_table": right_table,
            "left_endpoint": left_id,
            "right_endpoint": right_id,
        }
        adjacency[left_table].append((right_table, payload))
        adjacency[right_table].append((left_table, payload))
    return adjacency, by_id, table_to_id


def join_node_support(relation_rows):
    row = relation_rows.get("JOIN_BRIDGE")
    if not row:
        return {}
    top_key = next((key for key in row if key.startswith("top_")), None)
    if not top_key:
        return {}
    return {
        int(candidate["id"]): 1.0 / rank
        for rank, candidate in enumerate(row.get(top_key, []), start=1)
    }


def shortest_table_path(adjacency, source, target, node_support, support_weight):
    counter = itertools.count()
    queue = [(0.0, next(counter), source, [])]
    best = {source: 0.0}
    while queue:
        cost, _, table, path = heapq.heappop(queue)
        if table == target:
            return cost, path
        if cost > best.get(table, float("inf")):
            continue
        for neighbor, edge in adjacency.get(table, []):
            support = max(
                node_support.get(int(edge["left_endpoint"]), 0.0),
                node_support.get(int(edge["right_endpoint"]), 0.0),
            )
            edge_cost = max(0.25, 1.0 - support_weight * support)
            new_cost = cost + edge_cost
            if new_cost < best.get(neighbor, float("inf")):
                best[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, next(counter), neighbor, path + [edge]))
    return float("inf"), []


def metric_closure_mst_paths(adjacency, terminals, node_support, support_weight):
    closure = []
    for left_index, left in enumerate(terminals):
        for right in terminals[left_index + 1 :]:
            cost, path = shortest_table_path(
                adjacency, left, right, node_support, support_weight
            )
            if path:
                closure.append((cost, left, right, path))
    closure.sort(key=lambda item: item[0])
    parent = {terminal: terminal for terminal in terminals}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    paths = []
    for cost, left, right, path in closure:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            continue
        parent[right_root] = left_root
        paths.append({"terminals": [left, right], "cost": cost, "edges": path})
        if len(paths) == len(terminals) - 1:
            break
    return paths


def complete_join_path(graph_example, relation_rows, value_matches, args):
    adjacency, by_id, table_to_id = build_table_fk_graph(graph_example)
    terminals, terminal_scores, terminal_relations = select_terminal_tables(
        relation_rows,
        by_id,
        value_matches,
        args.terminal_top_per_relation,
        args.max_terminal_tables,
        args.terminal_min_ratio,
        args.value_terminal_weight,
    )
    if len(terminals) < 2:
        return [], {
            "terminal_tables": terminals,
            "terminal_scores": dict(terminal_scores),
            "terminal_relations": {key: sorted(value) for key, value in terminal_relations.items()},
            "paths": [],
        }
    support = join_node_support(relation_rows)
    paths = metric_closure_mst_paths(
        adjacency,
        terminals,
        support,
        args.join_support_weight,
    )
    candidates = defaultdict(float)
    path_debug = []
    for path_record in paths:
        path_length = len(path_record["edges"])
        path_score = 1.0 / max(path_length, 1)
        edge_debug = []
        for edge in path_record["edges"]:
            for table in [edge["left_table"], edge["right_table"]]:
                if table in table_to_id:
                    candidates[table_to_id[table]] = max(candidates[table_to_id[table]], path_score)
            for endpoint in [edge["left_endpoint"], edge["right_endpoint"]]:
                candidates[int(endpoint)] = max(candidates[int(endpoint)], path_score)
            edge_debug.append(edge)
        path_debug.append({**path_record, "path_score": path_score, "edges": edge_debug})
    ranked = [
        {"schema_item_id": item_id, "score": score}
        for item_id, score in sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    ]
    return ranked, {
        "terminal_tables": terminals,
        "terminal_scores": dict(sorted(terminal_scores.items(), key=lambda item: item[1], reverse=True)),
        "terminal_relations": {key: sorted(value) for key, value in terminal_relations.items()},
        "paths": path_debug,
    }


def prediction_top_key(row):
    return next((key for key in row if key.startswith("top_")), None)


def evidence_row(node, evidence_score, source, max_base_score):
    return {
        "id": int(node["id"]),
        "type": node["type"],
        "name": node["name"],
        "score": float(max_base_score + evidence_score),
        "evidence_source": source,
        f"{source}_score": float(evidence_score),
    }


def inject_candidates(row, candidates, by_id, budget, protected_prefix, source):
    if not row or not candidates or budget <= 0:
        return row
    row = dict(row)
    top_key = prediction_top_key(row)
    if not top_key:
        return row
    baseline = [dict(item) for item in row.get(top_key, [])]
    max_base_score = max([float(item.get("score", 0.0)) for item in baseline], default=0.0)
    protected = baseline[:protected_prefix]
    protected_ids = {int(item["id"]) for item in protected}
    injected = []
    for candidate in candidates:
        item_id = int(candidate["schema_item_id"])
        if item_id in protected_ids or item_id not in by_id:
            continue
        injected.append(
            evidence_row(by_id[item_id], float(candidate["score"]), source, max_base_score)
        )
        if len(injected) >= budget:
            break
    injected_ids = {int(item["id"]) for item in injected}
    remainder = [
        item for item in baseline[protected_prefix:] if int(item["id"]) not in injected_ids
    ]
    row[top_key] = (protected + injected + remainder)[: len(baseline)]
    return row


def fuse_gated_value_candidates(row, candidates, by_id, budget, protected_prefix):
    """Fuse value evidence without allowing weak evidence to introduce new IDs."""
    if not row or not candidates or budget <= 0:
        return row
    row = dict(row)
    top_key = prediction_top_key(row)
    if not top_key:
        return row
    baseline = [dict(item) for item in row.get(top_key, [])]
    baseline_ids = {int(item["id"]) for item in baseline}
    eligible = []
    for candidate in candidates:
        item_id = int(candidate["schema_item_id"])
        gate = candidate.get("value_gate")
        if item_id not in by_id or gate == "reject":
            continue
        if item_id not in baseline_ids and gate != "inject":
            continue
        eligible.append(candidate)
    return inject_candidates(
        row,
        eligible,
        by_id,
        budget,
        protected_prefix,
        "value_index_gated",
    )


def enhance_predictions(prediction_rows, aligned, value_index, args):
    grouped = group_relation_predictions(prediction_rows)
    enhanced = []
    debug_rows = []
    for item in aligned:
        record_index = int(item["record_index"])
        graph_example = item["graph_example"]
        inputs = graph_example["inference_inputs"]
        _, by_id, _ = node_indexes(graph_example)
        relation_rows = grouped.get(record_index, {})
        value_matches = []
        if args.enable_value_index:
            value_matches = value_index.query(
                inputs.get("db_id"),
                inputs.get("question"),
                inputs.get("evidence"),
                max_candidates=args.value_max_candidates,
                max_matches_per_column=args.max_matches_per_column,
                min_score=args.min_value_score,
            )
            value_matches = contextualize_value_matches(
                value_matches, relation_rows, by_id, args
            )
        value_injection_candidates = [
            match
            for match in value_matches
            if args.value_fusion_mode == "direct"
            or match.get("value_gate") in {"inject", "rerank"}
        ]
        value_terminal_matches = [
            match
            for match in value_matches
            if args.value_fusion_mode == "direct"
            or match.get("eligible_for_terminal", False)
        ]
        path_candidates = []
        path_debug = {"terminal_tables": [], "paths": []}
        if args.enable_join_path:
            path_candidates, path_debug = complete_join_path(
                graph_example,
                relation_rows,
                value_terminal_matches,
                args,
            )
        for relation, row in relation_rows.items():
            updated = row
            if args.enable_value_index and relation in {"PREDICATE_COLUMN", "VALUE_ANCHOR"}:
                if args.value_fusion_mode == "direct":
                    updated = inject_candidates(
                        updated,
                        value_matches,
                        by_id,
                        args.value_injection_budget,
                        args.protected_relation_prefix,
                        "value_index",
                    )
                else:
                    updated = fuse_gated_value_candidates(
                        updated,
                        value_matches,
                        by_id,
                        args.value_injection_budget,
                        args.protected_relation_prefix,
                    )
            if args.enable_join_path and relation == "JOIN_BRIDGE":
                updated = inject_candidates(
                    updated,
                    path_candidates,
                    by_id,
                    args.join_injection_budget,
                    args.protected_relation_prefix,
                    "join_path",
                )
            enhanced.append(updated)
        debug_rows.append(
            {
                "record_index": record_index,
                "db_id": inputs.get("db_id"),
                "question": inputs.get("question"),
                "value_matches": value_matches,
                "value_injection_candidates": value_injection_candidates,
                "value_terminal_matches": value_terminal_matches,
                "join_path_candidates": path_candidates,
                "join_path": path_debug,
            }
        )
    return enhanced, debug_rows


def evaluate_evidence_channels(debug_rows, baseline_assembled, enhanced_assembled, aligned, top_k):
    baseline_by_index = {int(row["record_index"]): row for row in baseline_assembled}
    enhanced_by_index = {int(row["record_index"]): row for row in enhanced_assembled}
    aligned_by_index = {int(item["record_index"]): item for item in aligned}
    stats = {
        "value_match_sample_count": 0,
        "value_candidate_count": 0,
        "value_gold_candidate_count": 0,
        "value_recoverable_missing_sample_count": 0,
        "join_path_sample_count": 0,
        "join_path_candidate_count": 0,
        "join_path_gold_candidate_count": 0,
        "join_path_recoverable_missing_sample_count": 0,
        "gated_value_candidate_count": 0,
        "gated_value_gold_candidate_count": 0,
        "value_terminal_candidate_count": 0,
        "value_terminal_gold_candidate_count": 0,
        "value_gate_inject_count": 0,
        "value_gate_rerank_count": 0,
        "value_gate_reject_count": 0,
        "complete_coverage_gained_samples": 0,
        "complete_coverage_lost_samples": 0,
    }
    for debug in debug_rows:
        index = int(debug["record_index"])
        baseline = baseline_by_index[index]
        enhanced = enhanced_by_index[index]
        relation_record = aligned_by_index[index]["clause_record"]
        gold = set(int(item_id) for item_id in relation_record.get("whole_sql_labels", []))
        baseline_ids = {
            int(item["id"]) for item in baseline.get(f"top_{top_k}", [])
        }
        missing = gold - baseline_ids
        value_ids = {
            int(item["schema_item_id"]) for item in debug.get("value_matches", [])
        }
        gated_value_ids = {
            int(item["schema_item_id"])
            for item in debug.get("value_injection_candidates", [])
        }
        value_terminal_ids = {
            int(item["schema_item_id"])
            for item in debug.get("value_terminal_matches", [])
        }
        join_ids = {
            int(item["schema_item_id"])
            for item in debug.get("join_path_candidates", [])
        }
        if value_ids:
            stats["value_match_sample_count"] += 1
            stats["value_candidate_count"] += len(value_ids)
            stats["value_gold_candidate_count"] += len(value_ids & gold)
            stats["value_recoverable_missing_sample_count"] += int(bool(value_ids & missing))
        if join_ids:
            stats["join_path_sample_count"] += 1
            stats["join_path_candidate_count"] += len(join_ids)
            stats["join_path_gold_candidate_count"] += len(join_ids & gold)
            stats["join_path_recoverable_missing_sample_count"] += int(bool(join_ids & missing))
        stats["gated_value_candidate_count"] += len(gated_value_ids)
        stats["gated_value_gold_candidate_count"] += len(gated_value_ids & gold)
        stats["value_terminal_candidate_count"] += len(value_terminal_ids)
        stats["value_terminal_gold_candidate_count"] += len(value_terminal_ids & gold)
        for match in debug.get("value_matches", []):
            gate_key = f"value_gate_{match.get('value_gate', 'reject')}_count"
            if gate_key in stats:
                stats[gate_key] += 1
        recall_key = f"assembled_recall@{top_k}"
        baseline_complete = (baseline.get(recall_key) or 0.0) >= 1.0
        enhanced_complete = (enhanced.get(recall_key) or 0.0) >= 1.0
        stats["complete_coverage_gained_samples"] += int(
            enhanced_complete and not baseline_complete
        )
        stats["complete_coverage_lost_samples"] += int(
            baseline_complete and not enhanced_complete
        )
    stats["value_candidate_gold_precision"] = (
        stats["value_gold_candidate_count"] / stats["value_candidate_count"]
        if stats["value_candidate_count"]
        else 0.0
    )
    stats["join_path_candidate_gold_precision"] = (
        stats["join_path_gold_candidate_count"] / stats["join_path_candidate_count"]
        if stats["join_path_candidate_count"]
        else 0.0
    )
    stats["gated_value_candidate_gold_precision"] = (
        stats["gated_value_gold_candidate_count"]
        / stats["gated_value_candidate_count"]
        if stats["gated_value_candidate_count"]
        else 0.0
    )
    stats["value_terminal_candidate_gold_precision"] = (
        stats["value_terminal_gold_candidate_count"]
        / stats["value_terminal_candidate_count"]
        if stats["value_terminal_candidate_count"]
        else 0.0
    )
    return stats


def build_coverage_transition_diagnostics(
    baseline_assembled, enhanced_assembled, aligned, top_k
):
    """Build gold-aware diagnostics after inference; never consumed by the model."""
    baseline_by_index = {int(row["record_index"]): row for row in baseline_assembled}
    enhanced_by_index = {int(row["record_index"]): row for row in enhanced_assembled}
    transitions = []
    for item in aligned:
        index = int(item["record_index"])
        baseline = baseline_by_index[index]
        enhanced = enhanced_by_index[index]
        gold = {
            int(item_id)
            for item_id in item["clause_record"].get("whole_sql_labels", [])
        }
        baseline_ids = {
            int(candidate["id"])
            for candidate in baseline.get(f"top_{top_k}", [])
        }
        enhanced_ids = {
            int(candidate["id"])
            for candidate in enhanced.get(f"top_{top_k}", [])
        }
        baseline_complete = gold.issubset(baseline_ids)
        enhanced_complete = gold.issubset(enhanced_ids)
        if baseline_complete == enhanced_complete:
            continue
        transition = "gained" if enhanced_complete else "lost"
        graph_inputs = item["graph_example"].get("inference_inputs", {})
        transitions.append(
            {
                "record_index": index,
                "db_id": graph_inputs.get("db_id"),
                "question": graph_inputs.get("question"),
                "transition": transition,
                "gold_ids": sorted(gold),
                "recovered_gold_ids": sorted((enhanced_ids - baseline_ids) & gold),
                "evicted_gold_ids": sorted((baseline_ids - enhanced_ids) & gold),
                "added_ids": sorted(enhanced_ids - baseline_ids),
                "removed_ids": sorted(baseline_ids - enhanced_ids),
            }
        )
    return transitions


def evaluate_typed_assembled(assembled_rows, aligned, top_k):
    """Evaluate table and column recall inside the final mixed schema Top-K.

    Samples without a gold item of the requested node type are excluded from that
    type's macro average.  This avoids treating an undefined recall as either zero
    or a free perfect score.
    """
    assembled_by_index = {int(row["record_index"]): row for row in assembled_rows}
    table_recalls = []
    column_recalls = []
    for item in aligned:
        index = int(item["record_index"])
        row = assembled_by_index[index]
        nodes = {
            int(node["id"]): node
            for node in item["graph_example"]
            .get("inference_inputs", {})
            .get("schema_nodes", [])
        }
        gold_ids = {
            int(item_id)
            for item_id in item["clause_record"].get("whole_sql_labels", [])
        }
        selected_ids = {
            int(candidate["id"])
            for candidate in row.get(f"top_{top_k}", [])
        }
        gold_table_ids = {
            item_id
            for item_id in gold_ids
            if nodes.get(item_id, {}).get("type") == "table"
        }
        gold_column_ids = {
            item_id
            for item_id in gold_ids
            if nodes.get(item_id, {}).get("type") == "column"
        }
        if gold_table_ids:
            table_recalls.append(
                len(gold_table_ids & selected_ids) / len(gold_table_ids)
            )
        if gold_column_ids:
            column_recalls.append(
                len(gold_column_ids & selected_ids) / len(gold_column_ids)
            )
    return {
        f"assembled_table_recall@{top_k}": (
            sum(table_recalls) / len(table_recalls) if table_recalls else 0.0
        ),
        f"assembled_column_recall@{top_k}": (
            sum(column_recalls) / len(column_recalls) if column_recalls else 0.0
        ),
        "assembled_table_recall_sample_count": len(table_recalls),
        "assembled_column_recall_sample_count": len(column_recalls),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Augment Stage 8G relation beliefs with indexed database values and "
            "FK-graph metric-closure join-path completion."
        )
    )
    parser.add_argument("--relation-predictions", required=True)
    parser.add_argument("--relation-file", required=True)
    parser.add_argument("--graph-file", required=True)
    parser.add_argument("--value-index", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-top-k", type=int, default=30)
    parser.add_argument(
        "--assembly-budgets",
        type=parse_budget_text,
        default=(
            "OUTPUT_TARGET:7,JOIN_BRIDGE:8,PREDICATE_COLUMN:7,"
            "METRIC_TARGET:4,VALUE_ANCHOR:2,TEMPORAL_FILTER:1,ORDER_KEY:1"
        ),
    )
    parser.add_argument("--enable-value-index", action="store_true")
    parser.add_argument("--enable-join-path", action="store_true")
    parser.add_argument("--value-max-candidates", type=int, default=500)
    parser.add_argument("--max-matches-per-column", type=int, default=3)
    parser.add_argument("--min-value-score", type=float, default=0.8)
    parser.add_argument("--value-injection-budget", type=int, default=3)
    parser.add_argument(
        "--value-fusion-mode",
        choices=["gated", "direct"],
        default="gated",
        help="Use confidence-gated fusion by default; direct reproduces Stage 9.",
    )
    parser.add_argument("--value-table-context-weight", type=float, default=0.35)
    parser.add_argument("--value-relation-context-weight", type=float, default=0.25)
    parser.add_argument("--value-ambiguity-weight", type=float, default=0.55)
    parser.add_argument("--value-support-weight", type=float, default=0.30)
    parser.add_argument("--value-margin-weight", type=float, default=0.15)
    parser.add_argument("--value-rerank-threshold", type=float, default=0.40)
    parser.add_argument("--value-injection-threshold", type=float, default=0.55)
    parser.add_argument("--value-terminal-threshold", type=float, default=0.65)
    parser.add_argument("--value-terminal-min-support", type=float, default=0.20)
    parser.add_argument("--join-injection-budget", type=int, default=8)
    parser.add_argument("--protected-relation-prefix", type=int, default=2)
    parser.add_argument("--terminal-top-per-relation", type=int, default=3)
    parser.add_argument("--max-terminal-tables", type=int, default=4)
    parser.add_argument("--terminal-min-ratio", type=float, default=0.45)
    parser.add_argument("--value-terminal-weight", type=float, default=1.0)
    parser.add_argument("--join-support-weight", type=float, default=0.4)
    args = parser.parse_args()
    if not args.enable_value_index and not args.enable_join_path:
        raise ValueError("Enable at least one of --enable-value-index or --enable-join-path")
    if args.enable_value_index and not args.value_index:
        raise ValueError("--enable-value-index requires --value-index")
    if not (
        0.0 <= args.value_rerank_threshold
        <= args.value_injection_threshold
        <= args.value_terminal_threshold
        <= 1.0
    ):
        raise ValueError(
            "Require 0 <= rerank <= injection <= terminal <= 1 for value gates"
        )
    if not 0.0 <= args.value_terminal_min_support <= 1.0:
        raise ValueError("--value-terminal-min-support must be in [0, 1]")

    aligned = load_aligned_records(
        Path(args.relation_file), Path(args.graph_file), args.limit
    )
    predictions = read_jsonl(Path(args.relation_predictions))
    allowed_indices = {int(item["record_index"]) for item in aligned}
    predictions = [
        row for row in predictions if int(row.get("record_index", -1)) in allowed_indices
    ]
    value_index = ValueIndex(args.value_index) if args.enable_value_index else None
    try:
        enhanced, debug_rows = enhance_predictions(predictions, aligned, value_index, args)
    finally:
        if value_index is not None:
            value_index.close()

    baseline_assembled = assemble_predictions(
        predictions,
        aligned,
        output_top_k=args.output_top_k,
        budgets=args.assembly_budgets,
    )
    enhanced_assembled = assemble_predictions(
        enhanced,
        aligned,
        output_top_k=args.output_top_k,
        budgets=args.assembly_budgets,
    )
    baseline_metrics = evaluate_assembled(baseline_assembled)
    enhanced_metrics = evaluate_assembled(enhanced_assembled)
    baseline_metrics.update(
        evaluate_typed_assembled(
            baseline_assembled,
            aligned,
            args.output_top_k,
        )
    )
    enhanced_metrics.update(
        evaluate_typed_assembled(
            enhanced_assembled,
            aligned,
            args.output_top_k,
        )
    )
    deltas = {
        key: enhanced_metrics[key] - baseline_metrics[key]
        for key in baseline_metrics
        if isinstance(baseline_metrics[key], (int, float))
    }
    evidence_metrics = evaluate_evidence_channels(
        debug_rows,
        baseline_assembled,
        enhanced_assembled,
        aligned,
        args.output_top_k,
    )
    transition_diagnostics = build_coverage_transition_diagnostics(
        baseline_assembled,
        enhanced_assembled,
        aligned,
        args.output_top_k,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "enhanced_relation_predictions.jsonl", enhanced)
    write_jsonl(output_dir / "baseline_assembled_predictions.jsonl", baseline_assembled)
    write_jsonl(output_dir / "enhanced_assembled_predictions.jsonl", enhanced_assembled)
    write_jsonl(output_dir / "evidence_debug.jsonl", debug_rows)
    write_jsonl(
        output_dir / "coverage_transition_diagnostics.jsonl",
        transition_diagnostics,
    )
    summary = {
        "config": vars(args),
        "base_sample_count": len(aligned),
        "relation_prediction_count": len(predictions),
        "baseline_metrics": baseline_metrics,
        "enhanced_metrics": enhanced_metrics,
        "deltas": deltas,
        "evidence_metrics": evidence_metrics,
        "method": {
            "value_index": (
                "Normalized database values are fused into PREDICATE_COLUMN and "
                f"VALUE_ANCHOR with {args.value_fusion_mode} confidence control."
            ),
            "join_path": (
                "Semantic/value terminal tables are connected by a metric-closure MST over "
                "the FK graph; expanded paths inject bridge tables and FK endpoint columns."
            ),
        },
        "generalization_boundary": (
            "Enhancement uses only question, evidence, database contents, schema graph, and "
            "grounder predictions. Gold labels are read only by the evaluator after inference."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
