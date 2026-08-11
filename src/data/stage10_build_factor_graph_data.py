import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5_train_dsg_grounder import write_json, write_jsonl  # noqa: E402
from src.training.stage5g_train_clause_grounder import (  # noqa: E402
    assemble_predictions,
    parse_budget_text,
)
from src.training.stage5j_train_relation_grounder import (  # noqa: E402
    DEFAULT_RELATIONS,
    load_aligned_records,
)


FACTOR_KINDS = ["relation", "value", "join_path"]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prediction_top_key(row):
    keys = [key for key in row if key.startswith("top_")]
    return max(keys, key=lambda key: int(key.split("_", 1)[1])) if keys else None


def group_relation_predictions(rows):
    grouped = defaultdict(dict)
    for row in rows:
        relation = row.get("relation_type") or row.get("clause")
        grouped[int(row["record_index"])][relation] = row
    return grouped


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def calibrated_relation_candidates(row, relation_top_m):
    top_key = prediction_top_key(row)
    candidates = list(row.get(top_key, []))[:relation_top_m] if top_key else []
    if not candidates:
        return []
    scores = [float(item.get("score", 0.0)) for item in candidates]
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    std = math.sqrt(variance) or 1.0
    return [
        {
            **item,
            "id": int(item["id"]),
            "reciprocal_rank": 1.0 / rank,
            "z_probability": sigmoid((float(item.get("score", 0.0)) - mean) / std),
            "rank": rank,
        }
        for rank, item in enumerate(candidates, start=1)
    ]


def path_schema_ids(path_record, nodes, table_to_id):
    result = set()
    for edge in path_record.get("edges", []):
        for key in ["left_endpoint", "right_endpoint"]:
            if edge.get(key) is not None:
                result.add(int(edge[key]))
        for key in ["left_table", "right_table"]:
            table_id = table_to_id.get(edge.get(key))
            if table_id is not None:
                result.add(table_id)
    return {item_id for item_id in result if item_id in nodes}


def build_one_factor_graph(
    item,
    relation_rows,
    baseline_row,
    evidence_debug,
    relation_types,
    args,
):
    graph_example = item["graph_example"]
    inputs = graph_example["inference_inputs"]
    schema_nodes = inputs.get("schema_nodes", [])
    nodes = {int(node["id"]): node for node in schema_nodes}
    positions = {int(node["id"]): position for position, node in enumerate(schema_nodes)}
    table_to_id = {
        node.get("name"): int(node["id"])
        for node in schema_nodes
        if node.get("type") == "table"
    }
    relation_to_id = {name: index for index, name in enumerate(relation_types)}
    candidate = defaultdict(
        lambda: {
            "priority": 0.0,
            "baseline": 0.0,
            "owner_closure": 0.0,
            "relation_source": 0.0,
            "value_source": 0.0,
            "join_source": 0.0,
            "relation_rr": [0.0] * len(relation_types),
            "relation_z": [0.0] * len(relation_types),
            "value_confidence": 0.0,
            "raw_value_score": 0.0,
            "value_ambiguity": 0.0,
            "value_margin": 0.0,
            "value_semantic_support": 0.0,
            "value_inject": 0.0,
            "value_terminal": 0.0,
            "join_score": 0.0,
            "join_path_count": 0.0,
        }
    )
    factor_specs = []

    baseline_key = prediction_top_key(baseline_row)
    baseline_ids = []
    for rank, row in enumerate(baseline_row.get(baseline_key, []), start=1):
        item_id = int(row["id"])
        if item_id not in nodes:
            continue
        baseline_ids.append(item_id)
        candidate[item_id]["baseline"] = 1.0
        candidate[item_id]["priority"] += 2.0 + 1.0 / rank

    for relation, row in relation_rows.items():
        if relation not in relation_to_id:
            continue
        calibrated = calibrated_relation_candidates(row, args.relation_top_m)
        edges = []
        for entry in calibrated:
            item_id = int(entry["id"])
            if item_id not in nodes:
                continue
            relation_id = relation_to_id[relation]
            rr = float(entry["reciprocal_rank"])
            zp = float(entry["z_probability"])
            candidate[item_id]["relation_source"] = 1.0
            candidate[item_id]["relation_rr"][relation_id] = max(
                candidate[item_id]["relation_rr"][relation_id], rr
            )
            candidate[item_id]["relation_z"][relation_id] = max(
                candidate[item_id]["relation_z"][relation_id], zp
            )
            candidate[item_id]["priority"] += rr + 0.5 * zp
            edges.append((item_id, rr))
        if edges:
            factor_specs.append(
                {
                    "key": f"relation:{relation}",
                    "kind": "relation",
                    "relation_id": relation_to_id[relation],
                    "edges": edges,
                }
            )

    value_groups = defaultdict(list)
    for match in (evidence_debug or {}).get("value_matches", []):
        item_id = int(match["schema_item_id"])
        if item_id not in nodes:
            continue
        confidence = float(match.get("value_confidence", match.get("score", 0.0)))
        candidate[item_id]["value_source"] = 1.0
        candidate[item_id]["value_confidence"] = max(
            candidate[item_id]["value_confidence"], confidence
        )
        candidate[item_id]["raw_value_score"] = max(
            candidate[item_id]["raw_value_score"],
            float(match.get("raw_value_score", 0.0)),
        )
        candidate[item_id]["value_ambiguity"] = max(
            candidate[item_id]["value_ambiguity"],
            float(match.get("value_ambiguity_score", 0.0)),
        )
        candidate[item_id]["value_margin"] = max(
            candidate[item_id]["value_margin"],
            float(match.get("value_margin_score", 0.0)),
        )
        candidate[item_id]["value_semantic_support"] = max(
            candidate[item_id]["value_semantic_support"],
            float(match.get("value_semantic_support", 0.0)),
        )
        candidate[item_id]["value_inject"] = max(
            candidate[item_id]["value_inject"],
            float(bool(match.get("eligible_for_injection"))),
        )
        candidate[item_id]["value_terminal"] = max(
            candidate[item_id]["value_terminal"],
            float(bool(match.get("eligible_for_terminal"))),
        )
        candidate[item_id]["priority"] += args.value_priority_weight * confidence
        anchor = match.get("value_anchor") or f"anonymous:{item_id}"
        value_groups[str(anchor)].append((item_id, confidence))
    for anchor, edges in value_groups.items():
        factor_specs.append(
            {
                "key": f"value:{anchor}",
                "kind": "value",
                "relation_id": -1,
                "edges": edges,
            }
        )

    join_candidates = {
        int(match["schema_item_id"]): float(match.get("score", 0.0))
        for match in (evidence_debug or {}).get("join_path_candidates", [])
        if int(match["schema_item_id"]) in nodes
    }
    for item_id, score in join_candidates.items():
        candidate[item_id]["join_source"] = 1.0
        candidate[item_id]["join_score"] = max(candidate[item_id]["join_score"], score)
        candidate[item_id]["priority"] += args.join_priority_weight * score
    covered_by_path = set()
    for path_index, path_record in enumerate(
        (evidence_debug or {}).get("join_path", {}).get("paths", [])
    ):
        path_ids = path_schema_ids(path_record, nodes, table_to_id)
        edges = []
        for item_id in path_ids:
            score = join_candidates.get(item_id, float(path_record.get("path_score", 0.0)))
            candidate[item_id]["join_source"] = 1.0
            candidate[item_id]["join_score"] = max(candidate[item_id]["join_score"], score)
            candidate[item_id]["join_path_count"] += 1.0
            candidate[item_id]["priority"] += args.join_priority_weight * score
            edges.append((item_id, score))
        if edges:
            covered_by_path.update(path_ids)
            factor_specs.append(
                {
                    "key": f"join_path:{path_index}",
                    "kind": "join_path",
                    "relation_id": -1,
                    "edges": edges,
                }
            )
    residual_join = [(item_id, score) for item_id, score in join_candidates.items() if item_id not in covered_by_path]
    if residual_join:
        factor_specs.append(
            {
                "key": "join_path:residual",
                "kind": "join_path",
                "relation_id": -1,
                "edges": residual_join,
            }
        )

    protected = set(baseline_ids)
    ranked_remaining = sorted(
        (item_id for item_id in candidate if item_id not in protected),
        key=lambda item_id: candidate[item_id]["priority"],
        reverse=True,
    )
    selected_ids = list(baseline_ids)
    for item_id in ranked_remaining:
        if len(selected_ids) >= args.max_candidates:
            break
        selected_ids.append(item_id)
    selected_set = set(selected_ids)

    # Owner closure is inference-safe and prevents an isolated column candidate from
    # entering the reranker without the table needed by the constrained decoder.
    for item_id in list(selected_ids):
        node = nodes[item_id]
        if node.get("type") != "column":
            continue
        owner_id = table_to_id.get(node.get("table"))
        if owner_id is not None and owner_id not in selected_set:
            selected_ids.append(owner_id)
            selected_set.add(owner_id)
            candidate[owner_id]["owner_closure"] = 1.0
            candidate[owner_id]["priority"] += 0.25

    local_by_id = {item_id: index for index, item_id in enumerate(selected_ids)}
    candidate_nodes = []
    for item_id in selected_ids:
        node = nodes[item_id]
        state = candidate[item_id]
        numeric = [
            float(node.get("type") == "table"),
            float(node.get("type") == "column"),
            state["baseline"],
            state["owner_closure"],
            state["relation_source"],
            state["value_source"],
            state["join_source"],
            *state["relation_rr"],
            *state["relation_z"],
            state["value_confidence"],
            state["raw_value_score"],
            state["value_ambiguity"],
            state["value_margin"],
            state["value_semantic_support"],
            state["value_inject"],
            state["value_terminal"],
            state["join_score"],
            min(state["join_path_count"] / 4.0, 1.0),
        ]
        owner_id = (
            table_to_id.get(node.get("table")) if node.get("type") == "column" else item_id
        )
        candidate_nodes.append(
            {
                "local_id": local_by_id[item_id],
                "schema_item_id": item_id,
                "schema_position": positions[item_id],
                "name": node.get("name"),
                "type": node.get("type"),
                "owner_table_id": owner_id,
                "owner_local_id": local_by_id.get(owner_id),
                "numeric_features": numeric,
                "priority": state["priority"],
            }
        )

    schema_edges = []
    for edge in inputs.get("schema_edges", []):
        src = int(edge["src"])
        dst = int(edge["dst"])
        if src in local_by_id and dst in local_by_id:
            schema_edges.append(
                {
                    "src": local_by_id[src],
                    "dst": local_by_id[dst],
                    "type": edge.get("type"),
                }
            )

    factors = []
    factor_edges = []
    for spec in factor_specs:
        kept_edges = [(item_id, weight) for item_id, weight in spec["edges"] if item_id in local_by_id]
        if not kept_edges:
            continue
        factor_id = len(factors)
        weights = [float(weight) for _, weight in kept_edges]
        factors.append(
            {
                "id": factor_id,
                "key": spec["key"],
                "kind": spec["kind"],
                "kind_id": FACTOR_KINDS.index(spec["kind"]),
                "relation_id": spec["relation_id"],
                "numeric_features": [
                    min(len(kept_edges) / max(args.max_candidates, 1), 1.0),
                    max(weights, default=0.0),
                    sum(weights) / len(weights),
                ],
            }
        )
        for item_id, weight in kept_edges:
            factor_edges.append(
                {
                    "schema": local_by_id[item_id],
                    "factor": factor_id,
                    "type": spec["kind"],
                    "type_id": FACTOR_KINDS.index(spec["kind"]),
                    "weight": float(weight),
                }
            )

    relation_record = item["clause_record"]
    gold_ids = {int(item_id) for item_id in relation_record.get("whole_sql_labels", [])}
    gold_table_ids = {
        item_id for item_id in gold_ids if nodes.get(item_id, {}).get("type") == "table"
    }
    gold_column_ids = {
        item_id for item_id in gold_ids if nodes.get(item_id, {}).get("type") == "column"
    }
    relation_gold = relation_record.get("relation_labels", {})
    whole_labels = [float(node["schema_item_id"] in gold_ids) for node in candidate_nodes]
    role_labels = [
        [
            float(node["schema_item_id"] in set(relation_gold.get(relation, [])))
            for relation in relation_types
        ]
        for node in candidate_nodes
    ]
    candidate_gold = gold_ids & selected_set
    return {
        "example_id": graph_example.get("example_id"),
        "record_index": int(item["record_index"]),
        "db_id": inputs.get("db_id"),
        "question_id": relation_record.get("question_id"),
        "question": inputs.get("question"),
        "evidence": inputs.get("evidence"),
        "candidate_nodes": candidate_nodes,
        "schema_edges": schema_edges,
        "factors": factors,
        "factor_edges": factor_edges,
        "baseline_selected_ids": baseline_ids,
        "whole_labels": whole_labels,
        "role_labels": role_labels,
        "gold_ids": sorted(gold_ids),
        "gold_table_ids": sorted(gold_table_ids),
        "gold_column_ids": sorted(gold_column_ids),
        "candidate_gold_ids": sorted(candidate_gold),
        "missing_gold_ids": sorted(gold_ids - selected_set),
        "candidate_oracle_recall": len(candidate_gold) / len(gold_ids) if gold_ids else 1.0,
    }


def build_factor_graph_dataset(aligned, prediction_rows, evidence_rows, args):
    grouped = group_relation_predictions(prediction_rows)
    evidence_by_index = {
        int(row["record_index"]): row for row in (evidence_rows or [])
    }
    baselines = assemble_predictions(
        prediction_rows,
        aligned,
        output_top_k=args.baseline_top_k,
        budgets=args.assembly_budgets,
    )
    baseline_by_index = {int(row["record_index"]): row for row in baselines}
    examples = []
    for item in aligned:
        index = int(item["record_index"])
        examples.append(
            build_one_factor_graph(
                item,
                grouped.get(index, {}),
                baseline_by_index[index],
                evidence_by_index.get(index),
                args.relation_types,
                args,
            )
        )
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Build Stage 10-A candidate heterogeneous factor graphs."
    )
    parser.add_argument("--relation-predictions", required=True)
    parser.add_argument("--relation-file", required=True)
    parser.add_argument("--graph-file", required=True)
    parser.add_argument("--evidence-debug", default=None)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--relation-types", default=",".join(DEFAULT_RELATIONS))
    parser.add_argument("--relation-top-m", type=int, default=20)
    parser.add_argument("--baseline-top-k", type=int, default=30)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--value-priority-weight", type=float, default=1.0)
    parser.add_argument("--join-priority-weight", type=float, default=0.8)
    parser.add_argument(
        "--assembly-budgets",
        type=parse_budget_text,
        default=(
            "OUTPUT_TARGET:7,JOIN_BRIDGE:8,PREDICATE_COLUMN:7,"
            "METRIC_TARGET:4,VALUE_ANCHOR:2,TEMPORAL_FILTER:1,ORDER_KEY:1"
        ),
    )
    args = parser.parse_args()
    args.relation_types = [
        part.strip() for part in args.relation_types.split(",") if part.strip()
    ]
    aligned = load_aligned_records(
        Path(args.relation_file), Path(args.graph_file), args.limit
    )
    predictions = read_jsonl(args.relation_predictions)
    allowed = {int(item["record_index"]) for item in aligned}
    predictions = [
        row for row in predictions if int(row.get("record_index", -1)) in allowed
    ]
    evidence = read_jsonl(args.evidence_debug) if args.evidence_debug else []
    examples = build_factor_graph_dataset(aligned, predictions, evidence, args)
    output_file = Path(args.output_file)
    write_jsonl(output_file, examples)
    summary_file = (
        Path(args.summary_file)
        if args.summary_file
        else output_file.with_name(output_file.stem + "_summary.json")
    )
    candidate_counts = [len(row["candidate_nodes"]) for row in examples]
    factor_counts = [len(row["factors"]) for row in examples]
    oracle_recalls = [row["candidate_oracle_recall"] for row in examples]
    empty_candidate_rows = [row for row in examples if not row["candidate_nodes"]]
    summary = {
        "config": vars(args),
        "sample_count": len(examples),
        "avg_candidate_count": sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
        "max_candidate_count": max(candidate_counts, default=0),
        "avg_factor_count": sum(factor_counts) / len(factor_counts) if factor_counts else 0.0,
        "candidate_oracle_recall": sum(oracle_recalls) / len(oracle_recalls) if oracle_recalls else 0.0,
        "complete_candidate_coverage": (
            sum(value >= 1.0 for value in oracle_recalls) / len(oracle_recalls)
            if oracle_recalls
            else 0.0
        ),
        "numeric_feature_dim": len(examples[0]["candidate_nodes"][0]["numeric_features"]) if examples and examples[0]["candidate_nodes"] else 0,
        "factor_numeric_dim": 3,
        "factor_kinds": FACTOR_KINDS,
        "relation_types": args.relation_types,
        "gold_injection": False,
        "empty_candidate_count": len(empty_candidate_rows),
        "empty_candidate_with_gold_count": sum(
            bool(row.get("gold_ids")) for row in empty_candidate_rows
        ),
    }
    write_json(summary_file, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_file}")


if __name__ == "__main__":
    main()
