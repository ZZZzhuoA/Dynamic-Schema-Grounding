import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5_train_dsg_grounder import (  # noqa: E402
    read_jsonl,
    write_json,
    write_jsonl,
)
from src.training.stage5j_train_relation_grounder import DEFAULT_RELATIONS  # noqa: E402


COMPLETION_FEATURES = [
    "table_conditioned_completion_source",
    "table_conditioned_query_similarity",
    "table_conditioned_anchor_belief",
    "table_conditioned_reciprocal_rank",
]


def import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Stage 10-G requires numpy.") from exc
    return np


def load_embedding_cache(cache_dir, split, np):
    cache_dir = Path(cache_dir)
    index_rows = json.loads(
        (cache_dir / f"{split}_index.json").read_text(encoding="utf-8")
    )
    return {
        "query": np.load(
            cache_dir / f"{split}_query_embeddings.npy", mmap_mode="r"
        ),
        "nodes": np.load(
            cache_dir / f"{split}_node_embeddings.npy", mmap_mode="r"
        ),
        "index": {int(row["example_index"]): row for row in index_rows},
    }


def embedding_rows(cache, record_index):
    row = cache["index"].get(int(record_index))
    if row is None:
        raise KeyError(f"Embedding cache has no record_index={record_index}")
    query_index = int(row["query_embedding_index"])
    query = cache["query"][query_index]
    if "node_embedding_indices" in row:
        indices = [int(index) for index in row["node_embedding_indices"]]
        nodes = cache["nodes"][indices]
    else:
        start = int(row["node_embedding_start"])
        count = int(row["node_count"])
        nodes = cache["nodes"][start : start + count]
    return query, nodes


def cosine_similarity(query, matrix, np):
    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    matrix_norm = np.linalg.norm(matrix, axis=1)
    denominator = np.maximum(matrix_norm * max(query_norm, 1e-12), 1e-12)
    return (matrix @ query) / denominator


def minmax(values):
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def index_rows(rows, kind):
    result = {}
    for fallback_index, row in enumerate(rows):
        index = int(row.get("record_index", fallback_index))
        if index in result:
            raise ValueError(f"Duplicate {kind} record_index={index}")
        result[index] = row
    return result


def table_anchor_beliefs(factor_graph, full_nodes, similarities):
    """Estimate table belief using only inference-time candidate state."""
    baseline_ids = {int(value) for value in factor_graph.get("baseline_selected_ids", [])}
    tables = [
        node for node in factor_graph.get("candidate_nodes", []) if node.get("type") == "table"
    ]
    if not tables:
        return []
    priorities = [float(node.get("priority", 0.0)) for node in tables]
    priority_scores = minmax(priorities)
    semantic_values = [
        float(similarities[int(node["schema_position"])]) for node in tables
    ]
    semantic_scores = minmax(semantic_values)
    anchors = []
    for node, priority_score, semantic_score, raw_similarity in zip(
        tables, priority_scores, semantic_scores, semantic_values
    ):
        schema_id = int(node["schema_item_id"])
        # Equal-rank fusion avoids a dev-tuned calibration constant.  Baseline
        # membership is only a deterministic tie-break because it is already
        # reflected in the upstream candidate priority.
        belief = 0.50 * priority_score + 0.50 * semantic_score
        anchors.append(
            {
                "schema_item_id": schema_id,
                "schema_position": int(node["schema_position"]),
                "name": node.get("name"),
                "belief": belief,
                "query_similarity": raw_similarity,
            }
        )
    return sorted(
        anchors,
        key=lambda item: (
            -item["belief"],
            -float(item["schema_item_id"] in baseline_ids),
            -item["query_similarity"],
            item["schema_item_id"],
        ),
    )


def completion_candidates(
    factor_graph,
    full_nodes,
    similarities,
    max_anchor_tables,
    columns_per_table,
    max_additions,
):
    existing_ids = {
        int(node["schema_item_id"]) for node in factor_graph.get("candidate_nodes", [])
    }
    anchors = table_anchor_beliefs(factor_graph, full_nodes, similarities)[
        :max_anchor_tables
    ]
    anchors_by_name = {anchor["name"]: anchor for anchor in anchors}
    per_table = {}
    for table_name, anchor in anchors_by_name.items():
        candidates = []
        for position, node in enumerate(full_nodes):
            schema_id = int(node["id"])
            if (
                node.get("type") != "column"
                or node.get("table") != table_name
                or schema_id in existing_ids
            ):
                continue
            candidates.append(
                {
                    "schema_item_id": schema_id,
                    "schema_position": position,
                    "name": node.get("name"),
                    "table": table_name,
                    "owner_table_id": int(anchor["schema_item_id"]),
                    "anchor_belief": float(anchor["belief"]),
                    "query_similarity": float(similarities[position]),
                }
            )
        candidates.sort(
            key=lambda item: (-item["query_similarity"], item["schema_item_id"])
        )
        for rank, item in enumerate(candidates[:columns_per_table], start=1):
            item["within_table_rank"] = rank
            item["reciprocal_rank"] = 1.0 / rank
            # This score is exposed to RGTA as a feature, not used as a direct
            # final-selection score.
            item["completion_score"] = (
                0.50 * ((item["query_similarity"] + 1.0) / 2.0)
                + 0.50 * item["anchor_belief"]
            )
        per_table[table_name] = candidates[:columns_per_table]
    # Round-robin allocation prevents a wide table from consuming the entire
    # completion budget. Anchor rank and within-table semantic rank are the only
    # ordering decisions; no dev-calibrated score threshold is used.
    selected = []
    for rank_index in range(columns_per_table):
        for anchor in anchors:
            table_rows = per_table.get(anchor["name"], [])
            if rank_index < len(table_rows):
                selected.append(table_rows[rank_index])
    return anchors, selected[:max_additions]


def augment_factor_graph(
    factor_graph,
    full_graph,
    relation_record,
    query_embedding,
    node_embeddings,
    np,
    relation_types,
    max_anchor_tables=4,
    columns_per_table=8,
    max_additions=24,
):
    inputs = full_graph.get("inference_inputs", {})
    full_nodes = inputs.get("schema_nodes", [])
    if len(full_nodes) != len(node_embeddings):
        raise ValueError(
            f"Schema/cache node mismatch at record_index={factor_graph.get('record_index')}: "
            f"schema={len(full_nodes)} cache={len(node_embeddings)}"
        )
    similarities = cosine_similarity(query_embedding, node_embeddings, np)
    anchors, additions = completion_candidates(
        factor_graph,
        full_nodes,
        similarities,
        max_anchor_tables=max_anchor_tables,
        columns_per_table=columns_per_table,
        max_additions=max_additions,
    )

    result = dict(factor_graph)
    old_candidates = factor_graph.get("candidate_nodes", [])
    old_numeric_dim = len(old_candidates[0].get("numeric_features", [])) if old_candidates else 0
    candidates = []
    for expected_local_id, node in enumerate(old_candidates):
        copied = dict(node)
        if int(copied.get("local_id", expected_local_id)) != expected_local_id:
            raise ValueError(
                f"Non-contiguous candidate local ids at record_index={factor_graph.get('record_index')}"
            )
        copied["numeric_features"] = list(copied.get("numeric_features", [])) + [0.0] * len(
            COMPLETION_FEATURES
        )
        candidates.append(copied)

    local_by_schema_id = {
        int(node["schema_item_id"]): int(node["local_id"]) for node in candidates
    }
    addition_by_id = {int(item["schema_item_id"]): item for item in additions}
    for item in additions:
        schema_id = int(item["schema_item_id"])
        owner_id = int(item["owner_table_id"])
        if owner_id not in local_by_schema_id:
            continue
        local_id = len(candidates)
        base_numeric = [0.0] * old_numeric_dim
        if old_numeric_dim >= 2:
            base_numeric[1] = 1.0
        numeric = base_numeric + [
            1.0,
            (float(item["query_similarity"]) + 1.0) / 2.0,
            float(item["anchor_belief"]),
            float(item["reciprocal_rank"]),
        ]
        candidates.append(
            {
                "local_id": local_id,
                "schema_item_id": schema_id,
                "schema_position": int(item["schema_position"]),
                "name": item.get("name"),
                "type": "column",
                "owner_table_id": owner_id,
                "owner_local_id": local_by_schema_id[owner_id],
                "numeric_features": numeric,
                "priority": float(item["completion_score"]),
                "completion_source": "table_conditioned_dense_retrieval",
            }
        )
        local_by_schema_id[schema_id] = local_id

    # Rebuild the induced schema graph so newly retrieved columns can exchange
    # messages with their owner tables in Schema-RGTA.
    schema_edges = []
    for edge in inputs.get("schema_edges", []):
        src = int(edge["src"])
        dst = int(edge["dst"])
        if src in local_by_schema_id and dst in local_by_schema_id:
            schema_edges.append(
                {
                    "src": local_by_schema_id[src],
                    "dst": local_by_schema_id[dst],
                    "type": edge.get("type"),
                }
            )

    gold_ids = {int(value) for value in factor_graph.get("gold_ids", [])}
    relation_gold = (relation_record or {}).get("relation_labels", {})
    selected_ids = set(local_by_schema_id)
    result["candidate_nodes"] = candidates
    result["schema_edges"] = schema_edges
    result["whole_labels"] = [
        float(int(node["schema_item_id"]) in gold_ids) for node in candidates
    ]
    result["role_labels"] = [
        [
            float(int(node["schema_item_id"]) in set(relation_gold.get(relation, [])))
            for relation in relation_types
        ]
        for node in candidates
    ]
    candidate_gold = gold_ids & selected_ids
    result["candidate_gold_ids"] = sorted(candidate_gold)
    result["missing_gold_ids"] = sorted(gold_ids - selected_ids)
    result["candidate_oracle_recall"] = (
        len(candidate_gold) / len(gold_ids) if gold_ids else 1.0
    )
    result["completion"] = {
        "method": "table_conditioned_dense_retrieval",
        "anchor_tables": anchors,
        "added_columns": [
            {
                key: value
                for key, value in item.items()
                if key
                in {
                    "schema_item_id",
                    "name",
                    "table",
                    "owner_table_id",
                    "query_similarity",
                    "anchor_belief",
                    "within_table_rank",
                    "completion_score",
                }
            }
            for item in additions
            if int(item["schema_item_id"]) in addition_by_id
        ],
        "feature_names": COMPLETION_FEATURES,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 10-G: augment RGTA candidate graphs with columns retrieved from "
            "high-belief tables without using gold labels for retrieval."
        )
    )
    parser.add_argument("--factor-graph-file", required=True)
    parser.add_argument("--full-graph-file", required=True)
    parser.add_argument("--relation-file", required=True)
    parser.add_argument(
        "--evaluation-label-file",
        default=None,
        help=(
            "Optional scope-aware labels used only for post-hoc candidate-ceiling "
            "metrics. They never affect anchors, retrieval, or output training labels."
        ),
    )
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", default=None)
    parser.add_argument("--relation-types", default=",".join(DEFAULT_RELATIONS))
    parser.add_argument("--max-anchor-tables", type=int, default=4)
    parser.add_argument("--columns-per-table", type=int, default=8)
    parser.add_argument("--max-additions", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    relation_types = [value.strip() for value in args.relation_types.split(",") if value.strip()]
    if min(args.max_anchor_tables, args.columns_per_table, args.max_additions) <= 0:
        raise ValueError("Completion budgets must all be positive")

    np = import_numpy()
    factor_rows = read_jsonl(Path(args.factor_graph_file), limit=args.limit)
    full_rows = read_jsonl(Path(args.full_graph_file))
    relation_rows = read_jsonl(Path(args.relation_file))
    evaluation_rows = (
        read_jsonl(Path(args.evaluation_label_file))
        if args.evaluation_label_file
        else []
    )
    full_by_index = index_rows(full_rows, "full graph")
    relation_by_index = index_rows(relation_rows, "relation")
    evaluation_by_index = index_rows(evaluation_rows, "evaluation label")
    cache = load_embedding_cache(args.embedding_cache_dir, args.split, np)

    outputs = []
    pre_recalls = []
    post_recalls = []
    added_counts = []
    recovered_complete = 0
    anchor_gold_table_hits = 0
    anchor_gold_table_total = 0
    evaluation_pre_recalls = []
    evaluation_post_recalls = []
    evaluation_recovered_complete = 0
    for factor_graph in factor_rows:
        record_index = int(factor_graph["record_index"])
        full_graph = full_by_index.get(record_index)
        relation_record = relation_by_index.get(record_index)
        if full_graph is None or relation_record is None:
            raise KeyError(f"Missing aligned full graph/relation row for record_index={record_index}")
        graph_db = full_graph.get("inference_inputs", {}).get("db_id")
        if factor_graph.get("db_id") != graph_db or relation_record.get("db_id") != graph_db:
            raise ValueError(f"db_id alignment failure at record_index={record_index}")
        query, nodes = embedding_rows(cache, record_index)
        output = augment_factor_graph(
            factor_graph,
            full_graph,
            relation_record,
            query,
            nodes,
            np,
            relation_types,
            max_anchor_tables=args.max_anchor_tables,
            columns_per_table=args.columns_per_table,
            max_additions=args.max_additions,
        )
        outputs.append(output)
        pre = float(factor_graph.get("candidate_oracle_recall", 0.0))
        post = float(output.get("candidate_oracle_recall", 0.0))
        pre_recalls.append(pre)
        post_recalls.append(post)
        added_counts.append(len(output["candidate_nodes"]) - len(factor_graph["candidate_nodes"]))
        recovered_complete += int(pre < 1.0 - 1e-9 and post >= 1.0 - 1e-9)
        gold_tables = {int(value) for value in factor_graph.get("gold_table_ids", [])}
        anchor_ids = {
            int(item["schema_item_id"])
            for item in output.get("completion", {}).get("anchor_tables", [])[: args.max_anchor_tables]
        }
        anchor_gold_table_hits += len(gold_tables & anchor_ids)
        anchor_gold_table_total += len(gold_tables)
        if evaluation_rows:
            evaluation_row = evaluation_by_index.get(record_index)
            if evaluation_row is None:
                raise KeyError(
                    f"Missing evaluation label row for record_index={record_index}"
                )
            evaluation_gold = {
                int(value) for value in evaluation_row.get("whole_sql_labels", [])
            }
            pre_ids = {
                int(node["schema_item_id"])
                for node in factor_graph.get("candidate_nodes", [])
            }
            post_ids = {
                int(node["schema_item_id"])
                for node in output.get("candidate_nodes", [])
            }
            evaluation_pre = (
                len(evaluation_gold & pre_ids) / len(evaluation_gold)
                if evaluation_gold
                else 1.0
            )
            evaluation_post = (
                len(evaluation_gold & post_ids) / len(evaluation_gold)
                if evaluation_gold
                else 1.0
            )
            evaluation_pre_recalls.append(evaluation_pre)
            evaluation_post_recalls.append(evaluation_post)
            evaluation_recovered_complete += int(
                evaluation_pre < 1.0 - 1e-9
                and evaluation_post >= 1.0 - 1e-9
            )

    output_file = Path(args.output_file)
    write_jsonl(output_file, outputs)
    summary_file = (
        Path(args.summary_file)
        if args.summary_file
        else output_file.with_name(output_file.stem + "_summary.json")
    )
    count = len(outputs)
    summary = {
        "config": vars(args),
        "sample_count": count,
        "avg_candidate_count_before": (
            sum(len(row.get("candidate_nodes", [])) for row in factor_rows) / count if count else 0.0
        ),
        "avg_candidate_count_after": (
            sum(len(row.get("candidate_nodes", [])) for row in outputs) / count if count else 0.0
        ),
        "avg_added_columns": sum(added_counts) / count if count else 0.0,
        "candidate_oracle_recall_before": sum(pre_recalls) / count if count else 0.0,
        "candidate_oracle_recall_after": sum(post_recalls) / count if count else 0.0,
        "complete_candidate_coverage_before": (
            sum(value >= 1.0 - 1e-9 for value in pre_recalls) / count if count else 0.0
        ),
        "complete_candidate_coverage_after": (
            sum(value >= 1.0 - 1e-9 for value in post_recalls) / count if count else 0.0
        ),
        "recovered_complete_samples": recovered_complete,
        "anchor_gold_table_recall": (
            anchor_gold_table_hits / anchor_gold_table_total if anchor_gold_table_total else 1.0
        ),
        "scope_aware_evaluation": (
            {
                "label_file": args.evaluation_label_file,
                "candidate_oracle_recall_before": (
                    sum(evaluation_pre_recalls) / len(evaluation_pre_recalls)
                ),
                "candidate_oracle_recall_after": (
                    sum(evaluation_post_recalls) / len(evaluation_post_recalls)
                ),
                "complete_candidate_coverage_before": (
                    sum(value >= 1.0 - 1e-9 for value in evaluation_pre_recalls)
                    / len(evaluation_pre_recalls)
                ),
                "complete_candidate_coverage_after": (
                    sum(value >= 1.0 - 1e-9 for value in evaluation_post_recalls)
                    / len(evaluation_post_recalls)
                ),
                "recovered_complete_samples": evaluation_recovered_complete,
                "consumed_by_retrieval": False,
            }
            if evaluation_pre_recalls
            else None
        ),
        "numeric_feature_dim": (
            len(outputs[0]["candidate_nodes"][0]["numeric_features"])
            if outputs and outputs[0].get("candidate_nodes")
            else 0
        ),
        "completion_feature_names": COMPLETION_FEATURES,
        "gold_injection": False,
        "protocol": (
            "Anchors and added columns use only factor-graph inference state plus frozen "
            "query/schema embeddings. Gold labels are consumed only by post-hoc ceiling metrics."
        ),
        "leaderboard_safety": (
            "No database-specific aliases, dev error corrections, or gold-conditioned thresholds. "
            "The same fixed completion rule must be used for train and dev."
        ),
    }
    write_json(summary_file, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_file}")


if __name__ == "__main__":
    main()
