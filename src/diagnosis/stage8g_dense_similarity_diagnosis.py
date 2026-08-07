import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5_train_dsg_grounder import (  # noqa: E402
    mean,
    precision_at_k,
    ranked_prediction_rows,
    recall_at_k,
    reciprocal_rank,
    write_json,
    write_jsonl,
)
from src.training.stage5g_train_clause_grounder import parse_clause_list  # noqa: E402
from src.training.stage5j_train_relation_grounder import (  # noqa: E402
    DEFAULT_RELATIONS,
    load_aligned_records,
    make_relation_examples,
)


def import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Stage 8G-C1 diagnosis requires numpy.") from exc
    return np


def load_embedding_cache(cache_dir: Path, split: str, np):
    query_file = cache_dir / f"{split}_query_embeddings.npy"
    node_file = cache_dir / f"{split}_node_embeddings.npy"
    index_file = cache_dir / f"{split}_index.json"
    missing = [str(path) for path in [query_file, node_file, index_file] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing embedding cache files: {missing}")

    query_embeddings = np.load(query_file, mmap_mode="r")
    node_embeddings = np.load(node_file, mmap_mode="r")
    index_rows = json.loads(index_file.read_text(encoding="utf-8"))
    by_example_index = {int(row["example_index"]): row for row in index_rows}
    return {
        "query_embeddings": query_embeddings,
        "node_embeddings": node_embeddings,
        "index_rows": index_rows,
        "by_example_index": by_example_index,
        "query_file": str(query_file),
        "node_file": str(node_file),
        "index_file": str(index_file),
    }


def normalized_dot_scores(query_embedding, node_embeddings, np, normalize=True):
    query = np.asarray(query_embedding, dtype="float32")
    nodes = np.asarray(node_embeddings, dtype="float32")
    if query.ndim == 2:
        query = query.reshape(-1)
    if normalize:
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        node_norms = np.linalg.norm(nodes, axis=1, keepdims=True)
        node_norms[node_norms == 0] = 1.0
        nodes = nodes / node_norms
    return np.matmul(nodes, query).astype("float32").tolist()


def scores_for_example(record_index, cache, np, normalize=True):
    index_row = cache["by_example_index"].get(int(record_index))
    if index_row is None:
        raise KeyError(f"No embedding-cache index for example_index={record_index}")
    query_index = int(index_row["query_embedding_index"])
    node_count = int(index_row["node_count"])
    query_embedding = cache["query_embeddings"][query_index]
    if "node_embedding_indices" in index_row:
        node_indices = [int(index) for index in index_row["node_embedding_indices"]]
        if len(node_indices) != node_count:
            raise ValueError(
                f"Deduplicated cache node-count mismatch for record_index={record_index}: "
                f"indices={len(node_indices)} expected={node_count}"
            )
        node_embeddings = cache["node_embeddings"][node_indices]
    else:
        node_start = int(index_row["node_embedding_start"])
        node_embeddings = cache["node_embeddings"][node_start : node_start + node_count]
    scores = normalized_dot_scores(query_embedding, node_embeddings, np, normalize=normalize)
    if len(scores) != node_count:
        raise ValueError(f"Score/node mismatch at example {record_index}: scores={len(scores)} nodes={node_count}")
    return scores


def node_lookup(example):
    return {int(node["id"]): node for node in example["inference_inputs"].get("schema_nodes", [])}


def label_names(example, gold_ids):
    nodes = node_lookup(example)
    return [nodes[item_id]["name"] for item_id in gold_ids if item_id in nodes]


def split_gold_ids(example, gold_ids):
    nodes = node_lookup(example)
    table_ids = []
    column_ids = []
    for item_id in gold_ids:
        node_type = nodes.get(int(item_id), {}).get("type")
        if node_type == "table":
            table_ids.append(int(item_id))
        elif node_type == "column":
            column_ids.append(int(item_id))
    return table_ids, column_ids


def ranked_column_ids(example, ranked_ids):
    nodes = node_lookup(example)
    return [item_id for item_id in ranked_ids if nodes.get(int(item_id), {}).get("type") == "column"]


def evaluate_rankings(rows, prediction_key, gold_key="gold_label_ids", prefix="cosine_schema"):
    metrics = {
        f"{prefix}_recall@5": [],
        f"{prefix}_recall@10": [],
        f"{prefix}_recall@20": [],
        f"{prefix}_recall@30": [],
        f"{prefix}_precision@10": [],
        f"{prefix}_precision@30": [],
        f"{prefix}_mrr": [],
        "cosine_table_recall@5": [],
        "cosine_column_recall@20": [],
        "cosine_column_recall@30": [],
    }
    missing_at_30 = 0
    for row in rows:
        gold_ids = [int(item_id) for item_id in row.get(gold_key, [])]
        ranked_ids = [int(item["id"]) for item in row[prediction_key]]
        full_ranked_ids = [int(item_id) for item_id in row.get("_ranked_ids", ranked_ids)]
        table_gold_ids = row.get("gold_table_ids", [])
        column_gold_ids = row.get("gold_column_ids", [])
        full_ranked_column_ids = row.get("_ranked_column_ids", ranked_ids)
        metrics[f"{prefix}_recall@5"].append(recall_at_k(gold_ids, full_ranked_ids, 5))
        metrics[f"{prefix}_recall@10"].append(recall_at_k(gold_ids, full_ranked_ids, 10))
        metrics[f"{prefix}_recall@20"].append(recall_at_k(gold_ids, full_ranked_ids, 20))
        recall_30 = recall_at_k(gold_ids, full_ranked_ids, 30)
        metrics[f"{prefix}_recall@30"].append(recall_30)
        metrics[f"{prefix}_precision@10"].append(precision_at_k(gold_ids, full_ranked_ids, 10))
        metrics[f"{prefix}_precision@30"].append(precision_at_k(gold_ids, full_ranked_ids, 30))
        metrics[f"{prefix}_mrr"].append(reciprocal_rank(gold_ids, full_ranked_ids))
        metrics["cosine_table_recall@5"].append(recall_at_k(table_gold_ids, full_ranked_ids, 5))
        metrics["cosine_column_recall@20"].append(recall_at_k(column_gold_ids, full_ranked_column_ids, 20))
        metrics["cosine_column_recall@30"].append(recall_at_k(column_gold_ids, full_ranked_column_ids, 30))
        if recall_30 is not None and recall_30 < 1.0:
            missing_at_30 += 1
    return {key: mean(values) for key, values in metrics.items()} | {"missing_samples@30": missing_at_30}


def build_whole_sql_predictions(aligned_records, cache, np, args):
    predictions = []
    missing_rows = []
    for item in aligned_records:
        graph_example = item["graph_example"]
        record_index = int(item["record_index"])
        scores = scores_for_example(record_index, cache, np, normalize=not args.no_normalize)
        top_rows, ranked_ids = ranked_prediction_rows(graph_example, scores, args.top_k)
        gold_ids = [int(item_id) for item_id in item["clause_record"].get("whole_sql_labels", [])]
        gold_table_ids, gold_column_ids = split_gold_ids(graph_example, gold_ids)
        ranked_cols = ranked_column_ids(graph_example, ranked_ids)
        recall_30 = recall_at_k(gold_ids, ranked_ids, min(args.top_k, 30))
        row = {
            "example_id": graph_example.get("example_id"),
            "record_index": record_index,
            "db_id": item["clause_record"].get("db_id"),
            "question_id": item["clause_record"].get("question_id"),
            "question": graph_example.get("inference_inputs", {}).get("question"),
            "evidence": graph_example.get("inference_inputs", {}).get("evidence"),
            "gold_label_ids": gold_ids,
            "gold_label_names": label_names(graph_example, gold_ids),
            "gold_table_ids": gold_table_ids,
            "gold_column_ids": gold_column_ids,
            f"top_{args.top_k}": top_rows,
            "_ranked_ids": ranked_ids,
            "_ranked_column_ids": ranked_cols,
        }
        predictions.append(row)
        if recall_30 is not None and recall_30 < 1.0:
            top_id_set = {int(top_item["id"]) for top_item in top_rows}
            missing_gold_ids = [item_id for item_id in gold_ids if item_id not in top_id_set]
            missing_rows.append(
                {
                    "example_id": row["example_id"],
                    "record_index": record_index,
                    "db_id": row["db_id"],
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "evidence": row["evidence"],
                    "recall@30": recall_30,
                    "missing_gold_ids": missing_gold_ids,
                    "missing_gold_names": label_names(graph_example, missing_gold_ids),
                    f"top_{args.top_k}": top_rows,
                }
            )
    return predictions, missing_rows


def build_relation_predictions(aligned_records, cache, np, args):
    relation_examples = make_relation_examples(
        aligned_records,
        args.relation_types,
        include_empty_relation_examples=args.include_empty_relation_examples,
    )
    predictions = []
    for example in relation_examples:
        record_index = int(example["record_index"])
        scores = scores_for_example(record_index, cache, np, normalize=not args.no_normalize)
        top_rows, ranked_ids = ranked_prediction_rows(example, scores, args.top_k)
        gold_ids = [int(item_id) for item_id in example.get("training_targets", {}).get("grounding_label_ids", [])]
        gold_table_ids, gold_column_ids = split_gold_ids(example, gold_ids)
        predictions.append(
            {
                "example_id": example["example_id"],
                "base_example_id": example["base_example_id"],
                "record_index": record_index,
                "db_id": example["metadata"].get("db_id"),
                "question_id": example["metadata"].get("question_id"),
                "relation_type": example["relation_type"],
                "gold_label_ids": gold_ids,
                "gold_label_names": label_names(example, gold_ids),
                "gold_table_ids": gold_table_ids,
                "gold_column_ids": gold_column_ids,
                f"top_{args.top_k}": top_rows,
                "_ranked_ids": ranked_ids,
                "_ranked_column_ids": ranked_column_ids(example, ranked_ids),
            }
        )
    return predictions


def relation_metrics(relation_predictions, top_k):
    metrics = {
        "relation_example_count": len(relation_predictions),
        "note": (
            "Relation metrics use the same dense query-schema similarity as whole-SQL diagnosis; "
            "the query embedding is not relation-conditioned here."
        ),
    }
    for relation in DEFAULT_RELATIONS:
        rows = [row for row in relation_predictions if row["relation_type"] == relation]
        metrics[f"{relation}_example_count"] = len(rows)
        if not rows:
            continue
        values = evaluate_rankings(rows, f"top_{top_k}", prefix=f"{relation}_cosine")
        for key, value in values.items():
            if key.startswith("cosine_"):
                key = key.replace("cosine_", f"{relation}_cosine_", 1)
            metrics[key] = value
    return metrics


def strip_internal_rows(rows):
    public_rows = []
    for row in rows:
        row = dict(row)
        row.pop("_ranked_ids", None)
        row.pop("_ranked_column_ids", None)
        public_rows.append(row)
    return public_rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 8G-C1: diagnose whether dense question/schema embeddings alone can retrieve "
            "gold schema nodes before adding them as priors to the graph grounder."
        )
    )
    parser.add_argument("--relation-file", default="experiments/stage5j_relation_labels_v1/dev_relation_labels.jsonl")
    parser.add_argument("--graph-file", default="experiments/stage8f_dsg_data_compact_llm_cards_qwen25_train1000_dev100/dev_examples.jsonl")
    parser.add_argument("--embedding-cache-dir", default="experiments/stage8g_embedding_cache_compact_qwen3_06b_train1000_dev100")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output-dir", default="experiments/stage8g_dense_similarity_diagnosis_compact_dev100")
    parser.add_argument("--relation-types", type=parse_clause_list, default=",".join(DEFAULT_RELATIONS))
    parser.add_argument("--include-empty-relation-examples", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    np = import_numpy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned_records = load_aligned_records(Path(args.relation_file), Path(args.graph_file), args.limit)
    cache = load_embedding_cache(Path(args.embedding_cache_dir), args.split, np)
    if len(cache["index_rows"]) < len(aligned_records):
        raise ValueError(
            f"Embedding cache has {len(cache['index_rows'])} examples, but diagnosis needs {len(aligned_records)}."
        )

    whole_predictions, missing_rows = build_whole_sql_predictions(aligned_records, cache, np, args)
    whole_metrics = evaluate_rankings(whole_predictions, f"top_{args.top_k}", prefix="cosine_schema")
    whole_metrics.update(
        {
            "split": args.split,
            "sample_count": len(whole_predictions),
            "top_k": args.top_k,
            "normalized": not args.no_normalize,
        }
    )

    relation_predictions = build_relation_predictions(aligned_records, cache, np, args)
    rel_metrics = relation_metrics(relation_predictions, args.top_k)
    rel_metrics.update({"split": args.split, "top_k": args.top_k, "normalized": not args.no_normalize})

    summary = {
        "config": vars(args),
        "cache": {
            "query_file": cache["query_file"],
            "node_file": cache["node_file"],
            "index_file": cache["index_file"],
            "query_shape": list(cache["query_embeddings"].shape),
            "node_shape": list(cache["node_embeddings"].shape),
        },
        "whole_sql": whole_metrics,
        "relations": rel_metrics,
        "diagnosis_note": (
            "If cosine_schema_recall@30 is already high, dense similarity is a useful prior for Stage 8G-C2. "
            "If it is low, the bottleneck is the embedding text/pooling rather than GNN assembly."
        ),
    }

    write_json(output_dir / "whole_similarity_metrics.json", whole_metrics)
    write_json(output_dir / "relation_similarity_metrics.json", rel_metrics)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "whole_similarity_predictions.jsonl", strip_internal_rows(whole_predictions))
    write_jsonl(output_dir / "relation_similarity_predictions.jsonl", strip_internal_rows(relation_predictions))
    write_jsonl(output_dir / "missing_whole_examples.jsonl", missing_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
