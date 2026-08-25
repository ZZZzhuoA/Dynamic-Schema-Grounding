import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5_train_dsg_grounder import write_json, write_jsonl  # noqa: E402


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index_fallback_graphs(rows):
    indexed = {}
    for position, row in enumerate(rows):
        index = int(row.get("record_index", row.get("metadata", {}).get("record_index", position)))
        if index in indexed:
            raise ValueError(f"Duplicate fallback graph record_index={index}")
        indexed[index] = row
    return indexed


def fallback_prediction(graph, record_index, fold_id, top_k=30):
    candidates = graph.get("candidate_nodes", [])
    by_schema_id = {
        int(node["schema_item_id"]): node
        for node in candidates
        if node.get("schema_item_id") is not None
    }
    ranked_ids = []
    for item_id in graph.get("baseline_selected_ids", []):
        item_id = int(item_id)
        if item_id in by_schema_id and item_id not in ranked_ids:
            ranked_ids.append(item_id)
    remaining = sorted(
        (
            node for node in candidates
            if int(node.get("schema_item_id", -1)) not in ranked_ids
        ),
        key=lambda node: (
            -float(node.get("priority", 0.0)),
            int(node.get("schema_item_id", 10**9)),
        ),
    )
    ranked_ids.extend(int(node["schema_item_id"]) for node in remaining)
    ranked_ids = ranked_ids[:top_k]
    question_id = graph.get("question_id")
    if question_id is None:
        question_id = graph.get("metadata", {}).get("question_id")
    return {
        "record_index": int(record_index),
        "db_id": graph.get("db_id")
        or graph.get("inference_inputs", {}).get("db_id"),
        "question_id": question_id,
        "question": graph.get("question")
        or graph.get("inference_inputs", {}).get("question"),
        f"top_{top_k}": [
            {
                "schema_item_id": item_id,
                "score": float(len(ranked_ids) - rank),
                "source": "inference_safe_baseline_fallback",
            }
            for rank, item_id in enumerate(ranked_ids)
        ],
        "raw_top_ids": ranked_ids,
        "baseline_top_ids": ranked_ids,
        "selector_debug": {
            "status": "no_trainable_candidate_fallback",
            "selected_count": len(ranked_ids),
        },
        "oof_fold_id": int(fold_id),
        "grounding_provenance": "strict_oof_inference_safe_fallback",
        "oof_fallback_reason": "heldout_record_skipped_by_empty_candidate_filter",
    }


def merge_oof_predictions(
    manifest,
    fold_output_dir,
    prediction_name,
    fallback_graphs=None,
):
    merged = []
    seen = set()
    fold_summaries = []
    for fold in manifest.get("folds", []):
        fold_id = int(fold["fold_id"])
        heldout_payload = read_json(fold["heldout_index_file"])
        expected = {int(value) for value in heldout_payload["record_indices"]}
        prediction_path = Path(fold_output_dir) / f"fold_{fold_id}" / prediction_name
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing fold prediction file: {prediction_path}")
        rows = read_jsonl(prediction_path)
        observed = {int(row["record_index"]) for row in rows}
        foreign = observed - expected
        missing = expected - observed
        if foreign:
            raise ValueError(
                f"Fold {fold_id} held-out mismatch: foreign={sorted(foreign)[:10]}"
            )
        fallback_rows = []
        if missing:
            if fallback_graphs is None:
                raise ValueError(
                    f"Fold {fold_id} held-out mismatch: missing={sorted(missing)[:10]}, "
                    "foreign=[]; provide --fallback-graph-file to represent skipped "
                    "empty-candidate records explicitly."
                )
            for index in sorted(missing):
                graph = fallback_graphs.get(index)
                if graph is None:
                    raise ValueError(
                        f"Fallback graph file has no record_index={index} for fold {fold_id}"
                    )
                fallback_rows.append(fallback_prediction(graph, index, fold_id))
            rows.extend(fallback_rows)
        for row in rows:
            index = int(row["record_index"])
            if index in seen:
                raise ValueError(f"Duplicate final OOF schema prediction: {index}")
            seen.add(index)
            # Remove training-only diagnostics before this artifact can become an
            # SFT inference input. Stage 16-A independently strips them again.
            cleaned = {
                key: value
                for key, value in row.items()
                if key not in {"gold_ids", "candidate_oracle_recall"}
            }
            cleaned["oof_fold_id"] = fold_id
            cleaned.setdefault(
                "grounding_provenance", "strict_database_disjoint_oof"
            )
            merged.append(cleaned)
        fold_summaries.append(
            {
                "fold_id": fold_id,
                "heldout_record_count": len(expected),
                "prediction_count": len(rows),
                "model_prediction_count": len(observed),
                "fallback_prediction_count": len(fallback_rows),
                "db_ids": heldout_payload.get("db_ids", fold.get("db_ids", [])),
            }
        )
    expected_count = int(manifest.get("record_count", len(seen)))
    expected_indices = set(range(expected_count))
    if seen != expected_indices:
        raise ValueError(
            "Incomplete final OOF schema merge: "
            f"missing={sorted(expected_indices-seen)[:10]}, "
            f"extra={sorted(seen-expected_indices)[:10]}"
        )
    merged.sort(key=lambda row: int(row["record_index"]))
    return merged, fold_summaries


def main():
    parser = argparse.ArgumentParser(
        description="Merge strict database-disjoint final Schema-RGTA OOF predictions."
    )
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--fold-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-name", default="dev_predictions.jsonl")
    parser.add_argument(
        "--fallback-graph-file",
        default=None,
        help=(
            "Inference-safe factor graph used only to represent held-out records "
            "skipped because their candidate graph was empty."
        ),
    )
    args = parser.parse_args()
    manifest = read_json(args.fold_manifest)
    if manifest.get("integrity", {}).get("database_disjoint") is not True:
        raise ValueError("Fold manifest does not certify database_disjoint=true")
    fallback_graphs = (
        index_fallback_graphs(read_jsonl(args.fallback_graph_file))
        if args.fallback_graph_file
        else None
    )
    predictions, folds = merge_oof_predictions(
        manifest,
        args.fold_output_dir,
        args.prediction_name,
        fallback_graphs=fallback_graphs,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "train_oof_schema_predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    summary = {
        "config": vars(args),
        "strict_oof": True,
        "partition_unit": "db_id",
        "checkpoint_policy": "last",
        "record_count": len(predictions),
        "fallback_prediction_count": sum(
            fold["fallback_prediction_count"] for fold in folds
        ),
        "folds": folds,
        "integrity": {
            "database_disjoint": True,
            "all_heldout_predictions_present": True,
            "no_duplicate_predictions": True,
            "training_diagnostics_removed": True,
            "skipped_records_explicitly_represented": True,
        },
        "outputs": {"schema_predictions": str(prediction_path)},
        "note": (
            "This certificate is valid only when every fold model was trained with "
            "--checkpoint-policy last and its held-out index file was used solely as dev."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
