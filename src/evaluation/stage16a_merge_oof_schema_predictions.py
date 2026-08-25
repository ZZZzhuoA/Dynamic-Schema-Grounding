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


def merge_oof_predictions(manifest, fold_output_dir, prediction_name):
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
        if observed != expected:
            raise ValueError(
                f"Fold {fold_id} held-out mismatch: missing={sorted(expected-observed)[:10]}, "
                f"foreign={sorted(observed-expected)[:10]}"
            )
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
            cleaned["grounding_provenance"] = "strict_database_disjoint_oof"
            merged.append(cleaned)
        fold_summaries.append(
            {
                "fold_id": fold_id,
                "heldout_record_count": len(expected),
                "prediction_count": len(rows),
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
    args = parser.parse_args()
    manifest = read_json(args.fold_manifest)
    if manifest.get("integrity", {}).get("database_disjoint") is not True:
        raise ValueError("Fold manifest does not certify database_disjoint=true")
    predictions, folds = merge_oof_predictions(
        manifest, args.fold_output_dir, args.prediction_name
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
        "folds": folds,
        "integrity": {
            "database_disjoint": True,
            "all_heldout_predictions_present": True,
            "no_duplicate_predictions": True,
            "training_diagnostics_removed": True,
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
