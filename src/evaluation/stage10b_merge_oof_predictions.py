import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5_train_dsg_grounder import (  # noqa: E402
    read_jsonl,
    write_json,
    write_jsonl,
)


def prediction_key(row):
    return int(row["record_index"]), str(row.get("relation_type") or row.get("clause"))


def main():
    parser = argparse.ArgumentParser(
        description="Merge and validate strict Stage 10-B out-of-fold predictions."
    )
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--fold-output-dir", required=True)
    parser.add_argument("--relation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--include-empty-relation-examples",
        action="store_true",
        help="Expect predictions for relation roles with empty gold labels as well.",
    )
    args = parser.parse_args()

    manifest = json.loads(Path(args.fold_manifest).read_text(encoding="utf-8"))
    relation_records = read_jsonl(Path(args.relation_file))
    relation_types = sorted(
        {
            relation
            for record in relation_records
            for relation in record.get("relation_labels", {})
        }
    )
    expected_keys = set()
    for record_index, record in enumerate(relation_records):
        labels = record.get("relation_labels", {})
        for relation in relation_types:
            if args.include_empty_relation_examples or labels.get(relation):
                expected_keys.add((record_index, relation))

    merged_relation = []
    merged_assembled = []
    seen_keys = set()
    seen_assembled = set()
    fold_summaries = []
    for fold in manifest["folds"]:
        fold_id = int(fold["fold_id"])
        expected_indices = set(
            json.loads(Path(fold["heldout_index_file"]).read_text(encoding="utf-8"))[
                "record_indices"
            ]
        )
        fold_dir = Path(args.fold_output_dir) / f"fold_{fold_id}"
        relation_path = fold_dir / "dev_relation_predictions.jsonl"
        assembled_path = fold_dir / "dev_assembled_predictions.jsonl"
        if not relation_path.exists() or not assembled_path.exists():
            raise FileNotFoundError(
                f"Missing fold outputs for fold {fold_id}: "
                f"relation={relation_path.exists()}, assembled={assembled_path.exists()}"
            )
        relation_rows = read_jsonl(relation_path)
        assembled_rows = read_jsonl(assembled_path)
        foreign_relation = sorted(
            {
                int(row["record_index"])
                for row in relation_rows
                if int(row["record_index"]) not in expected_indices
            }
        )
        foreign_assembled = sorted(
            {
                int(row["record_index"])
                for row in assembled_rows
                if int(row["record_index"]) not in expected_indices
            }
        )
        if foreign_relation or foreign_assembled:
            raise ValueError(
                f"Fold {fold_id} contains predictions outside its held-out records: "
                f"relation={foreign_relation[:10]}, assembled={foreign_assembled[:10]}"
            )
        for row in relation_rows:
            key = prediction_key(row)
            if key in seen_keys:
                raise ValueError(f"Duplicate OOF relation prediction: {key}")
            seen_keys.add(key)
            merged_relation.append(row)
        for row in assembled_rows:
            index = int(row["record_index"])
            if index in seen_assembled:
                raise ValueError(f"Duplicate OOF assembled prediction: {index}")
            seen_assembled.add(index)
            merged_assembled.append(row)
        fold_summaries.append(
            {
                "fold_id": fold_id,
                "heldout_record_count": len(expected_indices),
                "relation_prediction_count": len(relation_rows),
                "assembled_prediction_count": len(assembled_rows),
            }
        )

    missing_keys = sorted(expected_keys - seen_keys)
    unexpected_keys = sorted(seen_keys - expected_keys)
    expected_indices = set(range(len(relation_records)))
    missing_assembled = sorted(expected_indices - seen_assembled)
    unexpected_assembled = sorted(seen_assembled - expected_indices)
    if missing_keys or unexpected_keys or missing_assembled or unexpected_assembled:
        raise ValueError(
            "Incomplete OOF merge: "
            f"missing_relation={missing_keys[:10]}, "
            f"unexpected_relation={unexpected_keys[:10]}, "
            f"missing_assembled={missing_assembled[:10]}, "
            f"unexpected_assembled={unexpected_assembled[:10]}"
        )

    relation_order = {relation: index for index, relation in enumerate(relation_types)}
    merged_relation.sort(
        key=lambda row: (
            int(row["record_index"]),
            relation_order.get(str(row.get("relation_type") or row.get("clause")), 10**6),
        )
    )
    merged_assembled.sort(key=lambda row: int(row["record_index"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    relation_output = output_dir / "train_oof_relation_predictions.jsonl"
    assembled_output = output_dir / "train_oof_assembled_predictions.jsonl"
    write_jsonl(relation_output, merged_relation)
    write_jsonl(assembled_output, merged_assembled)

    relation_counts = Counter(
        str(row.get("relation_type") or row.get("clause"))
        for row in merged_relation
    )
    predictions_per_record = defaultdict(int)
    for row in merged_relation:
        predictions_per_record[int(row["record_index"])] += 1
    summary = {
        "config": vars(args),
        "strict_oof": True,
        "record_count": len(relation_records),
        "relation_prediction_count": len(merged_relation),
        "assembled_prediction_count": len(merged_assembled),
        "relation_counts": dict(sorted(relation_counts.items())),
        "min_relation_predictions_per_record": min(predictions_per_record.values()),
        "max_relation_predictions_per_record": max(predictions_per_record.values()),
        "folds": fold_summaries,
        "integrity": {
            "all_expected_relation_predictions_present": True,
            "all_records_have_one_assembled_prediction": True,
            "duplicate_relation_predictions": 0,
            "duplicate_assembled_predictions": 0,
        },
        "outputs": {
            "relation_predictions": str(relation_output),
            "assembled_predictions": str(assembled_output),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
