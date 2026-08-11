import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage5_train_dsg_grounder import read_jsonl, write_json  # noqa: E402


def stable_tie_break(seed, db_id):
    payload = f"{seed}:{db_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assign_database_folds(records, fold_count, seed):
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    by_database = defaultdict(list)
    for record_index, record in enumerate(records):
        db_id = record.get("db_id")
        if not db_id:
            raise ValueError(f"Missing db_id at record_index={record_index}")
        by_database[str(db_id)].append(record_index)
    if fold_count > len(by_database):
        raise ValueError(
            f"fold_count={fold_count} exceeds database_count={len(by_database)}"
        )

    # Largest-database-first bin packing balances question counts while preserving
    # the database-disjoint constraint. The hash only resolves equal-size ties.
    database_groups = sorted(
        by_database.items(),
        key=lambda item: (-len(item[1]), stable_tie_break(seed, item[0])),
    )
    folds = [
        {"fold_id": fold_id, "db_ids": [], "record_indices": []}
        for fold_id in range(fold_count)
    ]
    for db_id, indices in database_groups:
        target = min(
            folds,
            key=lambda fold: (
                len(fold["record_indices"]),
                len(fold["db_ids"]),
                fold["fold_id"],
            ),
        )
        target["db_ids"].append(db_id)
        target["record_indices"].extend(indices)

    all_indices = set(range(len(records)))
    for fold in folds:
        fold["db_ids"].sort()
        fold["record_indices"].sort()
        heldout = set(fold["record_indices"])
        fold["train_record_indices"] = sorted(all_indices - heldout)
        fold["record_count"] = len(fold["record_indices"])
        fold["train_record_count"] = len(fold["train_record_indices"])
        fold["db_count"] = len(fold["db_ids"])
    return folds


def validate_folds(records, folds):
    expected = set(range(len(records)))
    heldout_seen = set()
    db_to_fold = {}
    duplicate_indices = []
    duplicate_databases = []
    for fold in folds:
        for record_index in fold["record_indices"]:
            if record_index in heldout_seen:
                duplicate_indices.append(record_index)
            heldout_seen.add(record_index)
        for db_id in fold["db_ids"]:
            if db_id in db_to_fold:
                duplicate_databases.append(db_id)
            db_to_fold[db_id] = fold["fold_id"]
        train_databases = {
            str(records[index]["db_id"])
            for index in fold["train_record_indices"]
        }
        overlap = train_databases & set(fold["db_ids"])
        if overlap:
            raise ValueError(
                f"Database leakage in fold {fold['fold_id']}: {sorted(overlap)}"
            )
    missing = sorted(expected - heldout_seen)
    extra = sorted(heldout_seen - expected)
    if duplicate_indices or duplicate_databases or missing or extra:
        raise ValueError(
            "Invalid OOF partition: "
            f"duplicate_indices={duplicate_indices[:10]}, "
            f"duplicate_databases={duplicate_databases[:10]}, "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return {
        "heldout_record_count": len(heldout_seen),
        "heldout_record_coverage": len(heldout_seen) / len(records) if records else 0.0,
        "database_count": len(db_to_fold),
        "database_disjoint": True,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build balanced database-disjoint folds for Stage 10-B OOF grounding."
    )
    parser.add_argument("--relation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = read_jsonl(Path(args.relation_file))
    if not records:
        raise ValueError("relation_file is empty")
    folds = assign_database_folds(records, args.fold_count, args.seed)
    integrity = validate_folds(records, folds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_folds = []
    for fold in folds:
        fold_dir = output_dir / f"fold_{fold['fold_id']}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_file = fold_dir / "train_indices.json"
        heldout_file = fold_dir / "heldout_indices.json"
        write_json(
            train_file,
            {
                "kind": "oof_train",
                "fold_id": fold["fold_id"],
                "record_indices": fold["train_record_indices"],
            },
        )
        write_json(
            heldout_file,
            {
                "kind": "oof_heldout",
                "fold_id": fold["fold_id"],
                "db_ids": fold["db_ids"],
                "record_indices": fold["record_indices"],
            },
        )
        manifest_folds.append(
            {
                "fold_id": fold["fold_id"],
                "db_ids": fold["db_ids"],
                "db_count": fold["db_count"],
                "heldout_record_count": fold["record_count"],
                "train_record_count": fold["train_record_count"],
                "train_index_file": str(train_file),
                "heldout_index_file": str(heldout_file),
            }
        )

    summary = {
        "config": {
            "relation_file": args.relation_file,
            "output_dir": args.output_dir,
            "fold_count": args.fold_count,
            "seed": args.seed,
            "partition_unit": "db_id",
            "balancing_target": "question_count",
        },
        "record_count": len(records),
        "folds": manifest_folds,
        "integrity": integrity,
        "note": (
            "Each database occurs in exactly one held-out fold. Record indices retain "
            "their positions in the original full training files and embedding cache."
        ),
    }
    write_json(output_dir / "fold_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
