import argparse
import json
import sqlite3
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_sqlite_files(root: Path):
    return sorted(root.rglob("*.sqlite"))


def build_sqlite_index(sqlite_files):
    index = {}
    for path in sqlite_files:
        index.setdefault(path.stem, path)
    return index


def normalize_record(split: str, record: dict):
    if split == "train":
        return {
            "db_id": record.get("db_name"),
            "question": record.get("question"),
            "evidence": record.get("evidence"),
            "sql": record.get("sql_query"),
        }
    return {
        "db_id": record.get("db_id"),
        "question": record.get("question"),
        "evidence": record.get("evidence"),
        "sql": record.get("SQL"),
        "difficulty": record.get("difficulty"),
        "question_id": record.get("question_id"),
    }


def execute_sqlite_query(db_path: Path, sql: str, fetch_limit: int = 3):
    connection = sqlite3.connect(str(db_path))
    try:
        cursor = connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchmany(fetch_limit)
        return {"ok": True, "row_preview_count": len(rows), "error": None}
    except Exception as exc:
        return {"ok": False, "row_preview_count": 0, "error": str(exc)}
    finally:
        connection.close()


def check_split(base_dir: Path, split: str, sample_limit: int):
    if split == "train":
        json_path = base_dir / "train.json"
        db_root = base_dir / "train_databases"
    else:
        json_path = base_dir / "dev.json"
        db_root = base_dir / "dev_databases"

    records = read_json(json_path)
    sqlite_files = find_sqlite_files(db_root)
    sqlite_index = build_sqlite_index(sqlite_files)
    normalized = [normalize_record(split, record) for record in records]
    unique_db_ids = sorted({record["db_id"] for record in normalized if record["db_id"]})
    missing_db_ids = sorted([db_id for db_id in unique_db_ids if db_id not in sqlite_index])

    execution_checks = []
    ok_count = 0
    attempted = 0
    for record in normalized[:sample_limit]:
        db_id = record["db_id"]
        sql = record["sql"]
        item = {
            "db_id": db_id,
            "question": record["question"],
            "sql": sql,
            "ok": False,
            "error": None,
        }
        if not db_id or db_id not in sqlite_index:
            item["error"] = "missing sqlite database"
            execution_checks.append(item)
            continue
        if not sql:
            item["error"] = "missing SQL"
            execution_checks.append(item)
            continue
        attempted += 1
        result = execute_sqlite_query(sqlite_index[db_id], sql)
        item["ok"] = result["ok"]
        item["error"] = result["error"]
        if result["ok"]:
            ok_count += 1
        execution_checks.append(item)

    return {
        "split": split,
        "json_path": str(json_path),
        "db_root": str(db_root),
        "record_count": len(records),
        "unique_db_count": len(unique_db_ids),
        "sqlite_file_count": len(sqlite_files),
        "missing_db_ids": missing_db_ids,
        "sample_execution": {
            "sample_limit": sample_limit,
            "attempted": attempted,
            "ok": ok_count,
            "failed": len([item for item in execution_checks if not item["ok"]]),
            "failed_examples": [item for item in execution_checks if not item["ok"]][:10],
        },
        "first_record": normalized[0] if normalized else None,
    }


def check_dev_tables(base_dir: Path):
    path = base_dir / "dev_tables.json"
    if not path.exists():
        return {"exists": False}
    data = read_json(path)
    return {
        "exists": True,
        "path": str(path),
        "db_count": len(data),
        "db_ids": [item.get("db_id") for item in data],
        "first_schema_keys": list(data[0].keys()) if data else [],
    }


def check_bird_schema_files(base_dir: Path):
    schema_dir = base_dir / "bird-schema"
    files = {
        "filled_db_train_tables": schema_dir / "filled_db_train_tables.json",
        "filled_db_test_tables": schema_dir / "filled_db_test_tables.json",
        "train_question_answer": schema_dir / "train_question_answer.json",
    }
    result = {"schema_dir": str(schema_dir), "exists": schema_dir.exists(), "files": {}}
    for name, path in files.items():
        item = {"path": str(path), "exists": path.exists()}
        if path.exists():
            try:
                data = read_json(path)
                item["type"] = type(data).__name__
                item["count"] = len(data) if hasattr(data, "__len__") else None
                if isinstance(data, list) and data:
                    item["first_keys"] = list(data[0].keys())
            except Exception as exc:
                item["error"] = str(exc)
        result["files"][name] = item
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird-dir", default="Data/BIRD")
    parser.add_argument("--output-dir", default="experiments/stage0_data_check")
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument(
        "--splits",
        default="train,dev",
        help="Comma-separated splits to check. Use train, dev, or train,dev.",
    )
    args = parser.parse_args()

    base_dir = Path(args.bird_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "bird_dir": str(base_dir),
        "exists": base_dir.exists(),
        "train": None,
        "dev": None,
        "dev_tables": None,
        "bird_schema": None,
    }

    splits = {item.strip() for item in args.splits.split(",") if item.strip()}

    if not base_dir.exists():
        report["error"] = "BIRD directory does not exist."
    else:
        if "train" in splits:
            report["train"] = check_split(base_dir, "train", args.sample_limit)
        if "dev" in splits:
            report["dev"] = check_split(base_dir, "dev", args.sample_limit)
        report["dev_tables"] = check_dev_tables(base_dir)
        report["bird_schema"] = check_bird_schema_files(base_dir)

    split_name = "_".join(sorted(splits)) if splits else "none"
    report_path = output_dir / f"bird_stage0_report_{split_name}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
