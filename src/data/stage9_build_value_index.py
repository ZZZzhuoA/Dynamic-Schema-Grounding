import argparse
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grounding.value_index import (  # noqa: E402
    find_sqlite_files,
    index_database,
    initialize_index,
    read_schema_catalog,
)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Build a test-time-safe normalized database-value to schema-column index."
    )
    parser.add_argument("--graph-file", required=True)
    parser.add_argument("--db-root", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-values-per-column", type=int, default=20000)
    parser.add_argument("--max-value-chars", type=int, default=128)
    parser.add_argument("--include-numeric-values", action="store_true")
    parser.add_argument("--db-limit", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.rebuild and output_file.exists():
        output_file.unlink()
    catalog = read_schema_catalog(Path(args.graph_file))
    sqlite_files = find_sqlite_files(Path(args.db_root))
    connection = sqlite3.connect(str(output_file))
    initialize_index(connection)
    existing = {
        row[0] for row in connection.execute("SELECT db_id FROM indexed_databases").fetchall()
    }
    summaries = {}
    missing = []
    db_ids = sorted(catalog)
    if args.db_limit is not None:
        db_ids = db_ids[: args.db_limit]
    for position, db_id in enumerate(db_ids, start=1):
        if db_id in existing:
            row = connection.execute(
                "SELECT indexed_column_count, indexed_value_count FROM indexed_databases WHERE db_id=?",
                (db_id,),
            ).fetchone()
            summaries[db_id] = {
                "indexed_column_count": int(row[0]),
                "indexed_value_count": int(row[1]),
                "event": "resume_skip",
            }
            print(json.dumps({"db_id": db_id, "event": "resume_skip"}, ensure_ascii=False))
            continue
        sqlite_path = sqlite_files.get(db_id)
        if sqlite_path is None:
            missing.append(db_id)
            print(json.dumps({"db_id": db_id, "error": "sqlite_not_found"}, ensure_ascii=False))
            continue
        stats = index_database(connection, db_id, sqlite_path, catalog[db_id], args)
        summaries[db_id] = stats
        print(json.dumps({"index": position, "db_id": db_id, **stats}, ensure_ascii=False))
    connection.close()
    summary = {
        "config": vars(args),
        "database_count": len(summaries),
        "missing_database_count": len(missing),
        "missing_databases": missing,
        "indexed_column_count": sum(item["indexed_column_count"] for item in summaries.values()),
        "indexed_value_count": sum(item["indexed_value_count"] for item in summaries.values()),
        "databases": summaries,
        "generalization_boundary": (
            "The index uses database contents and schema available at inference time only. "
            "Gold SQL and schema labels are never read."
        ),
    }
    write_json(output_file.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
