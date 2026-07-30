import argparse
import json
import sqlite3
from pathlib import Path


def read_jsonl(path: Path, limit=None):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_sqlite_index(db_root: Path):
    return {path.stem: path for path in db_root.rglob("*.sqlite")}


def execute_sql(db_path: Path, sql: str):
    if not sql or not sql.strip():
        return {"ok": False, "rows": None, "error": "empty SQL"}
    connection = sqlite3.connect(str(db_path))
    try:
        cursor = connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        normalized = sorted([tuple(str(value) for value in row) for row in rows])
        return {"ok": True, "rows": normalized, "error": None}
    except Exception as exc:
        return {"ok": False, "rows": None, "error": str(exc)}
    finally:
        connection.close()


def evaluate_file(generation_file: Path, db_root: Path, output_dir: Path, limit=None):
    records = read_jsonl(generation_file, limit=limit)
    sqlite_index = find_sqlite_index(db_root)
    details = []
    correct = 0
    pred_exec_ok = 0
    gold_exec_ok = 0
    api_errors = 0

    for item in records:
        db_id = item.get("db_id")
        db_path = sqlite_index.get(db_id)
        detail = {
            "question_id": item.get("question_id"),
            "db_id": db_id,
            "setting": item.get("setting"),
            "generated_sql": item.get("generated_sql"),
            "gold_sql": item.get("gold_sql"),
            "api_error": item.get("error"),
            "pred_error": None,
            "gold_error": None,
            "execution_correct": False,
        }
        if item.get("error"):
            api_errors += 1
        if not db_path:
            detail["pred_error"] = "missing database"
            detail["gold_error"] = "missing database"
            details.append(detail)
            continue
        pred = execute_sql(db_path, item.get("generated_sql") or "")
        gold = execute_sql(db_path, item.get("gold_sql") or "")
        detail["pred_error"] = pred["error"]
        detail["gold_error"] = gold["error"]
        if pred["ok"]:
            pred_exec_ok += 1
        if gold["ok"]:
            gold_exec_ok += 1
        if pred["ok"] and gold["ok"] and pred["rows"] == gold["rows"]:
            correct += 1
            detail["execution_correct"] = True
        details.append(detail)

    total = len(records)
    metrics = {
        "generation_file": str(generation_file),
        "sample_count": total,
        "execution_accuracy": correct / total if total else 0,
        "pred_execution_success_rate": pred_exec_ok / total if total else 0,
        "gold_execution_success_rate": gold_exec_ok / total if total else 0,
        "api_error_rate": api_errors / total if total else 0,
        "correct_count": correct,
        "pred_exec_ok_count": pred_exec_ok,
        "gold_exec_ok_count": gold_exec_ok,
        "api_error_count": api_errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = generation_file.stem
    with (output_dir / f"{stem}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_jsonl(output_dir / f"{stem}_execution_details.jsonl", details)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-file", required=True)
    parser.add_argument("--db-root", default="Data/BIRD/dev_databases")
    parser.add_argument("--output-dir", default="experiments/stage3_online_llm_generation/evaluation")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    metrics = evaluate_file(
        Path(args.generation_file),
        Path(args.db_root),
        Path(args.output_dir),
        limit=args.limit,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
