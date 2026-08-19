"""Execute real LLM SQL candidates and parse each into its own typed plan."""

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stage5g_build_clause_labels import build_schema_index, clause_labels_from_sql
from src.data.stage13_build_typed_ra_data import build_typed_plan
from src.data.stage13b_prepare_typed_trajectories import prepare_steps


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def graph_index(path):
    result = {}
    for fallback, row in enumerate(read_jsonl(path)):
        index = int(row.get("record_index", row.get("metadata", {}).get("record_index", fallback)))
        result[index] = row
    return result


def sqlite_index(root):
    return {path.stem: path for path in Path(root).rglob("*.sqlite")}


def execute_read_only(path, sql, timeout_seconds=30.0, max_rows=100000):
    if not str(sql or "").strip():
        return {"ok": False, "rows": None, "error": "empty SQL", "elapsed_seconds": 0.0}
    start = time.monotonic()
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA query_only = ON")

    def interrupted():
        return int(time.monotonic() - start > timeout_seconds)

    connection.set_progress_handler(interrupted, 10000)
    try:
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return {"ok": False, "rows": None, "error": f"row limit exceeded: {max_rows}", "elapsed_seconds": time.monotonic() - start}
        normalized = sorted(tuple(str(value) for value in row) for row in rows)
        return {"ok": True, "rows": normalized, "error": None, "elapsed_seconds": time.monotonic() - start}
    except Exception as exc:
        return {"ok": False, "rows": None, "error": str(exc), "elapsed_seconds": time.monotonic() - start}
    finally:
        connection.close()


def candidate_record(base, sql):
    inputs = base.get("inference_inputs", base)
    schema_items = inputs.get("schema_items") or inputs.get("schema_nodes") or []
    schema_edges = inputs.get("schema_edges", [])
    schema = build_schema_index(schema_items)
    labels, _ = clause_labels_from_sql(sql, schema, candidate_column_ids=None)
    pointer_ids = sorted({item_id for ids in labels.values() for item_id in ids})
    by_id = {int(item["id"]): item for item in schema_items}
    names = [by_id[item_id]["name"] for item_id in pointer_ids if item_id in by_id]
    return {
        "split": base.get("split", "dev"),
        "question_id": base.get("question_id"),
        "db_id": base.get("db_id"),
        "question": inputs.get("question", base.get("question", "")),
        "evidence": inputs.get("evidence", base.get("evidence", "")),
        "sql": sql,
        "schema_items": schema_items,
        "schema_edges": schema_edges,
        "whole_sql_labels": pointer_ids,
        "label_sources": {"sql_parse": names, "foreign_key": []},
    }


def parse_candidate(base, graph, sql, record_index):
    if not str(sql or "").strip().lower().startswith(("select", "with")):
        return None, "not_select_or_with"
    try:
        graph_inputs = graph.get("inference_inputs", graph)
        parse_base = {
            **base,
            "inference_inputs": {
                **graph_inputs,
                "question": base.get("question", graph_inputs.get("question", "")),
                "evidence": base.get("evidence", graph_inputs.get("evidence", "")),
            },
        }
        plan = build_typed_plan(candidate_record(parse_base, sql), graph, record_index)
        if not plan["training_targets"]["action_sequence"]:
            return None, "empty_action_sequence"
        return {
            "steps": prepare_steps(plan),
            "parse_status": plan["audit"]["parse_status"],
            "unsupported_features": plan["audit"]["unsupported_features"],
            "schema_label_coverage": plan["audit"]["schema_label_coverage"],
        }, None
    except Exception as exc:
        return None, repr(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-file", required=True)
    parser.add_argument("--graph-file", required=True)
    parser.add_argument("--db-root", default="Data/BIRD/dev_databases")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-rows", type=int, default=100000)
    args = parser.parse_args()

    groups = read_jsonl(args.generation_file)
    graphs = graph_index(args.graph_file)
    databases = sqlite_index(args.db_root)
    output, stats = [], Counter()
    for group in groups:
        index = int(group["record_index"])
        graph = graphs.get(index)
        database = databases.get(group.get("db_id"))
        if graph is None:
            stats["missing_graph"] += 1
            continue
        if database is None:
            stats["missing_database"] += 1
            continue
        gold = execute_read_only(database, group.get("gold_sql"), args.timeout_seconds, args.max_rows)
        candidates = []
        for raw in group.get("candidates", []):
            candidate = dict(raw)
            execution = execute_read_only(database, candidate.get("generated_sql"), args.timeout_seconds, args.max_rows)
            parsed, parse_error = parse_candidate(group, graph, candidate.get("generated_sql"), index)
            candidate.update(
                {
                    "execution_ok": execution["ok"],
                    "execution_error": execution["error"],
                    "execution_seconds": execution["elapsed_seconds"],
                    "execution_correct": bool(execution["ok"] and gold["ok"] and execution["rows"] == gold["rows"]),
                    "label": int(execution["ok"] and gold["ok"] and execution["rows"] == gold["rows"]),
                    "parse_ok": parsed is not None,
                    "parse_error": parse_error,
                    "steps": parsed["steps"] if parsed else [],
                    "parse_status": parsed["parse_status"] if parsed else "failed",
                    "unsupported_features": parsed["unsupported_features"] if parsed else [],
                }
            )
            stats["candidate_count"] += 1
            stats["execution_ok_count"] += int(candidate["execution_ok"])
            stats["execution_correct_count"] += int(candidate["execution_correct"])
            stats["parse_ok_count"] += int(candidate["parse_ok"])
            stats[f"parse_status::{candidate['parse_status']}"] += 1
            candidates.append(candidate)
        inputs = graph.get("inference_inputs", graph)
        output.append(
            {
                "split": "dev",
                "record_index": index,
                "question_id": group.get("question_id"),
                "db_id": group.get("db_id"),
                "gold_sql": group.get("gold_sql"),
                "gold_execution_ok": gold["ok"],
                "gold_execution_error": gold["error"],
                "inference_inputs": inputs,
                "candidates": candidates,
                "metadata": {
                    "candidate_data_version": "stage15b_real_llm_sql_v1",
                    "candidate_count": len(candidates),
                    "correct_candidate_count": sum(c["execution_correct"] for c in candidates),
                },
            }
        )
        stats["group_count"] += 1
        stats["oracle_group_count"] += int(any(c["execution_correct"] for c in candidates))
        stats["gold_execution_ok_count"] += int(gold["ok"])

    write_jsonl(args.output_file, output)
    summary = {
        "config": vars(args),
        **dict(stats),
        "candidate_execution_rate": stats["execution_ok_count"] / stats["candidate_count"] if stats["candidate_count"] else 0.0,
        "candidate_parse_rate": stats["parse_ok_count"] / stats["candidate_count"] if stats["candidate_count"] else 0.0,
        "oracle_ex_at_k": stats["oracle_group_count"] / stats["group_count"] if stats["group_count"] else 0.0,
        "leakage_note": "Gold execution results produce evaluation labels only; candidate typed plans are parsed solely from each candidate SQL.",
    }
    summary_path = Path(args.summary_file) if args.summary_file else Path(args.output_file).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
