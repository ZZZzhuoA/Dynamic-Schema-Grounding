import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
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


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_by_question_id(path: Path, limit=None):
    records = read_jsonl(path, limit=limit)
    result = {}
    for index, item in enumerate(records):
        question_id = item.get("question_id")
        if question_id is None:
            question_id = index
        result[int(question_id)] = item
    return result


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


def normalize_name(name: str):
    return re.sub(r"\s+", " ", str(name).strip().lower())


def normalize_sql(sql: str):
    sql = str(sql or "").lower()
    sql = sql.replace("`", "").replace("[", "").replace("]", "").replace('"', "")
    return re.sub(r"\s+", " ", sql)


def owner_table(name: str):
    if "." in name:
        return name.split(".", 1)[0]
    return name


def split_schema_items(label_record):
    name_to_type = {}
    table_names = set()
    column_names = set()
    column_to_table = {}
    for item in label_record.get("schema_items", []):
        name = item.get("name")
        item_type = item.get("type")
        if not name:
            continue
        name_to_type[name] = item_type
        if item_type == "table":
            table_names.add(name)
        elif item_type == "column":
            column_names.add(name)
            column_to_table[name] = owner_table(name)
    return name_to_type, table_names, column_names, column_to_table


def gold_sets(label_record):
    name_to_type, _, _, _ = split_schema_items(label_record)
    labels = set(label_record.get("label_names", []))
    gold_tables = set()
    gold_columns = set()
    for name in labels:
        item_type = name_to_type.get(name)
        if item_type == "table":
            gold_tables.add(name)
        elif item_type == "column":
            gold_columns.add(name)
            gold_tables.add(owner_table(name))
        elif "." in name:
            gold_columns.add(name)
            gold_tables.add(owner_table(name))
        else:
            gold_tables.add(name)
    fk_endpoints = set(label_record.get("label_sources", {}).get("foreign_key", []))
    return {
        "all": labels,
        "tables": gold_tables,
        "columns": gold_columns,
        "fk_endpoints": fk_endpoints,
    }


def predicted_sets(prediction_record, k):
    top_key = f"top_{k}"
    if top_key in prediction_record:
        top = prediction_record.get(top_key, [])[:k]
    elif k <= 30:
        top = prediction_record.get("top_30", [])[:k]
    else:
        top = []
    names = {item.get("name") for item in top if item.get("name")}
    tables = set()
    columns = set()
    for item in top:
        name = item.get("name")
        item_type = item.get("type")
        if not name:
            continue
        if item_type == "table":
            tables.add(name)
        elif item_type == "column":
            columns.add(name)
            tables.add(owner_table(name))
        elif "." in name:
            columns.add(name)
            tables.add(owner_table(name))
        else:
            tables.add(name)
    return {"all": names, "tables": tables, "columns": columns}


def recall(gold, pred):
    if not gold:
        return None
    return len(set(gold) & set(pred)) / len(set(gold))


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def extract_schema_used_by_sql(sql, label_record):
    sql_norm = normalize_sql(sql)
    name_to_type, table_names, column_names, _ = split_schema_items(label_record)
    used_tables = set()
    used_columns = set()

    for table in table_names:
        pattern = r"(?<![\w])" + re.escape(normalize_name(table)) + r"(?![\w])"
        if re.search(pattern, sql_norm):
            used_tables.add(table)

    for column in column_names:
        table, column_name = column.split(".", 1) if "." in column else ("", column)
        full_norm = normalize_name(column)
        col_norm = normalize_name(column_name)
        full_hit = full_norm in sql_norm
        col_pattern = r"(?<![\w])" + re.escape(col_norm) + r"(?![\w])"
        col_hit = re.search(col_pattern, sql_norm) is not None
        if full_hit or col_hit:
            used_columns.add(column)
            if table:
                used_tables.add(table)

    return {
        "tables": sorted(used_tables),
        "columns": sorted(used_columns),
        "all": sorted(used_tables | used_columns),
    }


def evaluate_generation_records(generation_records, labels_by_qid, db_root: Path):
    sqlite_index = find_sqlite_index(db_root)
    details = {}
    for item in generation_records:
        qid = item.get("question_id")
        if qid is None:
            continue
        qid = int(qid)
        db_path = sqlite_index.get(item.get("db_id"))
        label_record = labels_by_qid.get(qid)
        pred_error = "missing database"
        gold_error = "missing database"
        execution_correct = False
        pred_ok = False
        gold_ok = False
        if db_path:
            pred = execute_sql(db_path, item.get("generated_sql") or "")
            gold = execute_sql(db_path, item.get("gold_sql") or "")
            pred_error = pred["error"]
            gold_error = gold["error"]
            pred_ok = pred["ok"]
            gold_ok = gold["ok"]
            execution_correct = pred["ok"] and gold["ok"] and pred["rows"] == gold["rows"]
        used_schema = extract_schema_used_by_sql(item.get("generated_sql") or "", label_record) if label_record else {}
        details[qid] = {
            "question_id": qid,
            "db_id": item.get("db_id"),
            "setting": item.get("setting"),
            "question": item.get("question"),
            "evidence": item.get("evidence"),
            "generated_sql": item.get("generated_sql"),
            "gold_sql": item.get("gold_sql"),
            "api_error": item.get("error"),
            "pred_error": pred_error,
            "gold_error": gold_error,
            "pred_exec_ok": pred_ok,
            "gold_exec_ok": gold_ok,
            "execution_correct": execution_correct,
            "used_schema": used_schema,
        }
    return details


def summarize_generation(details):
    total = len(details)
    correct = sum(1 for item in details.values() if item["execution_correct"])
    pred_ok = sum(1 for item in details.values() if item["pred_exec_ok"])
    api_errors = sum(1 for item in details.values() if item["api_error"])
    return {
        "sample_count": total,
        "execution_accuracy": correct / total if total else 0,
        "pred_execution_success_rate": pred_ok / total if total else 0,
        "api_error_rate": api_errors / total if total else 0,
        "correct_count": correct,
        "pred_exec_ok_count": pred_ok,
        "api_error_count": api_errors,
    }


def error_bucket(detail):
    if detail.get("api_error"):
        return "api_error"
    error = detail.get("pred_error")
    if error:
        lowered = error.lower()
        if "no such column" in lowered:
            return "no_such_column"
        if "no such table" in lowered:
            return "no_such_table"
        if "syntax" in lowered or "near " in lowered or "unrecognized token" in lowered:
            return "syntax_error"
        return "execution_error"
    if not detail.get("execution_correct"):
        return "semantic_mismatch"
    return "correct"


def build_case(
    qid,
    label_record,
    full_detail,
    rgta_detail,
    lexical_detail,
    rgta_pred,
    lexical_pred,
):
    gold = gold_sets(label_record)
    rgta30 = predicted_sets(rgta_pred, 30) if rgta_pred else {"all": set(), "tables": set(), "columns": set()}
    lexical30 = predicted_sets(lexical_pred, 30) if lexical_pred else {"all": set(), "tables": set(), "columns": set()}
    missing_rgta_columns = sorted(gold["columns"] - rgta30["columns"])
    missing_rgta_tables = sorted(gold["tables"] - rgta30["tables"])
    rgta_used = set(rgta_detail.get("used_schema", {}).get("all", [])) if rgta_detail else set()
    wrong_used = sorted(rgta_used - gold["all"])
    return {
        "question_id": qid,
        "db_id": label_record.get("db_id"),
        "difficulty": label_record.get("difficulty"),
        "question": label_record.get("question"),
        "evidence": label_record.get("evidence"),
        "gold_sql": label_record.get("sql"),
        "gold_labels": sorted(gold["all"]),
        "gold_tables": sorted(gold["tables"]),
        "gold_columns": sorted(gold["columns"]),
        "rgta_top30_hit_tables": sorted(gold["tables"] & rgta30["tables"]),
        "rgta_top30_missing_tables": missing_rgta_tables,
        "rgta_top30_hit_columns": sorted(gold["columns"] & rgta30["columns"]),
        "rgta_top30_missing_columns": missing_rgta_columns,
        "rgta_top30_recall_all": recall(gold["all"], rgta30["all"]),
        "rgta_top30_recall_tables": recall(gold["tables"], rgta30["tables"]),
        "rgta_top30_recall_columns": recall(gold["columns"], rgta30["columns"]),
        "lexical_top30_recall_all": recall(gold["all"], lexical30["all"]),
        "full_schema_correct": full_detail.get("execution_correct") if full_detail else None,
        "rgta_correct": rgta_detail.get("execution_correct") if rgta_detail else None,
        "lexical_correct": lexical_detail.get("execution_correct") if lexical_detail else None,
        "rgta_error_bucket": error_bucket(rgta_detail) if rgta_detail else "missing_generation",
        "rgta_pred_error": rgta_detail.get("pred_error") if rgta_detail else None,
        "rgta_generated_sql": rgta_detail.get("generated_sql") if rgta_detail else None,
        "rgta_used_schema": rgta_detail.get("used_schema") if rgta_detail else {},
        "rgta_wrong_used_schema": wrong_used,
        "full_generated_sql": full_detail.get("generated_sql") if full_detail else None,
        "lexical_generated_sql": lexical_detail.get("generated_sql") if lexical_detail else None,
    }


def grounding_recall_report(labels, predictions_by_name, ks):
    report = {}
    for method, predictions in predictions_by_name.items():
        max_available = 0
        if predictions:
            for item in predictions:
                for key, value in item.items():
                    if key.startswith("top_") and isinstance(value, list):
                        max_available = max(max_available, len(value))
        method_report = {"max_available_k": max_available, "by_k": {}}
        for k in ks:
            if k > max_available:
                method_report["by_k"][str(k)] = {"available": False}
                continue
            all_recalls = []
            table_recalls = []
            column_recalls = []
            fk_recalls = []
            empty_gold_columns = 0
            for label_record, pred_record in zip(labels, predictions):
                gold = gold_sets(label_record)
                pred = predicted_sets(pred_record, k)
                all_recalls.append(recall(gold["all"], pred["all"]))
                table_recalls.append(recall(gold["tables"], pred["tables"]))
                column_recall = recall(gold["columns"], pred["columns"])
                if column_recall is None:
                    empty_gold_columns += 1
                column_recalls.append(column_recall)
                fk_recalls.append(recall(gold["fk_endpoints"], pred["columns"]))
            method_report["by_k"][str(k)] = {
                "available": True,
                "schema_recall": mean(all_recalls),
                "table_recall": mean(table_recalls),
                "column_recall": mean(column_recalls),
                "fk_endpoint_recall": mean(fk_recalls),
                "empty_gold_column_count": empty_gold_columns,
            }
        report[method] = method_report
    return report


def write_summary_md(path: Path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Stage 3D-A Grounding Bottleneck Diagnosis\n")
    lines.append("## Generation results\n")
    lines.append("| Setting | EX | Exec success | Correct | Pred exec ok |")
    lines.append("|---|---:|---:|---:|---:|")
    for setting, metrics in report["generation_metrics"].items():
        lines.append(
            f"| {setting} | {metrics['execution_accuracy']:.4f} | "
            f"{metrics['pred_execution_success_rate']:.4f} | "
            f"{metrics['correct_count']} | {metrics['pred_exec_ok_count']} |"
        )
    lines.append("")
    lines.append("## Grounding recall\n")
    for method, method_report in report["grounding_recall"].items():
        lines.append(f"### {method}\n")
        lines.append(f"Max available k: {method_report['max_available_k']}\n")
        lines.append("| k | available | schema recall | table recall | column recall | FK endpoint recall |")
        lines.append("|---:|:---:|---:|---:|---:|---:|")
        for k, row in method_report["by_k"].items():
            if not row.get("available"):
                lines.append(f"| {k} | no | - | - | - | - |")
            else:
                lines.append(
                    f"| {k} | yes | {row['schema_recall']:.4f} | "
                    f"{row['table_recall']:.4f} | {row['column_recall']:.4f} | "
                    f"{row['fk_endpoint_recall']:.4f} |"
                )
        lines.append("")
    lines.append("## Case-level comparisons\n")
    for key, value in report["case_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## R-GTA error buckets\n")
    for key, value in report["rgta_error_buckets"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Preliminary diagnosis\n")
    lines.append(report["preliminary_diagnosis"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-file", default="experiments/stage1_label_extraction/bird_dev_grounding_labels.jsonl")
    parser.add_argument(
        "--rgta-predictions",
        default="experiments/stage2_rgta_torch_grounding_hybrid_500/rgcn_torch_dev_predictions.jsonl",
    )
    parser.add_argument(
        "--lexical-predictions",
        default="experiments/stage2_static_grounding/lexical_dev_predictions.jsonl",
    )
    parser.add_argument(
        "--full-generation",
        default="experiments/stage3_online_llm_generation/generations_full_schema_v2_limit100.jsonl",
    )
    parser.add_argument(
        "--rgta-generation",
        default="experiments/stage3_online_llm_generation/generations_rgta_top30_v2_limit100.jsonl",
    )
    parser.add_argument(
        "--lexical-generation",
        default="experiments/stage3_online_llm_generation/generations_lexical_top30_v2_limit100.jsonl",
    )
    parser.add_argument("--db-root", default="Data/BIRD/dev_databases")
    parser.add_argument("--output-dir", default="experiments/stage3d_diagnosis")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    labels = read_jsonl(Path(args.labels_file), limit=args.limit)
    labels_by_qid = {}
    for index, item in enumerate(labels):
        qid = item.get("question_id")
        if qid is None:
            qid = index
        labels_by_qid[int(qid)] = item

    rgta_predictions = read_jsonl(Path(args.rgta_predictions), limit=args.limit)
    lexical_predictions = read_jsonl(Path(args.lexical_predictions), limit=args.limit)
    rgta_by_qid = {int(labels[index].get("question_id", index)): item for index, item in enumerate(rgta_predictions)}
    lexical_by_qid = {int(labels[index].get("question_id", index)): item for index, item in enumerate(lexical_predictions)}

    full_generation = read_jsonl(Path(args.full_generation), limit=args.limit)
    rgta_generation = read_jsonl(Path(args.rgta_generation), limit=args.limit)
    lexical_generation = read_jsonl(Path(args.lexical_generation), limit=args.limit)
    full_details = evaluate_generation_records(full_generation, labels_by_qid, Path(args.db_root))
    rgta_details = evaluate_generation_records(rgta_generation, labels_by_qid, Path(args.db_root))
    lexical_details = evaluate_generation_records(lexical_generation, labels_by_qid, Path(args.db_root))

    generation_metrics = {
        "full_schema": summarize_generation(full_details),
        "rgta_top30": summarize_generation(rgta_details),
        "lexical_top30": summarize_generation(lexical_details),
    }

    recall_report = grounding_recall_report(
        labels,
        {
            "rgta": rgta_predictions,
            "lexical": lexical_predictions,
        },
        ks=[10, 20, 30, 50, 80],
    )

    full_correct_rgta_wrong = []
    rgta_correct_full_wrong = []
    rgta_correct_lexical_wrong = []
    lexical_correct_rgta_wrong = []
    all_cases = []
    for qid, label_record in labels_by_qid.items():
        full_detail = full_details.get(qid)
        rgta_detail = rgta_details.get(qid)
        lexical_detail = lexical_details.get(qid)
        case = build_case(
            qid,
            label_record,
            full_detail,
            rgta_detail,
            lexical_detail,
            rgta_by_qid.get(qid),
            lexical_by_qid.get(qid),
        )
        all_cases.append(case)
        if full_detail and rgta_detail and full_detail["execution_correct"] and not rgta_detail["execution_correct"]:
            full_correct_rgta_wrong.append(case)
        if full_detail and rgta_detail and rgta_detail["execution_correct"] and not full_detail["execution_correct"]:
            rgta_correct_full_wrong.append(case)
        if rgta_detail and lexical_detail and rgta_detail["execution_correct"] and not lexical_detail["execution_correct"]:
            rgta_correct_lexical_wrong.append(case)
        if rgta_detail and lexical_detail and lexical_detail["execution_correct"] and not rgta_detail["execution_correct"]:
            lexical_correct_rgta_wrong.append(case)

    rgta_error_buckets = Counter(error_bucket(item) for item in rgta_details.values())
    full_correct_rgta_wrong_missing_column_count = sum(
        1 for case in full_correct_rgta_wrong if case["rgta_top30_missing_columns"]
    )
    full_correct_rgta_wrong_missing_table_count = sum(
        1 for case in full_correct_rgta_wrong if case["rgta_top30_missing_tables"]
    )

    if full_correct_rgta_wrong:
        miss_col_rate = full_correct_rgta_wrong_missing_column_count / len(full_correct_rgta_wrong)
        miss_table_rate = full_correct_rgta_wrong_missing_table_count / len(full_correct_rgta_wrong)
    else:
        miss_col_rate = 0
        miss_table_rate = 0

    preliminary = (
        f"Among full_schema-correct but rgta_top30-wrong cases, "
        f"{full_correct_rgta_wrong_missing_column_count}/{len(full_correct_rgta_wrong)} "
        f"({miss_col_rate:.1%}) miss at least one gold column in R-GTA top30, and "
        f"{full_correct_rgta_wrong_missing_table_count}/{len(full_correct_rgta_wrong)} "
        f"({miss_table_rate:.1%}) miss at least one gold table. "
        "If missing-column rate is high, top-k recall is the primary bottleneck; "
        "otherwise prompt/reasoning/value grounding is likely dominant."
    )

    report = {
        "config": vars(args),
        "generation_metrics": generation_metrics,
        "grounding_recall": recall_report,
        "case_counts": {
            "full_correct_rgta_wrong": len(full_correct_rgta_wrong),
            "rgta_correct_full_wrong": len(rgta_correct_full_wrong),
            "rgta_correct_lexical_wrong": len(rgta_correct_lexical_wrong),
            "lexical_correct_rgta_wrong": len(lexical_correct_rgta_wrong),
            "full_correct_rgta_wrong_missing_column_count": full_correct_rgta_wrong_missing_column_count,
            "full_correct_rgta_wrong_missing_table_count": full_correct_rgta_wrong_missing_table_count,
        },
        "rgta_error_buckets": dict(rgta_error_buckets),
        "preliminary_diagnosis": preliminary,
    }

    write_json(output_dir / "grounding_recall_report.json", report)
    write_jsonl(output_dir / "all_cases.jsonl", all_cases)
    write_jsonl(output_dir / "full_correct_rgta_wrong_cases.jsonl", full_correct_rgta_wrong)
    write_jsonl(output_dir / "rgta_correct_full_wrong_cases.jsonl", rgta_correct_full_wrong)
    write_jsonl(output_dir / "rgta_correct_lexical_wrong_cases.jsonl", rgta_correct_lexical_wrong)
    write_jsonl(output_dir / "lexical_correct_rgta_wrong_cases.jsonl", lexical_correct_rgta_wrong)
    write_summary_md(output_dir / "stage3d_summary.md", report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
