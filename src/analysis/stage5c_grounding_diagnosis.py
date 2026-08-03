import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CLAUSES = ["select", "from", "join", "where", "group_by", "order_by", "having"]


def read_jsonl(path: Path, limit=None):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_name(text):
    text = str(text or "").strip().lower()
    text = text.replace("`", "").replace('"', "").replace("[", "").replace("]", "")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_sql(sql):
    text = str(sql or "").lower()
    text = text.replace("`", " ").replace('"', " ").replace("[", " ").replace("]", " ")
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def label_name_by_id(label_record):
    return {item["id"]: item["name"] for item in label_record.get("schema_items", [])}


def label_type_by_id(label_record):
    return {item["id"]: item["type"] for item in label_record.get("schema_items", [])}


def owner_table(label_name):
    return label_name.split(".", 1)[0] if "." in label_name else label_name


def top_rows(prediction, k):
    key = f"top_{k}"
    if key in prediction:
        return prediction[key][:k]
    available = sorted(
        int(field.split("_", 1)[1])
        for field in prediction
        if field.startswith("top_") and field.split("_", 1)[1].isdigit()
    )
    for candidate in available:
        if candidate >= k:
            return prediction[f"top_{candidate}"][:k]
    return []


def pred_ids(prediction, k):
    return {item["id"] for item in top_rows(prediction, k)}


def pred_names(prediction, k):
    return {item["name"] for item in top_rows(prediction, k)}


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def recall(gold, pred):
    gold = set(gold)
    pred = set(pred)
    if not gold:
        return None
    return len(gold & pred) / len(gold)


def precision(gold, pred):
    pred = set(pred)
    if not pred:
        return None
    return len(set(gold) & pred) / len(pred)


def split_gold_ids(label_record):
    type_by_id = label_type_by_id(label_record)
    gold = set(label_record.get("whole_sql_labels", []))
    tables = {item_id for item_id in gold if type_by_id.get(item_id) == "table"}
    columns = {item_id for item_id in gold if type_by_id.get(item_id) == "column"}
    return gold, tables, columns


def clause_spans(sql):
    normalized = normalize_sql(sql)
    clause_patterns = [
        ("select", r"\bselect\b"),
        ("from", r"\bfrom\b"),
        ("join", r"\bjoin\b"),
        ("where", r"\bwhere\b"),
        ("group_by", r"\bgroup\s+by\b"),
        ("order_by", r"\border\s+by\b"),
        ("having", r"\bhaving\b"),
    ]
    matches = []
    for name, pattern in clause_patterns:
        for match in re.finditer(pattern, normalized):
            matches.append((match.start(), name))
    matches.sort()
    spans = defaultdict(str)
    for idx, (start, name) in enumerate(matches):
        end = matches[idx + 1][0] if idx + 1 < len(matches) else len(normalized)
        spans[name] += " " + normalized[start:end]
    return {key: value.strip() for key, value in spans.items()}


def label_appears_in_text(label_name, text):
    if not text:
        return False
    parts = label_name.split(".", 1)
    column_name = parts[1] if len(parts) == 2 else label_name
    candidates = {normalize_name(label_name), normalize_name(column_name)}
    normalized_text = f" {normalize_sql(text)} "
    for candidate in candidates:
        if candidate and re.search(rf"(?<![a-z0-9_]){re.escape(candidate)}(?![a-z0-9_])", normalized_text):
            return True
        # Multi-word quoted identifiers become separated tokens in normalized SQL.
        loose = " ".join(token for token in candidate.split("_") if token)
        if loose and f" {loose} " in normalized_text:
            return True
    return False


def classify_gold_column_clauses(label_record):
    spans = clause_spans(label_record.get("sql") or "")
    name_by_id = label_name_by_id(label_record)
    _, _, gold_columns = split_gold_ids(label_record)
    result = {}
    for item_id in gold_columns:
        name = name_by_id.get(item_id, "")
        matched = []
        for clause in CLAUSES:
            if label_appears_in_text(name, spans.get(clause, "")):
                matched.append(clause)
        if not matched:
            matched = ["other"]
        result[item_id] = matched
    return result


def summarize_method(labels, predictions, ks):
    by_k = {}
    missing_by_k = {}
    for k in ks:
        schema_recalls = []
        schema_precisions = []
        table_recalls = []
        column_recalls = []
        clause_recalls = defaultdict(list)
        missing_counter = Counter()
        missing_clause_counter = Counter()
        samples_missing_any_column = 0
        samples_missing_any_schema = 0

        for label_record, pred_record in zip(labels, predictions):
            gold, gold_tables, gold_columns = split_gold_ids(label_record)
            predicted = pred_ids(pred_record, k)
            name_by_id = label_name_by_id(label_record)
            clauses_by_column = classify_gold_column_clauses(label_record)

            schema_recalls.append(recall(gold, predicted))
            schema_precisions.append(precision(gold, predicted))
            table_recalls.append(recall(gold_tables, predicted))
            column_recalls.append(recall(gold_columns, predicted))

            missing = gold - predicted
            missing_columns = gold_columns - predicted
            if missing:
                samples_missing_any_schema += 1
            if missing_columns:
                samples_missing_any_column += 1

            for item_id in missing:
                missing_counter[name_by_id.get(item_id, str(item_id))] += 1
            for item_id in gold_columns:
                for clause in clauses_by_column.get(item_id, ["other"]):
                    clause_recalls[clause].append(1.0 if item_id in predicted else 0.0)
                    if item_id not in predicted:
                        missing_clause_counter[clause] += 1

        by_k[str(k)] = {
            "schema_recall": mean(schema_recalls),
            "schema_precision": mean(schema_precisions),
            "table_recall": mean(table_recalls),
            "column_recall": mean(column_recalls),
            "samples_missing_any_schema": samples_missing_any_schema,
            "samples_missing_any_column": samples_missing_any_column,
            "clause_column_recall": {
                clause: mean(values) for clause, values in sorted(clause_recalls.items())
            },
        }
        missing_by_k[str(k)] = {
            "top_missing_labels": missing_counter.most_common(30),
            "missing_clause_counts": dict(sorted(missing_clause_counter.items())),
        }
    return {"by_k": by_k, "missing_by_k": missing_by_k}


def compare_methods(labels, left_predictions, right_predictions, left_name, right_name, k):
    cases = []
    left_better = 0
    right_better = 0
    tied = 0
    for index, (label_record, left_pred, right_pred) in enumerate(
        zip(labels, left_predictions, right_predictions)
    ):
        gold, _, gold_columns = split_gold_ids(label_record)
        left_ids = pred_ids(left_pred, k)
        right_ids = pred_ids(right_pred, k)
        left_col_recall = recall(gold_columns, left_ids)
        right_col_recall = recall(gold_columns, right_ids)
        if left_col_recall > right_col_recall:
            left_better += 1
        elif right_col_recall > left_col_recall:
            right_better += 1
        else:
            tied += 1
        name_by_id = label_name_by_id(label_record)
        cases.append(
            {
                "index": index,
                "question_id": label_record.get("question_id", index),
                "db_id": label_record.get("db_id"),
                "question": label_record.get("question"),
                "gold_labels": [name_by_id[item_id] for item_id in sorted(gold)],
                f"{left_name}_missing_columns": [
                    name_by_id[item_id] for item_id in sorted(gold_columns - left_ids)
                ],
                f"{right_name}_missing_columns": [
                    name_by_id[item_id] for item_id in sorted(gold_columns - right_ids)
                ],
                f"{left_name}_column_recall": left_col_recall,
                f"{right_name}_column_recall": right_col_recall,
            }
        )
    return {
        "k": k,
        "left": left_name,
        "right": right_name,
        "left_better_count": left_better,
        "right_better_count": right_better,
        "tied_count": tied,
        "cases": cases,
    }


def write_summary(path: Path, report):
    def fmt(value):
        return "-" if value is None else f"{value:.4f}"

    lines = ["# Stage 5-C v2 Grounding Diagnosis", ""]
    lines.append("## Overall recall")
    lines.append("")
    lines.append("| Method | k | Schema R | Schema P | Table R | Column R | Missing-column samples |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method, method_report in report["methods"].items():
        for k, metrics in method_report["by_k"].items():
            lines.append(
                f"| {method} | {k} | {fmt(metrics['schema_recall'])} | "
                f"{fmt(metrics['schema_precision'])} | {fmt(metrics['table_recall'])} | "
                f"{fmt(metrics['column_recall'])} | {metrics['samples_missing_any_column']} |"
            )
    lines.append("")
    lines.append("## Clause-level column recall at k=30")
    lines.append("")
    lines.append("| Method | Clause | Recall |")
    lines.append("|---|---|---:|")
    for method, method_report in report["methods"].items():
        clause_rows = method_report["by_k"].get("30", {}).get("clause_column_recall", {})
        for clause, value in clause_rows.items():
            lines.append(f"| {method} | {clause} | {fmt(value)} |")
    lines.append("")
    lines.append("## DSG vs R-GTA column recall at k=30")
    cmp = report["comparisons"].get("dsg_vs_rgta_top30")
    if cmp:
        lines.append(f"- DSG better: {cmp['left_better_count']}")
        lines.append(f"- R-GTA better: {cmp['right_better_count']}")
        lines.append(f"- Tied: {cmp['tied_count']}")
    lines.append("")
    lines.append("## Top missing DSG labels at k=30")
    for name, count in report["methods"].get("dsg", {}).get("missing_by_k", {}).get("30", {}).get(
        "top_missing_labels", []
    )[:20]:
        lines.append(f"- `{name}`: {count}")
    lines.append("")
    lines.append("## Diagnosis")
    lines.append(report["diagnosis"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-file",
        default="experiments/stage1_label_extraction_v2/bird_dev_grounding_labels.jsonl",
    )
    parser.add_argument(
        "--dsg-predictions",
        default="experiments/stage5_dsg_grounder_hardneg_v2_1000/dev_predictions.jsonl",
    )
    parser.add_argument(
        "--rgta-predictions",
        default="experiments/stage2_rgta_torch_grounding_hybrid_500_top80/rgcn_torch_dev_predictions.jsonl",
    )
    parser.add_argument(
        "--lexical-predictions",
        default="experiments/stage2_static_grounding/lexical_dev_predictions.jsonl",
    )
    parser.add_argument("--output-dir", default="experiments/stage5c_grounding_diagnosis_v2_100")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--ks", default="10,20,30,50,80")
    args = parser.parse_args()

    labels = read_jsonl(Path(args.labels_file), args.limit)
    ks = [int(value.strip()) for value in args.ks.split(",") if value.strip()]
    prediction_paths = {
        "dsg": Path(args.dsg_predictions),
        "rgta": Path(args.rgta_predictions),
        "lexical": Path(args.lexical_predictions),
    }
    predictions = {
        method: read_jsonl(path, args.limit)
        for method, path in prediction_paths.items()
        if path.exists()
    }
    for method, rows in predictions.items():
        if len(rows) != len(labels):
            raise ValueError(f"Length mismatch for {method}: labels={len(labels)}, predictions={len(rows)}")

    method_reports = {
        method: summarize_method(labels, rows, ks)
        for method, rows in predictions.items()
    }
    comparisons = {}
    if "dsg" in predictions and "rgta" in predictions:
        comparisons["dsg_vs_rgta_top30"] = compare_methods(
            labels, predictions["dsg"], predictions["rgta"], "dsg", "rgta", 30
        )
    if "dsg" in predictions and "lexical" in predictions:
        comparisons["dsg_vs_lexical_top30"] = compare_methods(
            labels, predictions["dsg"], predictions["lexical"], "dsg", "lexical", 30
        )

    dsg_30 = method_reports.get("dsg", {}).get("by_k", {}).get("30", {})
    diagnosis = (
        "Under v2 labels, this report measures real column-level coverage after fixing quoted "
        "identifier parsing. Focus on column_recall@30 and clause-level recalls: if WHERE/ORDER "
        "columns are low, the next model change should be clause-aware or value-aware grounding; "
        "if SELECT columns dominate misses, improve question/evidence lexical alignment."
    )
    if dsg_30:
        clause = dsg_30.get("clause_column_recall", {})
        lowest = sorted(clause.items(), key=lambda x: x[1])[:3]
        diagnosis += " DSG weakest clause recalls at k=30: " + ", ".join(
            f"{name}={value:.4f}" for name, value in lowest
        )

    report = {
        "config": vars(args),
        "methods": method_reports,
        "comparisons": {
            key: {k: v for k, v in value.items() if k != "cases"}
            for key, value in comparisons.items()
        },
        "diagnosis": diagnosis,
    }
    output_dir = Path(args.output_dir)
    write_json(output_dir / "stage5c_report.json", report)
    for key, value in comparisons.items():
        write_jsonl(output_dir / f"{key}_cases.jsonl", value["cases"])
    write_summary(output_dir / "stage5c_summary.md", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
