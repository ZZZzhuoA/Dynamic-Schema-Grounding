"""Diagnose Stage 17-B full-schema coverage failures.

This script only reads leakage-free prediction rows plus offline gold labels for
post-hoc evaluation. It is intended for analysis, not inference.
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict, deque
from pathlib import Path


TOP_BUCKETS = (
    ("31_40", 31, 40),
    ("41_50", 41, 50),
    ("51_80", 51, 80),
    ("81_120", 81, 120),
    ("gt_120", 121, math.inf),
)


def read_jsonl(path, limit=None):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    return records


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def assignment(value):
    if "=" not in value:
        raise ValueError(f"Expected SEED=DIR or NAME=FILE, got {value!r}")
    key, path = value.split("=", 1)
    return key.strip(), Path(path)


def prediction_key(record):
    question_id = record.get("question_id")
    if question_id is None:
        return None
    return str(record.get("db_id")), str(question_id)


def rows_by_record_and_key(records, name):
    by_record = {}
    by_key = {}
    for index, record in enumerate(records):
        record_index = int(
            record.get("record_index", record.get("metadata", {}).get("record_index", index))
        )
        if record_index in by_record:
            raise ValueError(f"Duplicate {name} record_index={record_index}")
        by_record[record_index] = record
        key = prediction_key(record)
        if key is not None:
            if key in by_key:
                raise ValueError(f"Duplicate {name} key={key}")
            by_key[key] = record
    return by_record, by_key


def graph_metadata(graph, fallback_index):
    inputs = graph.get("inference_inputs", {})
    metadata = graph.get("metadata", {})
    return {
        "record_index": int(metadata.get("record_index", fallback_index)),
        "db_id": inputs.get("db_id"),
        "question_id": metadata.get("question_id"),
        "question": inputs.get("question"),
        "nodes": inputs.get("schema_nodes", []),
        "edges": inputs.get("schema_edges", []),
    }


def align_records(predictions, labels, graphs):
    label_by_record, label_by_key = rows_by_record_and_key(labels, "label")
    graph_rows = [graph_metadata(graph, index) for index, graph in enumerate(graphs)]
    graph_by_record, graph_by_key = rows_by_record_and_key(graph_rows, "graph")
    aligned = []
    used = set()
    for prediction in predictions:
        record_index = int(prediction["record_index"])
        key = prediction_key(prediction)
        label = label_by_key.get(key) if key is not None else None
        graph = graph_by_key.get(key) if key is not None else None
        if label is None:
            label = label_by_record.get(record_index)
        if graph is None:
            graph = graph_by_record.get(record_index)
        if label is None or graph is None:
            raise ValueError(
                f"Could not align prediction record_index={record_index}, key={key}"
            )
        if record_index in used:
            raise ValueError(f"Prediction record_index={record_index} aligned twice")
        used.add(record_index)
        if str(prediction.get("db_id")) != str(label.get("db_id")):
            raise ValueError(f"Prediction/label db mismatch at record_index={record_index}")
        if str(graph.get("db_id")) != str(label.get("db_id")):
            raise ValueError(f"Graph/label db mismatch at record_index={record_index}")
        schema_items = label.get("schema_items", [])
        if int(prediction.get("schema_node_count", -1)) != len(schema_items):
            raise ValueError(f"Schema count mismatch at record_index={record_index}")
        aligned.append((prediction, label, graph))
    return aligned


def schema_size_bucket(size):
    if size <= 50:
        return "le_50"
    if size <= 100:
        return "51_100"
    if size <= 200:
        return "101_200"
    return "gt_200"


def rank_bucket(rank):
    for name, low, high in TOP_BUCKETS:
        if low <= rank <= high:
            return name
    return "unranked"


def tokens(text):
    return {tok for tok in re.split(r"[^a-z0-9]+", str(text).lower()) if len(tok) >= 2}


def node_table_name(node):
    if node.get("type") == "table":
        return str(node.get("name"))
    return str(node.get("table") or node.get("normalized_table") or "")


def build_node_maps(nodes, edges):
    node_by_id = {int(node["id"]): node for node in nodes}
    table_id_by_name = {
        str(node.get("name")): int(node["id"])
        for node in nodes
        if node.get("type") == "table"
    }
    parent_table_id = {}
    for node_id, node in node_by_id.items():
        if node.get("type") == "column":
            table_name = node.get("table")
            if table_name in table_id_by_name:
                parent_table_id[node_id] = table_id_by_name[table_name]
    adjacency = defaultdict(list)
    for edge in edges:
        src = int(edge["src"])
        dst = int(edge["dst"])
        rel = str(edge.get("type"))
        if src == dst:
            continue
        adjacency[src].append((dst, rel))
        adjacency[dst].append((src, rel))
    return node_by_id, parent_table_id, adjacency


def nearest_distance(start, targets, adjacency, max_depth=3):
    if start in targets:
        return 0
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        node_id, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nxt, _ in adjacency.get(node_id, []):
            if nxt in seen:
                continue
            if nxt in targets:
                return depth + 1
            seen.add(nxt)
            queue.append((nxt, depth + 1))
    return None


def ranked_maps(prediction):
    ranked = prediction.get("ranked_schema", [])
    rows = {}
    ranked_ids = []
    for rank, row in enumerate(ranked, start=1):
        item_id = int(row["schema_item_id"])
        ranked_ids.append(item_id)
        rows[item_id] = {
            **row,
            "rank": int(row.get("rank", rank)),
            "logit": float(row.get("logit", 0.0)),
            "probability": float(row.get("probability", 0.0)),
        }
    return ranked_ids, rows


def node_summary(node, row=None):
    payload = {
        "schema_item_id": int(node["id"]),
        "name": node.get("name"),
        "type": node.get("type"),
        "table": node.get("table") if node.get("type") == "column" else node.get("name"),
        "is_primary_key": bool(node.get("is_primary_key")),
        "is_foreign_key_endpoint": bool(node.get("is_foreign_key_endpoint")),
        "is_identifier_column": is_identifier_column(node),
        "has_official_description": bool(
            str(node.get("official_column_description") or "").strip()
        ),
        "has_official_value_description": bool(
            str(node.get("official_value_description") or "").strip()
        ),
    }
    if row:
        payload.update(
            {
                "rank": int(row["rank"]),
                "logit": float(row["logit"]),
                "probability": float(row["probability"]),
            }
        )
    return payload


def is_identifier_column(node):
    if node.get("type") != "column":
        return False
    name = str(node.get("column") or node.get("name") or "")
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    pieces = [piece for piece in re.split(r"[^a-z0-9]+", expanded.casefold()) if piece]
    return bool(pieces and (pieces[-1] in {"id", "identifier"} or name.casefold() == "id"))


def table_context(item_id, node_by_id, parent_table_id):
    node = node_by_id[item_id]
    table_id = item_id if node.get("type") == "table" else parent_table_id.get(item_id)
    table_name = node_table_name(node)
    return table_id, table_name


def has_question_overlap(node, question_tokens):
    text = " ".join(
        str(node.get(key) or "")
        for key in ("name", "table", "column", "normalized_name", "normalized_column")
    )
    return bool(tokens(text) & question_tokens)


def analyze_one(prediction, label, graph, top_k):
    nodes = graph["nodes"]
    node_by_id, parent_table_id, adjacency = build_node_maps(nodes, graph["edges"])
    ranked_ids, rank_rows = ranked_maps(prediction)
    gold = sorted({int(value) for value in label.get("whole_sql_labels", [])})
    if not gold:
        return None
    gold_set = set(gold)
    selected = set(ranked_ids[:top_k])
    missing = sorted(gold_set - selected, key=lambda item_id: rank_rows[item_id]["rank"])
    false_pos = [
        item_id for item_id in ranked_ids[:top_k] if item_id not in gold_set
    ]
    question_tokens = tokens(graph.get("question"))
    gold_tables = {
        table_context(item_id, node_by_id, parent_table_id)[0]
        for item_id in gold_set
        if table_context(item_id, node_by_id, parent_table_id)[0] is not None
    }
    selected_gold = selected & gold_set
    selected_tables = {
        table_context(item_id, node_by_id, parent_table_id)[0]
        for item_id in selected
        if table_context(item_id, node_by_id, parent_table_id)[0] is not None
    }
    boundary = rank_rows.get(ranked_ids[top_k - 1]) if len(ranked_ids) >= top_k else None
    boundary_logit = float(boundary["logit"]) if boundary else None
    gold_ranks = [rank_rows[item_id]["rank"] for item_id in gold]
    worst_gold_rank = max(gold_ranks)

    missing_details = []
    for item_id in missing:
        node = node_by_id[item_id]
        row = rank_rows[item_id]
        table_id, _ = table_context(item_id, node_by_id, parent_table_id)
        nearest_gold_selected = nearest_distance(item_id, selected_gold, adjacency)
        detail = {
            **node_summary(node, row),
            "rank_bucket": rank_bucket(int(row["rank"])),
            "margin_below_rank_k": (
                float(boundary_logit - row["logit"]) if boundary_logit is not None else None
            ),
            "parent_table_id": table_id,
            "parent_table_selected": table_id in selected_tables if table_id is not None else False,
            "parent_table_gold": table_id in gold_tables if table_id is not None else False,
            "nearest_selected_gold_distance_le3": nearest_gold_selected,
            "question_token_overlap": has_question_overlap(node, question_tokens),
        }
        missing_details.append(detail)

    false_positive_details = []
    for item_id in false_pos[: min(20, len(false_pos))]:
        node = node_by_id[item_id]
        row = rank_rows[item_id]
        table_id, _ = table_context(item_id, node_by_id, parent_table_id)
        false_positive_details.append(
            {
                **node_summary(node, row),
                "parent_table_id": table_id,
                "parent_table_selected": table_id in selected_tables if table_id is not None else False,
                "parent_table_gold": table_id in gold_tables if table_id is not None else False,
                "nearest_gold_distance_le3": nearest_distance(item_id, gold_set, adjacency),
                "question_token_overlap": has_question_overlap(node, question_tokens),
            }
        )

    return {
        "record_index": int(prediction["record_index"]),
        "db_id": prediction.get("db_id"),
        "question_id": prediction.get("question_id"),
        "question": graph.get("question"),
        "schema_node_count": len(nodes),
        "schema_size_bucket": schema_size_bucket(len(nodes)),
        "gold_count": len(gold),
        "top_k": top_k,
        "topk_budget_ratio": top_k / max(len(nodes), 1),
        "complete_coverage": not missing,
        "recall": len(selected & gold_set) / len(gold_set),
        "precision": len(selected & gold_set) / min(top_k, len(ranked_ids)),
        "gold_ranks": gold_ranks,
        "worst_gold_rank": worst_gold_rank,
        "rank_k_logit": boundary_logit,
        "missing_gold_count": len(missing),
        "false_positive_count": len(false_pos),
        "missing_gold": missing_details,
        "top_false_positives": false_positive_details,
        "gold_schema": [
            node_summary(node_by_id[item_id], rank_rows.get(item_id)) for item_id in gold
        ],
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return None
    index = min(len(values) - 1, max(0, round((len(values) - 1) * pct)))
    return values[index]


def summarize_group(rows):
    failures = [row for row in rows if not row["complete_coverage"]]
    summary = {
        "sample_count": len(rows),
        "complete_coverage@30": mean(row["complete_coverage"] for row in rows),
        "schema_recall@30": mean(row["recall"] for row in rows),
        "schema_precision@30": mean(row["precision"] for row in rows),
        "avg_schema_node_count": mean(row["schema_node_count"] for row in rows),
        "avg_gold_count": mean(row["gold_count"] for row in rows),
        "avg_topk_budget_ratio": mean(row["topk_budget_ratio"] for row in rows),
        "failed_count": len(failures),
        "avg_missing_gold_per_failed": mean(row["missing_gold_count"] for row in failures),
        "avg_false_positives@30": mean(row["false_positive_count"] for row in rows),
        "avg_worst_gold_rank": mean(row["worst_gold_rank"] for row in rows),
        "p90_worst_gold_rank": percentile([row["worst_gold_rank"] for row in rows], 0.9),
        "gold_count_gt_30_count": sum(1 for row in rows if row["gold_count"] > 30),
    }
    categories = {
        "primary_key": "is_primary_key",
        "foreign_key_endpoint": "is_foreign_key_endpoint",
        "identifier_column": "is_identifier_column",
        "official_description_column": "has_official_description",
        "official_value_description_column": "has_official_value_description",
    }
    for category, field in categories.items():
        gold_items = [item for row in rows for item in row["gold_schema"] if item.get(field)]
        selected = sum(1 for item in gold_items if int(item.get("rank", 10**9)) <= 30)
        missing = [item for item in gold_items if int(item.get("rank", 10**9)) > 30]
        summary[f"{category}_recall@30"] = selected / len(gold_items) if gold_items else None
        summary[f"{category}_gold_count"] = len(gold_items)
        summary[f"{category}_missing_count"] = len(missing)
        summary[f"{category}_rank_31_40_missing_count"] = sum(
            31 <= int(item.get("rank", 10**9)) <= 40 for item in missing
        )
        summary[f"{category}_rank_41_50_missing_count"] = sum(
            41 <= int(item.get("rank", 10**9)) <= 50 for item in missing
        )
    return summary


def row_identity(row):
    question_id = row.get("question_id")
    if question_id is not None:
        return str(row.get("db_id")), str(question_id)
    return "record_index", int(row["record_index"])


def compare_runs(baseline_rows, candidate_rows):
    baseline = {row_identity(row): row for row in baseline_rows}
    candidate = {row_identity(row): row for row in candidate_rows}
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate), key=str)
        missing_baseline = sorted(set(candidate) - set(baseline), key=str)
        raise ValueError(
            "Comparison sample mismatch: "
            f"missing_candidate={missing_candidate[:5]} missing_baseline={missing_baseline[:5]}"
        )
    recovered = []
    regressed = []
    unchanged_complete = 0
    unchanged_incomplete = 0
    for key in sorted(baseline, key=str):
        before = baseline[key]
        after = candidate[key]
        payload = {
            "db_id": after.get("db_id"),
            "question_id": after.get("question_id"),
            "record_index": after.get("record_index"),
            "baseline_worst_gold_rank": before.get("worst_gold_rank"),
            "candidate_worst_gold_rank": after.get("worst_gold_rank"),
        }
        if not before["complete_coverage"] and after["complete_coverage"]:
            recovered.append(payload)
        elif before["complete_coverage"] and not after["complete_coverage"]:
            regressed.append(payload)
        elif after["complete_coverage"]:
            unchanged_complete += 1
        else:
            unchanged_incomplete += 1
    return {
        "sample_count": len(baseline),
        "recovered_complete_count": len(recovered),
        "regressed_complete_count": len(regressed),
        "net_recovered_complete_count": len(recovered) - len(regressed),
        "unchanged_complete_count": unchanged_complete,
        "unchanged_incomplete_count": unchanged_incomplete,
        "recovered_samples": recovered,
        "regressed_samples": regressed,
    }


def count_categories(rows):
    missing = Counter()
    false_pos = Counter()
    fp_items = Counter()
    fn_items = Counter()
    for row in rows:
        for item in row["missing_gold"]:
            missing[f"type::{item['type']}"] += 1
            missing[f"rank_bucket::{item['rank_bucket']}"] += 1
            if item["parent_table_selected"]:
                missing["parent_table_selected"] += 1
            if item["parent_table_gold"]:
                missing["parent_table_gold"] += 1
            if item["question_token_overlap"]:
                missing["question_token_overlap"] += 1
            dist = item["nearest_selected_gold_distance_le3"]
            missing[f"nearest_selected_gold_distance::{dist if dist is not None else 'gt3'}"] += 1
            fn_items[(row["db_id"], item["schema_item_id"], item["name"], item["type"])] += 1
        for item in row["top_false_positives"]:
            false_pos[f"type::{item['type']}"] += 1
            if item["parent_table_gold"]:
                false_pos["parent_table_gold"] += 1
            if item["parent_table_selected"]:
                false_pos["parent_table_selected"] += 1
            if item["question_token_overlap"]:
                false_pos["question_token_overlap"] += 1
            dist = item["nearest_gold_distance_le3"]
            false_pos[f"nearest_gold_distance::{dist if dist is not None else 'gt3'}"] += 1
            fp_items[(row["db_id"], item["schema_item_id"], item["name"], item["type"])] += 1
    return {
        "missing_gold_categories": dict(sorted(missing.items())),
        "false_positive_categories_top20_per_failure": dict(sorted(false_pos.items())),
        "frequent_missing_gold": [
            {
                "db_id": key[0],
                "schema_item_id": key[1],
                "name": key[2],
                "type": key[3],
                "count": count,
            }
            for key, count in fn_items.most_common(30)
        ],
        "frequent_false_positives": [
            {
                "db_id": key[0],
                "schema_item_id": key[1],
                "name": key[2],
                "type": key[3],
                "count": count,
            }
            for key, count in fp_items.most_common(30)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", help="SEED=experiment_dir with dev_predictions.jsonl")
    parser.add_argument(
        "--prediction-file",
        action="append",
        help="NAME=prediction_file; useful for comparing a single file without a run dir",
    )
    parser.add_argument("--dev-graph-file", required=True)
    parser.add_argument("--dev-label-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--case-limit", type=int, default=200)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument(
        "--comparison",
        action="append",
        help="BASELINE=CANDIDATE run names for paired recovered/regressed analysis.",
    )
    args = parser.parse_args()

    if not args.run and not args.prediction_file:
        raise ValueError("Provide at least one --run or --prediction-file")

    graphs = read_jsonl(args.dev_graph_file, args.dev_limit)
    labels = read_jsonl(args.dev_label_file)
    output_dir = Path(args.output_dir)
    all_rows = []
    per_run = {}
    per_run_rows = {}

    run_specs = []
    for value in args.run or []:
        seed, run_dir = assignment(value)
        run_specs.append((seed, run_dir / "dev_predictions.jsonl", str(run_dir)))
    for value in args.prediction_file or []:
        name, path = assignment(value)
        run_specs.append((name, path, str(path)))

    for run_name, prediction_file, run_path in run_specs:
        predictions = read_jsonl(prediction_file)
        aligned = align_records(predictions, labels, graphs)
        rows = []
        for prediction, label, graph in aligned:
            row = analyze_one(prediction, label, graph, args.top_k)
            if row is None:
                continue
            row["run"] = str(run_name)
            rows.append(row)
        per_run[str(run_name)] = summarize_group(rows)
        per_run_rows[str(run_name)] = rows
        all_rows.extend(rows)

    comparisons = {}
    for value in args.comparison or []:
        baseline_name, candidate_path = assignment(value)
        candidate_name = str(candidate_path)
        if baseline_name not in per_run_rows or candidate_name not in per_run_rows:
            raise ValueError(
                f"Unknown comparison {value!r}; available runs={sorted(per_run_rows)}"
            )
        comparisons[f"{baseline_name}->{candidate_name}"] = compare_runs(
            per_run_rows[baseline_name], per_run_rows[candidate_name]
        )

    by_database_rows = defaultdict(list)
    by_size_rows = defaultdict(list)
    for row in all_rows:
        by_database_rows[row["db_id"]].append(row)
        by_size_rows[row["schema_size_bucket"]].append(row)

    by_database = {
        db_id: summarize_group(rows) for db_id, rows in sorted(by_database_rows.items())
    }
    by_schema_size = {
        bucket: summarize_group(rows) for bucket, rows in sorted(by_size_rows.items())
    }
    categories = count_categories(all_rows)
    low_databases = sorted(
        (
            {"db_id": db_id, **summary}
            for db_id, summary in by_database.items()
        ),
        key=lambda item: (item["complete_coverage@30"], -item["avg_schema_node_count"]),
    )
    high_databases = sorted(
        (
            {"db_id": db_id, **summary}
            for db_id, summary in by_database.items()
        ),
        key=lambda item: (-item["complete_coverage@30"], item["avg_schema_node_count"]),
    )
    failure_cases = sorted(
        [row for row in all_rows if not row["complete_coverage"]],
        key=lambda row: (
            row["missing_gold_count"],
            row["worst_gold_rank"],
            row["schema_node_count"],
        ),
        reverse=True,
    )
    near_miss_cases = sorted(
        [
            row
            for row in all_rows
            if not row["complete_coverage"] and row["worst_gold_rank"] <= args.top_k + 10
        ],
        key=lambda row: (row["worst_gold_rank"], row["missing_gold_count"]),
    )

    summary = {
        "top_k": args.top_k,
        "run_count": len(run_specs),
        "runs": {str(name): {"prediction_file": str(path), "run_path": run_path} for name, path, run_path in run_specs},
        "per_run": per_run,
        "paired_comparisons": comparisons,
        "overall": summarize_group(all_rows),
        "by_database": by_database,
        "by_schema_size": by_schema_size,
        "lowest_coverage_databases": low_databases[:15],
        "highest_coverage_databases": high_databases[:15],
        **categories,
        "interpretation_hints": [
            "Low coverage with low avg_topk_budget_ratio suggests schema-size budget pressure.",
            "Many missing gold nodes in rank_bucket::31_40 indicates near-boundary rescue loss may help.",
            "High false_positive parent_table_gold or nearest_gold_distance<=2 suggests over-selecting local neighbors around correct anchors.",
            "High missing parent_table_selected means the model finds the table but misses required columns.",
            "High question_token_overlap among false positives suggests lexical matching is too strong.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "failure_cases.jsonl", failure_cases[: args.case_limit])
    write_jsonl(output_dir / "near_miss_cases.jsonl", near_miss_cases[: args.case_limit])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
