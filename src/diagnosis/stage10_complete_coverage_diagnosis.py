import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stage13_build_typed_ra_data import explicit_sql_join_edges  # noqa: E402
from src.data.stage5g_build_clause_labels import transform_record  # noqa: E402


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_index(row, fallback):
    for key in ["record_index", "question_id", "index"]:
        if row.get(key) is not None:
            return int(row[key])
    return int(fallback)


def index_rows(rows):
    result = {}
    for fallback, row in enumerate(rows):
        key = row_index(row, fallback)
        if key in result:
            raise ValueError(f"Duplicate record index: {key}")
        result[key] = row
    return result


def prediction_top_key(row):
    keys = [
        key
        for key, value in row.items()
        if key.startswith("top_") and isinstance(value, list)
    ]
    return max(keys, key=lambda key: int(key.split("_", 1)[1])) if keys else None


def item_id(item):
    if isinstance(item, int):
        return int(item)
    for key in ["schema_item_id", "id"]:
        if item.get(key) is not None:
            return int(item[key])
    raise ValueError(f"Prediction item has no schema id: {item}")


def prediction_sets(row):
    top_key = prediction_top_key(row)
    constrained = {item_id(item) for item in row.get(top_key, [])} if top_key else set()
    return {
        "top_key": top_key,
        "constrained": constrained,
        "raw": {int(value) for value in row.get("raw_top_ids", [])},
        "baseline": {int(value) for value in row.get("baseline_top_ids", [])},
    }


def label_target(row):
    return {int(value) for value in row.get("whole_sql_labels", [])}


def schema_names(row):
    return {int(item["id"]): item.get("name") for item in row.get("schema_items", [])}


def semantic_and_join_targets(label_row):
    """Separate exact semantic pointers from one reference SQL join realization."""
    transformed = transform_record(label_row)
    by_id = {int(item["id"]): item for item in label_row.get("schema_items", [])}
    whole = label_target(label_row)
    required_tables = {
        schema_id for schema_id in whole if by_id.get(schema_id, {}).get("type") == "table"
    }
    semantic_columns = set()
    for clause in ["select", "where", "group_by", "having", "order_by"]:
        semantic_columns.update(
            int(schema_id)
            for schema_id in transformed.get("clause_labels", {}).get(clause, [])
            if by_id.get(int(schema_id), {}).get("type") == "column"
        )
    reference_join_columns = {
        int(schema_id)
        for schema_id in transformed.get("clause_labels", {}).get("join", [])
        if by_id.get(int(schema_id), {}).get("type") == "column"
    }
    return {
        "semantic": required_tables | semantic_columns,
        "required_tables": required_tables,
        "reference_join": reference_join_columns,
        "reference_join_edges": explicit_sql_join_edges(
            label_row.get("sql") or "", label_row.get("schema_items", [])
        ),
    }


def selected_join_connected(record, selected):
    """Test connectivity of required tables in the selected induced schema graph."""
    required_tables = set(record.get("required_tables", []))
    if not required_tables:
        return True
    if not required_tables.issubset(selected):
        return False
    if len(required_tables) == 1:
        return True

    nodes = record.get("candidate_nodes", [])
    local_to_schema = {
        int(node["local_id"]): int(node["schema_item_id"]) for node in nodes
    }
    adjacency = defaultdict(set)

    def add_edge(left, right):
        left, right = int(left), int(right)
        if left in selected and right in selected:
            adjacency[left].add(right)
            adjacency[right].add(left)

    for edge in record.get("schema_edges", []):
        left = local_to_schema.get(int(edge["src"]))
        right = local_to_schema.get(int(edge["dst"]))
        if left is not None and right is not None:
            add_edge(left, right)
    # Explicit non-FK equalities used by the reference SQL are also legal edges.
    for edge in record.get("reference_join_edges", []):
        add_edge(edge["left_column_id"], edge["right_column_id"])

    start = next(iter(required_tables))
    visited, frontier = set(), [start]
    while frontier:
        node = frontier.pop()
        if node in visited:
            continue
        visited.add(node)
        frontier.extend(adjacency[node] - visited)
    return required_tables.issubset(visited)


def summarize_structural_coverage(records):
    stages = ["candidate", "raw", "constrained", "baseline"]
    counts = {
        stage: Counter(
            semantic_complete=0,
            reference_join_complete=0,
            join_connected=0,
            grounding_complete=0,
        )
        for stage in stages
    }
    join_required_samples = sum(len(record.get("required_tables", [])) > 1 for record in records)
    transitions = Counter()
    failures = []
    for record in records:
        semantic = set(record.get("semantic", []))
        reference_join = set(record.get("reference_join", []))
        stage_details = {}
        for stage in stages:
            selected = set(record[stage])
            semantic_complete = semantic.issubset(selected)
            reference_complete = reference_join.issubset(selected)
            connected = selected_join_connected(record, selected)
            grounding_complete = semantic_complete and connected
            counts[stage]["semantic_complete"] += int(semantic_complete)
            counts[stage]["reference_join_complete"] += int(reference_complete)
            counts[stage]["join_connected"] += int(connected)
            counts[stage]["grounding_complete"] += int(grounding_complete)
            stage_details[stage] = {
                "semantic_complete": semantic_complete,
                "reference_join_complete": reference_complete,
                "join_connected": connected,
                "grounding_complete": grounding_complete,
            }
        constrained = stage_details["constrained"]
        if constrained["grounding_complete"] and not constrained["reference_join_complete"]:
            transitions["alternate_join_path_accepted"] += 1
        if constrained["reference_join_complete"] and not constrained["join_connected"]:
            transitions["reference_columns_present_but_disconnected"] += 1
        if not constrained["grounding_complete"]:
            if not constrained["semantic_complete"]:
                reason = "semantic_missing"
            else:
                reason = "join_not_connected"
            transitions[reason] += 1
            failures.append(
                {
                    "record_index": record["record_index"],
                    "db_id": record.get("db_id"),
                    "question": record.get("question"),
                    "reason": reason,
                    "semantic_missing_ids": sorted(semantic - record["constrained"]),
                    "semantic_missing_names": [
                        record["names"].get(schema_id, f"id:{schema_id}")
                        for schema_id in sorted(semantic - record["constrained"])
                    ],
                    "required_table_ids": sorted(record.get("required_tables", [])),
                    "reference_join_ids": sorted(reference_join),
                    "reference_join_missing_ids": sorted(reference_join - record["constrained"]),
                }
            )

    sample_count = len(records)
    metrics = {"sample_count": sample_count, "join_required_samples": join_required_samples}
    for stage in stages:
        metrics[stage] = {
            f"{name}_samples": value
            for name, value in counts[stage].items()
        }
        metrics[stage].update(
            {
                f"{name}_coverage": value / sample_count if sample_count else 0.0
                for name, value in counts[stage].items()
            }
        )
    metrics["constrained_outcomes"] = dict(transitions)
    return metrics, failures


def summarize_policy(records, policy, top_k):
    count = len(records)
    complete = Counter()
    recall_sums = Counter()
    failure_reasons = Counter()
    missing_names = Counter()
    failures = []

    for record in records:
        gold = record[policy]
        candidate = record["candidate"]
        raw = record["raw"]
        constrained = record["constrained"]
        baseline = record["baseline"]
        sets = {
            "candidate": candidate,
            "raw": raw,
            "constrained": constrained,
            "baseline": baseline,
        }
        for name, selected in sets.items():
            complete[name] += int(gold.issubset(selected))
            recall_sums[name] += len(gold & selected) / len(gold) if gold else 1.0

        if gold.issubset(constrained):
            continue
        if len(gold) > top_k:
            reason = "target_exceeds_top_k"
        elif not gold.issubset(candidate):
            reason = "candidate_missing"
        elif gold.issubset(raw):
            reason = "constrained_lost_after_raw_complete"
        else:
            reason = "reranker_raw_missing"
        failure_reasons[reason] += 1
        missing = sorted(gold - constrained)
        for schema_id in missing:
            missing_names[record["names"].get(schema_id, f"id:{schema_id}")] += 1
        failures.append(
            {
                "record_index": record["record_index"],
                "db_id": record.get("db_id"),
                "question": record.get("question"),
                "policy": policy,
                "reason": reason,
                "gold_count": len(gold),
                "candidate_missing_ids": sorted(gold - candidate),
                "raw_missing_ids": sorted(gold - raw),
                "constrained_missing_ids": missing,
                "constrained_missing_names": [
                    record["names"].get(schema_id, f"id:{schema_id}")
                    for schema_id in missing
                ],
            }
        )

    metrics = {
        "sample_count": count,
        "top_k": top_k,
        "candidate_complete_samples": complete["candidate"],
        "candidate_complete_coverage": complete["candidate"] / count if count else 0.0,
        "candidate_schema_recall": recall_sums["candidate"] / count if count else 0.0,
        "raw_complete_samples": complete["raw"],
        "raw_complete_coverage": complete["raw"] / count if count else 0.0,
        "raw_schema_recall": recall_sums["raw"] / count if count else 0.0,
        "constrained_complete_samples": complete["constrained"],
        "constrained_complete_coverage": complete["constrained"] / count if count else 0.0,
        "constrained_schema_recall": recall_sums["constrained"] / count if count else 0.0,
        "baseline_complete_samples": complete["baseline"],
        "baseline_complete_coverage": complete["baseline"] / count if count else 0.0,
        "failure_reasons": dict(failure_reasons),
        "top_missing_names": missing_names.most_common(50),
    }
    return metrics, failures


def align_records(graph_rows, prediction_rows, label_sets):
    graphs = index_rows(graph_rows)
    predictions = index_rows(prediction_rows)
    labels = {name: index_rows(rows) for name, rows in label_sets.items()}
    common = set(graphs) & set(predictions)
    for rows in labels.values():
        common &= set(rows)
    if not common:
        raise ValueError("No aligned records across graph, prediction, and label files")

    aligned = []
    for index in sorted(common):
        graph = graphs[index]
        prediction = predictions[index]
        selected = prediction_sets(prediction)
        names = {}
        targets = {}
        for policy, rows in labels.items():
            targets[policy] = label_target(rows[index])
            names.update(schema_names(rows[index]))
        exact_targets = semantic_and_join_targets(labels["exact"][index])
        aligned.append(
            {
                "record_index": index,
                "db_id": graph.get("db_id") or prediction.get("db_id"),
                "question": graph.get("question") or prediction.get("question"),
                "candidate": {
                    int(node["schema_item_id"])
                    for node in graph.get("candidate_nodes", [])
                },
                "candidate_nodes": graph.get("candidate_nodes", []),
                "schema_edges": graph.get("schema_edges", []),
                "raw": selected["raw"],
                "constrained": selected["constrained"],
                "baseline": selected["baseline"],
                "names": names,
                **exact_targets,
                **targets,
            }
        )
    return aligned, {
        "graph_count": len(graphs),
        "prediction_count": len(predictions),
        "label_counts": {name: len(rows) for name, rows in labels.items()},
        "aligned_count": len(aligned),
    }


def policy_transition(records, old_policy, new_policy):
    counts = Counter()
    examples = defaultdict(list)
    for record in records:
        old_complete = record[old_policy].issubset(record["constrained"])
        new_complete = record[new_policy].issubset(record["constrained"])
        transition = f"old_{'complete' if old_complete else 'missing'}__new_{'complete' if new_complete else 'missing'}"
        counts[transition] += 1
        if old_complete != new_complete:
            examples[transition].append(record["record_index"])
    return {"counts": dict(counts), "changed_indices": dict(examples)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-graph-file", required=True)
    parser.add_argument("--prediction-file", required=True)
    parser.add_argument("--legacy-label-file", required=True)
    parser.add_argument("--exact-label-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    aligned, integrity = align_records(
        read_jsonl(args.factor_graph_file),
        read_jsonl(args.prediction_file),
        {
            "legacy": read_jsonl(args.legacy_label_file),
            "exact": read_jsonl(args.exact_label_file),
        },
    )
    legacy_metrics, legacy_failures = summarize_policy(aligned, "legacy", args.top_k)
    exact_metrics, exact_failures = summarize_policy(aligned, "exact", args.top_k)
    structural_metrics, structural_failures = summarize_structural_coverage(aligned)
    summary = {
        "config": vars(args),
        "integrity": integrity,
        "legacy": legacy_metrics,
        "exact": exact_metrics,
        "structural": structural_metrics,
        "delta_exact_minus_legacy": {
            key: exact_metrics[key] - legacy_metrics[key]
            for key in [
                "candidate_complete_coverage",
                "candidate_schema_recall",
                "raw_complete_coverage",
                "raw_schema_recall",
                "constrained_complete_coverage",
                "constrained_schema_recall",
                "baseline_complete_coverage",
            ]
        },
        "transition": policy_transition(aligned, "legacy", "exact"),
        "interpretation": (
            "Candidate missing is a retrieval ceiling; reranker raw missing is a score/order failure; "
            "constrained_lost_after_raw_complete isolates decoder-induced loss. Gold labels are used "
            "only for offline diagnosis."
        ),
    }
    output_dir = Path(args.output_dir)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "legacy_failures.jsonl", legacy_failures)
    write_jsonl(output_dir / "exact_failures.jsonl", exact_failures)
    write_jsonl(output_dir / "structural_failures.jsonl", structural_failures)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
