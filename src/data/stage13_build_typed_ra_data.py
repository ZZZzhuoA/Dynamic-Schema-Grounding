"""Build typed relational-algebra supervision for Stage 13.

The output deliberately separates operator actions, schema pointers, exact value
copies, and join-path edges.  It is training data: gold SQL is never placed in
the inference input block.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stage5g_build_clause_labels import (
    scan_clause_events,
    transform_record,
)


AGGREGATE_PATTERN = re.compile(r"(?i)\b(count|sum|avg|min|max|group_concat)\s*\(")
FUNCTION_PATTERN = re.compile(
    r"(?i)\b(cast|coalesce|strftime|date|datetime|julianday|round|substr|replace|length|lower|upper)\s*\("
)
COMPARISON_PATTERN = re.compile(
    r"(?i)(?:<>|!=|<=|>=|=|<|>|\bbetween\b|\blike\b|\bin\b|\bis\s+(?:not\s+)?null\b)"
)
STRING_LITERAL_PATTERN = re.compile(r"'(?:''|[^'])*'")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_])")
SUBQUERY_PATTERN = re.compile(r"(?is)\(\s*select\b")
SET_OPERATOR_PATTERN = re.compile(r"(?i)\b(union|intersect|except)\b")


def read_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def graph_by_record_index(path, limit=None):
    if not path:
        return {}
    result = {}
    for fallback_index, row in enumerate(read_jsonl(path, limit)):
        metadata = row.get("metadata", {})
        index = int(row.get("record_index", metadata.get("record_index", fallback_index)))
        result[index] = row
    return result


def top_level_segments(sql):
    events = sorted(scan_clause_events(sql or "", include_nested=False), key=lambda row: row["start"])
    segments = defaultdict(list)
    for index, event in enumerate(events):
        end = events[index + 1]["start"] if index + 1 < len(events) else len(sql)
        segments[event["clause"]].append(sql[event["content_start"] : end].strip())
    return dict(segments)


def schema_partition(schema_items, ids):
    by_id = {int(item["id"]): item for item in schema_items}
    tables, columns = [], []
    for item_id in sorted({int(value) for value in ids or []}):
        item = by_id.get(item_id)
        if not item:
            continue
        if item.get("type") == "table":
            tables.append(item_id)
        elif item.get("type") == "column":
            columns.append(item_id)
    return tables, columns


def source_match(value, question, evidence):
    for source, text in (("question", question or ""), ("evidence", evidence or "")):
        start = text.find(value)
        if start >= 0:
            return {"source": source, "match": "exact", "start": start, "end": start + len(value)}
    folded = value.casefold()
    for source, text in (("question", question or ""), ("evidence", evidence or "")):
        start = text.casefold().find(folded)
        if start >= 0:
            return {
                "source": source,
                "match": "casefold_only",
                "start": start,
                "end": start + len(value),
                "source_surface": text[start : start + len(value)],
            }
    return {"source": "database_value_required", "match": "none", "start": None, "end": None}


def literal_targets(text, question, evidence, clause):
    targets = []
    occupied = [False] * len(text)
    for match in STRING_LITERAL_PATTERN.finditer(text):
        for index in range(match.start(), match.end()):
            occupied[index] = True
        raw = match.group(0)
        value = raw[1:-1].replace("''", "'")
        targets.append(
            {
                "clause": clause,
                "kind": "string",
                "raw_sql_literal": raw,
                "canonical_value": value,
                "case_sensitive": any(char.isalpha() for char in value),
                **source_match(value, question, evidence),
            }
        )
    for match in NUMBER_PATTERN.finditer(text):
        if any(occupied[match.start() : match.end()]):
            continue
        raw = match.group(0)
        targets.append(
            {
                "clause": clause,
                "kind": "number",
                "raw_sql_literal": raw,
                "canonical_value": raw,
                "case_sensitive": False,
                **source_match(raw, question, evidence),
            }
        )
    return targets


def operator_targets(text):
    targets = []
    targets.extend(match.group(1).upper() for match in AGGREGATE_PATTERN.finditer(text))
    targets.extend(match.group(1).upper() for match in FUNCTION_PATTERN.finditer(text))
    targets.extend(match.group(0).upper() for match in COMPARISON_PATTERN.finditer(text))
    masked = list(text)
    quote = None
    for index, char in enumerate(text):
        if quote:
            masked[index] = " "
            if char == quote:
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
            masked[index] = " "
        elif char == "[":
            quote = "]"
            masked[index] = " "
    targets.extend(re.findall(r"[+*/-]", "".join(masked)))
    lowered = text.lower()
    if re.search(r"\bdistinct\b", lowered):
        targets.append("DISTINCT")
    if re.search(r"\bdesc\b", lowered):
        targets.append("DESC")
    elif re.search(r"\basc\b", lowered):
        targets.append("ASC")
    return list(dict.fromkeys(targets))


def inference_graph_inputs(graph):
    return graph.get("inference_inputs", graph) if graph else {}


def foreign_key_edges(graph, schema_items):
    inputs = inference_graph_inputs(graph)
    by_id = {int(item["id"]): item for item in schema_items}
    edges, seen = [], set()
    for edge in inputs.get("schema_edges", []):
        if edge.get("type") not in {"foreign_key_forward", "foreign_key_backward"}:
            continue
        left, right = int(edge["src"]), int(edge["dst"])
        key = tuple(sorted((left, right)))
        if key in seen or left not in by_id or right not in by_id:
            continue
        seen.add(key)
        left_item, right_item = by_id[left], by_id[right]
        edges.append(
            {
                "left_column_id": left,
                "right_column_id": right,
                "left_table": left_item.get("table"),
                "right_table": right_item.get("table"),
            }
        )
    return edges


def selected_join_edges(graph, schema_items, table_ids, join_column_ids):
    by_id = {int(item["id"]): item for item in schema_items}
    table_names = {str(by_id[item_id].get("name")) for item_id in table_ids if item_id in by_id}
    join_columns = set(join_column_ids)
    candidates = foreign_key_edges(graph, schema_items)
    selected = []
    for edge in candidates:
        endpoints_selected = {
            edge["left_column_id"], edge["right_column_id"]
        }.issubset(join_columns)
        tables_selected = (
            edge["left_table"] in table_names and edge["right_table"] in table_names
        )
        # Prefer explicit ON/FK endpoint supervision. Table-only closure is a
        # fallback for records whose SQL label extraction found no join columns.
        if endpoints_selected or (not join_columns and tables_selected):
            selected.append(edge)
    return selected


def join_path_connected(schema_items, table_ids, edges):
    if len(table_ids) <= 1:
        return True
    by_id = {int(item["id"]): item for item in schema_items}
    names = {str(by_id[item_id].get("name")) for item_id in table_ids if item_id in by_id}
    adjacency = defaultdict(set)
    for edge in edges:
        left, right = edge.get("left_table"), edge.get("right_table")
        if left in names and right in names:
            adjacency[left].add(right)
            adjacency[right].add(left)
    if not names:
        return False
    visited, frontier = set(), [next(iter(names))]
    while frontier:
        table = frontier.pop()
        if table in visited:
            continue
        visited.add(table)
        frontier.extend(adjacency[table] - visited)
    return names.issubset(visited)


def make_node(node_id, operator, input_id, schema_items, pointer_ids, expression="", values=None):
    tables, columns = schema_partition(schema_items, pointer_ids)
    by_id = {int(item["id"]): item for item in schema_items}
    operators = operator_targets(expression)
    if input_id is None:
        inputs = []
    elif isinstance(input_id, (list, tuple)):
        inputs = list(input_id)
    else:
        inputs = [input_id]
    return {
        "node_id": node_id,
        "operator": operator,
        "inputs": inputs,
        "output_type": "relation",
        "table_pointer_ids": tables,
        "column_pointer_ids": columns,
        "operator_targets": operators,
        "type_constraints": {
            "column_types": [
                {"column_id": column_id, "data_type": by_id[column_id].get("data_type")}
                for column_id in columns if column_id in by_id
            ],
            "numeric_expression": any(
                value in {"SUM", "AVG", "+", "-", "*", "/"} for value in operators
            ),
            "explicit_cast": "CAST" in operators,
            "real_division": bool(
                "/" in operators and re.search(r"(?i)\bcast\s*\([^)]*\bas\s+real\b", expression)
            ),
        },
        "value_targets": values or [],
        "gold_expression": expression,
    }


def build_typed_plan(record, graph, record_index):
    clause_record = transform_record(record)
    sql = str(record.get("sql") or "").strip()
    schema_items = record.get("schema_items", [])
    clause_labels = clause_record.get("clause_labels", {})
    segments = top_level_segments(sql)
    question, evidence = record.get("question") or "", record.get("evidence") or ""
    nodes, action_sequence = [], []

    join_ids = clause_labels.get("join", [])
    table_ids, join_columns = schema_partition(schema_items, join_ids)
    join_expression = " ".join(segments.get("from", []) + segments.get("join", []))
    scan_node_ids = []
    for table_id in table_ids or [None]:
        pointers = [] if table_id is None else [table_id]
        scan_node = make_node(
            f"scan_{len(nodes)}", "SCAN", None, schema_items, pointers, ""
        )
        nodes.append(scan_node)
        scan_node_ids.append(scan_node["node_id"])
    current = scan_node_ids[0]

    join_edges = selected_join_edges(graph, schema_items, table_ids, join_columns)
    if len(table_ids) > 1 or segments.get("join"):
        join_node = make_node(
            f"join_{len(nodes)}", "JOIN", scan_node_ids, schema_items, join_ids, join_expression
        )
        join_node["join_edge_targets"] = join_edges
        nodes.append(join_node)
        current = join_node["node_id"]

    where_expression = " ".join(segments.get("where", []))
    if where_expression:
        values = literal_targets(where_expression, question, evidence, "where")
        node = make_node(
            f"filter_{len(nodes)}", "FILTER", current, schema_items,
            clause_labels.get("where", []), where_expression, values,
        )
        nodes.append(node)
        current = node["node_id"]

    select_expression = " ".join(segments.get("select", []))
    group_expression = " ".join(segments.get("group_by", []))
    having_expression = " ".join(segments.get("having", []))
    has_aggregate = bool(AGGREGATE_PATTERN.search(select_expression + " " + having_expression))
    if has_aggregate or group_expression:
        aggregate_ids = set(clause_labels.get("select", []))
        aggregate_ids.update(clause_labels.get("group_by", []))
        aggregate_expression = " ".join(
            value for value in (select_expression, group_expression) if value
        )
        node = make_node(
            f"aggregate_{len(nodes)}", "AGGREGATE", current, schema_items,
            aggregate_ids, aggregate_expression,
        )
        node["group_key_pointer_ids"] = schema_partition(
            schema_items, clause_labels.get("group_by", [])
        )[1]
        nodes.append(node)
        current = node["node_id"]

    if having_expression:
        values = literal_targets(having_expression, question, evidence, "having")
        node = make_node(
            f"having_filter_{len(nodes)}", "HAVING_FILTER", current, schema_items,
            clause_labels.get("having", []), having_expression, values,
        )
        nodes.append(node)
        current = node["node_id"]

    project_values = literal_targets(select_expression, question, evidence, "select")
    project = make_node(
        f"project_{len(nodes)}", "PROJECT", current, schema_items,
        clause_labels.get("select", []), select_expression, project_values,
    )
    nodes.append(project)
    current = project["node_id"]

    order_expression = " ".join(segments.get("order_by", []))
    if order_expression:
        node = make_node(
            f"sort_{len(nodes)}", "SORT", current, schema_items,
            clause_labels.get("order_by", []), order_expression,
        )
        nodes.append(node)
        current = node["node_id"]

    limit_expression = " ".join(segments.get("limit", []))
    if limit_expression:
        values = literal_targets(limit_expression, question, evidence, "limit")
        node = make_node(
            f"limit_{len(nodes)}", "LIMIT", current, schema_items, [], limit_expression, values
        )
        nodes.append(node)
        current = node["node_id"]

    set_operators = [match.group(1).upper() for match in SET_OPERATOR_PATTERN.finditer(sql)]
    has_subquery = bool(SUBQUERY_PATTERN.search(sql))
    parse_status = "supported_flat"
    unsupported = []
    if has_subquery:
        parse_status = "partial_nested"
        unsupported.append("nested_subquery")
    if set_operators:
        parse_status = "partial_set_query"
        unsupported.append("set_operator")

    for step_index, node in enumerate(nodes):
        action_sequence.append(
            {
                "step_index": step_index,
                "action": node["operator"],
                "table_pointer_ids": node["table_pointer_ids"],
                "column_pointer_ids": node["column_pointer_ids"],
                "operator_targets": node["operator_targets"],
                "value_targets": node["value_targets"],
                "input_node_ids": node["inputs"],
            }
        )

    whole = set(record.get("whole_sql_labels", []))
    assigned = {
        pointer
        for node in nodes
        for pointer in node["table_pointer_ids"] + node["column_pointer_ids"]
    }
    all_values = [value for node in nodes for value in node["value_targets"]]
    join_connected = join_path_connected(schema_items, table_ids, join_edges)
    return {
        "split": record.get("split"),
        "record_index": record_index,
        "question_id": record.get("question_id"),
        "db_id": record.get("db_id"),
        "inference_inputs": {
            "question": question,
            "evidence": evidence,
            "schema_items": schema_items,
            "schema_edges": inference_graph_inputs(graph).get("schema_edges", []),
        },
        "training_targets": {
            "gold_sql": sql,
            "relational_algebra": {
                "version": "typed_ra_v1",
                "nodes": nodes,
                "root_node_id": current,
                "set_operator_targets": set_operators,
            },
            "action_sequence": action_sequence,
            "join_path": {
                "table_pointer_ids": table_ids,
                "column_pointer_ids": join_columns,
                "edge_targets": join_edges,
                "connected": join_connected,
            },
            "value_copy_targets": all_values,
        },
        "audit": {
            "graph_attached": bool(graph),
            "parse_status": parse_status,
            "unsupported_features": unsupported,
            "has_subquery": has_subquery,
            "set_operators": set_operators,
            "whole_schema_label_count": len(whole),
            "assigned_schema_label_count": len(whole & assigned),
            "schema_label_coverage": len(whole & assigned) / len(whole) if whole else 1.0,
            "unassigned_schema_ids": sorted(whole - assigned),
            "join_path_connected": join_connected,
            "value_target_count": len(all_values),
        },
    }


def audit_issues(row):
    issues = []
    audit = row["audit"]
    if audit["parse_status"] != "supported_flat":
        issues.append({"type": audit["parse_status"], "severity": "exclude_from_v1_training"})
    if audit["schema_label_coverage"] < 1.0:
        issues.append({"type": "incomplete_schema_assignment", "severity": "error"})
    join_path = row["training_targets"]["join_path"]
    if len(join_path["table_pointer_ids"]) > 1 and not join_path["connected"]:
        severity = "error" if audit.get("graph_attached") else "graph_unavailable"
        issues.append({"type": "disconnected_join_path", "severity": severity})
    database_values = sum(
        value.get("source") == "database_value_required"
        for value in row["training_targets"]["value_copy_targets"]
    )
    if database_values:
        issues.append(
            {"type": "database_value_lookup_required", "severity": "expected", "count": database_values}
        )
    return issues


def summarize(rows):
    parse_status = Counter(row["audit"]["parse_status"] for row in rows)
    operator_counts = Counter(
        action["action"]
        for row in rows
        for action in row["training_targets"]["action_sequence"]
    )
    value_sources = Counter(
        value["source"]
        for row in rows
        for value in row["training_targets"]["value_copy_targets"]
    )
    values = [
        value
        for row in rows
        for value in row["training_targets"]["value_copy_targets"]
    ]
    join_rows = [
        row for row in rows
        if len(row["training_targets"]["join_path"]["table_pointer_ids"]) > 1
    ]
    issue_counts = Counter(
        issue["type"] for row in rows for issue in audit_issues(row)
    )
    return {
        "example_count": len(rows),
        "parse_status_counts": dict(parse_status),
        "audit_issue_counts": dict(issue_counts),
        "graph_attachment_rate": (
            sum(row["audit"].get("graph_attached", False) for row in rows) / len(rows)
            if rows else 0.0
        ),
        "supported_flat_rate": parse_status.get("supported_flat", 0) / len(rows) if rows else 0.0,
        "avg_action_count": (
            sum(len(row["training_targets"]["action_sequence"]) for row in rows) / len(rows)
            if rows else 0.0
        ),
        "operator_counts": dict(operator_counts),
        "avg_schema_label_coverage": (
            sum(row["audit"]["schema_label_coverage"] for row in rows) / len(rows)
            if rows else 0.0
        ),
        "full_schema_label_coverage_rate": (
            sum(row["audit"]["schema_label_coverage"] >= 1.0 for row in rows) / len(rows)
            if rows else 0.0
        ),
        "multi_table_example_count": len(join_rows),
        "join_path_connected_rate": (
            sum(row["audit"]["join_path_connected"] for row in join_rows) / len(join_rows)
            if join_rows else 1.0
        ),
        "value_target_count": len(values),
        "value_source_counts": dict(value_sources),
        "exact_value_copy_rate": (
            sum(value.get("match") == "exact" for value in values) / len(values) if values else 1.0
        ),
        "case_sensitive_value_count": sum(value.get("case_sensitive", False) for value in values),
    }


def build_split(label_path, graph_path, output_path, limit=None):
    records = read_jsonl(label_path, limit)
    graphs = graph_by_record_index(graph_path, limit)
    rows = [
        build_typed_plan(record, graphs.get(index), index)
        for index, record in enumerate(records)
    ]
    write_jsonl(output_path, rows)
    issue_rows = []
    for row in rows:
        issues = audit_issues(row)
        if issues:
            issue_rows.append(
                {
                    "record_index": row["record_index"],
                    "db_id": row["db_id"],
                    "question": row["inference_inputs"]["question"],
                    "issues": issues,
                }
            )
    output_path = Path(output_path)
    write_jsonl(
        output_path.with_name(output_path.stem + "_audit_issues.jsonl"), issue_rows
    )
    return summarize(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-labels",
        default="experiments/stage1_label_extraction_corrected/bird_train_grounding_labels.jsonl",
    )
    parser.add_argument(
        "--dev-labels",
        default="experiments/stage1_label_extraction_corrected/bird_dev_grounding_labels.jsonl",
    )
    parser.add_argument(
        "--train-graphs",
        default="experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl",
    )
    parser.add_argument(
        "--dev-graphs",
        default="experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl",
    )
    parser.add_argument("--output-dir", default="experiments/stage13a_typed_ra_data")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_summary = build_split(
        args.train_labels, args.train_graphs, output_dir / "train_typed_ra.jsonl", args.train_limit
    )
    dev_summary = build_split(
        args.dev_labels, args.dev_graphs, output_dir / "dev_typed_ra.jsonl", args.dev_limit
    )
    summary = {"config": vars(args), "train": train_summary, "dev": dev_summary}
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
