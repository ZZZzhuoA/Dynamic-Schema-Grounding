"""Build causal partial-SQL grounding trajectories for Stage 11.

Each event ends immediately after a top-level SQL clause keyword.  Its target is
the schema set needed by that clause, so identifiers inside the target clause are
never copied into the controller input.
"""

import argparse
import json
import re
from pathlib import Path


CLAUSE_SPECS = [
    ("select", "PROJECT", re.compile(r"\bselect\b", re.I)),
    ("join", "JOIN", re.compile(r"\b(from|join)\b", re.I)),
    ("where", "FILTER", re.compile(r"\bwhere\b", re.I)),
    ("group_by", "GROUP", re.compile(r"\bgroup\s+by\b", re.I)),
    ("having", "FILTER", re.compile(r"\bhaving\b", re.I)),
    ("order_by", "ORDER", re.compile(r"\border\s+by\b", re.I)),
]


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def top_level_mask(sql):
    """Return whether every character is outside quotes and parentheses."""
    mask = [False] * len(sql)
    depth = 0
    quote = None
    index = 0
    while index < len(sql):
        char = sql[index]
        if quote:
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "[":
            quote = "]"
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        else:
            mask[index] = depth == 0
        index += 1
    return mask


def clause_keyword_events(sql):
    mask = top_level_mask(sql)
    events = []
    seen = set()
    for clause, operation, pattern in CLAUSE_SPECS:
        for match in pattern.finditer(sql):
            if match.start() < len(mask) and mask[match.start()]:
                # Multiple JOIN keywords share one join-path target. The first FROM
                # event is the causal point where source planning begins.
                if clause in seen:
                    continue
                seen.add(clause)
                events.append(
                    {
                        "clause": clause,
                        "operation": operation,
                        "keyword_start": match.start(),
                        "prefix_end": match.end(),
                    }
                )
                break
    return sorted(events, key=lambda row: row["keyword_start"])


def build_trajectory(graph, labels):
    sql = str(labels.get("sql") or "").strip()
    clause_labels = labels.get("clause_labels") or {}
    candidate_by_schema = {
        int(node["schema_item_id"]): int(node["local_id"])
        for node in graph.get("candidate_nodes", [])
    }
    steps = []
    observed = set()
    for step_index, event in enumerate(clause_keyword_events(sql)):
        target_schema = {
            int(item_id) for item_id in clause_labels.get(event["clause"], [])
        }
        target_local = sorted(
            candidate_by_schema[item_id]
            for item_id in target_schema
            if item_id in candidate_by_schema
        )
        steps.append(
            {
                "step_index": step_index,
                "clause": event["clause"],
                "operation": event["operation"],
                "partial_sql": sql[: event["prefix_end"]].strip(),
                "target_schema_ids": sorted(target_schema),
                "target_local_ids": target_local,
                "missing_target_schema_ids": sorted(target_schema - candidate_by_schema.keys()),
                "target_candidate_recall": (
                    len(target_local) / len(target_schema) if target_schema else 1.0
                ),
                "observed_schema_ids": sorted(observed),
                "observed_local_ids": sorted(
                    candidate_by_schema[item_id]
                    for item_id in observed
                    if item_id in candidate_by_schema
                ),
            }
        )
        observed.update(target_schema)
    return {
        **graph,
        "gold_sql": sql,
        "trajectory_steps": steps,
        "trajectory_target_ids": sorted(
            {item for step in steps for item in step["target_schema_ids"]}
        ),
    }


def build_dataset(graph_file, label_file, output_file, limit=None):
    graphs = read_jsonl(graph_file)
    labels = read_jsonl(label_file)
    labels_by_index = {index: row for index, row in enumerate(labels)}
    rows = []
    for graph in graphs:
        index = int(graph["record_index"])
        if index not in labels_by_index:
            raise ValueError(f"No clause labels for record_index={index}")
        rows.append(build_trajectory(graph, labels_by_index[index]))
        if limit is not None and len(rows) >= limit:
            break
    write_jsonl(output_file, rows)
    steps = [step for row in rows for step in row["trajectory_steps"]]
    summary = {
        "example_count": len(rows),
        "step_count": len(steps),
        "avg_steps": len(steps) / len(rows) if rows else 0.0,
        "empty_target_steps": sum(not step["target_schema_ids"] for step in steps),
        "fully_recalled_target_steps": sum(
            step["target_candidate_recall"] >= 1.0 for step in steps
        ),
        "operation_counts": {
            operation: sum(step["operation"] == operation for step in steps)
            for operation in sorted({step["operation"] for step in steps})
        },
    }
    Path(output_file).with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-file", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = build_dataset(
        args.graph_file, args.label_file, args.output_file, args.limit
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {args.output_file}")


if __name__ == "__main__":
    main()
