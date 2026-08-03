import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def load_dev_tables(path: Path):
    return {entry["db_id"]: entry for entry in read_json(path)}


def schema_node_maps(example):
    nodes = example["inference_inputs"]["schema_nodes"]
    by_id = {int(node["id"]): node for node in nodes}
    return nodes, by_id


def quote_col(column):
    return f"`{column}`" if any(ch in str(column) for ch in " ()%/-") else str(column)


def selected_ids(prediction, budget):
    return [int(row["id"]) for row in prediction[f"top_{budget}"][:budget]]


def add_fk_endpoint_closure(example, selected):
    _, by_id = schema_node_maps(example)
    selected = set(selected)
    selected_tables = set()
    for item_id in list(selected):
        node = by_id.get(item_id)
        if not node:
            continue
        if node.get("type") == "table":
            selected_tables.add(node.get("name"))
        elif node.get("type") == "column":
            selected_tables.add(node.get("table"))
    for node in by_id.values():
        if node.get("type") == "table" and node.get("name") in selected_tables:
            selected.add(int(node["id"]))
    for edge in example["inference_inputs"].get("schema_edges", []):
        if edge.get("type") not in {"foreign_key_forward", "foreign_key_backward"}:
            continue
        src = by_id.get(int(edge["src"]))
        dst = by_id.get(int(edge["dst"]))
        if not src or not dst:
            continue
        if src.get("type") != "column" or dst.get("type") != "column":
            continue
        both_tables = src.get("table") in selected_tables and dst.get("table") in selected_tables
        either_endpoint = int(src["id"]) in selected or int(dst["id"]) in selected
        if both_tables or either_endpoint:
            selected.add(int(src["id"]))
            selected.add(int(dst["id"]))
    return sorted(selected)


def group_schema(example, selected):
    _, by_id = schema_node_maps(example)
    grouped = defaultdict(list)
    selected = set(selected)
    for item_id in selected:
        node = by_id.get(item_id)
        if not node:
            continue
        if node.get("type") == "table":
            grouped.setdefault(node["name"], [])
        elif node.get("type") == "column":
            grouped[node.get("table")].append(node)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def fk_lines(example, selected):
    _, by_id = schema_node_maps(example)
    selected = set(selected)
    lines = set()
    for edge in example["inference_inputs"].get("schema_edges", []):
        if edge.get("type") != "foreign_key_forward":
            continue
        src = by_id.get(int(edge["src"]))
        dst = by_id.get(int(edge["dst"]))
        if not src or not dst:
            continue
        if int(src["id"]) in selected and int(dst["id"]) in selected:
            lines.add(f"{src['table']}.{src['column']} = {dst['table']}.{dst['column']}")
    return sorted(lines)


def compact_semantic(node):
    parts = []
    if node.get("semantic_name"):
        parts.append(node["semantic_name"])
    if node.get("value_type"):
        parts.append("type: " + str(node["value_type"]))
    return "; ".join(parts)


def schema_text(example, selected):
    grouped = group_schema(example, selected)
    lines = []
    for table, columns in grouped.items():
        lines.append(f"Table {table}:")
        if not columns:
            lines.append("- [selected table]")
        for node in sorted(columns, key=lambda x: x.get("column") or x.get("name")):
            dtype = f" ({node.get('data_type')})" if node.get("data_type") else ""
            semantic = compact_semantic(node)
            suffix = f" -- {semantic}" if semantic else ""
            lines.append(f"- {quote_col(node.get('column'))}{dtype}{suffix}")
        lines.append("")
    fks = fk_lines(example, selected)
    if fks:
        lines.append("Foreign keys:")
        for fk in fks:
            lines.append(f"- {fk}")
        lines.append("")
    return "\n".join(lines).strip()


def relation_rows(prediction, example, selected):
    _, by_id = schema_node_maps(example)
    rows_by_role = defaultdict(list)
    for row in prediction.get("top_30", []):
        item_id = int(row["id"])
        if item_id not in selected:
            continue
        node = by_id.get(item_id)
        if not node or node.get("type") != "column":
            continue
        roles = []
        explicit_relation = row.get("relation_type")
        if explicit_relation:
            roles.append(explicit_relation)
        source = row.get("source_clause") or ""
        if source.startswith("relation_fusion:"):
            roles.append(source.split(":", 1)[1])
        if row.get("value_hints"):
            roles.append("VALUE_HINT")
        semantic_relations = set(node.get("relation_types", []) or [])
        value_type = node.get("value_type")
        if value_type == "numeric_metric" or "METRIC_TARGET" in semantic_relations:
            roles.append("METRIC_TARGET")
        if value_type in {"entity_text", "text"} or explicit_relation == "OUTPUT_TARGET":
            roles.append("OUTPUT_TARGET")
        if value_type in {"categorical", "temporal"} or "PREDICATE_COLUMN" in semantic_relations:
            roles.append("PREDICATE_COLUMN")
        if "VALUE_ANCHOR" in semantic_relations:
            roles.append("VALUE_ANCHOR")
        if "ORDER_KEY" in semantic_relations and value_type in {"numeric_metric", "temporal"}:
            roles.append("ORDER_KEY")
        if "FORMULA_COMPONENT" in semantic_relations:
            roles.append("FORMULA_COMPONENT")
        if not roles and source in {"select", "where", "order_by", "join"}:
            roles.append(source.upper())
        role_priority = [
            "VALUE_HINT",
            "OUTPUT_TARGET",
            "PREDICATE_COLUMN",
            "METRIC_TARGET",
            "ORDER_KEY",
            "FORMULA_COMPONENT",
            "VALUE_ANCHOR",
        ]
        role_set = set(roles)
        chosen_roles = [role for role in role_priority if role in role_set]
        for role in chosen_roles[:3]:
            rows_by_role[role].append((row, node))
    return rows_by_role


def role_evidence_text(prediction, example, selected, max_per_role=5):
    rows_by_role = relation_rows(prediction, example, selected)
    preferred_order = [
        "OUTPUT_TARGET",
        "ENTITY_NAME",
        "PREDICATE_COLUMN",
        "VALUE_ANCHOR",
        "VALUE_HINT",
        "METRIC_TARGET",
        "ORDER_KEY",
        "FORMULA_COMPONENT",
        "SELECT",
        "WHERE",
        "ORDER_BY",
    ]
    lines = []
    used_roles = set()
    for role in preferred_order + sorted(rows_by_role):
        if role in used_roles or role not in rows_by_role:
            continue
        used_roles.add(role)
        lines.append(f"{role}:")
        entries = sorted(rows_by_role[role], key=lambda pair: float(pair[0].get("score", 0.0)), reverse=True)
        for row, node in entries[:max_per_role]:
            value_text = ""
            if row.get("value_hints"):
                vals = [str(h["value"]) for h in row["value_hints"][:2]]
                value_text = f" | matched values: {', '.join(vals)}"
            semantic = compact_semantic(node)
            semantic_text = f" | {semantic}" if semantic else ""
            lines.append(f"- {node['name']}{semantic_text}{value_text}")
        lines.append("")
    return "\n".join(lines).strip()


def build_role_prompt(question, evidence, schema, role_evidence):
    evidence_block = f"\nQuestion evidence:\n{evidence}\n" if evidence else ""
    grounding_block = f"\nGrounding evidence by SQL role:\n{role_evidence}\n" if role_evidence else ""
    return (
        "You are an expert SQLite SQL generator.\n\n"
        "Rules:\n"
        "1. Use only the exact table and column names listed in the schema.\n"
        "2. Do not invent columns or tables.\n"
        "3. Use SQLite syntax only.\n"
        "4. Quote column names with backticks if they contain spaces, parentheses, %, /, or hyphens.\n"
        "5. Treat the grounding evidence as guidance: output columns are for SELECT, predicate/value columns are for WHERE, metric/order columns are for aggregation or ordering, and foreign keys are for JOINs.\n"
        "6. If the evidence defines a formula, implement the formula explicitly in SQL.\n"
        "7. Return only one SQL query, with no explanation.\n\n"
        "Database schema with semantic descriptions:\n"
        f"{schema}\n\n"
        f"Question:\n{question}\n"
        f"{evidence_block}"
        f"{grounding_block}\n"
        "Return only the SQL query."
    )


def build_records(predictions, examples, budget):
    rows = []
    for pred, example in zip(predictions, examples):
        selected = add_fk_endpoint_closure(example, selected_ids(pred, budget))
        schema = schema_text(example, selected)
        role_evidence = role_evidence_text(pred, example, set(selected))
        prompt = build_role_prompt(pred.get("question"), pred.get("evidence"), schema, role_evidence)
        rows.append(
            {
                "question_id": pred.get("question_id"),
                "db_id": pred.get("db_id"),
                "setting": f"role_aware_relation_fusion_top{budget}",
                "question": pred.get("question"),
                "evidence": pred.get("evidence"),
                "selected_schema_item_count": len(set(selected)),
                "schema_text": schema,
                "role_evidence_text": role_evidence,
                "prompt": prompt,
                "gold_sql": example.get("training_targets", {}).get("sql"),
                "gold_labels": pred.get("gold_label_names", []),
            }
        )
    return rows


def summarize(rows):
    lengths = [len(row["prompt"]) for row in rows]
    return {
        "count": len(rows),
        "avg_prompt_chars": sum(lengths) / len(lengths) if lengths else 0,
        "max_prompt_chars": max(lengths) if lengths else 0,
        "avg_selected_schema_items": (
            sum(row["selected_schema_item_count"] for row in rows) / len(rows) if rows else 0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="experiments/stage5j_relation_fusion_selector_rgcn_1000_best/relation_fusion_predictions.jsonl",
    )
    parser.add_argument("--dsg-data", default="experiments/stage5i_dsg_data_semantic_v1/dev_examples.jsonl")
    parser.add_argument("--output-dir", default="experiments/stage6_role_aware_prompts")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--budgets", default="30")
    args = parser.parse_args()

    predictions = read_jsonl(Path(args.predictions), args.limit)
    examples = read_jsonl(Path(args.dsg_data), args.limit)
    if len(predictions) != len(examples):
        raise ValueError(f"Length mismatch predictions={len(predictions)} examples={len(examples)}")
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]

    output_dir = Path(args.output_dir)
    summary = {"config": vars(args), "budgets": {}}
    for budget in budgets:
        rows = build_records(predictions, examples, budget)
        path = output_dir / f"prompts_role_aware_relation_fusion_top{budget}_dev.jsonl"
        write_jsonl(path, rows)
        summary["budgets"][str(budget)] = summarize(rows)
    write_json(output_dir / "prompt_statistics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
