import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grounding.stage5e_value_aware_selector import (  # noqa: E402
    VALUE_COLUMN_KEYWORDS,
    build_prompt,
    fetch_matched_values,
    find_sqlite_index,
    is_forced_value_column,
    is_value_searchable,
    quote_identifier,
    salient_query_tokens,
    schema_text_from_rows,
    value_match_score,
    value_tokens,
    value_evidence_by_column,
    normalize_text as normalize_value_text,
)
from src.grounding.stage5f_conservative_value_tcce import (  # noqa: E402
    is_metric_or_operator_column,
    normalize_tokens,
)
from src.grounding.stage5g_clause_structural_assembly import (  # noqa: E402
    assemble_one,
    coverage,
    group_clause_predictions,
    read_jsonl,
    write_json,
    write_jsonl,
)


def node_indexes(example):
    nodes = example["inference_inputs"]["schema_nodes"]
    by_id = {int(node["id"]): node for node in nodes}
    columns_by_table = defaultdict(list)
    for node in nodes:
        if node.get("type") == "column":
            columns_by_table[node.get("table")].append(node)
    return by_id, columns_by_table


def selected_tables_from_rows(rows, by_id):
    tables = set()
    for row in rows:
        node = by_id.get(int(row["id"]))
        if not node:
            continue
        if node.get("type") == "table":
            tables.add(node.get("name"))
        elif node.get("type") == "column":
            tables.add(node.get("table"))
    return {table for table in tables if table}


def candidate_value_columns(example, selected_tables, by_id, columns_by_table, question, evidence, args):
    selected_ids = {int(row["id"]) for row in example.get("_selected_rows", [])}
    candidates = []
    for table in selected_tables:
        for node in columns_by_table.get(table, []):
            if not is_forced_value_column(node):
                continue
            candidates.append(node)

    def priority(node):
        col_tokens = normalize_tokens(node.get("column") or node.get("name"))
        value_keyword_hit = len(col_tokens & VALUE_COLUMN_KEYWORDS)
        already_selected = 1 if int(node["id"]) in selected_ids else 0
        metric_penalty = 1 if is_metric_or_operator_column(node, question, evidence) else 0
        return (already_selected, value_keyword_hit, -metric_penalty)

    return sorted(candidates, key=priority, reverse=True)[: args.value_search_columns]


def protected_ids_from_rows(rows, by_id, question, evidence):
    protected = set()
    for row in rows:
        item_id = int(row["id"])
        node = by_id.get(item_id)
        if not node:
            continue
        if node.get("type") == "table":
            protected.add(item_id)
        elif row.get("source_clause") in {"table_backbone", "join"}:
            protected.add(item_id)
        elif is_metric_or_operator_column(node, question, evidence):
            protected.add(item_id)
        elif row.get("value_hints"):
            protected.add(item_id)
    return protected


JOIN_KEY_TOKENS = {
    "id",
    "code",
    "key",
    "cds",
    "cdscode",
    "api",
    "race",
    "driver",
    "account",
    "client",
    "order",
}

SELECT_OUTPUT_TOKENS = {
    "name",
    "title",
    "school",
    "website",
    "email",
    "phone",
    "address",
    "street",
    "zip",
    "city",
    "state",
    "district",
    "county",
    "code",
}


def is_join_key_like(node):
    text = " ".join(normalize_tokens(f"{node.get('table', '')} {node.get('column', '')} {node.get('name', '')}"))
    compact = text.replace(" ", "")
    toks = set(text.split())
    if toks & JOIN_KEY_TOKENS:
        return True
    return any(token in compact for token in ["id", "code", "cds", "api"])


def is_select_likely_column(node, question):
    col_tokens = normalize_tokens(node.get("column") or node.get("name"))
    question_tokens = normalize_tokens(question)
    if not (col_tokens & SELECT_OUTPUT_TOKENS):
        return False
    return bool(
        question_tokens
        & {
            "what",
            "which",
            "list",
            "name",
            "names",
            "website",
            "email",
            "phone",
            "address",
            "zip",
            "city",
            "state",
            "district",
            "school",
        }
    )


def fk_endpoint_ids(example):
    endpoints = set()
    for edge in example["inference_inputs"].get("schema_edges", []):
        if edge.get("type") in {"foreign_key_forward", "foreign_key_backward"}:
            endpoints.add(int(edge["src"]))
            endpoints.add(int(edge["dst"]))
    return endpoints


def structural_protected_ids(example, rows, by_id, question, evidence, args):
    protected = protected_ids_from_rows(rows, by_id, question, evidence)
    if args.protect_fk_endpoints:
        protected.update(fk_endpoint_ids(example))
    scored_columns = []
    for row in rows:
        item_id = int(row["id"])
        node = by_id.get(item_id)
        if not node or node.get("type") != "column":
            continue
        if args.protect_join_key_like and is_join_key_like(node):
            protected.add(item_id)
        if args.protect_select_likely and is_select_likely_column(node, question):
            protected.add(item_id)
        scored_columns.append((float(row.get("score", 0.0)), item_id))
    scored_columns.sort(reverse=True)
    protected.update(item_id for _, item_id in scored_columns[: args.protect_top_structural_columns])
    return protected


def add_value_hints(rows, value_evidence, args):
    augmented = []
    for row in rows:
        new_row = dict(row)
        hints = [
            hint
            for hint in value_evidence.get(int(row["id"]), [])
            if hint.get("score", 0.0) >= args.hint_threshold
        ][: args.max_values_per_column]
        if hints:
            new_row["value_hints"] = hints
            new_row["value_score"] = max(h["score"] for h in hints)
            new_row["selection_source"] = "g3_value_hint"
        augmented.append(new_row)
    return augmented


class SQLiteValueEvidenceCache:
    def __init__(self):
        self.connections = {}
        self.value_cache = {}
        self.column_value_cache = {}
        self.query_count = 0
        self.column_query_count = 0
        self.cache_hits = 0
        self.column_cache_hits = 0

    def close(self):
        for connection in self.connections.values():
            connection.close()
        self.connections.clear()

    def connection(self, db_path):
        if db_path is None:
            return None
        key = str(db_path)
        if key not in self.connections:
            self.connections[key] = sqlite3.connect(key)
        return self.connections[key]

    def fetch_values(self, db_path, table, column, query_tokens, max_scan_values):
        if db_path is None:
            return []
        cache_key = (
            str(db_path),
            str(table),
            str(column),
            tuple(query_tokens),
            int(max_scan_values),
        )
        if cache_key in self.value_cache:
            self.cache_hits += 1
            return self.value_cache[cache_key]
        connection = self.connection(db_path)
        values = fetch_matched_values(connection, table, column, query_tokens, max_scan_values)
        self.value_cache[cache_key] = values
        self.query_count += 1
        return values

    def fetch_column_values(self, db_path, table, column, max_cached_values):
        if db_path is None:
            return []
        cache_key = (
            str(db_path),
            str(table),
            str(column),
            int(max_cached_values),
        )
        if cache_key in self.column_value_cache:
            self.column_cache_hits += 1
            return self.column_value_cache[cache_key]
        connection = self.connection(db_path)
        sql = (
            f"SELECT DISTINCT {quote_identifier(column)} "
            f"FROM {quote_identifier(table)} "
            f"WHERE {quote_identifier(column)} IS NOT NULL "
            f"LIMIT {int(max_cached_values)}"
        )
        try:
            cursor = connection.execute(sql)
            values = [str(row[0]) for row in cursor.fetchall() if row[0] is not None and str(row[0]).strip()]
        except sqlite3.Error:
            values = []
        self.column_value_cache[cache_key] = values
        self.column_query_count += 1
        return values

    def fetch_values_from_column_cache(self, db_path, table, column, query_tokens, max_scan_values, max_cached_values):
        if not query_tokens:
            return []
        values = self.fetch_column_values(db_path, table, column, max_cached_values)
        lowered_tokens = [str(token).lower() for token in query_tokens]
        matched = []
        for value in values:
            value_text = str(value).lower()
            if any(token in value_text for token in lowered_tokens):
                matched.append(value)
                if len(matched) >= max_scan_values:
                    break
        return matched

    def evidence_by_column(self, db_path, columns, question, evidence, args):
        if db_path is None:
            return {}
        query_tokens = salient_query_tokens(question, evidence, args.max_value_query_tokens)
        query_token_set = set(value_tokens(f"{question} {evidence}"))
        query_norm = normalize_value_text(f"{question} {evidence}")
        evidence_by_id = {}
        for node in columns:
            if not is_value_searchable(node):
                continue
            if args.value_cache_mode == "column":
                values = self.fetch_values_from_column_cache(
                    db_path,
                    node.get("table"),
                    node.get("column"),
                    query_tokens,
                    args.max_scan_values,
                    args.max_cached_distinct_values,
                )
            else:
                values = self.fetch_values(
                    db_path,
                    node.get("table"),
                    node.get("column"),
                    query_tokens,
                    args.max_scan_values,
                )
            scored = []
            for value in values:
                score = value_match_score(query_token_set, query_norm, value)
                if score >= args.min_value_score:
                    scored.append({"value": value, "score": score})
            scored.sort(key=lambda item: item["score"], reverse=True)
            if scored:
                evidence_by_id[node["id"]] = scored[: args.max_values_per_column]
        return evidence_by_id

    def stats(self):
        return {
            "sqlite_value_queries": self.query_count,
            "sqlite_value_cache_hits": self.cache_hits,
            "sqlite_value_cache_size": len(self.value_cache),
            "sqlite_column_value_queries": self.column_query_count,
            "sqlite_column_value_cache_hits": self.column_cache_hits,
            "sqlite_column_value_cache_size": len(self.column_value_cache),
            "sqlite_connection_count": len(self.connections),
        }


def low_utility_replacement_slots(rows, by_id, protected_ids, args):
    candidates = []
    for index, row in enumerate(rows):
        item_id = int(row["id"])
        if item_id in protected_ids:
            continue
        node = by_id.get(item_id)
        if not node or node.get("type") != "column":
            continue
        score = float(row.get("score", 0.0))
        table_belief = float(row.get("table_belief", 0.0))
        value_score = float(row.get("value_score", 0.0))
        candidates.append((score + 0.25 * table_belief + value_score, index, item_id))
    candidates.sort()
    return candidates[: args.max_replacements]


def apply_value_gate(example, base_rows, db_path, budget, args, value_cache=None):
    by_id, columns_by_table = node_indexes(example)
    question = example["inference_inputs"].get("question") or ""
    evidence = example["inference_inputs"].get("evidence") or ""
    selected_tables = selected_tables_from_rows(base_rows, by_id)
    tmp_example = dict(example)
    tmp_example["_selected_rows"] = base_rows
    value_candidates = candidate_value_columns(
        tmp_example, selected_tables, by_id, columns_by_table, question, evidence, args
    )
    if value_cache is not None:
        value_evidence = value_cache.evidence_by_column(db_path, value_candidates, question, evidence, args)
    else:
        value_evidence = value_evidence_by_column(db_path, value_candidates, question, evidence, args)
    augmented = add_value_hints(base_rows, value_evidence, args)

    selected_ids = {int(row["id"]) for row in augmented}
    protected = structural_protected_ids(example, augmented, by_id, question, evidence, args)
    replacements = []
    slots = low_utility_replacement_slots(augmented, by_id, protected, args)
    if slots:
        candidate_rows = []
        for node in value_candidates:
            item_id = int(node["id"])
            if item_id in selected_ids:
                continue
            hints = [
                hint
                for hint in value_evidence.get(item_id, [])
                if hint.get("score", 0.0) >= args.replacement_threshold
            ]
            if not hints:
                continue
            value_score = max(hint["score"] for hint in hints)
            # Candidate must be clearly value-like and live in an already supported table.
            table_supported = node.get("table") in selected_tables
            if not table_supported:
                continue
            candidate_rows.append((value_score, node, hints))
        candidate_rows.sort(key=lambda item: item[0], reverse=True)

        for (_, target_index, target_id), (_, node, hints) in zip(slots, candidate_rows):
            item_id = int(node["id"])
            if item_id in selected_ids:
                continue
            old = augmented[target_index]
            augmented[target_index] = {
                "id": item_id,
                "type": node["type"],
                "name": node["name"],
                "score": float(max(h["score"] for h in hints)),
                "source_clause": "conservative_value_replacement",
                "table_belief": float(old.get("table_belief", 0.0)),
                "value_score": float(max(h["score"] for h in hints)),
                "value_hints": hints[: args.max_values_per_column],
            }
            selected_ids.remove(target_id)
            selected_ids.add(item_id)
            replacements.append(
                {
                    "removed": old["name"],
                    "added": node["name"],
                    "values": [hint["value"] for hint in hints[: args.max_values_per_column]],
                }
            )

    return augmented[:budget], {
        "selected_tables": sorted(selected_tables),
        "value_candidate_count": len(value_candidates),
        "value_hint_columns": sum(1 for row in augmented if row.get("value_hints")),
        "replacement_count": len(replacements),
        "replacements": replacements,
    }


def average(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsg-data", default="experiments/stage5_dsg_data_v2/dev_examples.jsonl")
    parser.add_argument("--clause-labels", default="experiments/stage5g_clause_labels/dev_clause_labels.jsonl")
    parser.add_argument(
        "--clause-predictions",
        default="experiments/stage5g_clause_grounder_rgcn_1000/dev_clause_predictions.jsonl",
    )
    parser.add_argument("--db-root", default="Data/BIRD/dev_databases")
    parser.add_argument("--output-dir", default="experiments/stage5h_value_gated_clause_assembly_rgcn_1000")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--budgets", default="30")
    # Structural assembly defaults from best G3 run.
    parser.add_argument("--max-tables", type=int, default=5)
    parser.add_argument("--fk-budget", type=int, default=4)
    parser.add_argument("--select-budget", type=int, default=12)
    parser.add_argument("--where-budget", type=int, default=12)
    parser.add_argument("--order-budget", type=int, default=4)
    parser.add_argument("--global-clause-seed", type=int, default=20)
    parser.add_argument("--model-weight", type=float, default=0.58)
    parser.add_argument("--prior-weight", type=float, default=0.27)
    parser.add_argument("--table-weight", type=float, default=0.15)
    parser.add_argument("--join-table-weight", type=float, default=0.60)
    parser.add_argument("--column-support-weight", type=float, default=0.32)
    parser.add_argument("--table-lexical-weight", type=float, default=0.08)
    # Conservative value gate.
    parser.add_argument("--value-search-columns", type=int, default=24)
    parser.add_argument("--max-value-query-tokens", type=int, default=10)
    parser.add_argument("--max-scan-values", type=int, default=20)
    parser.add_argument("--max-values-per-column", type=int, default=2)
    parser.add_argument(
        "--value-cache-mode",
        choices=["column", "query"],
        default="column",
        help=(
            "column caches DISTINCT values per db/table/column and matches query tokens in memory; "
            "query preserves the older per-question SQL LIKE lookup."
        ),
    )
    parser.add_argument(
        "--max-cached-distinct-values",
        type=int,
        default=512,
        help="Maximum DISTINCT values cached per candidate value column when --value-cache-mode=column.",
    )
    parser.add_argument("--min-value-score", type=float, default=0.45)
    parser.add_argument("--hint-threshold", type=float, default=0.65)
    parser.add_argument("--replacement-threshold", type=float, default=0.88)
    parser.add_argument("--max-replacements", type=int, default=2)
    parser.add_argument("--protect-fk-endpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--protect-join-key-like", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--protect-select-likely", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--protect-top-structural-columns", type=int, default=8)
    args = parser.parse_args()

    examples = read_jsonl(Path(args.dsg_data), args.limit)
    labels = read_jsonl(Path(args.clause_labels), args.limit)
    clause_predictions = read_jsonl(Path(args.clause_predictions))
    grouped_predictions = group_clause_predictions(clause_predictions)
    sqlite_index = find_sqlite_index(Path(args.db_root))
    budgets = [int(value.strip()) for value in args.budgets.split(",") if value.strip()]
    value_cache = SQLiteValueEvidenceCache()

    if len(examples) != len(labels):
        raise ValueError(f"Length mismatch examples={len(examples)} labels={len(labels)}")

    rows = []
    prompts_by_budget = {budget: [] for budget in budgets}
    coverage_by_budget = {budget: defaultdict(list) for budget in budgets}
    debug_counter = Counter()
    try:
        for index, (example, label_record) in enumerate(zip(examples, labels)):
            db_id = example["inference_inputs"]["db_id"]
            db_path = sqlite_index.get(db_id)
            clause_rows = grouped_predictions.get(index, {})
            row = {
                "example_id": example["example_id"],
                "record_index": index,
                "db_id": db_id,
                "question_id": label_record.get("question_id"),
                "question": example["inference_inputs"].get("question"),
                "evidence": example["inference_inputs"].get("evidence"),
                "gold_label_ids": label_record.get("whole_sql_labels", []),
                "gold_label_names": label_record.get("whole_sql_label_names", []),
            }
            debug_by_budget = {}
            for budget in budgets:
                base_rows, base_debug = assemble_one(example, clause_rows, budget, args)
                selected_rows, value_debug = apply_value_gate(
                    example, base_rows, db_path, budget, args, value_cache=value_cache
                )
                row[f"top_{budget}"] = selected_rows
                debug_by_budget[str(budget)] = {"base": base_debug, "value_gate": value_debug}
                debug_counter[f"value_hint_columns_top{budget}"] += value_debug["value_hint_columns"]
                debug_counter[f"replacements_top{budget}"] += value_debug["replacement_count"]
                cov = coverage(label_record, selected_rows)
                for key, value in cov.items():
                    if key != "missing_gold_names":
                        coverage_by_budget[budget][key].append(value)

                schema_text = schema_text_from_rows(example, selected_rows)
                prompts_by_budget[budget].append(
                    {
                        "question_id": label_record.get("question_id"),
                        "db_id": db_id,
                        "setting": f"value_gated_clause_assembly_top{budget}",
                        "question": example["inference_inputs"].get("question"),
                        "evidence": example["inference_inputs"].get("evidence"),
                        "selected_schema_item_count": len(selected_rows),
                        "schema_text": schema_text,
                        "prompt": build_prompt(
                            example["inference_inputs"].get("question"),
                            example["inference_inputs"].get("evidence"),
                            schema_text,
                        ),
                        "gold_sql": example.get("training_targets", {}).get("sql"),
                        "gold_labels": label_record.get("whole_sql_label_names", []),
                    }
                )
            row["value_gated_clause_assembly_debug"] = debug_by_budget
            rows.append(row)
    finally:
        cache_stats = value_cache.stats()
        value_cache.close()

    statistics = {
        "config": vars(args),
        "sample_count": len(rows),
        "budgets": {},
        "debug_counts": dict(debug_counter),
        "cache_stats": cache_stats,
        "note": (
            "Stage 5-H1 keeps G3 learned structural assembly as the skeleton, "
            "then adds high-confidence value hints and limited low-utility replacements."
        ),
    }
    for budget in budgets:
        statistics["budgets"][str(budget)] = {
            key: average(values) for key, values in sorted(coverage_by_budget[budget].items())
        }
        statistics["budgets"][str(budget)]["missing_samples"] = sum(
            1 for value in coverage_by_budget[budget]["schema_recall"] if value is not None and value < 1.0
        )
        statistics["budgets"][str(budget)]["avg_value_hint_columns"] = (
            debug_counter[f"value_hint_columns_top{budget}"] / len(rows) if rows else 0.0
        )
        statistics["budgets"][str(budget)]["avg_replacements"] = (
            debug_counter[f"replacements_top{budget}"] / len(rows) if rows else 0.0
        )

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "value_gated_clause_assembly_predictions.jsonl", rows)
    for budget, prompts in prompts_by_budget.items():
        write_jsonl(output_dir / f"prompts_value_gated_clause_assembly_top{budget}_dev.jsonl", prompts)
    write_json(output_dir / "selection_statistics.json", statistics)
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
