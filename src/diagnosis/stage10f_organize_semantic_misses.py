"""Organize Stage 10-F semantic-core misses for human review.

This script is deliberately diagnostic-only.  It does not call an LLM, alter a
prediction, or report a new model score.  It aligns the exact semantic targets
with the candidate graph, frozen-LLM prior, raw ranking, and final Top-K set so
that every missing target has an attributable pipeline location.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stage5g_build_clause_labels import transform_record  # noqa: E402
from src.diagnosis.stage10_complete_coverage_diagnosis import (  # noqa: E402
    index_rows,
    item_id,
    prediction_top_key,
    read_jsonl,
    semantic_and_join_targets,
    write_json,
    write_jsonl,
)


SEMANTIC_ROLES = {
    "OUTPUT_TARGET",
    "ENTITY_NAME",
    "METRIC_TARGET",
    "PREDICATE_COLUMN",
    "VALUE_ANCHOR",
    "TEMPORAL_FILTER",
    "ORDER_KEY",
    "GROUP_KEY",
    "FORMULA_COMPONENT",
}
CLAUSES = ["select", "where", "group_by", "having", "order_by", "join"]


def graph_inputs(row):
    return row.get("inference_inputs", row)


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def normalized_surface(text):
    return "".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def node_text(node):
    parts = [
        node.get("name"),
        node.get("table"),
        node.get("column"),
        node.get("semantic_name"),
        node.get("semantic_description"),
        node.get("semantic_text"),
    ]
    parts.extend(node.get("semantic_aliases") or [])
    return " ".join(str(value) for value in parts if value)


def prior_map(row):
    return {
        int(item["schema_item_id"]): {
            str(role): float(score)
            for role, score in (item.get("role_scores") or {}).items()
        }
        for item in row.get("node_priors", [])
    }


def best_semantic_prior(role_scores):
    pairs = [
        (role, float(score))
        for role, score in role_scores.items()
        if role in SEMANTIC_ROLES
    ]
    return max(pairs, key=lambda pair: pair[1]) if pairs else (None, 0.0)


def clause_membership(label):
    transformed = transform_record(label)
    result = defaultdict(list)
    for clause in CLAUSES:
        for schema_id in transformed.get("clause_labels", {}).get(clause, []):
            result[int(schema_id)].append(clause)
    return result


def candidate_ids(row):
    return {
        int(node["schema_item_id"])
        for node in row.get("candidate_nodes", [])
    }


def selected_ids(prediction):
    key = prediction_top_key(prediction)
    return key, {
        item_id(item) for item in prediction.get(key, [])
    } if key else set()


def item_pipeline_reason(
    schema_id,
    semantic_target_count,
    top_k,
    candidates,
    raw,
    priors,
    llm_support_threshold,
):
    if semantic_target_count > top_k:
        return "topk_budget_infeasible"
    if schema_id not in candidates:
        return "candidate_generation_gap"
    if schema_id in raw:
        return "constrained_selector_drop"
    _, prior_score = best_semantic_prior(priors.get(schema_id, {}))
    if prior_score >= llm_support_threshold:
        return "llm_supported_but_ranking_drop"
    return "candidate_semantic_ranking_gap"


def review_flags(node, question, evidence, sql):
    question_overlap = sorted(tokens(node_text(node)) & tokens(question))
    evidence_overlap = sorted(tokens(node_text(node)) & tokens(evidence))
    name_surface = normalized_surface(node.get("column") or node.get("name"))
    sql_surface = normalized_surface(sql)
    flags = []
    if not question_overlap and not evidence_overlap:
        flags.append("implicit_or_unexpressed_mapping")
    if name_surface and name_surface not in sql_surface:
        flags.append("target_name_not_visible_in_gold_sql_surface")
    if node.get("type") == "column" and not node.get("data_type"):
        flags.append("missing_column_type_metadata")
    return flags, question_overlap, evidence_overlap


def sample_bucket(item_reasons, review_flags_all):
    reasons = set(item_reasons)
    if len(reasons) > 1:
        return "mixed_pipeline_failures"
    reason = next(iter(reasons), "unknown")
    if reason == "candidate_generation_gap":
        return "candidate_recall_review"
    if reason == "constrained_selector_drop":
        return "selection_policy_review"
    if reason == "llm_supported_but_ranking_drop":
        return "llm_fusion_review"
    if reason == "topk_budget_infeasible":
        return "budget_definition_review"
    if "target_name_not_visible_in_gold_sql_surface" in review_flags_all:
        return "label_or_sql_manual_review"
    return "semantic_mapping_review"


def organize(graphs, factor_graphs, predictions, priors, labels, threshold):
    common = sorted(
        set(graphs) & set(factor_graphs) & set(predictions) & set(priors) & set(labels)
    )
    cases = []
    for index in common:
        graph = graph_inputs(graphs[index])
        factor = factor_graphs[index]
        prediction = predictions[index]
        prior = priors[index]
        label = labels[index]
        targets = semantic_and_join_targets(label)
        semantic = set(targets["semantic"])
        top_key, selected = selected_ids(prediction)
        missing = sorted(semantic - selected)
        if not missing:
            continue

        top_k = int(top_key.split("_", 1)[1]) if top_key else 0
        nodes = {
            int(node["id"]): node for node in graph.get("schema_nodes", [])
        }
        # Exact labels retain the canonical schema metadata and are a safe
        # fallback when a diagnostic graph omitted optional fields.
        for node in label.get("schema_items", []):
            nodes.setdefault(int(node["id"]), node)
        candidates = candidate_ids(factor)
        raw = {int(value) for value in prediction.get("raw_top_ids", [])}
        baseline = {int(value) for value in prediction.get("baseline_top_ids", [])}
        priors_by_id = prior_map(prior)
        clauses = clause_membership(label)
        missing_items = []
        all_reasons, all_flags = [], []
        for schema_id in missing:
            node = nodes.get(schema_id, {"id": schema_id, "name": f"id:{schema_id}"})
            best_role, best_score = best_semantic_prior(priors_by_id.get(schema_id, {}))
            reason = item_pipeline_reason(
                schema_id,
                len(semantic),
                top_k,
                candidates,
                raw,
                priors_by_id,
                threshold,
            )
            flags, q_overlap, e_overlap = review_flags(
                node,
                label.get("question") or graph.get("question"),
                label.get("evidence") or graph.get("evidence"),
                label.get("sql"),
            )
            all_reasons.append(reason)
            all_flags.extend(flags)
            missing_items.append(
                {
                    "schema_item_id": schema_id,
                    "name": node.get("name"),
                    "type": node.get("type"),
                    "table": node.get("table"),
                    "column": node.get("column"),
                    "data_type": node.get("data_type"),
                    "clauses": clauses.get(schema_id, []),
                    "pipeline_reason": reason,
                    "in_candidate_graph": schema_id in candidates,
                    "in_raw_topk": schema_id in raw,
                    "in_baseline_topk": schema_id in baseline,
                    "best_llm_role": best_role,
                    "best_llm_semantic_score": best_score,
                    "llm_semantically_supported": best_score >= threshold,
                    "question_token_overlap": q_overlap,
                    "evidence_token_overlap": e_overlap,
                    "review_flags": flags,
                    "semantic_name": node.get("semantic_name"),
                    "semantic_description": node.get("semantic_description"),
                    "semantic_aliases": node.get("semantic_aliases") or [],
                }
            )
        case = {
            "record_index": index,
            "question_id": label.get("question_id"),
            "db_id": label.get("db_id") or graph.get("db_id"),
            "difficulty": label.get("difficulty"),
            "question": label.get("question") or graph.get("question"),
            "evidence": label.get("evidence") or graph.get("evidence"),
            "gold_sql": label.get("sql"),
            "top_k": top_k,
            "semantic_target_count": len(semantic),
            "semantic_missing_count": len(missing_items),
            "sample_bucket": sample_bucket(all_reasons, all_flags),
            "pipeline_reasons": sorted(set(all_reasons)),
            "review_flags": sorted(set(all_flags)),
            "missing_items": missing_items,
            "selected_schema_ids": sorted(selected),
        }
        cases.append(case)
    return cases, len(common)


def summarize(cases, aligned_count, config):
    pipeline = Counter()
    buckets = Counter()
    clauses = Counter()
    item_types = Counter()
    flags = Counter()
    databases = Counter()
    missing_names = Counter()
    missing_item_count = 0
    for case in cases:
        buckets[case["sample_bucket"]] += 1
        databases[case.get("db_id")] += 1
        flags.update(case.get("review_flags", []))
        for item in case["missing_items"]:
            missing_item_count += 1
            pipeline[item["pipeline_reason"]] += 1
            item_types[item.get("type") or "unknown"] += 1
            clauses.update(item.get("clauses") or ["unassigned"])
            missing_names[item.get("name") or f"id:{item['schema_item_id']}"] += 1
    return {
        "aligned_sample_count": aligned_count,
        "semantic_missing_sample_count": len(cases),
        "semantic_missing_item_count": missing_item_count,
        "sample_bucket_counts": dict(buckets),
        "item_pipeline_reason_counts": dict(pipeline),
        "missing_item_type_counts": dict(item_types),
        "missing_clause_counts": dict(clauses),
        "review_flag_counts": dict(flags),
        "database_counts": dict(databases),
        "top_missing_names": missing_names.most_common(50),
        "interpretation": {
            "candidate_generation_gap": "Correct semantic node never reached the Stage 10 candidate graph.",
            "constrained_selector_drop": "Raw Top-K contained the node but constrained selection removed it.",
            "llm_supported_but_ranking_drop": "Frozen LLM supported the node, but fused ranking missed Top-K.",
            "candidate_semantic_ranking_gap": "Node was a candidate but neither raw rank nor LLM prior was strong enough.",
            "label_or_sql_manual_review": "A review flag exists; this is a queue for inspection, not proof of bad annotation.",
        },
        "config": config,
        "leakage_note": "Gold SQL and labels are used only for post-hoc error organization.",
    }


def write_csv(path, cases):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "record_index", "question_id", "db_id", "difficulty", "sample_bucket",
        "semantic_target_count", "semantic_missing_count", "pipeline_reasons",
        "review_flags", "question", "evidence", "gold_sql", "missing_items",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = {key: case.get(key) for key in fields}
            for key in ["pipeline_reasons", "review_flags", "missing_items"]:
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)


def write_markdown(path, summary, cases):
    lines = [
        "# Stage 10-F Semantic-Missing Casebook",
        "",
        f"Aligned samples: {summary['aligned_sample_count']}",
        f"Semantic-missing samples: {summary['semantic_missing_sample_count']}",
        f"Missing semantic items: {summary['semantic_missing_item_count']}",
        "",
        "## Groups",
        "",
        "| Group | Samples |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {count} |"
        for name, count in sorted(summary["sample_bucket_counts"].items())
    )
    for case in cases:
        lines.extend(
            [
                "",
                f"## {case['record_index']} · {case.get('db_id')} · {case['sample_bucket']}",
                "",
                f"**Question:** {case.get('question') or ''}",
                "",
                f"**Evidence:** {case.get('evidence') or ''}",
                "",
                f"**Gold SQL:** `{case.get('gold_sql') or ''}`",
                "",
                "| Missing schema item | Type | Clause | Pipeline location | LLM role / score | Review flags |",
                "|---|---|---|---|---|---|",
            ]
        )
        for item in case["missing_items"]:
            role_score = f"{item.get('best_llm_role') or '-'} / {item.get('best_llm_semantic_score', 0.0):.3f}"
            lines.append(
                "| {name} | {type_} | {clauses} | `{reason}` | {role_score} | {flags} |".format(
                    name=str(item.get("name") or item["schema_item_id"]).replace("|", "\\|"),
                    type_=item.get("type") or "-",
                    clauses=", ".join(item.get("clauses") or ["-"]),
                    reason=item["pipeline_reason"],
                    role_score=role_score,
                    flags=", ".join(item.get("review_flags") or ["-"]),
                )
            )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Organize Stage 10-F semantic misses.")
    parser.add_argument("--full-graph-file", required=True)
    parser.add_argument("--factor-graph-file", required=True)
    parser.add_argument("--prediction-file", required=True)
    parser.add_argument("--prior-file", required=True)
    parser.add_argument("--exact-label-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-support-threshold", type=float, default=0.5)
    args = parser.parse_args()

    graphs = index_rows(read_jsonl(args.full_graph_file))
    factor_graphs = index_rows(read_jsonl(args.factor_graph_file))
    predictions = index_rows(read_jsonl(args.prediction_file))
    priors = index_rows(read_jsonl(args.prior_file))
    labels = index_rows(read_jsonl(args.exact_label_file))
    cases, aligned_count = organize(
        graphs, factor_graphs, predictions, priors, labels, args.llm_support_threshold
    )
    output_dir = Path(args.output_dir)
    summary = summarize(cases, aligned_count, vars(args))
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "semantic_missing_cases.jsonl", cases)
    write_csv(output_dir / "semantic_missing_cases.csv", cases)
    write_markdown(output_dir / "semantic_missing_casebook.md", summary, cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
