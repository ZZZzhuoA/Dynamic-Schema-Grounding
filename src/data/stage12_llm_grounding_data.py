"""Prompt and token/grounding alignment helpers for Stage 12."""

from collections import defaultdict
import re

from src.decoding.stage7_sql_state import detect_operation_state


def full_schema_text(graph_example):
    inputs = graph_example.get("inference_inputs", graph_example)
    nodes = inputs.get("schema_nodes", [])
    by_local = {int(node.get("id", index)): node for index, node in enumerate(nodes)}
    tables = defaultdict(list)
    for node in nodes:
        if node.get("type") == "table":
            tables.setdefault(str(node.get("name")), [])
        elif node.get("type") == "column":
            table = str(node.get("table") or "")
            column = node.get("column") or str(node.get("name", "")).split(".", 1)[-1]
            tables[table].append((str(column), node.get("data_type")))
    lines = []
    for table in sorted(tables):
        lines.append(f"Table {table}:")
        for column, data_type in sorted(tables[table]):
            suffix = f" ({data_type})" if data_type else ""
            lines.append(f"- `{column}`{suffix}")
        lines.append("")
    foreign_keys = set()
    for edge in inputs.get("schema_edges", []):
        if edge.get("type") not in {"foreign_key_forward", "foreign_key_backward"}:
            continue
        src, dst = by_local.get(int(edge["src"])), by_local.get(int(edge["dst"]))
        if not src or not dst or src.get("type") != "column" or dst.get("type") != "column":
            continue
        left, right = str(src.get("name")), str(dst.get("name"))
        foreign_keys.add(tuple(sorted((left, right))))
    if foreign_keys:
        lines.append("Foreign keys:")
        for left, right in sorted(foreign_keys):
            lines.append(f"- {left} = {right}")
    return "\n".join(lines).strip()


def build_full_schema_prompt(trajectory, graph_example):
    evidence = trajectory.get("evidence") or ""
    evidence_block = f"\nEvidence:\n{evidence}\n" if evidence else ""
    return (
        "Given the complete database schema and question, generate one valid SQLite SQL query.\n"
        "Use exact table and column names. Return SQL only.\n\n"
        f"Database schema:\n{full_schema_text(graph_example)}\n\n"
        f"Question:\n{trajectory.get('question', '')}\n"
        f"{evidence_block}\nReturn only the SQL query."
    )


def build_chat_prompt(tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are an expert SQLite SQL generator. Return only SQL."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    return f"System: {messages[0]['content']}\n\nUser: {prompt}\n\nAssistant:"


def operation_step_map(trajectory_steps):
    result = {}
    for index, step in enumerate(trajectory_steps):
        result.setdefault(str(step.get("operation", "UNKNOWN")), index)
    if "PROJECT" not in result and trajectory_steps:
        result["PROJECT"] = 0
    return result


def operation_for_partial_sql(partial_sql):
    """Clause state that also recognizes a just-generated terminal keyword."""
    text = re.sub(r"\s+", " ", str(partial_sql or "").strip().lower())
    if not text:
        return "PROJECT"
    patterns = {
        "PROJECT": r"\bselect\b",
        "JOIN": r"\b(?:from|join|on)\b",
        "FILTER": r"\b(?:where|having|and|or)\b",
        "GROUP": r"\bgroup\s+by\b",
        "ORDER": r"\border\s+by\b",
    }
    positions = {}
    for operation, pattern in patterns.items():
        matches = list(re.finditer(pattern, text))
        positions[operation] = matches[-1].start() if matches else -1
    operation, position = max(positions.items(), key=lambda item: item[1])
    if position >= 0:
        return operation
    detected = detect_operation_state(text).operation
    return "PROJECT" if detected in {"UNKNOWN", "COMPUTE"} else detected


def step_for_partial_sql(partial_sql, trajectory_steps):
    mapping = operation_step_map(trajectory_steps)
    operation = operation_for_partial_sql(partial_sql)
    if operation in mapping:
        return mapping[operation]
    return mapping.get("PROJECT", 0)


def teacher_forcing_token_steps(tokenizer, prompt_ids, sql_ids, trajectory_steps):
    """Map each hidden position to the grounding used to predict its next token."""
    total = len(prompt_ids) + len(sql_ids)
    token_steps = [-1] * total
    if not prompt_ids or not sql_ids or not trajectory_steps:
        return token_steps
    for target_offset in range(len(sql_ids)):
        prefix = tokenizer.decode(sql_ids[:target_offset], skip_special_tokens=True)
        step_id = step_for_partial_sql(prefix, trajectory_steps)
        prediction_position = len(prompt_ids) - 1 + target_offset
        if prediction_position < total:
            token_steps[prediction_position] = step_id
    return token_steps


def observed_candidate_mask(partial_sql, candidate_nodes):
    normalized = str(partial_sql or "").lower()
    mask = []
    for node in candidate_nodes:
        names = {str(node.get("name") or "").lower()}
        if node.get("column"):
            names.add(str(node["column"]).lower())
        matched = any(name and name in normalized for name in names)
        mask.append(float(matched))
    return mask
