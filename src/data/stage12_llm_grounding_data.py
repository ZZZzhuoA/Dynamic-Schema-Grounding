"""Prompt and token/grounding alignment helpers for Stage 12."""

from collections import defaultdict
import re

from src.decoding.stage7_sql_state import detect_operation_state


TOKEN_ROLE_BASE = 0
TOKEN_ROLE_SCHEMA = 1
TOKEN_ROLE_OPERATOR = 2
TOKEN_ROLE_VALUE = 3
TOKEN_ROLE_NAMES = {
    TOKEN_ROLE_BASE: "base",
    TOKEN_ROLE_SCHEMA: "schema",
    TOKEN_ROLE_OPERATOR: "operator",
    TOKEN_ROLE_VALUE: "value",
}

SEMANTIC_OPERATOR_PATTERN = re.compile(
    r"(?i)(?:\b(?:distinct|count|sum|avg|min|max|cast|coalesce|case|when|then|else|end|"
    r"group\s+by|order\s+by|having|limit|offset|union|intersect|except|join|on|like|in|"
    r"between|is\s+not\s+null|is\s+null|asc|desc)\b|<>|!=|<=|>=|=|<|>|\+|-|\*|/)"
)
VALUE_PATTERN = re.compile(
    r"(?:'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?![A-Za-z_]))"
)


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


def schema_identifier_names(graph_example):
    inputs = graph_example.get("inference_inputs", graph_example)
    names = set()
    for node in inputs.get("schema_nodes", []):
        node_type = node.get("type")
        if node_type == "table":
            value = str(node.get("name") or "").strip()
            if value:
                names.add(value)
        elif node_type == "column":
            table = str(node.get("table") or "").strip()
            column = str(
                node.get("column") or str(node.get("name") or "").split(".", 1)[-1]
            ).strip()
            if column:
                names.add(column)
            if table and column:
                names.add(f"{table}.{column}")
    return sorted(names, key=len, reverse=True)


def _identifier_pattern(identifier):
    escaped = re.escape(identifier)
    # Bare identifiers need token boundaries; quoted/bracketed forms are handled
    # explicitly so names containing spaces or punctuation remain matchable.
    return re.compile(
        rf"(?i)(?:`{escaped}`|\[{escaped}\]|\"{escaped}\"|(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_]))"
    )


def sql_semantic_spans(sql, graph_example):
    """Return non-exclusive character spans with schema taking highest priority."""
    text = str(sql or "")
    spans = []
    for match in VALUE_PATTERN.finditer(text):
        spans.append((match.start(), match.end(), TOKEN_ROLE_VALUE))
    for match in SEMANTIC_OPERATOR_PATTERN.finditer(text):
        spans.append((match.start(), match.end(), TOKEN_ROLE_OPERATOR))
    for identifier in schema_identifier_names(graph_example):
        for match in _identifier_pattern(identifier).finditer(text):
            spans.append((match.start(), match.end(), TOKEN_ROLE_SCHEMA))
    return spans


def sql_token_roles(tokenizer, sql, graph_example, append_eos=False):
    """Align SQL semantic roles to tokenizer IDs using character offsets."""
    text = str(sql or "").strip()
    try:
        encoded = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
    except (TypeError, NotImplementedError):
        encoded = tokenizer(text, add_special_tokens=False)
    token_ids = list(encoded["input_ids"])
    offsets = encoded.get("offset_mapping")
    roles = [TOKEN_ROLE_BASE] * len(token_ids)
    if offsets is not None:
        spans = sql_semantic_spans(text, graph_example)
        priority = {
            TOKEN_ROLE_BASE: 0,
            TOKEN_ROLE_OPERATOR: 1,
            TOKEN_ROLE_VALUE: 2,
            TOKEN_ROLE_SCHEMA: 3,
        }
        for token_index, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            for span_start, span_end, role in spans:
                if start < span_end and end > span_start and priority[role] > priority[roles[token_index]]:
                    roles[token_index] = role
    if append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
        roles.append(TOKEN_ROLE_BASE)
    return token_ids, roles
