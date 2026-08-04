"""SQL operation-state detection for Stage 7-B grounded decoding.

The detector is intentionally lightweight and model-agnostic.  It maps partial
SQL text to relational-operation states such as PROJECT, FILTER, JOIN, ORDER,
and COMPUTE.  These states are used as gates for schema grounding logits bias.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


OP_PROJECT = "PROJECT"
OP_FILTER = "FILTER"
OP_JOIN = "JOIN"
OP_GROUP = "GROUP"
OP_ORDER = "ORDER"
OP_COMPUTE = "COMPUTE"
OP_AGGREGATE = "AGGREGATE"
OP_WINDOW = "WINDOW"
OP_UNKNOWN = "UNKNOWN"


SQL_KEYWORDS = {
    "select",
    "from",
    "where",
    "join",
    "inner",
    "left",
    "right",
    "on",
    "and",
    "or",
    "group",
    "by",
    "order",
    "limit",
    "as",
    "desc",
    "asc",
    "cast",
    "distinct",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "rank",
    "over",
    "partition",
}


@dataclass(frozen=True)
class SQLOperationState:
    operation: str
    should_bias_schema: bool
    reason: str
    operation_should_bias_schema: bool


def normalize_sql(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def last_keyword(sql: str) -> str | None:
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", sql.lower())
    for token in reversed(tokens):
        if token in SQL_KEYWORDS:
            return token
    return None


def infer_clause(sql: str) -> str:
    norm = normalize_sql(sql)
    positions = {
        "WHERE": norm.rfind(" where "),
        "ORDER": norm.rfind(" order by "),
        "GROUP": norm.rfind(" group by "),
        "JOIN": max(norm.rfind(" join "), norm.rfind(" on ")),
        "FROM": norm.rfind(" from "),
        "SELECT": norm.rfind("select "),
    }
    best = max(positions.items(), key=lambda item: item[1])
    return best[0] if best[1] >= 0 else "UNKNOWN"


def recent_text(sql: str, chars: int = 96) -> str:
    return normalize_sql(sql)[-chars:]


def has_unclosed_identifier(sql: str) -> bool:
    # If a quoted/bracketed identifier is being generated, keep allowing schema
    # continuation bias.
    return sql.count("`") % 2 == 1 or sql.count("[") > sql.count("]")


def ends_like_identifier_position(sql: str) -> bool:
    """Heuristic gate for positions where schema identifiers are plausible."""

    raw = str(sql or "")
    tail = recent_text(raw)
    if not tail:
        return True
    if has_unclosed_identifier(raw):
        return True

    identifier_triggers = [
        "select",
        "where",
        "and",
        "or",
        "by",
        "on",
        "from",
        "join",
        ",",
        "(",
        "+",
        "-",
        "*",
        "/",
        "=",
        ">",
        "<",
        ">=",
        "<=",
        "<>",
    ]
    stripped = raw.rstrip()
    lower = stripped.lower()
    if not stripped:
        return True
    if stripped[-1] in {",", "(", "+", "-", "*", "/", "=", ">", "<"}:
        return True
    for trigger in identifier_triggers:
        if lower.endswith(" " + trigger) or lower.endswith(trigger + " "):
            return True
    # After SQL keywords with trailing whitespace.
    if re.search(r"(select|where|and|or|by|on|from|join)\s+$", raw, flags=re.I):
        return True
    return False


def operation_specific_should_bias(sql: str, operation: str) -> bool:
    """Operation-specific gate, less brittle than one global identifier gate."""

    raw = str(sql or "")
    tail = recent_text(raw, chars=128)
    if ends_like_identifier_position(raw):
        return True
    if has_unclosed_identifier(raw):
        return True

    if operation in {OP_PROJECT, OP_COMPUTE, OP_AGGREGATE}:
        return True
    if operation == OP_FILTER:
        # WHERE/FILTER needs stronger grounding, but only near predicate
        # construction sites.  After a full predicate has been generated, this
        # becomes false until AND/OR/comparison/function context appears again.
        if re.search(r"\b(where|and|or)\s+[`\\[a-zA-Z_]", tail):
            return True
        if re.search(r"(=|>|<|>=|<=|<>|!=)\s*$", tail):
            return True
        if tail.endswith(" and") or tail.endswith(" or") or tail.endswith(" where"):
            return True
        return False
    if operation == OP_JOIN:
        return " on " in tail or tail.endswith(" on") or " join " in tail or tail.endswith(" join")
    if operation == OP_ORDER:
        return " order by " in tail or tail.endswith(" order by") or tail.endswith(",")
    if operation == OP_GROUP:
        return " group by " in tail or tail.endswith(" group by") or tail.endswith(",")
    if operation == OP_WINDOW:
        return " over " in tail or " partition by " in tail or " order by " in tail
    return False


def detect_operation_state(partial_sql: str) -> SQLOperationState:
    sql = str(partial_sql or "")
    norm = normalize_sql(sql)
    tail = recent_text(sql)
    clause = infer_clause(sql)
    should_bias = ends_like_identifier_position(sql)

    if " over " in tail or tail.endswith(" over") or "partition by" in tail:
        op = OP_WINDOW
        return SQLOperationState(op, should_bias, "window keyword", operation_specific_should_bias(sql, op))
    if " order by " in tail or clause == "ORDER":
        op = OP_ORDER
        return SQLOperationState(op, should_bias, "order clause", operation_specific_should_bias(sql, op))
    if " group by " in tail or clause == "GROUP":
        op = OP_GROUP
        return SQLOperationState(op, should_bias, "group clause", operation_specific_should_bias(sql, op))
    if " join " in tail or " on " in tail or clause == "JOIN":
        op = OP_JOIN
        return SQLOperationState(op, should_bias, "join/on clause", operation_specific_should_bias(sql, op))
    if clause == "WHERE" or re.search(r"\b(where|and|or)\b", tail):
        op = OP_FILTER
        return SQLOperationState(op, should_bias, "filter clause", operation_specific_should_bias(sql, op))
    if any(op in tail for op in [" + ", " - ", " * ", " / ", "cast(", "avg(", "sum(", "max(", "min(", "count("]):
        op = OP_COMPUTE
        return SQLOperationState(op, should_bias, "expression/function context", operation_specific_should_bias(sql, op))
    if clause == "SELECT" or norm.startswith("select"):
        op = OP_PROJECT
        return SQLOperationState(op, should_bias, "select projection", operation_specific_should_bias(sql, op))
    if clause == "FROM":
        op = OP_JOIN
        return SQLOperationState(op, should_bias, "from source", operation_specific_should_bias(sql, op))
    return SQLOperationState(OP_UNKNOWN, should_bias, "fallback", operation_specific_should_bias(sql, OP_UNKNOWN))


ROLE_TO_OPERATIONS = {
    "OUTPUT_TARGET": {OP_PROJECT, OP_COMPUTE, OP_ORDER, OP_WINDOW},
    "METRIC_TARGET": {OP_COMPUTE, OP_ORDER, OP_PROJECT, OP_AGGREGATE},
    "PREDICATE_COLUMN": {OP_FILTER},
    "VALUE_ANCHOR": {OP_FILTER},
    "TEMPORAL_FILTER": {OP_FILTER, OP_ORDER},
    "ORDER_KEY": {OP_ORDER, OP_COMPUTE},
    "GROUP_KEY": {OP_GROUP, OP_PROJECT},
    "JOIN_END": {OP_JOIN},
    "JOIN_BRIDGE": {OP_JOIN},
    "FORMULA_COMPONENT": {OP_COMPUTE, OP_PROJECT, OP_ORDER},
    # Clause names from Stage 5g.
    "select": {OP_PROJECT, OP_COMPUTE},
    "where": {OP_FILTER},
    "join": {OP_JOIN},
    "order_by": {OP_ORDER},
}


def operations_for_role(role: str | None) -> set[str]:
    if not role:
        return set()
    return ROLE_TO_OPERATIONS.get(str(role), ROLE_TO_OPERATIONS.get(str(role).lower(), set()))


def role_allowed_for_operation(role: str | None, operation: str, allow_unknown_role: bool = True) -> bool:
    ops = operations_for_role(role)
    if not ops:
        return allow_unknown_role
    return operation in ops
