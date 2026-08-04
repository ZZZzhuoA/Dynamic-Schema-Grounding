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


def detect_operation_state(partial_sql: str) -> SQLOperationState:
    sql = str(partial_sql or "")
    norm = normalize_sql(sql)
    tail = recent_text(sql)
    clause = infer_clause(sql)
    should_bias = ends_like_identifier_position(sql)

    if " over " in tail or tail.endswith(" over") or "partition by" in tail:
        return SQLOperationState(OP_WINDOW, should_bias, "window keyword")
    if " order by " in tail or clause == "ORDER":
        return SQLOperationState(OP_ORDER, should_bias, "order clause")
    if " group by " in tail or clause == "GROUP":
        return SQLOperationState(OP_GROUP, should_bias, "group clause")
    if " join " in tail or " on " in tail or clause == "JOIN":
        return SQLOperationState(OP_JOIN, should_bias, "join/on clause")
    if clause == "WHERE" or re.search(r"\b(where|and|or)\b", tail):
        return SQLOperationState(OP_FILTER, should_bias, "filter clause")
    if any(op in tail for op in [" + ", " - ", " * ", " / ", "cast(", "avg(", "sum(", "max(", "min(", "count("]):
        return SQLOperationState(OP_COMPUTE, should_bias, "expression/function context")
    if clause == "SELECT" or norm.startswith("select"):
        return SQLOperationState(OP_PROJECT, should_bias, "select projection")
    if clause == "FROM":
        return SQLOperationState(OP_JOIN, should_bias, "from source")
    return SQLOperationState(OP_UNKNOWN, should_bias, "fallback")


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
