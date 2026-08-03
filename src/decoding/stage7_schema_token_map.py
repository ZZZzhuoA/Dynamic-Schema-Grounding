"""Schema-token mapping utilities for Stage 7 grounded decoding.

This module converts schema grounding records into tokenizer-level token
sequences.  The sequences are later used by a LogitsProcessor to bias LLM
decoding toward grounded schema identifiers without putting the grounding
scores into the prompt.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Iterable


def normalize_name(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def clean_identifier(text: str) -> str:
    text = normalize_name(text)
    if len(text) >= 2 and text[0] in "`[\"" and text[-1] in "`]\"":
        return text[1:-1]
    return text


def schema_name_variants(table: str | None, column: str | None = None) -> list[str]:
    """Build likely SQL surface forms for a table or column.

    SQLite identifiers in BIRD often require backticks or brackets.  We include
    both quoted and unquoted variants because different LLMs choose different
    styles even when the prompt uses one consistent format.
    """

    table = clean_identifier(table or "")
    column = clean_identifier(column or "")
    variants: list[str] = []
    if table and not column:
        variants.extend([table, f"`{table}`", f"[{table}]"])
    elif table and column:
        variants.extend(
            [
                column,
                f"`{column}`",
                f"[{column}]",
                f"{table}.{column}",
                f"`{table}`.`{column}`",
                f"{table}.`{column}`",
                f"{table}.[{column}]",
            ]
        )
    elif column:
        variants.extend([column, f"`{column}`", f"[{column}]"])

    # Preserve order while removing duplicates and empty strings.
    seen = set()
    deduped = []
    for item in variants:
        item = normalize_name(item)
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


@dataclass(frozen=True)
class GroundedSchemaItem:
    full_name: str
    table: str
    column: str | None
    score: float
    source: str = "unknown"
    role: str | None = None

    @property
    def is_table(self) -> bool:
        return self.column is None


def split_full_name(full_name: str) -> tuple[str, str | None]:
    full_name = clean_identifier(full_name)
    if "." not in full_name:
        return full_name, None
    table, column = full_name.split(".", 1)
    return clean_identifier(table), clean_identifier(column)


def _score_from_object(obj: dict[str, Any], default: float) -> float:
    for key in ["score", "prob", "probability", "logit", "final_score", "grounding_score"]:
        if key in obj:
            try:
                return float(obj[key])
            except (TypeError, ValueError):
                pass
    return default


def _append_item(
    items: list[GroundedSchemaItem],
    full_name: str,
    score: float,
    source: str,
    role: str | None = None,
):
    table, column = split_full_name(full_name)
    if not table:
        return
    items.append(
        GroundedSchemaItem(
            full_name=f"{table}.{column}" if column else table,
            table=table,
            column=column,
            score=float(score),
            source=source,
            role=role,
        )
    )


def items_from_selected_schema(record: dict[str, Any], default_score: float = 1.0) -> list[GroundedSchemaItem]:
    selected = record.get("selected_schema") or {}
    tables = selected.get("tables") or {}
    items: list[GroundedSchemaItem] = []
    for table_name, columns in tables.items():
        _append_item(items, table_name, default_score, "selected_schema")
        for column in columns or []:
            if isinstance(column, dict):
                full_name = column.get("full_name") or f"{table_name}.{column.get('name', '')}"
            else:
                full_name = f"{table_name}.{column}"
            _append_item(items, full_name, default_score, "selected_schema")
    return items


def items_from_prediction_record(record: dict[str, Any], top_k: int | None = None) -> list[GroundedSchemaItem]:
    """Extract grounded schema items from several prediction JSONL formats.

    The earlier stages in this project used multiple formats while the method
    evolved.  This function intentionally accepts all common variants:
    ``top_30``, ``predictions``, ``selected_schema``, relation/role buckets, and
    plain lists of full names.
    """

    items: list[GroundedSchemaItem] = []

    if "selected_schema" in record:
        items.extend(items_from_selected_schema(record))

    candidate_keys = [
        "predictions",
        "schema_predictions",
        "ranked_schema",
        "ranked_items",
        "items",
    ]
    if top_k is not None:
        candidate_keys = [f"top_{top_k}", f"top{top_k}"] + candidate_keys
    candidate_keys.extend([key for key in record if re.fullmatch(r"top_?\d+", str(key))])

    for key in candidate_keys:
        value = record.get(key)
        if not isinstance(value, list):
            continue
        for rank, obj in enumerate(value):
            default_score = 1.0 / (rank + 1)
            if isinstance(obj, str):
                _append_item(items, obj, default_score, key)
            elif isinstance(obj, dict):
                full_name = (
                    obj.get("full_name")
                    or obj.get("schema_item")
                    or obj.get("schema")
                    or obj.get("name")
                    or obj.get("id")
                )
                if not full_name and obj.get("table") and obj.get("column"):
                    full_name = f"{obj['table']}.{obj['column']}"
                if full_name:
                    _append_item(
                        items,
                        full_name,
                        _score_from_object(obj, default_score),
                        key,
                        role=obj.get("role") or obj.get("relation") or obj.get("clause"),
                    )

    # Relation-role outputs may be stored as buckets: {"WHERE": [...], ...}.
    for bucket_key in ["roles", "role_predictions", "relation_predictions", "clause_predictions"]:
        buckets = record.get(bucket_key)
        if not isinstance(buckets, dict):
            continue
        for role, value in buckets.items():
            if not isinstance(value, list):
                continue
            for rank, obj in enumerate(value):
                default_score = 1.0 / (rank + 1)
                if isinstance(obj, str):
                    _append_item(items, obj, default_score, bucket_key, role=role)
                elif isinstance(obj, dict):
                    full_name = obj.get("full_name") or obj.get("schema_item") or obj.get("name")
                    if not full_name and obj.get("table") and obj.get("column"):
                        full_name = f"{obj['table']}.{obj['column']}"
                    if full_name:
                        _append_item(
                            items,
                            full_name,
                            _score_from_object(obj, default_score),
                            bucket_key,
                            role=role,
                        )

    return dedupe_items(items)


def dedupe_items(items: Iterable[GroundedSchemaItem]) -> list[GroundedSchemaItem]:
    best: dict[tuple[str, str | None, str | None], GroundedSchemaItem] = {}
    for item in items:
        key = (item.table.lower(), item.column.lower() if item.column else None, item.role)
        old = best.get(key)
        if old is None or item.score > old.score:
            best[key] = item
    return list(best.values())


def apply_intervention(
    items: list[GroundedSchemaItem],
    mode: str = "none",
    seed: int = 13,
) -> list[GroundedSchemaItem]:
    if mode == "none":
        return items
    if mode == "zero":
        return [
            GroundedSchemaItem(item.full_name, item.table, item.column, 0.0, item.source, item.role)
            for item in items
        ]
    scores = [item.score for item in items]
    if mode == "random":
        rng = random.Random(seed)
        shuffled = scores[:]
        rng.shuffle(shuffled)
        return [
            GroundedSchemaItem(item.full_name, item.table, item.column, score, item.source, item.role)
            for item, score in zip(items, shuffled)
        ]
    if mode == "reverse":
        if not scores:
            return items
        lo, hi = min(scores), max(scores)
        return [
            GroundedSchemaItem(item.full_name, item.table, item.column, hi + lo - item.score, item.source, item.role)
            for item in items
        ]
    raise ValueError(f"Unsupported intervention mode: {mode}")


def normalize_scores(items: list[GroundedSchemaItem]) -> list[GroundedSchemaItem]:
    if not items:
        return []
    scores = [item.score for item in items]
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [
            GroundedSchemaItem(item.full_name, item.table, item.column, 1.0, item.source, item.role)
            for item in items
        ]
    return [
        GroundedSchemaItem(
            item.full_name,
            item.table,
            item.column,
            (item.score - lo) / (hi - lo),
            item.source,
            item.role,
        )
        for item in items
    ]


def build_schema_token_sequences(
    tokenizer: Any,
    items: list[GroundedSchemaItem],
    max_items: int = 80,
    min_score: float = 0.0,
    include_tables: bool = True,
) -> list[dict[str, Any]]:
    """Convert grounded schema items to token sequences with scalar weights."""

    normalized = normalize_scores(items)
    normalized = [item for item in normalized if item.score >= min_score]
    normalized.sort(key=lambda item: item.score, reverse=True)
    if max_items:
        normalized = normalized[:max_items]

    sequences: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for item in normalized:
        if item.is_table and not include_tables:
            continue
        variants = schema_name_variants(item.table, item.column)
        for variant in variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if not token_ids:
                continue
            key = tuple(int(token_id) for token_id in token_ids)
            if key in seen:
                continue
            seen.add(key)
            sequences.append(
                {
                    "schema": item.full_name,
                    "role": item.role,
                    "score": float(item.score),
                    "variant": variant,
                    "token_ids": list(key),
                }
            )
    return sequences
