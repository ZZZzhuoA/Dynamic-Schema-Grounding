"""Grounding-controlled logits processors for Stage 7 decoding."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
import math
import re

import torch

from stage7_sql_state import detect_operation_state, role_allowed_for_operation

try:
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover - imported on the server runtime.
    class LogitsProcessor:  # type: ignore[no-redef]
        pass


def _suffix_matches_prefix(generated: list[int], sequence: list[int]) -> int:
    """Return matched prefix length when generated ends with seq prefix.

    If no prefix matches, return 0.  A full match returns len(sequence), but the
    caller should not request another token for a completed sequence.
    """

    max_len = min(len(generated), len(sequence) - 1)
    for length in range(max_len, 0, -1):
        if generated[-length:] == sequence[:length]:
            return length
    return 0


def parse_operation_boosts(text: str | None) -> dict[str, float]:
    boosts: dict[str, float] = {}
    if not text:
        return boosts
    for chunk in str(text).split(","):
        if not chunk.strip() or ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        try:
            boosts[key.strip().upper()] = float(value)
        except ValueError:
            continue
    return boosts


def normalize_for_count(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def schema_mention_count(generated_text: str, schema_name: str) -> int:
    if not generated_text or not schema_name:
        return 0
    candidates = [schema_name]
    if "." in schema_name:
        candidates.append(schema_name.split(".", 1)[1])
    norm_text = f" {normalize_for_count(generated_text)} "
    count = 0
    for candidate in candidates:
        norm_candidate = normalize_for_count(candidate)
        if norm_candidate:
            count = max(count, norm_text.count(f" {norm_candidate} "))
    return count


class GroundedSchemaLogitsProcessor(LogitsProcessor):
    """Bias next-token logits toward schema token sequences.

    This is intentionally a decoding-time neural interface rather than prompt
    formatting.  Grounding scores are converted to token-level control signals
    and added directly to the model's next-token logits.
    """

    def __init__(
        self,
        schema_token_sequences: list[list[dict[str, Any]]],
        prompt_lengths: list[int],
        tokenizer: Any | None = None,
        boost: float = 1.5,
        first_token_boost_ratio: float = 0.35,
        continuation_boost_ratio: float = 1.0,
        max_bias_per_token: float = 6.0,
        enable_operation_gate: bool = False,
        allow_unknown_role: bool = True,
        require_identifier_position: bool = False,
        operation_boosts: dict[str, float] | None = None,
        use_operation_specific_gate: bool = False,
        repetition_decay: float = 0.0,
        max_schema_mentions: int = 0,
    ):
        self.schema_token_sequences = schema_token_sequences
        self.prompt_lengths = prompt_lengths
        self.tokenizer = tokenizer
        self.boost = float(boost)
        self.first_token_boost_ratio = float(first_token_boost_ratio)
        self.continuation_boost_ratio = float(continuation_boost_ratio)
        self.max_bias_per_token = float(max_bias_per_token)
        self.enable_operation_gate = bool(enable_operation_gate)
        self.allow_unknown_role = bool(allow_unknown_role)
        self.require_identifier_position = bool(require_identifier_position)
        self.operation_boosts = operation_boosts or {}
        self.use_operation_specific_gate = bool(use_operation_specific_gate)
        self.repetition_decay = float(repetition_decay)
        self.max_schema_mentions = int(max_schema_mentions)
        self.call_count = 0
        self.total_biased_tokens = 0
        self.max_observed_bias = 0.0
        self.gated_off_calls = 0
        self.operation_specific_gated_off_calls = 0
        self.operation_counts = defaultdict(int)
        self.role_filtered_sequences = 0
        self.repetition_filtered_sequences = 0
        self.repetition_decayed_sequences = 0

    def _generated_text(self, generated_ids: list[int]) -> str:
        if self.tokenizer is None or not generated_ids:
            return ""
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.call_count += 1
        batch_size = input_ids.shape[0]
        for row in range(batch_size):
            prompt_len = self.prompt_lengths[row] if row < len(self.prompt_lengths) else 0
            generated = input_ids[row, prompt_len:].tolist()
            generated_text = self._generated_text(generated)
            state = detect_operation_state(generated_text)
            self.operation_counts[state.operation] += 1
            if self.enable_operation_gate and self.require_identifier_position and not state.should_bias_schema:
                self.gated_off_calls += 1
                continue
            if self.enable_operation_gate and self.use_operation_specific_gate and not state.operation_should_bias_schema:
                self.operation_specific_gated_off_calls += 1
                continue
            token_bias = defaultdict(float)
            sequences = self.schema_token_sequences[row] if row < len(self.schema_token_sequences) else []

            for item in sequences:
                if self.enable_operation_gate and not role_allowed_for_operation(
                    item.get("role"),
                    state.operation,
                    allow_unknown_role=self.allow_unknown_role,
                ):
                    self.role_filtered_sequences += 1
                    continue
                token_ids = item.get("token_ids") or []
                if not token_ids:
                    continue
                weight = self.boost * float(item.get("score", 1.0))
                if state.operation in self.operation_boosts:
                    weight *= self.operation_boosts[state.operation]
                if weight <= 0:
                    continue
                mention_count = schema_mention_count(generated_text, str(item.get("schema") or ""))
                if self.max_schema_mentions > 0 and mention_count >= self.max_schema_mentions:
                    self.repetition_filtered_sequences += 1
                    continue
                if self.repetition_decay > 0 and mention_count > 0:
                    weight *= math.exp(-self.repetition_decay * mention_count)
                    self.repetition_decayed_sequences += 1
                    if weight <= 0:
                        continue

                matched = _suffix_matches_prefix(generated, token_ids)
                if matched > 0 and matched < len(token_ids):
                    token_bias[int(token_ids[matched])] += weight * self.continuation_boost_ratio
                elif matched == 0:
                    token_bias[int(token_ids[0])] += weight * self.first_token_boost_ratio

            if not token_bias:
                continue
            self.total_biased_tokens += len(token_bias)
            self.max_observed_bias = max(self.max_observed_bias, max(float(value) for value in token_bias.values()))
            for token_id, bias in token_bias.items():
                if 0 <= token_id < scores.shape[-1]:
                    scores[row, token_id] += min(float(bias), self.max_bias_per_token)
        return scores

    def diagnostics(self) -> dict[str, Any]:
        return {
            "processor_call_count": self.call_count,
            "total_biased_tokens": self.total_biased_tokens,
            "avg_biased_tokens_per_call": (
                self.total_biased_tokens / self.call_count if self.call_count else 0.0
            ),
            "max_observed_bias_before_cap": self.max_observed_bias,
            "max_bias_per_token": self.max_bias_per_token,
            "gated_off_calls": self.gated_off_calls,
            "operation_specific_gated_off_calls": self.operation_specific_gated_off_calls,
            "operation_counts": dict(self.operation_counts),
            "role_filtered_sequences": self.role_filtered_sequences,
            "repetition_filtered_sequences": self.repetition_filtered_sequences,
            "repetition_decayed_sequences": self.repetition_decayed_sequences,
            "enable_operation_gate": self.enable_operation_gate,
            "require_identifier_position": self.require_identifier_position,
            "use_operation_specific_gate": self.use_operation_specific_gate,
            "operation_boosts": self.operation_boosts,
            "repetition_decay": self.repetition_decay,
            "max_schema_mentions": self.max_schema_mentions,
        }


def summarize_token_sequences(schema_token_sequences: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    summary = []
    for item in schema_token_sequences[:limit]:
        summary.append(
            {
                "schema": item.get("schema"),
                "variant": item.get("variant"),
                "score": item.get("score"),
                "token_count": len(item.get("token_ids") or []),
                "role": item.get("role"),
            }
        )
    return summary
