"""Grounding-controlled logits processors for Stage 7 decoding."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

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
        boost: float = 1.5,
        first_token_boost_ratio: float = 0.35,
        continuation_boost_ratio: float = 1.0,
        max_bias_per_token: float = 6.0,
    ):
        self.schema_token_sequences = schema_token_sequences
        self.prompt_lengths = prompt_lengths
        self.boost = float(boost)
        self.first_token_boost_ratio = float(first_token_boost_ratio)
        self.continuation_boost_ratio = float(continuation_boost_ratio)
        self.max_bias_per_token = float(max_bias_per_token)
        self.call_count = 0
        self.total_biased_tokens = 0
        self.max_observed_bias = 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.call_count += 1
        batch_size = input_ids.shape[0]
        for row in range(batch_size):
            prompt_len = self.prompt_lengths[row] if row < len(self.prompt_lengths) else 0
            generated = input_ids[row, prompt_len:].tolist()
            token_bias = defaultdict(float)
            sequences = self.schema_token_sequences[row] if row < len(self.schema_token_sequences) else []

            for item in sequences:
                token_ids = item.get("token_ids") or []
                if not token_ids:
                    continue
                weight = self.boost * float(item.get("score", 1.0))
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
