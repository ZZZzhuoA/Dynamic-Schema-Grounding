"""Full-schema query-conditioned relational graph transformer.

Stage 17-A0 deliberately predicts only whether each schema node is used by
the target SQL.  It consumes every table/column in the database and never
performs candidate pruning before graph propagation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


QUERY_TO_TABLE = "query_to_table"
QUERY_TO_COLUMN = "query_to_column"


def _segment_softmax(scores, destination, destination_count):
    """Softmax over incoming edges independently for every head/destination."""
    if scores.numel() == 0:
        return scores
    edge_count, head_count = scores.shape
    head_ids = torch.arange(head_count, device=scores.device).view(1, -1)
    segment = destination.view(-1, 1) * head_count + head_ids
    segment = segment.expand(edge_count, -1).reshape(-1)
    flat_scores = scores.reshape(-1)
    segment_count = destination_count * head_count
    maxima = torch.full(
        (segment_count,), -torch.inf, dtype=scores.dtype, device=scores.device
    )
    maxima.scatter_reduce_(0, segment, flat_scores, reduce="amax", include_self=True)
    exponentials = torch.exp(flat_scores - maxima[segment])
    denominator = torch.zeros(
        (segment_count,), dtype=scores.dtype, device=scores.device
    )
    denominator.index_add_(0, segment, exponentials)
    return (exponentials / denominator[segment].clamp_min(1e-8)).view_as(scores)


class SparseQRGTAEncoderLayer(nn.Module):
    """Sparse multi-head relation-aware attention over full schema edges."""

    def __init__(self, hidden_dim, num_heads, relation_count, dropout):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_key = nn.Embedding(relation_count, hidden_dim)
        self.relation_value = nn.Embedding(relation_count, hidden_dim)
        self.relation_bias = nn.Embedding(relation_count, num_heads)
        self.query_similarity_scale = nn.Parameter(torch.ones(num_heads))
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        schema_states,
        query_state,
        schema_edge_index,
        schema_edge_type,
        schema_edge_scalar,
        query_edge_destination,
        query_edge_type,
        query_edge_similarity,
    ):
        node_count = schema_states.shape[0]
        device = schema_states.device

        # Source index 0 is the fixed query. Schema sources are shifted by one.
        source_states = torch.cat([query_state.unsqueeze(0), schema_states], dim=0)
        structural_source = schema_edge_index[0] + 1
        structural_destination = schema_edge_index[1]
        query_source = torch.zeros_like(query_edge_destination)
        source = torch.cat([structural_source, query_source], dim=0)
        destination = torch.cat(
            [structural_destination, query_edge_destination], dim=0
        )
        relation = torch.cat([schema_edge_type, query_edge_type], dim=0)
        scalar = torch.cat([schema_edge_scalar, query_edge_similarity], dim=0)
        is_query_edge = torch.cat(
            [
                torch.zeros_like(schema_edge_scalar, dtype=torch.bool),
                torch.ones_like(query_edge_similarity, dtype=torch.bool),
            ],
            dim=0,
        )

        if source.numel() == 0:
            message = torch.zeros_like(schema_states)
        else:
            query = self.q_proj(schema_states).view(
                node_count, self.num_heads, self.head_dim
            )[destination]
            key = self.k_proj(source_states).view(
                source_states.shape[0], self.num_heads, self.head_dim
            )[source]
            value = self.v_proj(source_states).view(
                source_states.shape[0], self.num_heads, self.head_dim
            )[source]
            relation_key = self.relation_key(relation).view(
                -1, self.num_heads, self.head_dim
            )
            relation_value = self.relation_value(relation).view(
                -1, self.num_heads, self.head_dim
            )
            scores = ((query * (key + relation_key)).sum(dim=-1) / math.sqrt(self.head_dim))
            scores = scores + self.relation_bias(relation)
            scores = scores + (
                scalar.unsqueeze(-1)
                * is_query_edge.to(scores.dtype).unsqueeze(-1)
                * self.query_similarity_scale.unsqueeze(0)
            )
            attention = _segment_softmax(scores, destination, node_count)
            weighted = (value + relation_value) * attention.unsqueeze(-1)
            message_heads = torch.zeros(
                (node_count, self.num_heads, self.head_dim),
                dtype=weighted.dtype,
                device=device,
            )
            message_heads.index_add_(0, destination, weighted)
            message = message_heads.reshape(node_count, self.hidden_dim)

        schema_states = self.attention_norm(
            schema_states + self.dropout(self.output(message))
        )
        schema_states = self.ffn_norm(
            schema_states + self.dropout(self.ffn(schema_states))
        )
        return schema_states


class ResidualNodeMLPLayer(nn.Module):
    """Depth-matched node-local baseline with no graph message passing."""

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, schema_states):
        return self.norm(schema_states + self.dropout(self.ffn(schema_states)))


class FullSchemaQRGTA(nn.Module):
    """Binary full-schema node grounder with an optional graph-free MLP mode."""

    def __init__(
        self,
        dense_dim,
        relation_count,
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
        model_type="qrgta",
    ):
        super().__init__()
        if model_type not in {"qrgta", "mlp", "mlp_residual"}:
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.model_type = model_type
        self.node_input = nn.Linear(dense_dim, hidden_dim)
        self.query_input = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_type = nn.Embedding(2, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [
                SparseQRGTAEncoderLayer(
                    hidden_dim, num_heads, relation_count, dropout
                )
                for _ in range(num_layers if model_type == "qrgta" else 0)
            ]
        )
        self.node_mlp_layers = nn.ModuleList(
            [
                ResidualNodeMLPLayer(hidden_dim, dropout)
                for _ in range(num_layers if model_type == "mlp_residual" else 0)
            ]
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        dense_nodes,
        node_types,
        query_embedding,
        schema_edge_index,
        schema_edge_type,
        query_edge_destination,
        query_edge_type,
        query_edge_similarity,
    ):
        schema_states = self.node_norm(
            self.node_input(dense_nodes) + self.node_type(node_types)
        )
        query_state = self.query_norm(
            self.query_input(query_embedding).reshape(-1, schema_states.shape[-1])[0]
        )
        schema_edge_scalar = dense_nodes.new_zeros(schema_edge_type.shape[0])
        for layer in self.layers:
            schema_states = layer(
                schema_states,
                query_state,
                schema_edge_index,
                schema_edge_type,
                schema_edge_scalar,
                query_edge_destination,
                query_edge_type,
                query_edge_similarity,
            )
        for layer in self.node_mlp_layers:
            schema_states = layer(schema_states)
        query_matrix = query_state.unsqueeze(0).expand_as(schema_states)
        pair = torch.cat(
            [
                schema_states,
                query_matrix,
                schema_states * query_matrix,
                torch.abs(schema_states - query_matrix),
            ],
            dim=-1,
        )
        logits = self.scorer(pair).squeeze(-1)
        return {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "schema_states": schema_states,
        }


def balanced_binary_loss(logits, labels):
    """Per-example class-balanced BCE used by Stage 17-A0."""
    labels = labels.to(dtype=logits.dtype)
    positive = labels > 0.5
    negative = ~positive
    if not positive.any():
        raise ValueError("balanced_binary_loss requires at least one positive label")
    positive_loss = F.softplus(-logits[positive]).mean()
    if negative.any():
        negative_loss = F.softplus(logits[negative]).mean()
        return 0.5 * positive_loss + 0.5 * negative_loss
    return positive_loss
