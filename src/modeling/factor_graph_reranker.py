import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_attention(
    query_states,
    source_states,
    source_index,
    destination_index,
    edge_type,
    edge_weight,
    q_proj,
    k_proj,
    v_proj,
    edge_key_embedding,
    edge_value_embedding,
):
    """Edge-constrained attention from source nodes into destination nodes."""
    destination_count = query_states.shape[0]
    hidden_dim = query_states.shape[1]
    if source_index.numel() == 0:
        return torch.zeros_like(query_states)
    query = q_proj(query_states)[destination_index]
    key = k_proj(source_states)[source_index] + edge_key_embedding(edge_type)
    value = v_proj(source_states)[source_index] + edge_value_embedding(edge_type)
    scores = (query * key).sum(dim=-1) / math.sqrt(hidden_dim)
    if edge_weight is not None:
        scores = scores + torch.log(edge_weight.clamp_min(1e-6))
    max_per_destination = torch.full(
        (destination_count,),
        -torch.inf,
        dtype=scores.dtype,
        device=scores.device,
    )
    max_per_destination.scatter_reduce_(
        0, destination_index, scores, reduce="amax", include_self=True
    )
    exp_scores = torch.exp(scores - max_per_destination[destination_index])
    denominator = torch.zeros(
        (destination_count,), dtype=scores.dtype, device=scores.device
    )
    denominator.index_add_(0, destination_index, exp_scores)
    attention = exp_scores / denominator[destination_index].clamp_min(1e-8)
    output = torch.zeros(
        (destination_count, hidden_dim),
        dtype=value.dtype,
        device=value.device,
    )
    output.index_add_(0, destination_index, value * attention.unsqueeze(-1))
    return output


class FactorGraphLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        schema_relation_count,
        factor_edge_type_count,
        dropout,
        use_schema_graph=True,
        use_factor_graph=True,
    ):
        super().__init__()
        self.use_schema_graph = use_schema_graph
        self.use_factor_graph = use_factor_graph
        self.schema_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.schema_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.schema_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.schema_edge_k = nn.Embedding(schema_relation_count, hidden_dim)
        self.schema_edge_v = nn.Embedding(schema_relation_count, hidden_dim)
        self.factor_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.factor_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.factor_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.factor_edge_k = nn.Embedding(factor_edge_type_count, hidden_dim)
        self.factor_edge_v = nn.Embedding(factor_edge_type_count, hidden_dim)
        self.reverse_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reverse_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reverse_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reverse_edge_k = nn.Embedding(factor_edge_type_count, hidden_dim)
        self.reverse_edge_v = nn.Embedding(factor_edge_type_count, hidden_dim)
        self.schema_gate = nn.Linear(hidden_dim * 3, 2)
        self.factor_gate = nn.Linear(hidden_dim * 2, 1)
        self.schema_out = nn.Linear(hidden_dim, hidden_dim)
        self.factor_out = nn.Linear(hidden_dim, hidden_dim)
        self.schema_norm = nn.LayerNorm(hidden_dim)
        self.factor_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        schema_states,
        factor_states,
        schema_edge_index,
        schema_edge_type,
        factor_edge_index,
        factor_edge_type,
        factor_edge_weight,
    ):
        schema_message = torch.zeros_like(schema_states)
        if self.use_schema_graph and schema_edge_index.numel():
            schema_message = edge_attention(
                schema_states,
                schema_states,
                schema_edge_index[0],
                schema_edge_index[1],
                schema_edge_type,
                None,
                self.schema_q,
                self.schema_k,
                self.schema_v,
                self.schema_edge_k,
                self.schema_edge_v,
            )

        factor_to_schema = torch.zeros_like(schema_states)
        if self.use_factor_graph and factor_states.shape[0] and factor_edge_index.numel():
            schema_ids = factor_edge_index[0]
            factor_ids = factor_edge_index[1]
            factor_message = edge_attention(
                factor_states,
                schema_states,
                schema_ids,
                factor_ids,
                factor_edge_type,
                factor_edge_weight,
                self.factor_q,
                self.factor_k,
                self.factor_v,
                self.factor_edge_k,
                self.factor_edge_v,
            )
            factor_gate = torch.sigmoid(
                self.factor_gate(torch.cat([factor_states, factor_message], dim=-1))
            )
            factor_states = self.factor_norm(
                factor_states
                + self.dropout(
                    self.factor_out(F.gelu(factor_message)) * factor_gate
                )
            )
            factor_to_schema = edge_attention(
                schema_states,
                factor_states,
                factor_ids,
                schema_ids,
                factor_edge_type,
                factor_edge_weight,
                self.reverse_q,
                self.reverse_k,
                self.reverse_v,
                self.reverse_edge_k,
                self.reverse_edge_v,
            )

        gates = torch.sigmoid(
            self.schema_gate(
                torch.cat([schema_states, schema_message, factor_to_schema], dim=-1)
            )
        )
        combined = (
            gates[:, :1] * schema_message
            + gates[:, 1:] * factor_to_schema
        )
        schema_states = self.schema_norm(
            schema_states
            + self.dropout(self.schema_out(F.gelu(combined)))
        )
        return schema_states, factor_states


class HeterogeneousFactorGraphReranker(nn.Module):
    """Query-conditioned schema/evidence factor-graph reranker."""

    def __init__(
        self,
        dense_dim,
        numeric_dim,
        factor_numeric_dim,
        relation_count,
        schema_relation_count,
        factor_kind_count=3,
        factor_edge_type_count=3,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
        model_type="factor_rgta",
    ):
        super().__init__()
        self.relation_count = relation_count
        self.model_type = model_type
        self.schema_dense = nn.Linear(dense_dim, hidden_dim)
        self.schema_numeric = nn.Linear(numeric_dim, hidden_dim)
        self.query_input = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.factor_kind = nn.Embedding(factor_kind_count, hidden_dim)
        self.factor_relation = nn.Embedding(relation_count + 1, hidden_dim)
        self.factor_numeric = nn.Linear(factor_numeric_dim, hidden_dim)
        self.schema_input_norm = nn.LayerNorm(hidden_dim)
        self.factor_input_norm = nn.LayerNorm(hidden_dim)
        use_schema_graph = model_type in {"schema_rgta", "factor_rgta"}
        use_factor_graph = model_type == "factor_rgta"
        self.layers = nn.ModuleList(
            [
                FactorGraphLayer(
                    hidden_dim,
                    schema_relation_count,
                    factor_edge_type_count,
                    dropout,
                    use_schema_graph=use_schema_graph,
                    use_factor_graph=use_factor_graph,
                )
                for _ in range(num_layers)
            ]
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4 + numeric_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.role_scorer = nn.Linear(hidden_dim, relation_count)

    def forward(
        self,
        dense_nodes,
        numeric_features,
        query_embedding,
        factor_kind,
        factor_relation,
        factor_numeric,
        schema_edge_index,
        schema_edge_type,
        factor_edge_index,
        factor_edge_type,
        factor_edge_weight,
    ):
        query = self.query_input(query_embedding).squeeze(0)
        schema_states = self.schema_input_norm(
            self.schema_dense(dense_nodes) + self.schema_numeric(numeric_features)
        )
        if factor_kind.numel():
            relation_index = torch.where(
                factor_relation >= 0,
                factor_relation,
                torch.full_like(factor_relation, self.relation_count),
            )
            factor_states = self.factor_input_norm(
                self.factor_kind(factor_kind)
                + self.factor_relation(relation_index)
                + self.factor_numeric(factor_numeric)
                + query.unsqueeze(0)
            )
        else:
            factor_states = schema_states.new_empty((0, schema_states.shape[1]))
        for layer in self.layers:
            schema_states, factor_states = layer(
                schema_states,
                factor_states,
                schema_edge_index,
                schema_edge_type,
                factor_edge_index,
                factor_edge_type,
                factor_edge_weight,
            )
        query_matrix = query.unsqueeze(0).expand_as(schema_states)
        pair = torch.cat(
            [
                query_matrix,
                schema_states,
                query_matrix * schema_states,
                torch.abs(query_matrix - schema_states),
                numeric_features,
            ],
            dim=-1,
        )
        logits = self.scorer(pair).squeeze(-1)
        return {
            "logits": logits,
            "role_logits": self.role_scorer(schema_states),
            "schema_states": schema_states,
            "factor_states": factor_states,
        }
