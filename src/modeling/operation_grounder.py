"""Operation-conditioned graph grounder for Stage 8-A.

This model upgrades static schema graph encoding to operation-conditioned
relational graph attention.  The operation embedding (e.g. FILTER, JOIN,
COMPUTE, ORDER) is injected into graph attention, so message passing itself
changes for different SQL relational operations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling.dsg_grounder import DEFAULT_RELATIONS


DEFAULT_OPERATIONS = [
    "OUTPUT_TARGET",
    "ENTITY_NAME",
    "METRIC_TARGET",
    "PREDICATE_COLUMN",
    "VALUE_ANCHOR",
    "TEMPORAL_FILTER",
    "ORDER_KEY",
    "GROUP_KEY",
    "JOIN_BRIDGE",
    "FORMULA_COMPONENT",
]


class OperationConditionedRGTALayer(nn.Module):
    """Relational graph attention conditioned on the target SQL operation."""

    def __init__(self, hidden_dim, relations, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.relations = relations
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.op_k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.op_v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rel_k = nn.ParameterDict(
            {relation: nn.Parameter(torch.empty(hidden_dim)) for relation in relations}
        )
        self.rel_v = nn.ParameterDict(
            {relation: nn.Parameter(torch.empty(hidden_dim)) for relation in relations}
        )
        self.rel_op_gate = nn.ParameterDict(
            {relation: nn.Parameter(torch.tensor(0.0)) for relation in relations}
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in list(self.rel_k.values()) + list(self.rel_v.values()):
            nn.init.normal_(parameter, mean=0.0, std=1.0 / math.sqrt(self.hidden_dim))

    def forward(self, h, edges, operation_state):
        node_count = h.shape[0]
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        op_k = self.op_k_proj(operation_state).unsqueeze(0)
        op_v = self.op_v_proj(operation_state).unsqueeze(0)
        messages = []
        destinations = []
        scores = []

        for relation in self.relations:
            if relation not in edges or edges[relation].numel() == 0:
                continue
            src, dst = edges[relation][0], edges[relation][1]
            gate = torch.sigmoid(self.rel_op_gate[relation])
            edge_k = k[src] + self.rel_k[relation].unsqueeze(0) + gate * op_k
            edge_v = v[src] + self.rel_v[relation].unsqueeze(0) + gate * op_v
            edge_scores = (q[dst] * edge_k).sum(dim=-1) / math.sqrt(self.hidden_dim)
            messages.append(edge_v)
            destinations.append(dst)
            scores.append(edge_scores)

        if not messages:
            return self.norm(h)

        all_messages = torch.cat(messages, dim=0)
        all_dst = torch.cat(destinations, dim=0)
        all_scores = torch.cat(scores, dim=0)
        max_per_dst = torch.full(
            (node_count,), -torch.inf, dtype=all_scores.dtype, device=all_scores.device
        )
        max_per_dst.scatter_reduce_(0, all_dst, all_scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(all_scores - max_per_dst[all_dst])
        denom = torch.zeros((node_count,), dtype=all_scores.dtype, device=all_scores.device)
        denom.index_add_(0, all_dst, exp_scores)
        attn = exp_scores / denom[all_dst].clamp_min(1e-8)

        weighted = all_messages * attn.unsqueeze(-1)
        out = torch.zeros((node_count, self.hidden_dim), dtype=h.dtype, device=h.device)
        out.index_add_(0, all_dst, weighted)
        out = self.out_proj(out)
        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(out + h)


class OperationConditionedGrounder(nn.Module):
    """Question-schema grounder with operation-conditioned graph propagation."""

    def __init__(
        self,
        hash_dim=256,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        relations=None,
        operations=None,
        lexical_dim=6,
    ):
        super().__init__()
        self.hash_dim = hash_dim
        self.hidden_dim = hidden_dim
        self.relations = relations or list(DEFAULT_RELATIONS)
        self.operations = operations or list(DEFAULT_OPERATIONS)
        self.operation_to_id = {name: idx for idx, name in enumerate(self.operations)}
        self.lexical_dim = lexical_dim
        self.schema_input = nn.Linear(hash_dim, hidden_dim)
        self.query_input = nn.Sequential(
            nn.Linear(hash_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.operation_embedding = nn.Embedding(len(self.operations), hidden_dim)
        self.operation_query_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [OperationConditionedRGTALayer(hidden_dim, self.relations, dropout) for _ in range(num_layers)]
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 5 + lexical_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_schema(self, node_features, edges, operation_id):
        h = self.schema_input(node_features)
        operation_state = self.operation_embedding(operation_id).squeeze(0)
        for layer in self.layers:
            h = layer(h, edges, operation_state)
        return h, operation_state

    def forward(self, query_features, node_features, edges, operation_id, lex_features=None):
        query = self.query_input(query_features).squeeze(0)
        node_states, operation_state = self.encode_schema(node_features, edges, operation_id)
        query = query + self.operation_query_gate(torch.cat([query, operation_state], dim=-1))
        query_matrix = query.unsqueeze(0).expand_as(node_states)
        operation_matrix = operation_state.unsqueeze(0).expand_as(node_states)
        pair = torch.cat(
            [
                query_matrix,
                node_states,
                operation_matrix,
                query_matrix * node_states,
                torch.abs(query_matrix - node_states),
            ],
            dim=-1,
        )
        if self.lexical_dim:
            if lex_features is None:
                lex_features = torch.zeros(
                    (node_states.shape[0], self.lexical_dim),
                    dtype=node_states.dtype,
                    device=node_states.device,
                )
            pair = torch.cat([pair, lex_features], dim=-1)
        logits = self.scorer(pair).squeeze(-1)
        return {
            "logits": logits,
            "schema_node_embeddings": node_states,
            "operation_embedding": operation_state,
        }
