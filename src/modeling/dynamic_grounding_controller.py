"""Recurrent partial-SQL-conditioned schema grounding controller."""

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


OPERATIONS = ["PROJECT", "JOIN", "FILTER", "GROUP", "ORDER", "COMPUTE", "UNKNOWN"]


def partial_sql_features(sql):
    """Model-independent structural features available during autoregressive decoding."""
    text = str(sql or "")
    lower = text.lower()
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|<=|>=|<>|!=|[(),.+*/=-]", lower)
    length = max(len(tokens), 1)
    keyword = lambda word: float(word in tokens)
    return [
        min(len(tokens) / 128.0, 1.0),
        min(len(text) / 512.0, 1.0),
        keyword("select"), keyword("from"), keyword("join"), keyword("where"),
        float("group by" in lower), float("order by" in lower), keyword("having"),
        float(any(name in tokens for name in ("count", "sum", "avg", "min", "max"))),
        keyword("distinct"), min(tokens.count(",") / 8.0, 1.0),
        min(sum(tokens.count(op) for op in ("=", "!=", "<>", "<", ">", "<=", ">=")) / 8.0, 1.0),
        min(max(text.count("(") - text.count(")"), 0) / 8.0, 1.0),
        min(text.count("select") / 4.0, 1.0),
        float(length > 1 and tokens[-1] in {"select", "from", "join", "where", "by", "on", ","}),
    ]


class StateConditionedRGTALayer(nn.Module):
    def __init__(self, hidden_dim, relation_count, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.state_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.state_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rel_k = nn.Embedding(relation_count, hidden_dim)
        self.rel_v = nn.Embedding(relation_count, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, edge_index, edge_type, state):
        if edge_index.numel() == 0:
            return self.norm(nodes)
        src, dst = edge_index
        # State belongs on the query side. Adding the same state key to every
        # incoming edge would cancel inside each destination's softmax and would
        # not actually change graph attention.
        q = self.q(nodes)[dst] + self.state_q(state).unsqueeze(0)
        k = self.k(nodes)[src] + self.rel_k(edge_type)
        v = self.v(nodes)[src] + self.rel_v(edge_type) + self.state_v(state)
        scores = (q * k).sum(-1) / math.sqrt(self.hidden_dim)
        maximum = torch.full(
            (nodes.shape[0],), -torch.inf, dtype=scores.dtype, device=scores.device
        )
        maximum.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
        weights = torch.exp(scores - maximum[dst])
        denominator = torch.zeros_like(maximum)
        denominator.index_add_(0, dst, weights)
        weights = weights / denominator[dst].clamp_min(1e-8)
        messages = torch.zeros_like(nodes)
        messages.index_add_(0, dst, v * weights.unsqueeze(-1))
        return self.norm(nodes + self.dropout(self.out(F.gelu(messages))))


class DynamicSchemaGroundingController(nn.Module):
    """Updates a latent grounding belief after every partial-SQL event."""

    def __init__(
        self,
        dense_dim=1024,
        numeric_dim=36,
        hidden_dim=256,
        relation_count=5,
        num_layers=2,
        state_feature_dim=16,
        dropout=0.1,
        recurrent=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.recurrent = recurrent
        self.operation_to_id = {name: index for index, name in enumerate(OPERATIONS)}
        self.node_input = nn.Linear(dense_dim, hidden_dim)
        self.numeric_input = nn.Linear(numeric_dim, hidden_dim)
        self.query_input = nn.Linear(dense_dim, hidden_dim)
        self.operation = nn.Embedding(len(OPERATIONS), hidden_dim)
        self.partial_state = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.initial_state = nn.Linear(hidden_dim, hidden_dim)
        self.transition = nn.GRUCell(hidden_dim * 5, hidden_dim)
        self.layers = nn.ModuleList(
            [StateConditionedRGTALayer(hidden_dim, relation_count, dropout) for _ in range(num_layers)]
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        self.bridge_gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def initialize(self, dense_nodes, numeric_features, query_embedding):
        nodes = self.node_input(dense_nodes) + self.numeric_input(numeric_features)
        query = self.query_input(query_embedding).squeeze(0)
        state = torch.tanh(self.initial_state(query))
        belief = torch.full(
            (nodes.shape[0],), 1.0 / max(nodes.shape[0], 1), device=nodes.device
        )
        return nodes, query, state, belief

    def step(
        self, base_nodes, query, previous_state, previous_belief, edges,
        operation_id, sql_features, observed_mask=None,
    ):
        summary = (previous_belief.unsqueeze(-1) * base_nodes).sum(0)
        if observed_mask is None:
            observed_mask = torch.zeros(
                base_nodes.shape[0], dtype=base_nodes.dtype, device=base_nodes.device
            )
        observed_total = observed_mask.sum().clamp_min(1.0)
        observed_summary = (
            observed_mask.unsqueeze(-1) * base_nodes
        ).sum(0) / observed_total
        operation = self.operation(operation_id).squeeze(0)
        partial = self.partial_state(sql_features.unsqueeze(0)).squeeze(0)
        state_input = torch.cat(
            [query, operation, partial, summary, observed_summary], dim=-1
        )
        state = self.transition(state_input.unsqueeze(0), previous_state.unsqueeze(0)).squeeze(0)
        nodes = base_nodes
        for layer in self.layers:
            nodes = layer(nodes, edges[0], edges[1], state)
        state_matrix = state.unsqueeze(0).expand_as(nodes)
        query_matrix = query.unsqueeze(0).expand_as(nodes)
        logits = self.scorer(
            torch.cat([nodes, state_matrix, nodes * state_matrix, query_matrix], dim=-1)
        ).squeeze(-1)
        belief = torch.softmax(logits, dim=0)
        gate = torch.sigmoid(self.bridge_gate(torch.cat([nodes, state_matrix], dim=-1)))
        grounding_tokens = gate * nodes + (1.0 - gate) * state_matrix
        steering_state = (belief.unsqueeze(-1) * grounding_tokens).sum(0)
        return {
            "logits": logits,
            "belief": belief,
            "controller_state": state,
            "schema_states": nodes,
            "grounding_tokens": grounding_tokens,
            "steering_state": steering_state,
        }

    def forward_trajectory(self, dense_nodes, numeric_features, query_embedding, edges, steps):
        base_nodes, query, state, belief = self.initialize(
            dense_nodes, numeric_features, query_embedding
        )
        initial_state, initial_belief = state, belief
        outputs = []
        for step in steps:
            previous_state = state if self.recurrent else initial_state
            previous_belief = belief if self.recurrent else initial_belief
            output = self.step(
                base_nodes, query, previous_state, previous_belief, edges,
                step["operation_id"], step["sql_features"], step.get("observed_mask"),
            )
            outputs.append(output)
            state, belief = output["controller_state"], output["belief"]
        return outputs


class GroundingLLMBridge(nn.Module):
    """Projects controller outputs into an arbitrary decoder hidden space."""

    def __init__(self, grounding_dim, llm_dim):
        super().__init__()
        self.key = nn.Linear(grounding_dim, llm_dim, bias=False)
        self.value = nn.Linear(grounding_dim, llm_dim, bias=False)
        self.steering = nn.Linear(grounding_dim, llm_dim, bias=False)
        self.gate = nn.Linear(llm_dim * 2, llm_dim)

    def forward(self, llm_hidden, grounding_tokens, steering_state):
        keys, values = self.key(grounding_tokens), self.value(grounding_tokens)
        attention = torch.softmax(llm_hidden @ keys.transpose(-1, -2) / math.sqrt(keys.shape[-1]), dim=-1)
        context = attention @ values
        steering = self.steering(steering_state)
        if steering.ndim == llm_hidden.ndim - 1:
            steering = steering.unsqueeze(-2)
        gate = torch.sigmoid(self.gate(torch.cat([llm_hidden, context], dim=-1)))
        steered_hidden = llm_hidden + gate * context + (1.0 - gate) * steering
        return {"hidden_states": steered_hidden, "cross_attention": attention, "gate": gate}
