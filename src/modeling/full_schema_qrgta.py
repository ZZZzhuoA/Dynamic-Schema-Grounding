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
MODEL_TYPES = {
    "qrgta",
    "path_qrgta",
    "persistent_path_qrgta",
    "table_competitive_path_qrgta",
    "enhanced_table_competitive_path_qrgta",
    "pk_residual_table_competitive_path_qrgta",
    "mlp",
    "mlp_residual",
}


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


class PathAwareSparseQRGTAEncoderLayer(nn.Module):
    """Sparse QRGTA layer with query-gated path and distance features."""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        relation_count,
        distance_bucket_count,
        path_signature_count,
        dropout,
    ):
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
        self.distance_key = nn.Embedding(distance_bucket_count, hidden_dim)
        self.distance_value = nn.Embedding(distance_bucket_count, hidden_dim)
        self.distance_bias = nn.Embedding(distance_bucket_count, num_heads)
        self.path_key = nn.Embedding(path_signature_count, hidden_dim)
        self.path_value = nn.Embedding(path_signature_count, hidden_dim)
        self.path_bias = nn.Embedding(path_signature_count, num_heads)
        self.path_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
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
        schema_distance_bucket,
        schema_path_signature,
        query_edge_destination,
        query_edge_type,
        query_edge_similarity,
        query_distance_bucket,
        query_path_signature,
        schema_primary_key_direction=None,
        zero_pk_modifier=False,
    ):
        node_count = schema_states.shape[0]
        device = schema_states.device

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
        distance_bucket = torch.cat(
            [schema_distance_bucket, query_distance_bucket], dim=0
        )
        path_signature = torch.cat(
            [schema_path_signature, query_path_signature], dim=0
        )
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
            modifier = self._primary_key_modifier(
                source_states,
                query_state,
                source,
                destination,
                schema_primary_key_direction,
                query_edge_destination,
                zero_pk_modifier,
            )
            if modifier is not None:
                key_delta, value_delta, bias_delta = modifier
                relation_key = relation_key + key_delta
                relation_value = relation_value + value_delta
            distance_key = self.distance_key(distance_bucket).view(
                -1, self.num_heads, self.head_dim
            )
            distance_value = self.distance_value(distance_bucket).view(
                -1, self.num_heads, self.head_dim
            )
            path_key = self.path_key(path_signature).view(
                -1, self.num_heads, self.head_dim
            )
            path_value = self.path_value(path_signature).view(
                -1, self.num_heads, self.head_dim
            )
            path_embedding = (
                self.distance_key(distance_bucket) + self.path_key(path_signature)
            )
            gate_input = torch.cat(
                [query_state.unsqueeze(0).expand_as(path_embedding), path_embedding],
                dim=-1,
            )
            path_gate = torch.sigmoid(self.path_gate(gate_input))
            gate_heads = path_gate.unsqueeze(-1)
            gated_key = relation_key + gate_heads * (distance_key + path_key)
            gated_value = relation_value + gate_heads * (distance_value + path_value)

            scores = ((query * (key + gated_key)).sum(dim=-1) / math.sqrt(self.head_dim))
            scores = scores + self.relation_bias(relation)
            if modifier is not None:
                scores = scores + bias_delta
            scores = scores + path_gate * (
                self.distance_bias(distance_bucket) + self.path_bias(path_signature)
            )
            scores = scores + (
                scalar.unsqueeze(-1)
                * is_query_edge.to(scores.dtype).unsqueeze(-1)
                * self.query_similarity_scale.unsqueeze(0)
            )
            attention = _segment_softmax(scores, destination, node_count)
            weighted = (value + gated_value) * attention.unsqueeze(-1)
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

    def _primary_key_modifier(
        self,
        source_states,
        query_state,
        source,
        destination,
        schema_primary_key_direction,
        query_edge_destination,
        zero_pk_modifier,
    ):
        return None


class PersistentGatedPathAwareSparseQRGTAEncoderLayer(PathAwareSparseQRGTAEncoderLayer):
    """Path-aware sparse layer with identity-preserving gated delta updates."""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        relation_count,
        distance_bucket_count,
        path_signature_count,
        dropout,
    ):
        super().__init__(
            hidden_dim,
            num_heads,
            relation_count,
            distance_bucket_count,
            path_signature_count,
            dropout,
        )
        gate_input_dim = hidden_dim * 5
        self.message_update_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ffn_update_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _gate_input(self, states, delta, query_state):
        query_matrix = query_state.unsqueeze(0).expand_as(states)
        return torch.cat(
            [states, delta, query_matrix, states * query_matrix, torch.abs(states - query_matrix)],
            dim=-1,
        )

    def forward(
        self,
        schema_states,
        query_state,
        schema_edge_index,
        schema_edge_type,
        schema_edge_scalar,
        schema_distance_bucket,
        schema_path_signature,
        query_edge_destination,
        query_edge_type,
        query_edge_similarity,
        query_distance_bucket,
        query_path_signature,
        zero_update_gates=False,
    ):
        node_count = schema_states.shape[0]
        device = schema_states.device

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
        distance_bucket = torch.cat(
            [schema_distance_bucket, query_distance_bucket], dim=0
        )
        path_signature = torch.cat(
            [schema_path_signature, query_path_signature], dim=0
        )
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
            distance_key = self.distance_key(distance_bucket).view(
                -1, self.num_heads, self.head_dim
            )
            distance_value = self.distance_value(distance_bucket).view(
                -1, self.num_heads, self.head_dim
            )
            path_key = self.path_key(path_signature).view(
                -1, self.num_heads, self.head_dim
            )
            path_value = self.path_value(path_signature).view(
                -1, self.num_heads, self.head_dim
            )
            path_embedding = (
                self.distance_key(distance_bucket) + self.path_key(path_signature)
            )
            gate_input = torch.cat(
                [query_state.unsqueeze(0).expand_as(path_embedding), path_embedding],
                dim=-1,
            )
            path_gate = torch.sigmoid(self.path_gate(gate_input))
            gate_heads = path_gate.unsqueeze(-1)
            gated_key = relation_key + gate_heads * (distance_key + path_key)
            gated_value = relation_value + gate_heads * (distance_value + path_value)

            scores = ((query * (key + gated_key)).sum(dim=-1) / math.sqrt(self.head_dim))
            scores = scores + self.relation_bias(relation)
            scores = scores + path_gate * (
                self.distance_bias(distance_bucket) + self.path_bias(path_signature)
            )
            scores = scores + (
                scalar.unsqueeze(-1)
                * is_query_edge.to(scores.dtype).unsqueeze(-1)
                * self.query_similarity_scale.unsqueeze(0)
            )
            attention = _segment_softmax(scores, destination, node_count)
            weighted = (value + gated_value) * attention.unsqueeze(-1)
            message_heads = torch.zeros(
                (node_count, self.num_heads, self.head_dim),
                dtype=weighted.dtype,
                device=device,
            )
            message_heads.index_add_(0, destination, weighted)
            message = message_heads.reshape(node_count, self.hidden_dim)

        message_delta = self.output(message)
        if zero_update_gates:
            message_gate = torch.zeros_like(schema_states)
            ffn_gate = torch.zeros_like(schema_states)
            next_states = schema_states
            ffn_delta = torch.zeros_like(schema_states)
        else:
            message_gate = torch.sigmoid(
                self.message_update_gate(
                    self._gate_input(schema_states, message_delta, query_state)
                )
            )
            h_msg = self.attention_norm(
                schema_states + message_gate * self.dropout(message_delta)
            )
            ffn_delta = self.ffn(h_msg)
            ffn_gate = torch.sigmoid(
                self.ffn_update_gate(self._gate_input(h_msg, ffn_delta, query_state))
            )
            next_states = self.ffn_norm(h_msg + ffn_gate * self.dropout(ffn_delta))

        diagnostics = {
            "message_gate": message_gate,
            "ffn_gate": ffn_gate,
            "layer_delta_norm": torch.linalg.vector_norm(next_states - schema_states, dim=-1),
            "message_delta_norm": torch.linalg.vector_norm(message_delta, dim=-1),
            "ffn_delta_norm": torch.linalg.vector_norm(ffn_delta, dim=-1),
        }
        return next_states, diagnostics


class TableScopedCompetitivePathQRGTAEncoderLayer(PathAwareSparseQRGTAEncoderLayer):
    """Path-aware sparse layer with encoder-internal table-scoped column competition."""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        relation_count,
        distance_bucket_count,
        path_signature_count,
        dropout,
        competition_hidden_dim=128,
        competition_dropout=0.1,
    ):
        super().__init__(
            hidden_dim,
            num_heads,
            relation_count,
            distance_bucket_count,
            path_signature_count,
            dropout,
        )
        self.competition_logit = nn.Sequential(
            nn.Linear(hidden_dim * 7, competition_hidden_dim),
            nn.GELU(),
            nn.Dropout(competition_dropout),
            nn.Linear(competition_hidden_dim, 1),
        )
        self.competition_delta = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 1, competition_hidden_dim),
            nn.GELU(),
            nn.Dropout(competition_dropout),
            nn.Linear(competition_hidden_dim, hidden_dim),
        )
        self.competition_gate = nn.Sequential(
            nn.Linear(hidden_dim * 5, competition_hidden_dim),
            nn.GELU(),
            nn.Dropout(competition_dropout),
            nn.Linear(competition_hidden_dim, hidden_dim),
        )
        self.competition_norm = nn.LayerNorm(hidden_dim)
        self.competition_dropout = nn.Dropout(competition_dropout)

    def _competition_gate_input(self, states, delta, query_state):
        query_matrix = query_state.unsqueeze(0).expand_as(states)
        return torch.cat(
            [states, delta, query_matrix, states * query_matrix, torch.abs(states - query_matrix)],
            dim=-1,
        )

    def _table_scoped_competition(
        self,
        schema_states,
        query_state,
        column_parent_table,
        is_column_node,
        zero_table_competition=False,
        zero_competition_gates=False,
    ):
        if (
            zero_table_competition
            or column_parent_table is None
            or is_column_node is None
            or not bool(is_column_node.any())
        ):
            return schema_states
        column_indices = torch.nonzero(is_column_node, as_tuple=False).reshape(-1)
        parent_indices = column_parent_table[column_indices]
        valid = parent_indices >= 0
        if not bool(valid.any()):
            return schema_states
        column_indices = column_indices[valid]
        parent_indices = parent_indices[valid]
        column_states = schema_states[column_indices]
        table_states = schema_states[parent_indices]
        query_matrix = query_state.unsqueeze(0).expand_as(column_states)
        competition_pair = torch.cat(
            [
                column_states,
                table_states,
                query_matrix,
                column_states * query_matrix,
                torch.abs(column_states - query_matrix),
                column_states * table_states,
                torch.abs(column_states - table_states),
            ],
            dim=-1,
        )
        logits = self.competition_logit(competition_pair)
        sibling_weight = _segment_softmax(logits, parent_indices, schema_states.shape[0])
        delta_input = torch.cat(
            [
                column_states,
                table_states,
                query_matrix,
                sibling_weight,
                column_states * query_matrix,
                torch.abs(column_states - query_matrix),
            ],
            dim=-1,
        )
        competition_delta = self.competition_delta(delta_input)
        if zero_competition_gates:
            gate = torch.zeros_like(competition_delta)
            refined_columns = column_states
        else:
            gate = torch.sigmoid(
                self.competition_gate(
                    self._competition_gate_input(column_states, competition_delta, query_state)
                )
            )
            refined_columns = self.competition_norm(
                column_states + gate * self.competition_dropout(competition_delta)
            )
        next_states = schema_states.clone()
        next_states[column_indices] = refined_columns
        return next_states

    def forward(
        self,
        schema_states,
        query_state,
        schema_edge_index,
        schema_edge_type,
        schema_edge_scalar,
        schema_distance_bucket,
        schema_path_signature,
        query_edge_destination,
        query_edge_type,
        query_edge_similarity,
        query_distance_bucket,
        query_path_signature,
        column_parent_table=None,
        is_column_node=None,
        zero_table_competition=False,
        zero_competition_gates=False,
        schema_primary_key_direction=None,
        zero_pk_modifier=False,
    ):
        schema_states = super().forward(
            schema_states,
            query_state,
            schema_edge_index,
            schema_edge_type,
            schema_edge_scalar,
            schema_distance_bucket,
            schema_path_signature,
            query_edge_destination,
            query_edge_type,
            query_edge_similarity,
            query_distance_bucket,
            query_path_signature,
            schema_primary_key_direction=schema_primary_key_direction,
            zero_pk_modifier=zero_pk_modifier,
        )
        return self._table_scoped_competition(
            schema_states,
            query_state,
            column_parent_table,
            is_column_node,
            zero_table_competition=zero_table_competition,
            zero_competition_gates=zero_competition_gates,
        )


class PrimaryKeyResidualTableScopedCompetitivePathQRGTAEncoderLayer(
    TableScopedCompetitivePathQRGTAEncoderLayer
):
    """Table-competitive layer with query-gated residuals on direct PK ownership."""

    PRIMARY_KEY_DIRECTION_COUNT = 3

    def __init__(
        self,
        hidden_dim,
        num_heads,
        relation_count,
        distance_bucket_count,
        path_signature_count,
        dropout,
        competition_hidden_dim=128,
        competition_dropout=0.1,
    ):
        super().__init__(
            hidden_dim,
            num_heads,
            relation_count,
            distance_bucket_count,
            path_signature_count,
            dropout,
            competition_hidden_dim=competition_hidden_dim,
            competition_dropout=competition_dropout,
        )
        # Keep the common Stage 17-E parameters identical under the same seed.
        with torch.random.fork_rng(devices=[]):
            self.primary_key_delta_key = nn.Embedding(
                self.PRIMARY_KEY_DIRECTION_COUNT, hidden_dim
            )
            self.primary_key_delta_value = nn.Embedding(
                self.PRIMARY_KEY_DIRECTION_COUNT, hidden_dim
            )
            self.primary_key_delta_bias = nn.Embedding(
                self.PRIMARY_KEY_DIRECTION_COUNT, num_heads
            )
            self.primary_key_gate = nn.Sequential(
                nn.Linear(hidden_dim * 5, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        nn.init.zeros_(self.primary_key_delta_key.weight)
        nn.init.zeros_(self.primary_key_delta_value.weight)
        nn.init.zeros_(self.primary_key_delta_bias.weight)

    def _primary_key_modifier(
        self,
        source_states,
        query_state,
        source,
        destination,
        schema_primary_key_direction,
        query_edge_destination,
        zero_pk_modifier,
    ):
        if schema_primary_key_direction is None:
            raise ValueError(
                "pk_residual_table_competitive_path_qrgta requires PK edge tensors"
            )
        query_directions = torch.zeros_like(query_edge_destination)
        direction = torch.cat(
            [schema_primary_key_direction, query_directions], dim=0
        )
        if direction.shape[0] != source.shape[0]:
            raise ValueError("PK edge tensor must align with schema attention edges")
        is_primary_key = direction.ne(0)
        primary_key_indices = torch.nonzero(
            is_primary_key, as_tuple=False
        ).reshape(-1)
        if primary_key_indices.numel() == 0:
            return None
        primary_key_source = source[primary_key_indices]
        primary_key_destination = destination[primary_key_indices]
        primary_key_direction = direction[primary_key_indices]
        source_edge_states = source_states[primary_key_source]
        destination_states = source_states[1:][primary_key_destination]
        query_matrix = query_state.unsqueeze(0).expand_as(source_edge_states)
        gate_input = torch.cat(
            [
                query_matrix,
                source_edge_states,
                destination_states,
                source_edge_states * query_matrix,
                destination_states * query_matrix,
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.primary_key_gate(gate_input))
        if zero_pk_modifier:
            gate = torch.zeros_like(gate)
        gate_heads = gate.unsqueeze(-1)
        key_delta = self.primary_key_delta_key(primary_key_direction).view(
            -1, self.num_heads, self.head_dim
        )
        value_delta = self.primary_key_delta_value(primary_key_direction).view(
            -1, self.num_heads, self.head_dim
        )
        bias_delta = self.primary_key_delta_bias(primary_key_direction)
        full_key_delta = key_delta.new_zeros(
            source.shape[0], self.num_heads, self.head_dim
        ).index_copy_(0, primary_key_indices, gate_heads * key_delta)
        full_value_delta = value_delta.new_zeros(
            source.shape[0], self.num_heads, self.head_dim
        ).index_copy_(0, primary_key_indices, gate_heads * value_delta)
        full_bias_delta = bias_delta.new_zeros(
            source.shape[0], self.num_heads
        ).index_copy_(0, primary_key_indices, gate * bias_delta)
        return (
            full_key_delta,
            full_value_delta,
            full_bias_delta,
        )


class EnhancedTableScopedCompetitivePathQRGTAEncoderLayer(
    TableScopedCompetitivePathQRGTAEncoderLayer
):
    """Table-scoped competition with tempered sibling weights and multi-winner gates."""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        relation_count,
        distance_bucket_count,
        path_signature_count,
        dropout,
        competition_hidden_dim=128,
        competition_dropout=0.1,
        competition_temperature=1.5,
        competition_residual_scale=0.5,
    ):
        if competition_temperature <= 0.0:
            raise ValueError("competition_temperature must be > 0")
        super().__init__(
            hidden_dim,
            num_heads,
            relation_count,
            distance_bucket_count,
            path_signature_count,
            dropout,
            competition_hidden_dim=competition_hidden_dim,
            competition_dropout=competition_dropout,
        )
        self.competition_temperature = float(competition_temperature)
        self.competition_residual_scale = float(competition_residual_scale)
        self.multi_winner_gate = nn.Sequential(
            nn.Linear(hidden_dim * 5, competition_hidden_dim),
            nn.GELU(),
            nn.Dropout(competition_dropout),
            nn.Linear(competition_hidden_dim, 1),
        )
        self.competition_delta = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 2, competition_hidden_dim),
            nn.GELU(),
            nn.Dropout(competition_dropout),
            nn.Linear(competition_hidden_dim, hidden_dim),
        )

    def _multi_winner_input(self, column_states, table_states, query_state):
        query_matrix = query_state.unsqueeze(0).expand_as(column_states)
        return torch.cat(
            [
                column_states,
                table_states,
                query_matrix,
                column_states * query_matrix,
                torch.abs(column_states - query_matrix),
            ],
            dim=-1,
        )

    def _competition_features(
        self,
        schema_states,
        query_state,
        column_parent_table,
        is_column_node,
    ):
        column_indices = torch.nonzero(is_column_node, as_tuple=False).reshape(-1)
        parent_indices = column_parent_table[column_indices]
        valid = parent_indices >= 0
        if not bool(valid.any()):
            return None
        column_indices = column_indices[valid]
        parent_indices = parent_indices[valid]
        column_states = schema_states[column_indices]
        table_states = schema_states[parent_indices]
        query_matrix = query_state.unsqueeze(0).expand_as(column_states)
        competition_pair = torch.cat(
            [
                column_states,
                table_states,
                query_matrix,
                column_states * query_matrix,
                torch.abs(column_states - query_matrix),
                column_states * table_states,
                torch.abs(column_states - table_states),
            ],
            dim=-1,
        )
        logits = self.competition_logit(competition_pair)
        sibling_weight = _segment_softmax(
            logits / self.competition_temperature,
            parent_indices,
            schema_states.shape[0],
        )
        multi_winner_gate = torch.sigmoid(
            self.multi_winner_gate(
                self._multi_winner_input(column_states, table_states, query_state)
            )
        )
        delta_input = torch.cat(
            [
                column_states,
                table_states,
                query_matrix,
                sibling_weight,
                multi_winner_gate,
                column_states * query_matrix,
                torch.abs(column_states - query_matrix),
            ],
            dim=-1,
        )
        competition_delta = self.competition_delta(delta_input)
        gate = torch.sigmoid(
            self.competition_gate(
                self._competition_gate_input(column_states, competition_delta, query_state)
            )
        )
        return {
            "column_indices": column_indices,
            "logits": logits,
            "sibling_weight": sibling_weight,
            "multi_winner_gate": multi_winner_gate,
            "competition_delta": competition_delta,
            "competition_gate": gate,
        }

    def _table_scoped_competition(
        self,
        schema_states,
        query_state,
        column_parent_table,
        is_column_node,
        zero_table_competition=False,
        zero_competition_gates=False,
    ):
        if (
            zero_table_competition
            or column_parent_table is None
            or is_column_node is None
            or not bool(is_column_node.any())
        ):
            return schema_states
        features = self._competition_features(
            schema_states, query_state, column_parent_table, is_column_node
        )
        if features is None:
            return schema_states
        column_indices = features["column_indices"]
        column_states = schema_states[column_indices]
        if zero_competition_gates or self.competition_residual_scale == 0.0:
            refined_columns = column_states
        else:
            refined_columns = self.competition_norm(
                column_states
                + self.competition_residual_scale
                * features["competition_gate"]
                * self.competition_dropout(features["competition_delta"])
            )
        next_states = schema_states.clone()
        next_states[column_indices] = refined_columns
        return next_states


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
        distance_bucket_count=1,
        path_signature_count=1,
        role_count=0,
        competition_hidden_dim=128,
        competition_dropout=0.1,
        competition_temperature=1.5,
        competition_residual_scale=0.5,
    ):
        super().__init__()
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.model_type = model_type
        self.role_count = int(role_count)
        self.node_input = nn.Linear(dense_dim, hidden_dim)
        self.query_input = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_type = nn.Embedding(2, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        if model_type == "qrgta":
            layer_cls = lambda: SparseQRGTAEncoderLayer(
                hidden_dim, num_heads, relation_count, dropout
            )
        elif model_type == "path_qrgta":
            layer_cls = lambda: PathAwareSparseQRGTAEncoderLayer(
                hidden_dim,
                num_heads,
                relation_count,
                distance_bucket_count,
                path_signature_count,
                dropout,
            )
        elif model_type == "persistent_path_qrgta":
            layer_cls = lambda: PersistentGatedPathAwareSparseQRGTAEncoderLayer(
                hidden_dim,
                num_heads,
                relation_count,
                distance_bucket_count,
                path_signature_count,
                dropout,
            )
        elif model_type == "table_competitive_path_qrgta":
            layer_cls = lambda: TableScopedCompetitivePathQRGTAEncoderLayer(
                hidden_dim,
                num_heads,
                relation_count,
                distance_bucket_count,
                path_signature_count,
                dropout,
                competition_hidden_dim=competition_hidden_dim,
                competition_dropout=competition_dropout,
            )
        elif model_type == "enhanced_table_competitive_path_qrgta":
            layer_cls = lambda: EnhancedTableScopedCompetitivePathQRGTAEncoderLayer(
                hidden_dim,
                num_heads,
                relation_count,
                distance_bucket_count,
                path_signature_count,
                dropout,
                competition_hidden_dim=competition_hidden_dim,
                competition_dropout=competition_dropout,
                competition_temperature=competition_temperature,
                competition_residual_scale=competition_residual_scale,
            )
        elif model_type == "pk_residual_table_competitive_path_qrgta":
            layer_cls = lambda: PrimaryKeyResidualTableScopedCompetitivePathQRGTAEncoderLayer(
                hidden_dim,
                num_heads,
                relation_count,
                distance_bucket_count,
                path_signature_count,
                dropout,
                competition_hidden_dim=competition_hidden_dim,
                competition_dropout=competition_dropout,
            )
        else:
            layer_cls = None
        self.layers = nn.ModuleList(
            [layer_cls() for _ in range(num_layers)] if layer_cls else []
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
        self.role_scorer = None
        if self.role_count > 0:
            self.role_scorer = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.role_count),
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
        schema_distance_bucket=None,
        schema_path_signature=None,
        query_distance_bucket=None,
        query_path_signature=None,
        schema_primary_key_direction=None,
        zero_update_gates=False,
        zero_table_competition=False,
        zero_competition_gates=False,
        zero_pk_modifier=False,
        column_parent_table=None,
        is_column_node=None,
        record_persistent_diagnostics=False,
    ):
        schema_states = self.node_norm(
            self.node_input(dense_nodes) + self.node_type(node_types)
        )
        initial_schema_states = schema_states
        query_state = self.query_norm(
            self.query_input(query_embedding).reshape(-1, schema_states.shape[-1])[0]
        )
        schema_edge_scalar = dense_nodes.new_zeros(schema_edge_type.shape[0])
        persistent_diagnostics = []
        for layer in self.layers:
            if self.model_type in {
                "path_qrgta",
                "persistent_path_qrgta",
                "table_competitive_path_qrgta",
                "enhanced_table_competitive_path_qrgta",
                "pk_residual_table_competitive_path_qrgta",
            }:
                if (
                    schema_distance_bucket is None
                    or schema_path_signature is None
                    or query_distance_bucket is None
                    or query_path_signature is None
                ):
                    raise ValueError(f"{self.model_type} requires path and distance tensors")
                if (
                    self.model_type == "pk_residual_table_competitive_path_qrgta"
                    and schema_primary_key_direction is None
                ):
                    raise ValueError(f"{self.model_type} requires PK edge tensors")
                if self.model_type == "persistent_path_qrgta":
                    schema_states, layer_diagnostics = layer(
                        schema_states,
                        query_state,
                        schema_edge_index,
                        schema_edge_type,
                        schema_edge_scalar,
                        schema_distance_bucket,
                        schema_path_signature,
                        query_edge_destination,
                        query_edge_type,
                        query_edge_similarity,
                        query_distance_bucket,
                        query_path_signature,
                        zero_update_gates=zero_update_gates,
                    )
                    persistent_diagnostics.append(layer_diagnostics)
                elif self.model_type in {
                    "table_competitive_path_qrgta",
                    "enhanced_table_competitive_path_qrgta",
                    "pk_residual_table_competitive_path_qrgta",
                }:
                    if column_parent_table is None or is_column_node is None:
                        raise ValueError(
                            f"{self.model_type} requires parent-table tensors"
                        )
                    schema_states = layer(
                        schema_states,
                        query_state,
                        schema_edge_index,
                        schema_edge_type,
                        schema_edge_scalar,
                        schema_distance_bucket,
                        schema_path_signature,
                        query_edge_destination,
                        query_edge_type,
                        query_edge_similarity,
                        query_distance_bucket,
                        query_path_signature,
                        schema_primary_key_direction=schema_primary_key_direction,
                        zero_pk_modifier=zero_pk_modifier,
                        column_parent_table=column_parent_table,
                        is_column_node=is_column_node,
                        zero_table_competition=zero_table_competition,
                        zero_competition_gates=zero_competition_gates,
                    )
                else:
                    schema_states = layer(
                        schema_states,
                        query_state,
                        schema_edge_index,
                        schema_edge_type,
                        schema_edge_scalar,
                        schema_distance_bucket,
                        schema_path_signature,
                        query_edge_destination,
                        query_edge_type,
                        query_edge_similarity,
                        query_distance_bucket,
                        query_path_signature,
                    )
            else:
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
        output = {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "schema_states": schema_states,
        }
        if self.role_scorer is not None:
            role_logits = self.role_scorer(pair)
            output["role_logits"] = role_logits
            output["role_probabilities"] = torch.sigmoid(role_logits)
        if self.model_type == "persistent_path_qrgta" and record_persistent_diagnostics:
            if persistent_diagnostics:
                message_gates = torch.cat(
                    [item["message_gate"].reshape(-1) for item in persistent_diagnostics]
                )
                ffn_gates = torch.cat(
                    [item["ffn_gate"].reshape(-1) for item in persistent_diagnostics]
                )
                layer_delta_norms = torch.cat(
                    [item["layer_delta_norm"].reshape(-1) for item in persistent_diagnostics]
                )
            else:
                message_gates = schema_states.new_zeros(1)
                ffn_gates = schema_states.new_zeros(1)
                layer_delta_norms = schema_states.new_zeros(1)
            output["persistent_diagnostics"] = {
                "avg_message_gate": message_gates.mean(),
                "avg_ffn_gate": ffn_gates.mean(),
                "min_message_gate": message_gates.min(),
                "max_message_gate": message_gates.max(),
                "avg_layer_delta_norm": layer_delta_norms.mean(),
                "avg_final_delta_norm": torch.linalg.vector_norm(
                    schema_states - initial_schema_states, dim=-1
                ).mean(),
                "avg_identity_cosine_input_to_final": F.cosine_similarity(
                    initial_schema_states, schema_states, dim=-1
                ).mean(),
            }
        return output


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
