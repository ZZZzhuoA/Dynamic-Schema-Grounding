"""Semantic-slot-conditioned RGTA schema binder for Stage 14B."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.stage13b_prepare_typed_trajectories import ACTIONS
from src.modeling.typed_ra_decoder import TypedRAPointerDecoder


class SemanticSlotGraphBinder(TypedRAPointerDecoder):
    """Bind one semantic SQL slot to a typed schema graph.

    Unlike the Stage 13B recurrent decoder, every slot starts from the same
    question state.  Its own dense semantic embedding conditions every RGTA
    layer and the pointer query.  This makes repeated FILTER/PROJECT actions
    distinguishable without relying on a potentially erroneous prior action.
    """

    def __init__(self, *args, slot_dim=1024, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hidden_dim
        self.slot_dim = int(slot_dim)
        self.slot_input = nn.Linear(self.slot_dim, hidden_dim, bias=False)
        self.slot_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.slot_norm = nn.LayerNorm(hidden_dim)
        # Start as conservative corrections; training has to earn larger effects.
        self.semantic_scale = nn.Parameter(torch.tensor(-2.252168))
        self.owner_scale = nn.Parameter(torch.tensor(-2.252168))

    def load_stage13b_state(self, state_dict):
        """Warm-start all shape-compatible Stage 13B parameters."""
        current = self.state_dict()
        compatible = {
            name: value for name, value in state_dict.items()
            if name in current and tuple(current[name].shape) == tuple(value.shape)
        }
        result = self.load_state_dict(compatible, strict=False)
        return {
            "loaded_parameter_count": len(compatible),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }

    def condition_state(self, state, query, slot_embedding):
        slot = self.slot_input(slot_embedding.squeeze(0))
        gate = self.slot_gate(torch.cat([state, query, slot], dim=-1))
        conditioned = self.slot_norm(state + gate * slot)
        return conditioned, slot, gate

    @staticmethod
    def _dense_similarity(dense_nodes, slot_embedding):
        slot = slot_embedding.squeeze(0).unsqueeze(0).expand_as(dense_nodes)
        return F.cosine_similarity(dense_nodes.float(), slot.float(), dim=-1).to(dense_nodes.dtype)

    def forward_slot(
        self,
        dense_nodes,
        query_embedding,
        slot_embedding,
        node_type_ids,
        edge_index,
        edge_type,
        join_edge_index,
        action,
        owner_table_index=None,
    ):
        if action not in self.action_to_id:
            raise ValueError(f"Unknown action: {action}")
        base_nodes, query, initial_state = self.initialize(
            dense_nodes, query_embedding, node_type_ids
        )
        state, slot_state, slot_gate = self.condition_state(
            initial_state, query, slot_embedding
        )
        nodes = self._encode_graph(base_nodes, state, edge_index, edge_type)
        action_id = self.action_to_id[action]
        action_tensor = torch.tensor(action_id, dtype=torch.long, device=state.device)
        action_state = self.action_embedding(action_tensor)
        pointer_query = self.pointer_query(torch.cat([state, action_state], dim=-1))

        similarity = self._dense_similarity(dense_nodes, slot_embedding)
        semantic_scale = F.softplus(self.semantic_scale).clamp(max=5.0)
        table_logits = self.table_key(nodes) @ pointer_query + semantic_scale * similarity
        column_logits = self.column_key(nodes) @ pointer_query + semantic_scale * similarity
        table_logits = self._masked(table_logits, node_type_ids == 0)
        column_logits = self._masked(column_logits, node_type_ids == 1)

        if owner_table_index is not None:
            valid_owner = owner_table_index.ge(0) & node_type_ids.eq(1)
            if valid_owner.any():
                owner_prior = torch.zeros_like(column_logits)
                owner_prior[valid_owner] = torch.sigmoid(
                    table_logits[owner_table_index[valid_owner]]
                )
                column_logits = column_logits + F.softplus(self.owner_scale).clamp(max=5.0) * owner_prior

        if join_edge_index.numel():
            left, right = join_edge_index
            edge_features = torch.cat(
                [nodes[left], nodes[right], nodes[left] * nodes[right], torch.abs(nodes[left] - nodes[right])],
                dim=-1,
            )
            join_edge_logits = self.edge_key(edge_features) @ pointer_query
        else:
            join_edge_logits = state.new_empty((0,))

        state_action = torch.cat([state, action_state], dim=-1)
        return {
            "action_logits": self.action_head(torch.cat([state, query], dim=-1)),
            "table_logits": table_logits,
            "column_logits": column_logits,
            "join_edge_logits": join_edge_logits,
            "value_route_logits": self.value_route_head(state_action),
            "operator_logits": self.operator_head(state_action),
            "schema_states": nodes,
            "slot_state": slot_state,
            "slot_gate": slot_gate,
            "semantic_similarity": similarity,
            "action": ACTIONS[action_id],
        }


def owner_table_indices(nodes, device):
    table_by_name = {
        node.get("name"): index
        for index, node in enumerate(nodes)
        if node.get("type") == "table"
    }
    return torch.tensor(
        [
            table_by_name.get(node.get("table"), -1)
            if node.get("type") == "column" else -1
            for node in nodes
        ],
        dtype=torch.long,
        device=device,
    )
