"""Action-synchronous RGTA decoder for typed relational-algebra plans."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.stage13b_prepare_typed_trajectories import ACTIONS, OPERATORS, VALUE_ROUTES
from src.modeling.dynamic_grounding_controller import StateConditionedRGTALayer


class TypedRAPointerDecoder(nn.Module):
    def __init__(
        self,
        dense_dim=1024,
        hidden_dim=256,
        relation_count=5,
        num_layers=2,
        dropout=0.1,
        plan_hidden_dim=0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_to_id = {name: index for index, name in enumerate(ACTIONS)}
        self.value_route_to_id = {name: index for index, name in enumerate(VALUE_ROUTES)}
        self.operator_to_id = {name: index for index, name in enumerate(OPERATORS)}
        self.node_input = nn.Linear(dense_dim, hidden_dim)
        self.query_input = nn.Linear(dense_dim, hidden_dim)
        self.node_type = nn.Embedding(2, hidden_dim)
        self.action_embedding = nn.Embedding(len(ACTIONS), hidden_dim)
        self.value_route_embedding = nn.Embedding(len(VALUE_ROUTES), hidden_dim)
        self.operator_embedding = nn.Embedding(len(OPERATORS), hidden_dim)
        self.plan_hidden = (
            nn.Linear(plan_hidden_dim, hidden_dim, bias=False) if plan_hidden_dim else None
        )
        self.initial_state = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.graph_layers = nn.ModuleList(
            [StateConditionedRGTALayer(hidden_dim, relation_count, dropout) for _ in range(num_layers)]
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(ACTIONS)),
        )
        self.table_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.column_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pointer_query = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.edge_key = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.value_route_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(VALUE_ROUTES)),
        )
        self.operator_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(OPERATORS)),
        )
        self.transition = nn.GRUCell(hidden_dim * 5, hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)

    def initialize(self, dense_nodes, query_embedding, node_type_ids):
        base_nodes = self.node_input(dense_nodes) + self.node_type(node_type_ids)
        query = self.query_input(query_embedding).squeeze(0)
        state = self.initial_state(query)
        return base_nodes, query, state

    def _encode_graph(self, base_nodes, state, edge_index, edge_type):
        nodes = base_nodes
        for layer in self.graph_layers:
            nodes = layer(nodes, edge_index, edge_type, state)
        return nodes

    def encode_static_memory(
        self, dense_nodes, query_embedding, node_type_ids, edge_index, edge_type
    ):
        """Expose the pretrained query-conditioned graph memory without decoding actions."""
        base_nodes, query, state = self.initialize(
            dense_nodes, query_embedding, node_type_ids
        )
        return self._encode_graph(base_nodes, state, edge_index, edge_type), query

    @staticmethod
    def _masked(logits, mask):
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def step(
        self,
        base_nodes,
        query,
        state,
        node_type_ids,
        edge_index,
        edge_type,
        join_edge_index,
        teacher_step=None,
        forced_action=None,
        plan_hidden=None,
    ):
        if teacher_step is not None and forced_action is not None:
            raise ValueError("teacher_step and forced_action are mutually exclusive")
        if plan_hidden is not None:
            if self.plan_hidden is None:
                raise ValueError("plan_hidden was provided but plan_hidden_dim=0")
            state = self.state_norm(state + self.plan_hidden(plan_hidden))
        nodes = self._encode_graph(base_nodes, state, edge_index, edge_type)
        action_logits = self.action_head(torch.cat([state, query], dim=-1))
        if forced_action is not None:
            if forced_action not in self.action_to_id:
                raise ValueError(f"Unknown forced action: {forced_action}")
            action_id = self.action_to_id[forced_action]
            action_tensor = torch.tensor(action_id, dtype=torch.long, device=state.device)
        elif teacher_step is None:
            action_id = int(action_logits.argmax().item())
            action_tensor = torch.tensor(action_id, dtype=torch.long, device=state.device)
        else:
            action_id = self.action_to_id[teacher_step["action"]]
            action_tensor = torch.tensor(action_id, dtype=torch.long, device=state.device)
        action_state = self.action_embedding(action_tensor)
        pointer_query = self.pointer_query(torch.cat([state, action_state], dim=-1))
        table_logits = self._masked(
            self.table_key(nodes) @ pointer_query, node_type_ids == 0
        )
        column_logits = self._masked(
            self.column_key(nodes) @ pointer_query, node_type_ids == 1
        )
        if join_edge_index.numel():
            left, right = join_edge_index
            edge_features = torch.cat(
                [nodes[left], nodes[right], nodes[left] * nodes[right], torch.abs(nodes[left] - nodes[right])],
                dim=-1,
            )
            join_edge_logits = self.edge_key(edge_features) @ pointer_query
        else:
            join_edge_logits = state.new_empty((0,))
        value_route_logits = self.value_route_head(torch.cat([state, action_state], dim=-1))
        operator_logits = self.operator_head(torch.cat([state, action_state], dim=-1))

        if teacher_step is None:
            selected_ids = []
            if ACTIONS[action_id] == "SCAN" and (node_type_ids == 0).any():
                selected_ids = [int(table_logits.argmax().item())]
            elif ACTIONS[action_id] in {"JOIN", "FILTER", "AGGREGATE", "HAVING_FILTER", "SORT", "PROJECT"}:
                if (node_type_ids == 1).any():
                    selected_ids = [int(column_logits.argmax().item())]
            route_ids = []
            operator_ids = []
            action_name = ACTIONS[action_id]
            if action_name in {"FILTER", "HAVING_FILTER", "LIMIT"}:
                route_ids = [int(value_route_logits.argmax().item())]
            if action_name in {
                "JOIN", "FILTER", "AGGREGATE", "HAVING_FILTER",
                "SORT", "LIMIT", "PROJECT",
            }:
                operator_ids = [int(operator_logits.argmax().item())]
        else:
            selected_ids = list(teacher_step.get("table_pointer_ids", []))
            selected_ids += list(teacher_step.get("column_pointer_ids", []))
            route_ids = [self.value_route_to_id[name] for name in teacher_step.get("value_routes", [])]
            operator_ids = [self.operator_to_id[name] for name in teacher_step.get("operator_targets", [])]
        if selected_ids:
            schema_summary = nodes[torch.tensor(selected_ids, device=nodes.device)].mean(0)
        else:
            schema_summary = torch.zeros_like(state)
        if route_ids:
            route_summary = self.value_route_embedding(
                torch.tensor(route_ids, dtype=torch.long, device=state.device)
            ).mean(0)
        else:
            route_summary = torch.zeros_like(state)
        if operator_ids:
            operator_summary = self.operator_embedding(
                torch.tensor(operator_ids, dtype=torch.long, device=state.device)
            ).mean(0)
        else:
            operator_summary = torch.zeros_like(state)
        transition_input = torch.cat(
            [query, action_state, schema_summary, route_summary, operator_summary], dim=-1
        )
        next_state = self.transition(transition_input.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
        return {
            "action_logits": action_logits,
            "table_logits": table_logits,
            "column_logits": column_logits,
            "join_edge_logits": join_edge_logits,
            "value_route_logits": value_route_logits,
            "operator_logits": operator_logits,
            "schema_states": nodes,
            "controller_state": next_state,
            "forced_action": forced_action,
        }

    def forward_trajectory(
        self,
        dense_nodes,
        query_embedding,
        node_type_ids,
        edge_index,
        edge_type,
        join_edge_index,
        teacher_steps,
        plan_hidden_states=None,
    ):
        base_nodes, query, state = self.initialize(dense_nodes, query_embedding, node_type_ids)
        outputs = []
        for index, teacher_step in enumerate(teacher_steps):
            plan_hidden = None if plan_hidden_states is None else plan_hidden_states[index]
            output = self.step(
                base_nodes, query, state, node_type_ids, edge_index, edge_type,
                join_edge_index, teacher_step=teacher_step, plan_hidden=plan_hidden,
            )
            outputs.append(output)
            state = output["controller_state"]
        return outputs


def multilabel_bce(logits, positive_indices, valid_mask=None, pos_weight=5.0):
    if logits.numel() == 0:
        return None
    target = torch.zeros_like(logits)
    if positive_indices:
        target[torch.tensor(positive_indices, dtype=torch.long, device=logits.device)] = 1.0
    if valid_mask is not None:
        logits, target = logits[valid_mask], target[valid_mask]
    if logits.numel() == 0:
        return None
    weight = torch.where(target > 0, torch.full_like(target, pos_weight), torch.ones_like(target))
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)
