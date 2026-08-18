"""Typed plan–schema graph compatibility model for Stage 15A."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.stage13b_prepare_typed_trajectories import ACTIONS, OPERATORS, VALUE_ROUTES
from src.modeling.dynamic_grounding_controller import StateConditionedRGTALayer


PLAN_RELATIONS = ["self", "next", "previous"]


class SQLHypothesisGraphVerifier(nn.Module):
    """Score a concrete SQL plan by matching it against a schema graph.

    Candidate corruption labels are deliberately absent from this interface.
    The candidate contributes only its actions, bindings, operators, value
    routes, and join edges.
    """

    def __init__(
        self,
        dense_dim=1024,
        hidden_dim=256,
        schema_relation_count=5,
        num_schema_layers=2,
        num_plan_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_to_id = {name: index for index, name in enumerate(ACTIONS)}
        self.operator_to_id = {name: index for index, name in enumerate(OPERATORS)}
        self.route_to_id = {name: index for index, name in enumerate(VALUE_ROUTES)}

        # Names intentionally match TypedRAPointerDecoder where possible, so a
        # Stage 13B RGTA checkpoint can warm-start the schema encoder.
        self.node_input = nn.Linear(dense_dim, hidden_dim)
        self.query_input = nn.Linear(dense_dim, hidden_dim)
        self.node_type = nn.Embedding(2, hidden_dim)
        self.initial_state = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.graph_layers = nn.ModuleList(
            [
                StateConditionedRGTALayer(hidden_dim, schema_relation_count, dropout)
                for _ in range(num_schema_layers)
            ]
        )

        self.action_embedding = nn.Embedding(len(ACTIONS), hidden_dim)
        self.operator_embedding = nn.Embedding(len(OPERATORS), hidden_dim)
        self.route_embedding = nn.Embedding(len(VALUE_ROUTES), hidden_dim)
        self.binding_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.binding_norm = nn.LayerNorm(hidden_dim)
        self.plan_layers = nn.ModuleList(
            [
                StateConditionedRGTALayer(
                    hidden_dim, len(PLAN_RELATIONS), dropout
                )
                for _ in range(num_plan_layers)
            ]
        )
        self.step_energy = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.join_energy = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.global_score = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def load_stage13b_state(self, state_dict):
        current = self.state_dict()
        prefixes = ("node_input.", "query_input.", "node_type.", "initial_state.", "graph_layers.")
        compatible = {
            name: value
            for name, value in state_dict.items()
            if name.startswith(prefixes)
            and name in current
            and tuple(current[name].shape) == tuple(value.shape)
        }
        result = self.load_state_dict(compatible, strict=False)
        return {
            "loaded_parameter_count": len(compatible),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }

    def encode_schema(self, dense_nodes, query_embedding, node_type_ids, edge_index, edge_type):
        nodes = self.node_input(dense_nodes) + self.node_type(node_type_ids)
        query = self.query_input(query_embedding).squeeze(0)
        state = self.initial_state(query)
        for layer in self.graph_layers:
            nodes = layer(nodes, edge_index, edge_type, state)
        return nodes, query

    @staticmethod
    def _mean_embeddings(embedding, ids, device):
        if not ids:
            return embedding.weight.new_zeros((embedding.embedding_dim,)).to(device)
        index = torch.tensor(ids, dtype=torch.long, device=device)
        return embedding(index).mean(0)

    @staticmethod
    def _plan_edges(step_count, device):
        pairs, types = [], []
        for index in range(step_count):
            pairs.append((index, index))
            types.append(0)
            if index + 1 < step_count:
                pairs.extend(((index, index + 1), (index + 1, index)))
                types.extend((1, 2))
        return (
            torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous(),
            torch.tensor(types, dtype=torch.long, device=device),
        )

    def score_candidate(self, schema_states, query, schema_items, candidate):
        device = schema_states.device
        id_to_position = {int(item["id"]): index for index, item in enumerate(schema_items)}
        step_states, raw_step_energies, join_scores = [], [], []
        valid_pointer_count, pointer_count = 0, 0

        for step in candidate.get("steps", []):
            action_id = self.action_to_id.get(step.get("action"), self.action_to_id["STOP"])
            action = self.action_embedding(torch.tensor(action_id, device=device))
            operator_ids = [
                self.operator_to_id[value] for value in step.get("operator_targets", [])
                if value in self.operator_to_id
            ]
            route_ids = [
                self.route_to_id[value] for value in step.get("value_routes", [])
                if value in self.route_to_id
            ]
            operator = self._mean_embeddings(self.operator_embedding, operator_ids, device)
            route = self._mean_embeddings(self.route_embedding, route_ids, device)

            table_ids = [int(value) for value in step.get("table_pointer_ids", [])]
            column_ids = [int(value) for value in step.get("column_pointer_ids", [])]
            pointer_ids = table_ids + column_ids
            positions = [id_to_position[value] for value in pointer_ids if value in id_to_position]
            pointer_count += len(pointer_ids)
            valid_pointer_count += len(positions)
            binding = (
                schema_states[torch.tensor(positions, dtype=torch.long, device=device)].mean(0)
                if positions else torch.zeros_like(query)
            )
            numeric = query.new_tensor(
                [
                    min(len(table_ids) / 4.0, 1.0),
                    min(len(column_ids) / 8.0, 1.0),
                    min(len(operator_ids) / 6.0, 1.0),
                    min(len(route_ids) / 4.0, 1.0),
                    float(bool(step.get("join_edge_targets"))),
                    float(bool(positions)),
                ]
            )
            fused = self.binding_norm(
                action
                + self.binding_fusion(
                    torch.cat([action, operator, route, binding, query, numeric], dim=-1)
                )
            )
            step_states.append(fused)
            raw_step_energies.append(
                self.step_energy(torch.cat([fused, binding, fused * binding, query], dim=-1)).squeeze()
            )

            for edge in step.get("join_edge_targets", []):
                left = id_to_position.get(int(edge["left_column_id"]))
                right = id_to_position.get(int(edge["right_column_id"]))
                if left is None or right is None:
                    continue
                left_state, right_state = schema_states[left], schema_states[right]
                join_scores.append(
                    self.join_energy(
                        torch.cat(
                            [
                                left_state,
                                right_state,
                                left_state * right_state,
                                torch.abs(left_state - right_state),
                                query,
                            ],
                            dim=-1,
                        )
                    ).squeeze()
                )

        if not step_states:
            zero = query.sum() * 0.0
            return {"score": zero, "step_energy": zero, "join_energy": zero}

        plan = torch.stack(step_states)
        plan_edges, plan_types = self._plan_edges(len(step_states), device)
        for layer in self.plan_layers:
            plan = layer(plan, plan_edges, plan_types, query)
        attention = torch.softmax(plan @ query / max(self.hidden_dim ** 0.5, 1.0), dim=0)
        plan_summary = (attention.unsqueeze(-1) * plan).sum(0)
        step_energy = torch.stack(raw_step_energies).mean()
        join_energy = torch.stack(join_scores).mean() if join_scores else query.sum() * 0.0
        pointer_validity = valid_pointer_count / max(pointer_count, 1)
        global_numeric = query.new_tensor(
            [min(len(step_states) / 12.0, 1.0), pointer_validity, float(bool(join_scores))]
        )
        score = self.global_score(
            torch.cat(
                [plan_summary, query, plan_summary * query, torch.abs(plan_summary - query), global_numeric],
                dim=-1,
            )
        ).squeeze()
        score = score + 0.25 * step_energy + 0.25 * join_energy
        return {
            "score": score,
            "step_energy": step_energy,
            "join_energy": join_energy,
            "plan_summary": plan_summary,
            "pointer_validity": pointer_validity,
        }

    def forward(
        self,
        dense_nodes,
        query_embedding,
        node_type_ids,
        edge_index,
        edge_type,
        schema_items,
        candidates,
    ):
        schema_states, query = self.encode_schema(
            dense_nodes, query_embedding, node_type_ids, edge_index, edge_type
        )
        outputs = [
            self.score_candidate(schema_states, query, schema_items, candidate)
            for candidate in candidates
        ]
        return {
            "scores": torch.stack([output["score"] for output in outputs]),
            "candidate_outputs": outputs,
            "schema_states": schema_states,
            "query_state": query,
        }


def grouped_ranking_loss(scores, labels, margin=0.5, margin_weight=0.5):
    positive = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
    negative = torch.nonzero(labels <= 0.5, as_tuple=False).flatten()
    if positive.numel() != 1:
        raise ValueError(f"Expected exactly one positive candidate, got {positive.numel()}")
    target = positive[:1]
    listwise = F.cross_entropy(scores.unsqueeze(0), target)
    if negative.numel():
        pairwise = F.relu(margin - scores[positive[0]] + scores[negative]).mean()
    else:
        pairwise = scores.sum() * 0.0
    return listwise + float(margin_weight) * pairwise, {
        "listwise_loss": listwise,
        "pairwise_loss": pairwise,
    }
