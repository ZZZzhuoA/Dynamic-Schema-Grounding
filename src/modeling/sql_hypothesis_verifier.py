"""Typed plan–schema graph compatibility model for Stage 15A."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.stage13b_prepare_typed_trajectories import ACTIONS, OPERATORS, VALUE_ROUTES
from src.modeling.dynamic_grounding_controller import StateConditionedRGTALayer


PLAN_RELATIONS = ["self", "next", "previous"]
CONSISTENCY_FEATURE_NAMES = [
    "owner_scan_coverage",
    "scan_required_precision",
    "join_scan_coverage",
    "fk_validity",
    "required_table_connectivity",
    "join_required_coverage",
    "multi_table_join_present",
    "extra_scan_ratio",
    "missing_owner_ratio",
    "invalid_join_ratio",
]


def plan_schema_consistency_features(schema_items, schema_edges, candidate):
    """Inference-safe consistency factors derived from a concrete plan binding."""
    by_id = {int(item["id"]): item for item in schema_items}
    table_name_to_id = {
        item.get("name"): item_id
        for item_id, item in by_id.items()
        if item.get("type") == "table"
    }

    def owner(column_id):
        item = by_id.get(int(column_id), {})
        return table_name_to_id.get(item.get("table")) if item.get("type") == "column" else None

    scan_tables = {
        int(table_id)
        for step in candidate.get("steps", [])
        if step.get("action") == "SCAN"
        for table_id in step.get("table_pointer_ids", [])
        if int(table_id) in by_id and by_id[int(table_id)].get("type") == "table"
    }
    referenced_owners = {
        table_id
        for step in candidate.get("steps", [])
        for column_id in step.get("column_pointer_ids", [])
        for table_id in [owner(column_id)]
        if table_id is not None
    }
    required_tables = scan_tables | referenced_owners
    join_pairs = [
        tuple(sorted((int(edge["left_column_id"]), int(edge["right_column_id"]))))
        for step in candidate.get("steps", [])
        for edge in step.get("join_edge_targets", [])
    ]
    valid_fk_pairs = {
        tuple(sorted((int(edge["src"]), int(edge["dst"]))))
        for edge in schema_edges or []
        if edge.get("type") in {"foreign_key_forward", "foreign_key_backward"}
    }
    join_owner_pairs = []
    for left, right in join_pairs:
        left_owner, right_owner = owner(left), owner(right)
        if left_owner is not None and right_owner is not None:
            join_owner_pairs.append((left_owner, right_owner))

    def ratio(numerator, denominator, empty=1.0):
        return numerator / denominator if denominator else empty

    owner_scan_coverage = ratio(len(referenced_owners & scan_tables), len(referenced_owners))
    scan_required_precision = (
        ratio(len(scan_tables & referenced_owners), len(scan_tables))
        if referenced_owners else 1.0
    )
    join_scan_coverage = ratio(
        sum(left in scan_tables and right in scan_tables for left, right in join_owner_pairs),
        len(join_owner_pairs),
    )
    fk_validity = ratio(sum(pair in valid_fk_pairs for pair in join_pairs), len(join_pairs))
    join_required_coverage = ratio(
        sum(left in required_tables and right in required_tables for left, right in join_owner_pairs),
        len(join_owner_pairs),
    )

    adjacency = {table_id: set() for table_id in required_tables}
    for left, right in join_owner_pairs:
        if left in required_tables and right in required_tables:
            adjacency[left].add(right)
            adjacency[right].add(left)
    if len(required_tables) <= 1:
        connectivity = 1.0
    else:
        largest = 0
        unseen = set(required_tables)
        while unseen:
            frontier = [unseen.pop()]
            size = 0
            while frontier:
                current = frontier.pop()
                size += 1
                for neighbor in adjacency.get(current, set()):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        frontier.append(neighbor)
            largest = max(largest, size)
        connectivity = largest / len(required_tables)

    multi_table_join_present = float(len(required_tables) <= 1 or bool(join_owner_pairs))
    extra_scan_ratio = (
        ratio(len(scan_tables - referenced_owners), len(scan_tables), empty=0.0)
        if referenced_owners else 0.0
    )
    missing_owner_ratio = 1.0 - owner_scan_coverage
    invalid_join_ratio = 1.0 - fk_validity
    values = [
        owner_scan_coverage,
        scan_required_precision,
        join_scan_coverage,
        fk_validity,
        connectivity,
        join_required_coverage,
        multi_table_join_present,
        extra_scan_ratio,
        missing_owner_ratio,
        invalid_join_ratio,
    ]
    return values, dict(zip(CONSISTENCY_FEATURE_NAMES, values))


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
        self.consistency_encoder = nn.Sequential(
            nn.Linear(len(CONSISTENCY_FEATURE_NAMES), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.consistency_energy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.global_score = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 3 + len(CONSISTENCY_FEATURE_NAMES), hidden_dim),
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

    def score_candidate(self, schema_states, query, schema_items, schema_edges, candidate):
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
        consistency_values, consistency_debug = plan_schema_consistency_features(
            schema_items, schema_edges, candidate
        )
        consistency_numeric = query.new_tensor(consistency_values)
        consistency_state = self.consistency_encoder(consistency_numeric)
        consistency_energy = self.consistency_energy(consistency_state).squeeze()
        score = self.global_score(
            torch.cat(
                [
                    plan_summary,
                    query,
                    plan_summary * query,
                    torch.abs(plan_summary - query),
                    consistency_state,
                    global_numeric,
                    consistency_numeric,
                ],
                dim=-1,
            )
        ).squeeze()
        score = score + 0.25 * step_energy + 0.25 * join_energy + 0.25 * consistency_energy
        return {
            "score": score,
            "step_energy": step_energy,
            "join_energy": join_energy,
            "plan_summary": plan_summary,
            "pointer_validity": pointer_validity,
            "consistency_energy": consistency_energy,
            "consistency_features": consistency_debug,
        }

    def forward(
        self,
        dense_nodes,
        query_embedding,
        node_type_ids,
        edge_index,
        edge_type,
        schema_items,
        schema_edges,
        candidates,
    ):
        schema_states, query = self.encode_schema(
            dense_nodes, query_embedding, node_type_ids, edge_index, edge_type
        )
        outputs = [
            self.score_candidate(schema_states, query, schema_items, schema_edges, candidate)
            for candidate in candidates
        ]
        return {
            "scores": torch.stack([output["score"] for output in outputs]),
            "candidate_outputs": outputs,
            "schema_states": schema_states,
            "query_state": query,
        }


def grouped_ranking_loss(
    scores,
    labels,
    margin=0.5,
    margin_weight=0.5,
    hardest_negative_weight=0.5,
):
    positive = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
    negative = torch.nonzero(labels <= 0.5, as_tuple=False).flatten()
    if positive.numel() != 1:
        raise ValueError(f"Expected exactly one positive candidate, got {positive.numel()}")
    target = positive[:1]
    listwise = F.cross_entropy(scores.unsqueeze(0), target)
    if negative.numel():
        pairwise = F.relu(margin - scores[positive[0]] + scores[negative]).mean()
        hardest = F.softplus(margin + scores[negative].max() - scores[positive[0]])
    else:
        pairwise = scores.sum() * 0.0
        hardest = scores.sum() * 0.0
    return (
        listwise
        + float(margin_weight) * pairwise
        + float(hardest_negative_weight) * hardest
    ), {
        "listwise_loss": listwise,
        "pairwise_loss": pairwise,
        "hardest_negative_loss": hardest,
    }
