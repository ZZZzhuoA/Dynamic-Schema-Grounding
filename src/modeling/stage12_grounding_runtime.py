"""Load and run the frozen Stage 11 controller for Stage 12 LLM integration."""

import json
from pathlib import Path

import torch

from src.data.stage12_llm_grounding_data import (
    observed_candidate_mask,
    operation_for_partial_sql,
)
from src.modeling.dynamic_grounding_controller import (
    DynamicSchemaGroundingController,
    partial_sql_features,
)
from src.training.stage11_train_dynamic_grounding_controller import trajectory_tensors


def load_controller(summary_path, checkpoint_path, dense_dim, numeric_dim, device):
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    config = summary.get("config", {})
    relations = summary.get("relations", [])
    model = DynamicSchemaGroundingController(
        dense_dim=dense_dim,
        numeric_dim=numeric_dim,
        hidden_dim=int(config.get("hidden_dim", 256)),
        relation_count=max(len(relations), 1),
        num_layers=int(config.get("num_layers", 2)),
        dropout=0.0,
        recurrent=False,
        history_mode="independent",
    ).to(device)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    incompatible = model.load_state_dict(state, strict=False)
    missing_base = [
        key for key in incompatible.missing_keys
        if not key.startswith(("history_delta.", "history_relevance.", "residual_norm."))
    ]
    if missing_base:
        raise ValueError("Controller checkpoint misses base keys: " + ", ".join(missing_base[:20]))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, {name: index for index, name in enumerate(relations)}, summary


def trajectory_grounding_context(
    controller, trajectory, cache, relation_to_id, runtime, device,
):
    dense, numeric, query, edges, steps = trajectory_tensors(
        trajectory, cache, relation_to_id, controller, runtime, device
    )
    with torch.no_grad():
        outputs = controller.forward_trajectory(dense, numeric, query, edges, steps)
    return (
        [output["grounding_tokens"].detach() for output in outputs],
        [output["steering_state"].detach() for output in outputs],
        (dense, numeric, query, edges),
    )


def partial_sql_grounding_context(
    controller, trajectory, tensors, partial_sql,
):
    dense, numeric, query_embedding, edges = tensors
    base_nodes, query, initial_state, initial_belief = controller.initialize(
        dense, numeric, query_embedding
    )
    operation = operation_for_partial_sql(partial_sql)
    operation_id = controller.operation_to_id.get(
        operation, controller.operation_to_id["UNKNOWN"]
    )
    observed = torch.tensor(
        observed_candidate_mask(partial_sql, trajectory.get("candidate_nodes", [])),
        dtype=dense.dtype,
        device=dense.device,
    )
    with torch.no_grad():
        output = controller.step(
            base_nodes, query, initial_state, initial_belief, edges,
            torch.tensor([operation_id], dtype=torch.long, device=dense.device),
            torch.tensor(partial_sql_features(partial_sql), dtype=dense.dtype, device=dense.device),
            observed_mask=observed,
            reference_state=initial_state,
            reference_belief=initial_belief,
            history_available=False,
        )
    return output["grounding_tokens"].detach(), output["steering_state"].detach(), operation
