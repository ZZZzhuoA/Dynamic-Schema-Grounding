"""Shared loading and tensor helpers for the frozen-GNN static adapter."""

import json
from pathlib import Path

import torch

from src.modeling.typed_ra_decoder import TypedRAPointerDecoder


def load_frozen_typed_graph_encoder(summary_path, checkpoint_path, dense_dim, device):
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    config = summary.get("config", {})
    relations = summary.get("relations", [])
    model = TypedRAPointerDecoder(
        dense_dim=dense_dim,
        hidden_dim=int(config.get("hidden_dim", 256)),
        relation_count=max(len(relations), 1),
        num_layers=int(config.get("num_layers", 2)),
        dropout=0.0,
    ).to(device)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, {name: index for index, name in enumerate(relations)}, summary


def graph_tensors(example, cache, relation_to_id, device):
    inputs = example.get("inference_inputs", example)
    nodes = inputs.get("schema_nodes") or inputs.get("schema_items") or []
    record_index = int(example.get("metadata", {}).get("record_index", example.get("record_index", 0)))
    # Stage 8G/13B caches expose ``by_index`` while the older Stage 10 helper
    # calls the same mapping ``index``.  Accept both so a cache's on-disk
    # format, rather than the Python loader used by the caller, is decisive.
    index = cache.get("by_index") or cache.get("index")
    if index is None:
        raise KeyError("Embedding cache has neither 'by_index' nor 'index'")
    row = index.get(record_index)
    if row is None:
        raise KeyError(f"Embedding cache has no record_index={record_index}")
    query_index = int(row["query_embedding_index"])
    node_count = int(row["node_count"])
    if "node_embedding_indices" in row:
        values = cache["node"][[int(value) for value in row["node_embedding_indices"]]]
    else:
        start = int(row["node_embedding_start"])
        values = cache["node"][start : start + node_count]
    if len(values) != len(nodes):
        raise ValueError(
            f"Node/cache mismatch at record_index={record_index}: graph={len(nodes)} cache={len(values)}"
        )
    dense = torch.tensor(values, dtype=torch.float32, device=device)
    query = torch.tensor(
        cache["query"][query_index : query_index + 1], dtype=torch.float32, device=device
    )
    node_types = torch.tensor(
        [0 if node.get("type") == "table" else 1 for node in nodes],
        dtype=torch.long,
        device=device,
    )
    pairs, types = [], []
    for edge in inputs.get("schema_edges", []):
        relation_id = relation_to_id.get(edge.get("type"))
        if relation_id is None:
            continue
        pairs.append((int(edge["src"]), int(edge["dst"])))
        types.append(relation_id)
    edge_index = (
        torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()
        if pairs else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    edge_type = torch.tensor(types, dtype=torch.long, device=device)
    return dense, query, node_types, edge_index, edge_type, nodes


def corrupt_destinations(edge_index, node_count):
    if edge_index.numel() == 0:
        return edge_index
    corrupted = edge_index.clone()
    corrupted[1] = torch.roll(corrupted[1], shifts=1)
    if torch.equal(corrupted[1], edge_index[1]):
        corrupted[1] = (corrupted[1] + 1) % max(node_count, 1)
    return corrupted


def frozen_graph_memory(model, tensors, corrupt=False):
    dense, query, node_types, edge_index, edge_type, _ = tensors
    if corrupt:
        edge_index = corrupt_destinations(edge_index, dense.shape[0])
    with torch.no_grad():
        memory, _ = model.encode_static_memory(
            dense, query, node_types, edge_index, edge_type
        )
    return memory.detach()
