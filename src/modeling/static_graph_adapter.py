"""Static schema-graph memory and decoder-layer adapters for code LLMs."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling.dynamic_grounded_llm import (
    layer_indices_from_fractions,
    resolve_decoder_layers,
)
from src.modeling.dynamic_grounding_controller import StateConditionedRGTALayer


class StaticSchemaGraphEncoder(nn.Module):
    """Encode a complete schema graph once for an entire SQL generation."""

    def __init__(
        self,
        dense_dim=1024,
        hidden_dim=256,
        relation_count=5,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_input = nn.Linear(dense_dim, hidden_dim)
        self.query_input = nn.Linear(dense_dim, hidden_dim)
        self.node_type = nn.Embedding(2, hidden_dim)
        self.layers = nn.ModuleList(
            StateConditionedRGTALayer(hidden_dim, relation_count, dropout)
            for _ in range(num_layers)
        )
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.structure_src = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.structure_dst = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.structure_relation = nn.Embedding(relation_count, hidden_dim)

    def forward(self, dense_nodes, query_embedding, node_type_ids, edge_index, edge_type):
        query = self.query_input(query_embedding).squeeze(0)
        nodes = self.node_input(dense_nodes) + self.node_type(node_type_ids)
        for layer in self.layers:
            nodes = layer(nodes, edge_index, edge_type, query)
        return self.memory_norm(nodes), query

    def structure_loss(self, nodes, edge_index, edge_type, max_edges=512):
        """Contrast true typed edges with destination-corrupted edges."""
        if edge_index.numel() == 0:
            return nodes.sum() * 0.0
        src, dst = edge_index
        # Self loops are useful to the encoder but trivial for reconstruction.
        keep = src.ne(dst)
        src, dst, relation = src[keep], dst[keep], edge_type[keep]
        if not src.numel():
            return nodes.sum() * 0.0
        if src.numel() > max_edges:
            selection = torch.linspace(
                0, src.numel() - 1, max_edges, device=src.device
            ).long()
            src, dst, relation = src[selection], dst[selection], relation[selection]
        negative_dst = torch.roll(dst, shifts=1)
        if dst.numel() == 1 or torch.equal(negative_dst, dst):
            negative_dst = (dst + 1) % nodes.shape[0]
        source = self.structure_src(nodes[src]) + self.structure_relation(relation)
        positive = (source * self.structure_dst(nodes[dst])).sum(-1) / math.sqrt(self.hidden_dim)
        negative = (source * self.structure_dst(nodes[negative_dst])).sum(-1) / math.sqrt(self.hidden_dim)
        return 0.5 * (F.softplus(-positive).mean() + F.softplus(negative).mean())


class StaticGraphCrossAdapter(nn.Module):
    """Read fixed graph memory and apply a token-wise gated residual update."""

    def __init__(
        self,
        llm_dim,
        graph_dim,
        num_heads=8,
        dropout=0.0,
        residual_scale_init=0.02,
        gate_bias_init=-2.0,
    ):
        super().__init__()
        if llm_dim % num_heads:
            raise ValueError("llm_dim must be divisible by num_heads")
        self.llm_dim = llm_dim
        self.num_heads = num_heads
        self.head_dim = llm_dim // num_heads
        self.norm = nn.LayerNorm(llm_dim)
        self.query = nn.Linear(llm_dim, llm_dim, bias=False)
        self.key = nn.Linear(graph_dim, llm_dim, bias=False)
        self.value = nn.Linear(graph_dim, llm_dim, bias=False)
        self.output = nn.Linear(llm_dim, llm_dim, bias=False)
        self.context_norm = nn.LayerNorm(llm_dim)
        self.gate = nn.Linear(llm_dim * 2, 1)
        nn.init.constant_(self.gate.bias, float(gate_bias_init))
        self.dropout = nn.Dropout(dropout)
        initial = max(min(float(residual_scale_init), 0.999), -0.999)
        self.residual_scale = nn.Parameter(torch.tensor(math.atanh(initial)))
        self.last_diagnostics = {}

    def forward(self, hidden, graph_memory):
        if hidden.shape[0] != 1:
            raise ValueError("Static graph adapters currently require batch_size=1")
        output_dtype = hidden.dtype
        working_dtype = self.norm.weight.dtype
        hidden = hidden.to(dtype=working_dtype)
        memory = graph_memory.to(hidden.device, working_dtype)
        batch, length, _ = hidden.shape
        normalized = self.norm(hidden)
        query = self.query(normalized).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.key(memory).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        value = self.value(memory).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        scores = torch.einsum("bhtd,hnd->bhtn", query, key) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        context = torch.einsum("bhtn,hnd->bhtd", attention, value)
        context = context.transpose(1, 2).contiguous().view(batch, length, self.llm_dim)
        context = self.context_norm(self.output(context))
        gate = torch.sigmoid(self.gate(torch.cat([normalized, context], dim=-1)))
        update = torch.tanh(self.residual_scale) * gate * context
        result = (hidden + self.dropout(update)).to(dtype=output_dtype)
        hidden_norm = hidden.detach().float().norm(dim=-1).mean().clamp_min(1e-8)
        self.last_diagnostics = {
            "mean_gate": float(gate.detach().mean().cpu()),
            "mean_attention_max": float(attention.detach().amax(-1).mean().cpu()),
            "mean_update_norm": float(update.detach().float().norm(dim=-1).mean().cpu()),
            "mean_update_ratio": float(
                (update.detach().float().norm(dim=-1).mean() / hidden_norm).cpu()
            ),
            "residual_scale": float(torch.tanh(self.residual_scale.detach()).cpu()),
        }
        return result


class GraphMemoryProjector(nn.Module):
    """Preserve node semantics while adding a gated graph-structure residual."""

    def __init__(
        self, graph_dim, llm_dim, semantic_dim=None, query_dim=None,
        dropout=0.0, structure_scale_init=0.1,
    ):
        super().__init__()
        semantic_dim = int(semantic_dim or graph_dim)
        query_dim = int(query_dim or semantic_dim)
        self.semantic_norm = nn.LayerNorm(semantic_dim)
        self.semantic_projection = nn.Linear(semantic_dim, llm_dim, bias=False)
        self.graph_norm = nn.LayerNorm(graph_dim)
        self.graph_projection = nn.Linear(graph_dim, llm_dim, bias=False)
        self.query_projection = nn.Linear(query_dim, graph_dim, bias=False)
        self.structure_gate = nn.Linear(graph_dim * 2, 1)
        nn.init.constant_(self.structure_gate.bias, -2.0)
        initial = max(min(float(structure_scale_init), 0.999), -0.999)
        self.structure_scale = nn.Parameter(torch.tensor(math.atanh(initial)))
        self.output_norm = nn.LayerNorm(llm_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_memory, semantic_memory=None, query_embedding=None, return_components=False):
        # Frozen graph encoders commonly emit FP32 while the code LLM runs in
        # BF16.  Keep the projector's own precision policy explicit instead of
        # relying on LayerNorm to accept mixed input/parameter dtypes.
        graph_memory = graph_memory.to(
            device=self.graph_norm.weight.device, dtype=self.graph_norm.weight.dtype
        )
        if semantic_memory is None:
            semantic_memory = graph_memory
        semantic_memory = semantic_memory.to(
            device=self.semantic_norm.weight.device, dtype=self.semantic_norm.weight.dtype
        )
        if query_embedding is None:
            query_embedding = semantic_memory.new_zeros((1, self.query_projection.in_features))
        query_embedding = query_embedding.to(
            device=self.query_projection.weight.device, dtype=self.query_projection.weight.dtype
        )
        graph_state = self.graph_norm(graph_memory)
        query_state = self.query_projection(query_embedding).mean(0)
        query_matrix = query_state.unsqueeze(0).expand_as(graph_state)
        structure_gate = torch.sigmoid(
            self.structure_gate(torch.cat([graph_state, query_matrix], dim=-1))
        )
        semantic = self.semantic_projection(self.semantic_norm(semantic_memory))
        structure = self.graph_projection(graph_state)
        structure = torch.tanh(self.structure_scale) * structure_gate * structure
        memory = self.output_norm(self.dropout(semantic + structure))
        if return_components:
            return {
                "memory": memory,
                "semantic_memory": self.output_norm(semantic),
                "structure_memory": structure,
                "structure_gate": structure_gate,
            }
        return memory


class StaticGraphConditionedCausalLM(nn.Module):
    """Attach static graph-memory adapters to selected frozen decoder layers."""

    def __init__(
        self,
        base_model,
        graph_dim,
        layer_indices=None,
        layer_fractions=(0.25, 0.5, 0.75, 1.0),
        num_heads=8,
        dropout=0.0,
        residual_scale_init=0.02,
        gate_bias_init=-2.0,
    ):
        super().__init__()
        self.base_model = base_model
        layers, self.layer_path = resolve_decoder_layers(base_model)
        if layer_indices is None:
            layer_indices = layer_indices_from_fractions(len(layers), layer_fractions)
        self.layer_indices = list(layer_indices)
        llm_dim = int(base_model.config.hidden_size)
        self.adapters = nn.ModuleDict()
        self._graph_memory = None
        self._handles = []
        for index in self.layer_indices:
            layer = layers[index]
            parameter = next(layer.parameters())
            adapter = StaticGraphCrossAdapter(
                llm_dim,
                graph_dim,
                num_heads=num_heads,
                dropout=dropout,
                residual_scale_init=residual_scale_init,
                gate_bias_init=gate_bias_init,
            ).to(device=parameter.device)
            self.adapters[str(index)] = adapter
            self._handles.append(layer.register_forward_hook(self._make_hook(index)))

    def _make_hook(self, index):
        def hook(_module, _inputs, output):
            if self._graph_memory is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            updated = self.adapters[str(index)](hidden, self._graph_memory)
            if isinstance(output, tuple):
                return (updated, *output[1:])
            return updated
        return hook

    def set_graph_memory(self, graph_memory):
        if graph_memory.ndim != 2:
            raise ValueError("graph_memory must have shape [node_count, graph_dim]")
        self._graph_memory = graph_memory

    def clear_graph_memory(self):
        self._graph_memory = None

    def freeze_base_model(self):
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        for parameter in self.adapters.parameters():
            parameter.requires_grad = True

    def adapter_state_dict(self):
        return {key: value.detach().cpu() for key, value in self.adapters.state_dict().items()}

    def diagnostics(self):
        return {index: dict(adapter.last_diagnostics) for index, adapter in self.adapters.items()}

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
