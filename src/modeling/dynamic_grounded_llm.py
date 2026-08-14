"""Neural adapters that inject dynamic schema grounding into decoder layers."""

import math

import torch
import torch.nn as nn


def resolve_decoder_layers(model):
    """Resolve common HuggingFace decoder layer layouts without model-specific imports."""
    candidates = [
        ("model.layers", lambda root: root.model.layers),
        ("model.model.layers", lambda root: root.model.model.layers),
        ("transformer.h", lambda root: root.transformer.h),
    ]
    for name, getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if layers is not None and len(layers):
            return layers, name
    raise ValueError("Could not locate decoder layers on the causal language model")


def layer_indices_from_fractions(layer_count, fractions):
    indices = []
    for fraction in fractions:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("Adapter layer fractions must be in (0, 1]")
        index = min(round(float(fraction) * layer_count) - 1, layer_count - 1)
        indices.append(max(index, 0))
    return sorted(set(indices))


class DynamicGroundingAdapter(nn.Module):
    """Cross-attention plus hidden steering for token-specific grounding states."""

    def __init__(self, llm_dim, grounding_dim, num_heads=8, dropout=0.0):
        super().__init__()
        if llm_dim % num_heads:
            raise ValueError("llm_dim must be divisible by num_heads")
        self.llm_dim = llm_dim
        self.num_heads = num_heads
        self.head_dim = llm_dim // num_heads
        self.norm = nn.LayerNorm(llm_dim)
        self.query = nn.Linear(llm_dim, llm_dim, bias=False)
        self.key = nn.Linear(grounding_dim, llm_dim, bias=False)
        self.value = nn.Linear(grounding_dim, llm_dim, bias=False)
        self.output = nn.Linear(llm_dim, llm_dim, bias=False)
        self.steering = nn.Linear(grounding_dim, llm_dim, bias=False)
        self.route = nn.Linear(llm_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        # Exact identity at initialization protects the frozen language model.
        self.cross_scale = nn.Parameter(torch.zeros(()))
        self.steering_scale = nn.Parameter(torch.zeros(()))

    def _attend(self, hidden, grounding):
        # hidden: [B, T, D_lm], grounding: [N, D_g]
        batch, length, _ = hidden.shape
        query = self.query(self.norm(hidden)).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.key(grounding).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        value = self.value(grounding).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        scores = torch.einsum("bhtd,hnd->bhtn", query, key) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        context = torch.einsum("bhtn,hnd->bhtd", attention, value)
        context = context.transpose(1, 2).contiguous().view(batch, length, self.llm_dim)
        return self.output(context), attention

    def forward(self, hidden, grounding_steps, steering_steps, token_step_ids):
        if hidden.shape[0] != 1:
            raise ValueError("Stage 12 adapters currently require batch_size=1")
        if token_step_ids.numel() != hidden.shape[1]:
            raise ValueError(
                f"token_step_ids length={token_step_ids.numel()} does not match hidden length={hidden.shape[1]}"
            )
        result = hidden
        diagnostics = []
        step_ids = token_step_ids.to(hidden.device)
        for step_id in torch.unique(step_ids).tolist():
            step_id = int(step_id)
            if step_id < 0:
                continue
            positions = (step_ids == step_id).nonzero(as_tuple=False).flatten()
            if not positions.numel():
                continue
            grounding = grounding_steps[step_id].to(hidden.device, hidden.dtype)
            steering = steering_steps[step_id].to(hidden.device, hidden.dtype)
            selected = result.index_select(1, positions)
            context, attention = self._attend(selected, grounding)
            steering = self.steering(steering).view(1, 1, -1).expand_as(context)
            route = torch.sigmoid(self.route(torch.cat([selected, context], dim=-1)))
            update = (
                torch.tanh(self.cross_scale) * route * context
                + torch.tanh(self.steering_scale) * (1.0 - route) * steering
            )
            updated = selected + self.dropout(update)
            result = result.index_copy(1, positions, updated)
            diagnostics.append(
                {
                    "step_id": step_id,
                    "token_count": int(positions.numel()),
                    "mean_route": float(route.detach().mean().cpu()),
                    "mean_attention_max": float(attention.detach().amax(-1).mean().cpu()),
                    "mean_context_norm": float(context.detach().float().norm(dim=-1).mean().cpu()),
                    "mean_steering_norm": float(steering.detach().float().norm(dim=-1).mean().cpu()),
                    "mean_update_norm": float(update.detach().float().norm(dim=-1).mean().cpu()),
                }
            )
        return result, diagnostics


class DynamicGroundedCausalLM(nn.Module):
    """Wrap a frozen HF causal LM and inject adapters through layer hooks."""

    def __init__(
        self, base_model, grounding_dim, layer_indices=None,
        layer_fractions=(0.25, 0.5, 0.75, 1.0), num_heads=8, dropout=0.0,
    ):
        super().__init__()
        self.base_model = base_model
        layers, self.layer_path = resolve_decoder_layers(base_model)
        if layer_indices is None:
            layer_indices = layer_indices_from_fractions(len(layers), layer_fractions)
        self.layer_indices = list(layer_indices)
        llm_dim = int(getattr(base_model.config, "hidden_size"))
        self.adapters = nn.ModuleDict()
        self._handles = []
        self._context = None
        self.last_diagnostics = {}
        for index in self.layer_indices:
            layer = layers[index]
            device = next(layer.parameters()).device
            adapter = DynamicGroundingAdapter(
                llm_dim, grounding_dim, num_heads=num_heads, dropout=dropout
            ).to(device=device, dtype=next(layer.parameters()).dtype)
            self.adapters[str(index)] = adapter
            self._handles.append(layer.register_forward_hook(self._make_hook(index)))

    def _make_hook(self, index):
        def hook(_module, _inputs, output):
            if self._context is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            adapter = self.adapters[str(index)]
            updated, diagnostics = adapter(
                hidden,
                self._context["grounding_steps"],
                self._context["steering_steps"],
                self._context["token_step_ids"],
            )
            self.last_diagnostics[str(index)] = diagnostics
            if isinstance(output, tuple):
                return (updated, *output[1:])
            return updated
        return hook

    def set_grounding_context(self, grounding_steps, steering_steps, token_step_ids):
        if len(grounding_steps) != len(steering_steps):
            raise ValueError("grounding_steps and steering_steps must have equal length")
        self._context = {
            "grounding_steps": grounding_steps,
            "steering_steps": steering_steps,
            "token_step_ids": token_step_ids,
        }

    def clear_grounding_context(self):
        self._context = None

    def freeze_base_model(self):
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        for parameter in self.adapters.parameters():
            parameter.requires_grad = True

    def adapter_state_dict(self):
        return {key: value.detach().cpu() for key, value in self.adapters.state_dict().items()}

    def adapter_scale_summary(self):
        result = {}
        for index, adapter in self.adapters.items():
            result[index] = {
                "cross_scale_raw": float(adapter.cross_scale.detach().float().cpu()),
                "cross_scale_effective": float(torch.tanh(adapter.cross_scale.detach()).float().cpu()),
                "steering_scale_raw": float(adapter.steering_scale.detach().float().cpu()),
                "steering_scale_effective": float(
                    torch.tanh(adapter.steering_scale.detach()).float().cpu()
                ),
            }
        return result

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
