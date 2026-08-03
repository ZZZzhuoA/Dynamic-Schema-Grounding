import hashlib
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_RELATIONS = [
    "self_loop",
    "table_to_column",
    "column_to_table",
    "foreign_key_forward",
    "foreign_key_backward",
    "same_table_column",
]


def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return normalize_text(text).split()


def stable_hash(token):
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def hash_vector(text, dim):
    vector = torch.zeros(dim, dtype=torch.float32)
    for token in tokenize(text):
        index = stable_hash(token) % dim
        sign = 1.0 if stable_hash("sign:" + token) % 2 == 0 else -1.0
        vector[index] += sign
    norm = vector.norm(p=2)
    if norm > 0:
        vector = vector / norm
    return vector


def query_text(inference_inputs):
    return f"{inference_inputs.get('question') or ''} {inference_inputs.get('evidence') or ''}"


def node_text(node):
    semantic_text = node.get("semantic_text")
    if node.get("type") == "table":
        base = f"table: {node.get('name', '')}"
        return f"{base}; semantic: {semantic_text}" if semantic_text else base
    base = (
        f"column: {node.get('table', '')}.{node.get('column', '')}; "
        f"type: {node.get('data_type', '') or ''}"
    )
    return f"{base}; semantic: {semantic_text}" if semantic_text else base


def phrase_bonus(query_norm, text):
    text = normalize_text(text)
    if not text:
        return 0.0
    return 1.0 if f" {text} " in query_norm else 0.0


def overlap_score(query_tokens, item_tokens):
    if not item_tokens:
        return 0.0
    return len(set(query_tokens) & set(item_tokens)) / len(set(item_tokens))


def lexical_features(inference_inputs):
    question = inference_inputs.get("question") or ""
    evidence = inference_inputs.get("evidence") or ""
    q_text = query_text(inference_inputs)
    q_tokens = tokenize(q_text)
    question_tokens = tokenize(question)
    evidence_tokens = tokenize(evidence)
    q_norm = f" {normalize_text(q_text)} "
    features = []
    for node in inference_inputs["schema_nodes"]:
        item_tokens = tokenize(node_text(node))
        if node["type"] == "table":
            table = node.get("name", "")
            column = ""
        else:
            table = node.get("table", "")
            column = node.get("column", "")
        features.append(
            [
                overlap_score(q_tokens, item_tokens),
                overlap_score(question_tokens, item_tokens),
                overlap_score(evidence_tokens, item_tokens),
                phrase_bonus(q_norm, table),
                phrase_bonus(q_norm, column),
                1.0 if node["type"] == "table" else 0.0,
            ]
        )
    return torch.tensor(features, dtype=torch.float32)


def make_query_features(inference_inputs, hash_dim):
    return hash_vector(query_text(inference_inputs), hash_dim).unsqueeze(0)


def make_node_features(inference_inputs, hash_dim):
    rows = [hash_vector(node_text(node), hash_dim) for node in inference_inputs["schema_nodes"]]
    return torch.stack(rows, dim=0) if rows else torch.empty((0, hash_dim), dtype=torch.float32)


def make_edge_tensors(inference_inputs, relations, device):
    grouped = {relation: [] for relation in relations}
    for edge in inference_inputs.get("schema_edges", []):
        relation = edge["type"]
        if relation not in grouped:
            continue
        grouped[relation].append((edge["src"], edge["dst"]))
    tensors = {}
    for relation, pairs in grouped.items():
        if pairs:
            tensors[relation] = torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()
        else:
            tensors[relation] = torch.empty((2, 0), dtype=torch.long, device=device)
    return tensors


class RGCNLayer(nn.Module):
    def __init__(self, hidden_dim, relations, dropout):
        super().__init__()
        self.relations = relations
        self.rel_linears = nn.ModuleDict(
            {relation: nn.Linear(hidden_dim, hidden_dim, bias=False) for relation in relations}
        )
        self.root = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def aggregate(self, h, edge_index):
        node_count, hidden_dim = h.shape
        if edge_index.numel() == 0:
            return torch.zeros_like(h)
        src, dst = edge_index[0], edge_index[1]
        out = torch.zeros((node_count, hidden_dim), dtype=h.dtype, device=h.device)
        out.index_add_(0, dst, h[src])
        deg = torch.zeros((node_count,), dtype=h.dtype, device=h.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        return out / deg.clamp_min(1.0).unsqueeze(-1)

    def forward(self, h, edges):
        out = self.root(h)
        for relation in self.relations:
            if relation not in edges:
                continue
            out = out + self.rel_linears[relation](self.aggregate(h, edges[relation]))
        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(out + h)


class RGTALayer(nn.Module):
    """Edge-constrained relational graph attention for schema nodes."""

    def __init__(self, hidden_dim, relations, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.relations = relations
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rel_k = nn.ParameterDict(
            {relation: nn.Parameter(torch.empty(hidden_dim)) for relation in relations}
        )
        self.rel_v = nn.ParameterDict(
            {relation: nn.Parameter(torch.empty(hidden_dim)) for relation in relations}
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in list(self.rel_k.values()) + list(self.rel_v.values()):
            nn.init.normal_(parameter, mean=0.0, std=1.0 / math.sqrt(self.hidden_dim))

    def forward(self, h, edges):
        node_count = h.shape[0]
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        messages = []
        destinations = []
        scores = []

        for relation in self.relations:
            if relation not in edges or edges[relation].numel() == 0:
                continue
            src, dst = edges[relation][0], edges[relation][1]
            edge_k = k[src] + self.rel_k[relation].unsqueeze(0)
            edge_v = v[src] + self.rel_v[relation].unsqueeze(0)
            edge_scores = (q[dst] * edge_k).sum(dim=-1) / math.sqrt(self.hidden_dim)
            messages.append(edge_v)
            destinations.append(dst)
            scores.append(edge_scores)

        if not messages:
            return self.norm(h)

        all_messages = torch.cat(messages, dim=0)
        all_dst = torch.cat(destinations, dim=0)
        all_scores = torch.cat(scores, dim=0)
        max_per_dst = torch.full(
            (node_count,), -torch.inf, dtype=all_scores.dtype, device=all_scores.device
        )
        max_per_dst.scatter_reduce_(0, all_dst, all_scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(all_scores - max_per_dst[all_dst])
        denom = torch.zeros((node_count,), dtype=all_scores.dtype, device=all_scores.device)
        denom.index_add_(0, all_dst, exp_scores)
        attn = exp_scores / denom[all_dst].clamp_min(1e-8)

        weighted = all_messages * attn.unsqueeze(-1)
        out = torch.zeros((node_count, self.hidden_dim), dtype=h.dtype, device=h.device)
        out.index_add_(0, all_dst, weighted)
        out = self.out_proj(out)
        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(out + h)


class DSGGrounder(nn.Module):
    """Question-schema graph grounding module for DSG-SQL.

    The model consumes only test-time available inputs: question/evidence,
    schema nodes, and schema edges. Gold grounding labels are used outside this
    module only as training loss targets.
    """

    def __init__(
        self,
        hash_dim=256,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        relations=None,
        encoder_type="rgta",
        lexical_dim=6,
    ):
        super().__init__()
        self.hash_dim = hash_dim
        self.hidden_dim = hidden_dim
        self.relations = relations or list(DEFAULT_RELATIONS)
        self.encoder_type = encoder_type
        self.lexical_dim = lexical_dim
        self.schema_input = nn.Linear(hash_dim, hidden_dim)
        self.query_input = nn.Sequential(
            nn.Linear(hash_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if encoder_type == "rgcn":
            layer_cls = RGCNLayer
        elif encoder_type == "rgta":
            layer_cls = RGTALayer
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.layers = nn.ModuleList([layer_cls(hidden_dim, self.relations, dropout) for _ in range(num_layers)])
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4 + lexical_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_schema(self, node_features, edges):
        h = self.schema_input(node_features)
        for layer in self.layers:
            h = layer(h, edges)
        return h

    def forward(self, query_features, node_features, edges, lex_features=None):
        query = self.query_input(query_features).squeeze(0)
        node_states = self.encode_schema(node_features, edges)
        query_matrix = query.unsqueeze(0).expand_as(node_states)
        pair = torch.cat(
            [
                query_matrix,
                node_states,
                query_matrix * node_states,
                torch.abs(query_matrix - node_states),
            ],
            dim=-1,
        )
        if self.lexical_dim:
            if lex_features is None:
                lex_features = torch.zeros(
                    (node_states.shape[0], self.lexical_dim),
                    dtype=node_states.dtype,
                    device=node_states.device,
                )
            pair = torch.cat([pair, lex_features], dim=-1)
        logits = self.scorer(pair).squeeze(-1)
        return {
            "logits": logits,
            "schema_node_embeddings": node_states,
        }
