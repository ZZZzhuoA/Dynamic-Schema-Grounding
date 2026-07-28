import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


RELATIONS = [
    "self_loop",
    "table_to_column",
    "column_to_table",
    "foreign_key_forward",
    "foreign_key_backward",
]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path, limit=None):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = text.lower()
    text = text.replace("`", " ").replace('"', " ").replace("[", " ").replace("]", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(text):
    text = normalize_text(text).replace(" ", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def tokenize(text):
    text = normalize_text(text)
    return [token for token in text.split() if token]


def stable_hash(token):
    value = 2166136261
    for ch in token:
        value ^= ord(ch)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def hash_vector(text, dim):
    vec = np.zeros(dim, dtype=np.float32)
    tokens = tokenize(text)
    if not tokens:
        return vec
    counts = Counter(tokens)
    for token, count in counts.items():
        h = stable_hash(token)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        vec[idx] += sign * (1.0 + math.log(count))
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def overlap_score(query_tokens, item_tokens):
    query_set = set(query_tokens)
    item_set = set(item_tokens)
    if not query_set or not item_set:
        return 0.0
    overlap = query_set & item_set
    return len(overlap) / max(len(item_set), 1)


def phrase_bonus(query_norm_with_padding, phrase):
    phrase_norm = normalize_text(phrase)
    if not phrase_norm:
        return 0.0
    if f" {phrase_norm} " in query_norm_with_padding:
        return 1.0
    phrase_tokens = phrase_norm.split()
    if len(phrase_tokens) >= 2 and all(
        f" {token} " in query_norm_with_padding for token in phrase_tokens
    ):
        return 0.3
    return 0.0


def schema_item_text(item):
    if item["type"] == "table":
        return f"table: {item.get('name', '')}"
    return (
        f"column: {item.get('table', '')}.{item.get('column', '')}; "
        f"type: {item.get('data_type', '') or ''}"
    )


def query_text(record):
    return f"{record.get('question') or ''} {record.get('evidence') or ''}"


def lexical_features(record):
    question = record.get("question") or ""
    evidence = record.get("evidence") or ""
    q_text = query_text(record)
    q_tokens = tokenize(q_text)
    question_tokens = tokenize(question)
    evidence_tokens = tokenize(evidence)
    q_norm = f" {normalize_text(q_text)} "
    features = []
    for item in record["schema_items"]:
        item_tokens = tokenize(schema_item_text(item))
        if item["type"] == "table":
            table = item.get("name", "")
            column = ""
        else:
            table = item.get("table", "")
            column = item.get("column", "")
        features.append(
            [
                overlap_score(q_tokens, item_tokens),
                overlap_score(question_tokens, item_tokens),
                overlap_score(evidence_tokens, item_tokens),
                phrase_bonus(q_norm, table),
                phrase_bonus(q_norm, column),
                1.0 if item["type"] == "table" else 0.0,
            ]
        )
    return np.asarray(features, dtype=np.float32)


def load_table_entries(path: Path):
    return {entry["db_id"]: entry for entry in read_json(path)}


def build_schema_graph(record, table_entry, include_same_table_edges=False):
    relations = list(RELATIONS)
    if include_same_table_edges and "same_table_column" not in relations:
        relations.append("same_table_column")

    schema_items = record["schema_items"]
    node_count = len(schema_items)
    table_name_to_id = {}
    column_key_to_id = {}
    table_to_columns = defaultdict(list)

    for item in schema_items:
        if item["type"] == "table":
            table_name_to_id[normalize_name(item["name"])] = item["id"]
        else:
            table = normalize_name(item.get("table", ""))
            column = normalize_name(item.get("column", ""))
            column_key_to_id[(table, column)] = item["id"]
            table_to_columns[table].append(item["id"])

    edges = {relation: [] for relation in relations}
    for item in schema_items:
        edges["self_loop"].append((item["id"], item["id"]))
        if item["type"] != "column":
            continue
        table = normalize_name(item.get("table", ""))
        table_id = table_name_to_id.get(table)
        if table_id is not None:
            edges["table_to_column"].append((table_id, item["id"]))
            edges["column_to_table"].append((item["id"], table_id))

    if include_same_table_edges:
        for column_ids in table_to_columns.values():
            for src in column_ids:
                for dst in column_ids:
                    if src != dst:
                        edges["same_table_column"].append((src, dst))

    if table_entry:
        tables = table_entry.get("table_names_original", [])
        columns = table_entry.get("column_names_original", [])
        for left_index, right_index in table_entry.get("foreign_keys", []):
            if left_index >= len(columns) or right_index >= len(columns):
                continue
            left_table_index, left_column = columns[left_index]
            right_table_index, right_column = columns[right_index]
            if left_table_index < 0 or right_table_index < 0:
                continue
            left_key = (normalize_name(tables[left_table_index]), normalize_name(left_column))
            right_key = (normalize_name(tables[right_table_index]), normalize_name(right_column))
            left_id = column_key_to_id.get(left_key)
            right_id = column_key_to_id.get(right_key)
            if left_id is None or right_id is None:
                continue
            edges["foreign_key_forward"].append((left_id, right_id))
            edges["foreign_key_backward"].append((right_id, left_id))

    return node_count, edges, relations


def edges_to_tensors(edges, device):
    tensors = {}
    for relation, pairs in edges.items():
        if pairs:
            index = torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()
        else:
            index = torch.empty((2, 0), dtype=torch.long, device=device)
        tensors[relation] = index
    return tensors


def build_schema_cache(records, table_entries, hash_dim, device, include_same_table_edges):
    cache = {}
    for record in records:
        db_id = record["db_id"]
        if db_id in cache:
            continue
        table_entry = table_entries.get(db_id)
        node_count, edges, relations = build_schema_graph(
            record, table_entry, include_same_table_edges
        )
        x = np.zeros((node_count, hash_dim), dtype=np.float32)
        for item in record["schema_items"]:
            x[item["id"]] = hash_vector(schema_item_text(item), hash_dim)
        cache[db_id] = {
            "node_features": torch.tensor(x, dtype=torch.float32, device=device),
            "edges": edges_to_tensors(edges, device),
            "relations": relations,
        }
    return cache


def build_record_cache(records, hash_dim, device):
    cached = []
    for record in records:
        cached.append(
            {
                "record": record,
                "query_features": make_query_features(record, hash_dim, device),
                "labels": labels_for_record(record, device),
                "lexical_features": torch.tensor(
                    lexical_features(record), dtype=torch.float32, device=device
                ),
            }
        )
    return cached


def labels_for_record(record, device):
    labels = torch.zeros(len(record["schema_items"]), dtype=torch.float32, device=device)
    for idx in record["whole_sql_labels"]:
        if idx < len(labels):
            labels[idx] = 1.0
    return labels


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
        out = out / deg.clamp_min(1.0).unsqueeze(-1)
        return out

    def forward(self, h, edges):
        out = self.root(h)
        for relation in self.relations:
            if relation not in edges:
                continue
            agg = self.aggregate(h, edges[relation])
            out = out + self.rel_linears[relation](agg)
        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(out + h)


class RGTALayer(nn.Module):
    """Relational Graph Transformer Attention over typed schema edges.

    For each directed edge j -> i with relation r, node i attends to node j using
    relation-specific key/value biases. This keeps attention constrained by the
    schema graph rather than becoming full self-attention.
    """

    def __init__(self, hidden_dim, relations, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.relations = relations
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rel_k = nn.ParameterDict(
            {
                relation: nn.Parameter(torch.empty(hidden_dim))
                for relation in relations
            }
        )
        self.rel_v = nn.ParameterDict(
            {
                relation: nn.Parameter(torch.empty(hidden_dim))
                for relation in relations
            }
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
            rel_k = self.rel_k[relation].unsqueeze(0)
            rel_v = self.rel_v[relation].unsqueeze(0)
            edge_k = k[src] + rel_k
            edge_v = v[src] + rel_v
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

        weighted_messages = all_messages * attn.unsqueeze(-1)
        out = torch.zeros((node_count, self.hidden_dim), dtype=h.dtype, device=h.device)
        out.index_add_(0, all_dst, weighted_messages)
        out = self.out_proj(out)
        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(out + h)


class RGCNGrounder(nn.Module):
    def __init__(
        self,
        hash_dim,
        hidden_dim,
        relations,
        num_layers,
        dropout,
        lexical_dim,
        encoder_type="rgcn",
    ):
        super().__init__()
        self.encoder_type = encoder_type
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
        self.layers = nn.ModuleList([layer_cls(hidden_dim, relations, dropout) for _ in range(num_layers)])
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

    def forward(self, query_features, node_features, edges, lexical_features=None):
        q = self.query_input(query_features).squeeze(0)
        z = self.encode_schema(node_features, edges)
        q_matrix = q.unsqueeze(0).expand_as(z)
        pair = torch.cat([q_matrix, z, q_matrix * z, torch.abs(q_matrix - z)], dim=-1)
        if lexical_features is not None:
            pair = torch.cat([pair, lexical_features], dim=-1)
        return self.scorer(pair).squeeze(-1)


def make_query_features(record, hash_dim, device):
    return torch.tensor(
        hash_vector(query_text(record), hash_dim)[None, :],
        dtype=torch.float32,
        device=device,
    )


def train_model(model, record_cache, schema_cache, args, log_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for step, item in enumerate(record_cache, start=1):
                record = item["record"]
                cached = schema_cache[record["db_id"]]
                logits = model(
                    item["query_features"],
                    cached["node_features"],
                    cached["edges"],
                    item["lexical_features"] if args.use_lexical_features else None,
                )
                labels = item["labels"]
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                total_loss += float(loss.detach().cpu())
            row = {"epoch": epoch, "loss": total_loss / max(len(record_cache), 1)}
            log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(row)


def recall_at_k(gold_ids, predicted_ids):
    if not gold_ids:
        return None
    return len(set(gold_ids) & set(predicted_ids)) / len(set(gold_ids))


def precision_at_k(gold_ids, predicted_ids):
    if not predicted_ids:
        return None
    return len(set(gold_ids) & set(predicted_ids)) / len(predicted_ids)


def reciprocal_rank(gold_ids, ranked_ids):
    gold = set(gold_ids)
    if not gold:
        return None
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in gold:
            return 1.0 / rank
    return 0.0


def mean(values):
    vals = [x for x in values if x is not None]
    return sum(vals) / len(vals) if vals else 0.0


@torch.no_grad()
def evaluate(model, records, schema_cache, hash_dim, device, output_dir, split, top_k_examples):
    model.eval()
    predictions = []
    schema_recall_10, schema_recall_20, schema_recall_30 = [], [], []
    schema_precision_10, schema_mrr = [], []
    table_recall_3, table_recall_5 = [], []
    column_recall_10, column_recall_20 = [], []

    for record in records:
        cached = schema_cache[record["db_id"]]
        query_features = make_query_features(record, hash_dim, device)
        lex = torch.tensor(lexical_features(record), dtype=torch.float32, device=device)
        logits = model(
            query_features,
            cached["node_features"],
            cached["edges"],
            lex if getattr(model, "use_lexical_features", False) else None,
        )
        scores = logits.detach().cpu().numpy()
        ranked_ids = list(np.argsort(-scores))
        schema_items_by_id = {item["id"]: item for item in record["schema_items"]}
        gold_ids = record["whole_sql_labels"]
        gold_table_ids = [i for i in gold_ids if schema_items_by_id[i]["type"] == "table"]
        gold_column_ids = [i for i in gold_ids if schema_items_by_id[i]["type"] == "column"]
        ranked_table_ids = [i for i in ranked_ids if schema_items_by_id[i]["type"] == "table"]
        ranked_column_ids = [i for i in ranked_ids if schema_items_by_id[i]["type"] == "column"]

        schema_recall_10.append(recall_at_k(gold_ids, ranked_ids[:10]))
        schema_recall_20.append(recall_at_k(gold_ids, ranked_ids[:20]))
        schema_recall_30.append(recall_at_k(gold_ids, ranked_ids[:30]))
        schema_precision_10.append(precision_at_k(gold_ids, ranked_ids[:10]))
        schema_mrr.append(reciprocal_rank(gold_ids, ranked_ids))
        table_recall_3.append(recall_at_k(gold_table_ids, ranked_table_ids[:3]))
        table_recall_5.append(recall_at_k(gold_table_ids, ranked_table_ids[:5]))
        column_recall_10.append(recall_at_k(gold_column_ids, ranked_column_ids[:10]))
        column_recall_20.append(recall_at_k(gold_column_ids, ranked_column_ids[:20]))

        top_30 = [
            {
                "id": int(i),
                "type": schema_items_by_id[i]["type"],
                "name": schema_items_by_id[i]["name"],
                "score": float(scores[i]),
            }
            for i in ranked_ids[:30]
        ]
        predictions.append(
            {
                "db_id": record["db_id"],
                "question": record.get("question"),
                "evidence": record.get("evidence"),
                "gold_label_names": record.get("label_names", []),
                "top_30": top_30,
            }
        )

    metrics = {
        "split": split,
        "sample_count": len(records),
        "schema_recall@10": mean(schema_recall_10),
        "schema_recall@20": mean(schema_recall_20),
        "schema_recall@30": mean(schema_recall_30),
        "schema_precision@10": mean(schema_precision_10),
        "schema_mrr": mean(schema_mrr),
        "table_recall@3": mean(table_recall_3),
        "table_recall@5": mean(table_recall_5),
        "column_recall@10": mean(column_recall_10),
        "column_recall@20": mean(column_recall_20),
        "note": "PyTorch trainable R-GCN static schema grounder.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / f"rgcn_torch_{split}_predictions.jsonl", predictions)
    with (output_dir / f"rgcn_torch_{split}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / f"rgcn_torch_{split}_topk_examples.md").open("w", encoding="utf-8") as f:
        for idx, item in enumerate(predictions[:top_k_examples], start=1):
            f.write(f"## Example {idx}\n\n")
            f.write(f"**DB:** `{item['db_id']}`\n\n")
            f.write(f"**Question:** {item['question']}\n\n")
            if item.get("evidence"):
                f.write(f"**Evidence:** {item['evidence']}\n\n")
            f.write("**Gold labels:**\n\n")
            for name in item["gold_label_names"]:
                f.write(f"- `{name}`\n")
            f.write("\n**Top predictions:**\n\n")
            for row in item["top_30"][:15]:
                f.write(f"- `{row['name']}` ({row['type']}), score={row['score']:.4f}\n")
            f.write("\n")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird-dir", default="Data/BIRD")
    parser.add_argument(
        "--train-input",
        default="experiments/stage1_label_extraction/bird_train_grounding_labels.jsonl",
    )
    parser.add_argument(
        "--dev-input",
        default="experiments/stage1_label_extraction/bird_dev_grounding_labels.jsonl",
    )
    parser.add_argument("--output-dir", default="experiments/stage2_rgcn_torch_grounding")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--top-k-examples", type=int, default=20)
    parser.add_argument("--include-same-table-edges", action="store_true")
    parser.add_argument("--use-lexical-features", action="store_true")
    parser.add_argument("--encoder-type", choices=["rgcn", "rgta"], default="rgcn")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_records = read_jsonl(Path(args.train_input), args.train_limit)
    dev_records = read_jsonl(Path(args.dev_input), args.dev_limit)
    bird_dir = Path(args.bird_dir)
    train_tables = load_table_entries(
        bird_dir / "train_databases" / "train_databases" / "train_tables.json"
    )
    dev_tables = load_table_entries(bird_dir / "dev_tables.json")

    relations = list(RELATIONS)
    if args.include_same_table_edges:
        relations.append("same_table_column")

    print("Building schema graph caches...")
    train_schema_cache = build_schema_cache(
        train_records, train_tables, args.hash_dim, device, args.include_same_table_edges
    )
    dev_schema_cache = build_schema_cache(
        dev_records, dev_tables, args.hash_dim, device, args.include_same_table_edges
    )
    print("Building record caches...")
    train_record_cache = build_record_cache(train_records, args.hash_dim, device)

    model = RGCNGrounder(
        args.hash_dim,
        args.hidden_dim,
        relations,
        args.num_layers,
        args.dropout,
        lexical_dim=6 if args.use_lexical_features else 0,
        encoder_type=args.encoder_type,
    ).to(device)
    model.use_lexical_features = args.use_lexical_features

    config = vars(args).copy()
    config["torch_version"] = torch.__version__
    config["relation_types"] = relations
    with (output_dir / "rgcn_torch_train_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    train_model(
        model,
        train_record_cache,
        train_schema_cache,
        args,
        output_dir / "rgcn_torch_train_log.jsonl",
    )
    metrics = evaluate(
        model,
        dev_records,
        dev_schema_cache,
        args.hash_dim,
        device,
        output_dir,
        "dev",
        args.top_k_examples,
    )
    torch.save(model.state_dict(), output_dir / "rgcn_torch_model.pt")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
