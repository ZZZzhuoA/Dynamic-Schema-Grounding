import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


RELATIONS = [
    "self_loop",
    "table_to_column",
    "column_to_table",
    "same_table_column",
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


def schema_item_text(item):
    if item["type"] == "table":
        return f"table: {item.get('name', '')}"
    return (
        f"column: {item.get('table', '')}.{item.get('column', '')}; "
        f"type: {item.get('data_type', '') or ''}"
    )


def query_text(record):
    return f"{record.get('question') or ''} {record.get('evidence') or ''}"


def load_table_entries(path: Path):
    return {entry["db_id"]: entry for entry in read_json(path)}


def build_schema_graph(record, table_entry, include_same_table_edges=False):
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

    edges = {relation: [] for relation in RELATIONS}
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

    return node_count, edges


class FixedRGCNEncoder:
    """R-GCN-style relation-specific message passing with fixed projections.

    This avoids PyTorch/DGL/PyG dependencies while still injecting database graph
    structure into schema node representations. The trainable part is the final
    grounding scorer.
    """

    def __init__(self, input_dim, hidden_dim, seed=13):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.w1 = {
            rel: rng.normal(0, 1.0 / math.sqrt(input_dim), size=(input_dim, hidden_dim)).astype(
                np.float32
            )
            for rel in RELATIONS
        }
        self.w2 = {
            rel: rng.normal(0, 1.0 / math.sqrt(hidden_dim), size=(hidden_dim, hidden_dim)).astype(
                np.float32
            )
            for rel in RELATIONS
        }

    @staticmethod
    def aggregate(features, node_count, edges):
        output = np.zeros((node_count, features.shape[1]), dtype=np.float32)
        degree = np.zeros(node_count, dtype=np.float32)
        for src, dst in edges:
            output[dst] += features[src]
            degree[dst] += 1.0
        mask = degree > 0
        output[mask] /= degree[mask, None]
        return output

    def layer(self, features, edges_by_relation, weights):
        node_count = features.shape[0]
        output = np.zeros((node_count, next(iter(weights.values())).shape[1]), dtype=np.float32)
        for relation, edges in edges_by_relation.items():
            if not edges:
                continue
            agg = self.aggregate(features, node_count, edges)
            output += agg @ weights[relation]
        output = np.maximum(output, 0.0)
        norm = np.linalg.norm(output, axis=1, keepdims=True)
        output = np.divide(output, np.maximum(norm, 1e-8))
        return output

    def encode(self, node_features, edges_by_relation):
        hidden = self.layer(node_features, edges_by_relation, self.w1)
        hidden = self.layer(hidden, edges_by_relation, self.w2)
        return hidden


def build_features(record, graph_encoder, table_entry, hash_dim, include_same_table_edges=False):
    node_count, edges = build_schema_graph(record, table_entry, include_same_table_edges)
    x = np.zeros((node_count, hash_dim), dtype=np.float32)
    for item in record["schema_items"]:
        x[item["id"]] = hash_vector(schema_item_text(item), hash_dim)
    z = graph_encoder.encode(x, edges)
    q = hash_vector(query_text(record), graph_encoder.hidden_dim)
    return q, z


def precompute_examples(records, table_entries, graph_encoder, hash_dim, include_same_table_edges):
    examples = []
    schema_cache = {}
    for record in records:
        db_id = record["db_id"]
        if db_id not in schema_cache:
            table_entry = table_entries.get(db_id)
            node_count, edges = build_schema_graph(record, table_entry, include_same_table_edges)
            x = np.zeros((node_count, hash_dim), dtype=np.float32)
            for item in record["schema_items"]:
                x[item["id"]] = hash_vector(schema_item_text(item), hash_dim)
            schema_cache[db_id] = graph_encoder.encode(x, edges)
        z = schema_cache[db_id]
        q = hash_vector(query_text(record), graph_encoder.hidden_dim)
        features = pair_features(q, z)
        labels = labels_for_record(record)
        examples.append((record, features, labels))
    return examples


def pair_features(q, z):
    q_matrix = np.repeat(q[None, :], z.shape[0], axis=0)
    return np.concatenate([q_matrix, z, q_matrix * z, np.abs(q_matrix - z)], axis=1)


def sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


class LogisticScorer:
    def __init__(self, dim, seed=7):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, size=(dim,)).astype(np.float32)
        self.b = np.float32(0.0)

    def predict_logits(self, features):
        return features @ self.w + self.b

    def train_step(self, features, labels, lr, pos_weight):
        logits = self.predict_logits(features)
        probs = sigmoid(logits)
        weights = np.where(labels > 0.5, pos_weight, 1.0).astype(np.float32)
        grad = (probs - labels) * weights / max(len(labels), 1)
        grad_w = features.T @ grad
        grad_b = grad.sum()
        self.w -= lr * grad_w.astype(np.float32)
        self.b -= np.float32(lr * grad_b)
        loss = -(
            weights
            * (labels * np.log(probs + 1e-8) + (1 - labels) * np.log(1 - probs + 1e-8))
        ).mean()
        return float(loss)


def labels_for_record(record):
    labels = np.zeros(len(record["schema_items"]), dtype=np.float32)
    for idx in record["whole_sql_labels"]:
        if idx < len(labels):
            labels[idx] = 1.0
    return labels


def train(precomputed_examples, scorer, epochs, lr, pos_weight, log_path):
    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            count = 0
            for _, features, labels in precomputed_examples:
                loss = scorer.train_step(features, labels, lr, pos_weight)
                total_loss += loss
                count += 1
            row = {"epoch": epoch, "loss": total_loss / max(count, 1)}
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


def evaluate(precomputed_examples, scorer, output_dir, split, top_k_examples):
    predictions = []
    schema_recall_10, schema_recall_20, schema_recall_30 = [], [], []
    schema_precision_10, schema_mrr = [], []
    table_recall_3, table_recall_5 = [], []
    column_recall_10, column_recall_20 = [], []

    for record, features, _ in precomputed_examples:
        logits = scorer.predict_logits(features)
        ranked_ids = list(np.argsort(-logits))
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
                "score": float(logits[i]),
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
        "sample_count": len(precomputed_examples),
        "schema_recall@10": mean(schema_recall_10),
        "schema_recall@20": mean(schema_recall_20),
        "schema_recall@30": mean(schema_recall_30),
        "schema_precision@10": mean(schema_precision_10),
        "schema_mrr": mean(schema_mrr),
        "table_recall@3": mean(table_recall_3),
        "table_recall@5": mean(table_recall_5),
        "column_recall@10": mean(column_recall_10),
        "column_recall@20": mean(column_recall_20),
        "note": "NumPy fixed-projection R-GCN-style encoder; scorer is trained.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / f"rgcn_{split}_predictions.jsonl", predictions)
    with (output_dir / f"rgcn_{split}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / f"rgcn_{split}_topk_examples.md").open("w", encoding="utf-8") as f:
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
    parser.add_argument("--output-dir", default="experiments/stage2_rgcn_grounding")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--top-k-examples", type=int, default=20)
    parser.add_argument("--include-same-table-edges", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_jsonl(Path(args.train_input), args.train_limit)
    dev_records = read_jsonl(Path(args.dev_input), args.dev_limit)
    bird_dir = Path(args.bird_dir)
    train_tables = load_table_entries(bird_dir / "train_databases" / "train_databases" / "train_tables.json")
    dev_tables = load_table_entries(bird_dir / "dev_tables.json")

    graph_encoder = FixedRGCNEncoder(args.hash_dim, args.hidden_dim)
    scorer = LogisticScorer(args.hidden_dim * 4)

    config = vars(args).copy()
    config["relation_types"] = RELATIONS
    config["encoder_note"] = "Fixed-projection NumPy R-GCN-style encoder; trainable logistic scorer."
    with (output_dir / "rgcn_train_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("Precomputing train graph features...")
    train_examples = precompute_examples(
        train_records,
        train_tables,
        graph_encoder,
        args.hash_dim,
        args.include_same_table_edges,
    )
    print("Precomputing dev graph features...")
    dev_examples = precompute_examples(
        dev_records,
        dev_tables,
        graph_encoder,
        args.hash_dim,
        args.include_same_table_edges,
    )

    train(
        train_examples,
        scorer,
        args.epochs,
        args.lr,
        args.pos_weight,
        output_dir / "rgcn_train_log.jsonl",
    )
    metrics = evaluate(
        dev_examples,
        scorer,
        output_dir,
        "dev",
        args.top_k_examples,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
