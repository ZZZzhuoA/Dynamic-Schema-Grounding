import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = text.replace("`", " ").replace('"', " ").replace("[", " ").replace("]", " ")
    text = text.replace("_", " ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    text = normalize_text(text)
    if not text:
        return []
    return [token for token in text.split() if token]


def schema_item_text(item):
    if item["type"] == "table":
        return item.get("name", "")
    parts = [
        item.get("table", ""),
        item.get("column", ""),
        item.get("data_type", "") or "",
        item.get("name", ""),
    ]
    return " ".join(parts)


def item_table_name(item):
    if item["type"] == "table":
        return item.get("name", "")
    return item.get("table", "")


def item_column_name(item):
    if item["type"] == "column":
        return item.get("column", "")
    return ""


def compute_idf(records):
    doc_freq = Counter()
    doc_count = 0
    for record in records:
        for item in record["schema_items"]:
            tokens = set(tokenize(schema_item_text(item)))
            if not tokens:
                continue
            doc_count += 1
            doc_freq.update(tokens)
    return {
        token: math.log((doc_count + 1) / (freq + 1)) + 1.0
        for token, freq in doc_freq.items()
    }


def overlap_score(query_tokens, item_tokens, idf):
    query_set = set(query_tokens)
    item_set = set(item_tokens)
    if not query_set or not item_set:
        return 0.0
    overlap = query_set & item_set
    weighted_overlap = sum(idf.get(token, 1.0) for token in overlap)
    weighted_item = sum(idf.get(token, 1.0) for token in item_set)
    weighted_query = sum(idf.get(token, 1.0) for token in query_set)
    recall_like = weighted_overlap / weighted_item if weighted_item else 0.0
    precision_like = weighted_overlap / weighted_query if weighted_query else 0.0
    return 2.0 * recall_like + precision_like


def phrase_bonus_from_normalized(normalized_query_with_padding, phrase):
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return 0.0
    if f" {normalized_phrase} " in normalized_query_with_padding:
        return 2.0
    phrase_tokens = normalized_phrase.split()
    if len(phrase_tokens) >= 2 and all(
        f" {token} " in normalized_query_with_padding for token in phrase_tokens
    ):
        return 0.6
    return 0.0


def score_item(query_features, item_features, idf):
    score = 0.0
    score += overlap_score(query_features["query_tokens"], item_features["tokens"], idf)
    score += 0.6 * overlap_score(query_features["evidence_tokens"], item_features["tokens"], idf)
    score += 0.3 * overlap_score(query_features["question_tokens"], item_features["tokens"], idf)

    score += phrase_bonus_from_normalized(
        query_features["normalized_query_with_padding"], item_features["table_name"]
    )
    score += 1.5 * phrase_bonus_from_normalized(
        query_features["normalized_query_with_padding"], item_features["column_name"]
    )

    if item_features["type"] == "table":
        # Tables are often not mentioned directly, so keep them competitive when
        # their name overlaps with the question/evidence.
        score += 0.2 * overlap_score(
            query_features["query_tokens"], item_features["table_tokens"], idf
        )
    else:
        # If a column is a likely hit, its owning table should also be ranked
        # reasonably in table-only metrics. The table itself is handled by table
        # items, but this keeps column ranking focused on the column phrase.
        score += 0.2 * overlap_score(
            query_features["query_tokens"], item_features["column_tokens"], idf
        )

    return score


def build_query_features(record):
    question = record.get("question") or ""
    evidence = record.get("evidence") or ""
    query_text = f"{question} {evidence}"
    return {
        "question_tokens": tokenize(question),
        "evidence_tokens": tokenize(evidence),
        "query_tokens": tokenize(query_text),
        "normalized_query_with_padding": f" {normalize_text(query_text)} ",
    }


def build_item_features(item):
    table_name = item_table_name(item)
    column_name = item_column_name(item)
    return {
        "id": item["id"],
        "type": item["type"],
        "name": item["name"],
        "tokens": tokenize(schema_item_text(item)),
        "table_name": table_name,
        "column_name": column_name,
        "table_tokens": tokenize(table_name),
        "column_tokens": tokenize(column_name),
    }


def ranked_items(record, idf):
    query_features = build_query_features(record)
    scored = []
    for item in record["schema_items"]:
        item_features = build_item_features(item)
        score = score_item(query_features, item_features, idf)
        scored.append(
            {
                "id": item["id"],
                "type": item["type"],
                "name": item["name"],
                "score": score,
            }
        )
    scored.sort(key=lambda row: (-row["score"], row["type"] != "table", row["name"]))
    return scored


def recall_at_k(gold_ids, predicted_ids):
    if not gold_ids:
        return None
    predicted = set(predicted_ids)
    return len(set(gold_ids) & predicted) / len(set(gold_ids))


def precision_at_k(gold_ids, predicted_ids):
    if not predicted_ids:
        return None
    return len(set(gold_ids) & set(predicted_ids)) / len(predicted_ids)


def reciprocal_rank(gold_ids, ranked_ids):
    gold = set(gold_ids)
    if not gold:
        return None
    for index, item_id in enumerate(ranked_ids, start=1):
        if item_id in gold:
            return 1.0 / index
    return 0.0


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def evaluate(records, output_dir: Path, split: str, top_k_examples: int, output_top_k: int):
    idf = compute_idf(records)
    predictions = []

    schema_recall_10 = []
    schema_recall_20 = []
    schema_recall_30 = []
    schema_precision_10 = []
    schema_mrr = []
    table_recall_3 = []
    table_recall_5 = []
    column_recall_10 = []
    column_recall_20 = []

    for record in records:
        ranked = ranked_items(record, idf)
        ranked_ids = [item["id"] for item in ranked]
        gold_ids = record["whole_sql_labels"]
        schema_items_by_id = {item["id"]: item for item in record["schema_items"]}
        gold_table_ids = [
            item_id for item_id in gold_ids if schema_items_by_id[item_id]["type"] == "table"
        ]
        gold_column_ids = [
            item_id for item_id in gold_ids if schema_items_by_id[item_id]["type"] == "column"
        ]
        ranked_table_ids = [item["id"] for item in ranked if item["type"] == "table"]
        ranked_column_ids = [item["id"] for item in ranked if item["type"] == "column"]

        schema_recall_10.append(recall_at_k(gold_ids, ranked_ids[:10]))
        schema_recall_20.append(recall_at_k(gold_ids, ranked_ids[:20]))
        schema_recall_30.append(recall_at_k(gold_ids, ranked_ids[:30]))
        schema_precision_10.append(precision_at_k(gold_ids, ranked_ids[:10]))
        schema_mrr.append(reciprocal_rank(gold_ids, ranked_ids))
        table_recall_3.append(recall_at_k(gold_table_ids, ranked_table_ids[:3]))
        table_recall_5.append(recall_at_k(gold_table_ids, ranked_table_ids[:5]))
        column_recall_10.append(recall_at_k(gold_column_ids, ranked_column_ids[:10]))
        column_recall_20.append(recall_at_k(gold_column_ids, ranked_column_ids[:20]))

        prediction = {
            "db_id": record["db_id"],
            "question": record.get("question"),
            "evidence": record.get("evidence"),
            "gold_label_names": record.get("label_names", []),
            "top_30": ranked[:30],
        }
        if output_top_k != 30:
            prediction[f"top_{output_top_k}"] = ranked[:output_top_k]
        predictions.append(prediction)

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
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / f"lexical_{split}_predictions.jsonl", predictions)
    with (output_dir / f"lexical_{split}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    examples = predictions[:top_k_examples]
    with (output_dir / f"lexical_{split}_topk_examples.md").open("w", encoding="utf-8") as f:
        for index, item in enumerate(examples, start=1):
            f.write(f"## Example {index}\n\n")
            f.write(f"**DB:** `{item['db_id']}`\n\n")
            f.write(f"**Question:** {item['question']}\n\n")
            if item.get("evidence"):
                f.write(f"**Evidence:** {item['evidence']}\n\n")
            f.write("**Gold labels:**\n\n")
            for name in item["gold_label_names"]:
                f.write(f"- `{name}`\n")
            f.write("\n**Top predictions:**\n\n")
            top_key = f"top_{output_top_k}" if f"top_{output_top_k}" in item else "top_30"
            for row in item[top_key][:15]:
                f.write(f"- `{row['name']}` ({row['type']}), score={row['score']:.4f}\n")
            f.write("\n")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="experiments/stage1_label_extraction/bird_dev_grounding_labels.jsonl",
    )
    parser.add_argument("--output-dir", default="experiments/stage2_static_grounding")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k-examples", type=int, default=20)
    parser.add_argument("--output-top-k", type=int, default=30)
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    if args.limit is not None:
        records = records[: args.limit]
    metrics = evaluate(records, Path(args.output_dir), args.split, args.top_k_examples, args.output_top_k)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
