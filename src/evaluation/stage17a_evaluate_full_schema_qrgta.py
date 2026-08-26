"""Recompute Stage 17-A0 metrics from leakage-free prediction rows."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.stage17a_train_full_schema_qrgta import (  # noqa: E402
    ranking_metrics,
    read_jsonl,
    write_json,
)


def prediction_key(record):
    question_id = record.get("question_id")
    if question_id is None:
        return None
    return str(record.get("db_id")), str(question_id)


def align_predictions(predictions, labels):
    by_key = {}
    by_record = {}
    for prediction in predictions:
        record_index = int(prediction["record_index"])
        if record_index in by_record:
            raise ValueError(f"Duplicate prediction record_index={record_index}")
        by_record[record_index] = prediction
        key = prediction_key(prediction)
        if key is not None:
            if key in by_key:
                raise ValueError(f"Duplicate prediction key={key}")
            by_key[key] = prediction

    examples = []
    rankings = {}
    skipped_empty = 0
    used_records = set()
    for label_index, label in enumerate(labels):
        key = prediction_key(label)
        prediction = by_key.get(key) if key is not None else None
        if prediction is None:
            prediction = by_record.get(label_index)
        if prediction is None:
            continue
        record_index = int(prediction["record_index"])
        if record_index in used_records:
            raise ValueError(f"Prediction record_index={record_index} aligned twice")
        used_records.add(record_index)
        if str(prediction.get("db_id")) != str(label.get("db_id")):
            raise ValueError(f"Database mismatch for record_index={record_index}")
        schema_items = label.get("schema_items", [])
        ranked = prediction.get("ranked_schema", [])
        if int(prediction.get("schema_node_count", -1)) != len(schema_items):
            raise ValueError(
                f"Schema count mismatch for record_index={record_index}: "
                f"prediction={prediction.get('schema_node_count')} label={len(schema_items)}"
            )
        expected = {int(item["id"]): str(item["name"]) for item in schema_items}
        observed = {}
        ranked_ids = []
        for rank, row in enumerate(ranked, start=1):
            item_id = int(row["schema_item_id"])
            if int(row.get("rank", rank)) != rank:
                raise ValueError(f"Non-contiguous rank at record_index={record_index}")
            if item_id in observed:
                raise ValueError(
                    f"Duplicate schema item {item_id} at record_index={record_index}"
                )
            observed[item_id] = str(row.get("name"))
            ranked_ids.append(item_id)
        if expected != observed:
            missing = sorted(set(expected) - set(observed))
            foreign = sorted(set(observed) - set(expected))
            name_mismatch = sorted(
                item_id
                for item_id in set(expected) & set(observed)
                if expected[item_id] != observed[item_id]
            )
            raise ValueError(
                f"Schema identity mismatch for record_index={record_index}: "
                f"missing={missing[:10]} foreign={foreign[:10]} "
                f"name_mismatch={name_mismatch[:10]}"
            )
        gold_ids = sorted({int(value) for value in label.get("whole_sql_labels", [])})
        if not gold_ids:
            skipped_empty += 1
            continue
        examples.append(
            {
                "record_index": record_index,
                "db_id": label.get("db_id"),
                "nodes": schema_items,
                "gold_ids": gold_ids,
            }
        )
        rankings[record_index] = ranked_ids
    if len(used_records) != len(predictions):
        unused = sorted(set(by_record) - used_records)
        raise ValueError(f"Predictions could not be aligned: {unused[:10]}")
    return examples, rankings, skipped_empty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    predictions = read_jsonl(args.prediction_file)
    labels = read_jsonl(args.label_file)
    examples, rankings, skipped_empty = align_predictions(predictions, labels)
    metrics = ranking_metrics(examples, rankings, split="recomputed_dev")
    metrics.update(
        {
            "prediction_file": args.prediction_file,
            "label_file": args.label_file,
            "prediction_count": len(predictions),
            "skipped_empty_gold_count": skipped_empty,
        }
    )
    write_json(args.output_file, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
