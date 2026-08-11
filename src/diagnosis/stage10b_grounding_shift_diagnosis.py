import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def top_key(row):
    keys = [key for key in row if key.startswith("top_")]
    return max(keys, key=lambda key: int(key.split("_", 1)[1])) if keys else None


def summarize(rows):
    aggregate = defaultdict(list)
    relation_metrics = defaultdict(lambda: defaultdict(list))
    record_ids = set()
    for row in rows:
        record_ids.add(int(row["record_index"]))
        relation = str(row.get("relation_type") or row.get("clause"))
        candidates = row.get(top_key(row), []) if top_key(row) else []
        ranked = [int(item["id"]) for item in candidates]
        scores = [float(item.get("score", 0.0)) for item in candidates]
        gold = {int(item_id) for item_id in row.get("gold_label_ids", [])}
        best_gold_rank = next(
            (rank for rank, item_id in enumerate(ranked, start=1) if item_id in gold),
            None,
        )
        values = {
            "top1_score": scores[0] if scores else 0.0,
            "top1_top2_margin": scores[0] - scores[1] if len(scores) > 1 else 0.0,
            "score_std": (
                math.sqrt(mean([(score - mean(scores)) ** 2 for score in scores]))
                if scores
                else 0.0
            ),
            "gold_recall@5": len(gold & set(ranked[:5])) / len(gold) if gold else 1.0,
            "gold_recall@10": len(gold & set(ranked[:10])) / len(gold) if gold else 1.0,
            "gold_recall@20": len(gold & set(ranked[:20])) / len(gold) if gold else 1.0,
            "gold_mrr": 1.0 / best_gold_rank if best_gold_rank else 0.0,
        }
        for name, value in values.items():
            aggregate[name].append(value)
            relation_metrics[relation][name].append(value)
    return {
        "relation_prediction_count": len(rows),
        "base_record_count": len(record_ids),
        "overall": {name: mean(values) for name, values in sorted(aggregate.items())},
        "by_relation": {
            relation: {
                "count": len(next(iter(metrics.values()), [])),
                **{name: mean(values) for name, values in sorted(metrics.items())},
            }
            for relation, metrics in sorted(relation_metrics.items())
        },
    }


def nested_delta(left, right):
    result = {}
    for key in sorted(set(left) & set(right)):
        if isinstance(left[key], dict) and isinstance(right[key], dict):
            result[key] = nested_delta(left[key], right[key])
        elif isinstance(left[key], (int, float)) and isinstance(right[key], (int, float)):
            result[key] = right[key] - left[key]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare in-sample, OOF, and dev Stage 8G prediction distributions."
    )
    parser.add_argument("--oof-predictions", required=True)
    parser.add_argument("--in-sample-predictions", default=None)
    parser.add_argument("--dev-predictions", default=None)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    summaries = {"oof": summarize(read_jsonl(args.oof_predictions))}
    if args.in_sample_predictions:
        summaries["in_sample"] = summarize(read_jsonl(args.in_sample_predictions))
    if args.dev_predictions:
        summaries["dev"] = summarize(read_jsonl(args.dev_predictions))
    comparisons = {}
    if "in_sample" in summaries:
        comparisons["in_sample_to_oof"] = nested_delta(
            summaries["in_sample"], summaries["oof"]
        )
    if "dev" in summaries:
        comparisons["oof_to_dev"] = nested_delta(summaries["oof"], summaries["dev"])
    report = {
        "config": vars(args),
        "summaries": summaries,
        "comparisons": comparisons,
        "delta_definition": "right_minus_left",
        "interpretation": (
            "OOF is useful when its ranking quality and confidence statistics are closer "
            "to unseen dev than in-sample training predictions are. This report is diagnostic "
            "only and is never consumed as model input."
        ),
    }
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nOutput written to: {output_file}")


if __name__ == "__main__":
    main()
