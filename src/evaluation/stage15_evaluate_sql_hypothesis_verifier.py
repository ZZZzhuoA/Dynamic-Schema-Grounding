"""Evaluate grouped SQL-hypothesis ranking predictions."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def ranking_metrics(rows):
    reciprocal_ranks = []
    hits_at_1 = 0
    pairwise_correct = 0
    pairwise_total = 0
    margins = []
    hardest_margins = []
    schema_control_gains = []
    top1_error_count = 0
    top1_error_by_corruption = defaultdict(int)
    by_corruption = defaultdict(
        lambda: {"correct": 0, "total": 0, "margin_sum": 0.0, "margins": []}
    )
    for row in rows:
        control = row.get("schema_control")
        if control is not None and control.get("positive_gain") is not None:
            schema_control_gains.append(float(control["positive_gain"]))
        candidates = row.get("candidates", [])
        positives = [candidate for candidate in candidates if int(candidate.get("label", 0)) == 1]
        if len(positives) != 1:
            raise ValueError(
                f"record_index={row.get('record_index')} has {len(positives)} positives"
            )
        positive = positives[0]
        positive_score = float(positive["score"])
        negative_scores = [
            float(candidate["score"])
            for candidate in candidates
            if int(candidate.get("label", 0)) == 0
        ]
        if negative_scores:
            hardest_negative_score = max(negative_scores)
            hardest_margins.append(positive_score - hardest_negative_score)
            if hardest_negative_score >= positive_score:
                top1_error_count += 1
                hardest_candidates = [
                    candidate
                    for candidate in candidates
                    if int(candidate.get("label", 0)) == 0
                    and float(candidate["score"]) == hardest_negative_score
                ]
                top1_error_by_corruption[
                    hardest_candidates[0].get("corruption_type", "unknown")
                ] += 1
        # Average rank for ties prevents a fixed candidate order from granting
        # an untrained all-equal scorer artificial Hits@1/MRR credit.
        greater = sum(score > positive_score for score in negative_scores)
        equal = sum(score == positive_score for score in negative_scores)
        rank = 1.0 + greater + 0.5 * equal
        reciprocal_ranks.append(1.0 / rank)
        hits_at_1 += int(greater == 0 and equal == 0)
        for candidate in candidates:
            if int(candidate.get("label", 0)) == 1:
                continue
            margin = positive_score - float(candidate["score"])
            corruption = candidate.get("corruption_type", "unknown")
            correct = int(margin > 0.0)
            pairwise_correct += correct
            pairwise_total += 1
            margins.append(margin)
            bucket = by_corruption[corruption]
            bucket["correct"] += correct
            bucket["total"] += 1
            bucket["margin_sum"] += margin
            bucket["margins"].append(margin)
    sample_count = len(rows)
    return {
        "sample_count": sample_count,
        "mrr": sum(reciprocal_ranks) / sample_count if sample_count else 0.0,
        "hits@1": hits_at_1 / sample_count if sample_count else 0.0,
        "pairwise_accuracy": pairwise_correct / pairwise_total if pairwise_total else 0.0,
        "mean_positive_margin": sum(margins) / len(margins) if margins else 0.0,
        "median_positive_margin": quantile(margins, 0.5),
        "positive_margin_p10": quantile(margins, 0.10),
        "positive_margin_p25": quantile(margins, 0.25),
        "mean_hardest_margin": (
            sum(hardest_margins) / len(hardest_margins) if hardest_margins else None
        ),
        "median_hardest_margin": quantile(hardest_margins, 0.5),
        "hardest_margin_p10": quantile(hardest_margins, 0.10),
        "top1_error_count": top1_error_count,
        "top1_error_rate": top1_error_count / sample_count if sample_count else 0.0,
        "top1_error_by_corruption": {
            name: {
                "count": count,
                "share_of_top1_errors": count / top1_error_count if top1_error_count else 0.0,
            }
            for name, count in sorted(top1_error_by_corruption.items())
        },
        "pairwise_count": pairwise_total,
        "schema_control_positive_gain": (
            sum(schema_control_gains) / len(schema_control_gains)
            if schema_control_gains else None
        ),
        "schema_control_win_rate": (
            sum(value > 0.0 for value in schema_control_gains) / len(schema_control_gains)
            if schema_control_gains else None
        ),
        "by_corruption": {
            name: {
                "count": values["total"],
                "pairwise_accuracy": values["correct"] / values["total"] if values["total"] else 0.0,
                "mean_positive_margin": values["margin_sum"] / values["total"] if values["total"] else 0.0,
                "median_positive_margin": quantile(values["margins"], 0.5),
                "positive_margin_p10": quantile(values["margins"], 0.10),
            }
            for name, values in sorted(by_corruption.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", required=True)
    parser.add_argument("--output-file")
    args = parser.parse_args()
    metrics = ranking_metrics(read_jsonl(args.prediction_file))
    metrics["prediction_file"] = args.prediction_file
    if args.output_file:
        path = Path(args.output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
