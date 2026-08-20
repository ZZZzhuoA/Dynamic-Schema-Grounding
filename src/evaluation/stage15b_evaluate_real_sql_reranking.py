"""Evaluate real-candidate reranking without using labels during selection."""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def zscores(values):
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    return [(value - mean) / scale if scale > 1e-12 else 0.0 for value in values]


def llm_prior(candidate):
    value = candidate.get("mean_logprob")
    return float(value) if value is not None else -float(candidate.get("llm_rank", 0))


def first_matching(candidates, predicate):
    return next((index for index, candidate in enumerate(candidates) if predicate(candidate)), None)


def highest(candidates, predicate, key):
    eligible = [(index, candidate) for index, candidate in enumerate(candidates) if predicate(candidate)]
    return max(eligible, key=lambda pair: key(pair[1], pair[0]))[0] if eligible else None


def verifier_value(candidate, score_field):
    if score_field == "verifier_score":
        return candidate.get("verifier_score")
    return candidate.get("control_scores", {}).get(score_field)


def baseline_index(candidates):
    return next(
        (index for index, candidate in enumerate(candidates) if candidate.get("generation_mode") == "greedy"),
        0 if candidates else None,
    )


def score_selections(candidates, alphas, score_field, prefix=""):
    selected = {
        f"{prefix}verifier": highest(
            candidates,
            lambda value: verifier_value(value, score_field) is not None,
            lambda value, _: float(verifier_value(value, score_field)),
        ),
        f"{prefix}verifier_execution_filter": highest(
            candidates,
            lambda value: value.get("execution_ok") and verifier_value(value, score_field) is not None,
            lambda value, _: float(verifier_value(value, score_field)),
        ),
    }
    eligible = [
        (index, candidate) for index, candidate in enumerate(candidates)
        if candidate.get("execution_ok") and verifier_value(candidate, score_field) is not None
    ]
    if eligible:
        verifier_z = zscores([float(verifier_value(candidate, score_field)) for _, candidate in eligible])
        llm_z = zscores([llm_prior(candidate) for _, candidate in eligible])
        for alpha in alphas:
            best = max(
                range(len(eligible)),
                key=lambda local: alpha * verifier_z[local] + (1.0 - alpha) * llm_z[local],
            )
            selected[f"{prefix}hybrid_{alpha:g}"] = eligible[best][0]
    else:
        for alpha in alphas:
            selected[f"{prefix}hybrid_{alpha:g}"] = None
    return selected


def selections(row, alphas):
    candidates = row.get("candidates", [])
    baseline = baseline_index(candidates)
    selected = {
        "llm_top1": baseline,
        "execution_filter": first_matching(candidates, lambda value: value.get("execution_ok")),
    }
    selected.update(score_selections(candidates, alphas, "verifier_score"))
    control_modes = sorted(
        {
            mode
            for candidate in candidates
            for mode in candidate.get("control_scores", {})
        }
    )
    for mode in control_modes:
        selected.update(
            score_selections(
                candidates, alphas, mode, prefix=f"control_{mode}_"
            )
        )
    return selected


def evaluate(rows, alphas):
    methods = defaultdict(lambda: {"correct": 0, "selected": 0, "changed": 0, "recovered": 0, "regressed": 0})
    oracle_count = baseline_correct = executable_oracle = 0
    candidate_count = parsed = executable = 0
    oracle_at = defaultdict(int)
    details = []
    for row in rows:
        candidates = row.get("candidates", [])
        candidate_count += len(candidates)
        parsed += sum(bool(candidate.get("parse_ok")) for candidate in candidates)
        executable += sum(bool(candidate.get("execution_ok")) for candidate in candidates)
        labels = [bool(candidate.get("execution_correct")) for candidate in candidates]
        oracle = any(labels)
        baseline_candidate_index = baseline_index(candidates)
        baseline = labels[baseline_candidate_index] if baseline_candidate_index is not None else False
        oracle_count += int(oracle)
        baseline_correct += int(baseline)
        executable_oracle += int(any(c.get("execution_ok") and c.get("execution_correct") for c in candidates))
        for k in (1, 2, 4, 8):
            oracle_at[k] += int(any(labels[:k]))
        chosen = selections(row, alphas)
        detail = {"record_index": row.get("record_index"), "db_id": row.get("db_id"), "oracle_correct": oracle, "selections": {}}
        for method, index in chosen.items():
            correct = bool(index is not None and labels[index])
            bucket = methods[method]
            bucket["selected"] += int(index is not None)
            bucket["correct"] += int(correct)
            bucket["changed"] += int(index is not None and index != baseline_candidate_index)
            bucket["recovered"] += int(oracle and not baseline and correct)
            bucket["regressed"] += int(baseline and not correct)
            detail["selections"][method] = {
                "candidate_index": index,
                "candidate_id": candidates[index].get("candidate_id") if index is not None else None,
                "execution_correct": correct,
            }
        details.append(detail)
    total = len(rows)
    baseline_wrong_oracle = sum(
        bool(any(c.get("execution_correct") for c in row.get("candidates", [])))
        and not bool(
            row.get("candidates")
            and row["candidates"][baseline_index(row["candidates"])].get("execution_correct")
        )
        for row in rows
    )
    metrics = {
        "sample_count": total,
        "candidate_count": candidate_count,
        "mean_candidate_count": candidate_count / total if total else 0.0,
        "candidate_parse_rate": parsed / candidate_count if candidate_count else 0.0,
        "candidate_execution_success_rate": executable / candidate_count if candidate_count else 0.0,
        "oracle_ex_at_k": {str(k): oracle_at[k] / total if total else 0.0 for k in (1, 2, 4, 8)},
        "oracle_ex_available": oracle_count / total if total else 0.0,
        "executable_oracle_rate": executable_oracle / total if total else 0.0,
        "baseline_wrong_but_oracle_available_count": baseline_wrong_oracle,
        "methods": {},
    }
    for method, values in methods.items():
        metrics["methods"][method] = {
            "execution_accuracy": values["correct"] / total if total else 0.0,
            "selection_rate": values["selected"] / total if total else 0.0,
            "changed_from_llm_top1_rate": values["changed"] / total if total else 0.0,
            "recovered_count": values["recovered"],
            "recovery_rate_on_available_errors": values["recovered"] / baseline_wrong_oracle if baseline_wrong_oracle else 0.0,
            "regressed_count": values["regressed"],
            "net_gain_over_llm_top1": values["correct"] - baseline_correct,
        }
    return metrics, details


def calibration_split(rows, fraction, seed):
    if fraction <= 0.0:
        return [], list(rows)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    calibration_count = max(1, min(len(rows) - 1, round(len(rows) * fraction)))
    calibration_indices = set(indices[:calibration_count])
    calibration = [row for index, row in enumerate(rows) if index in calibration_indices]
    heldout = [row for index, row in enumerate(rows) if index not in calibration_indices]
    return calibration, heldout


def choose_alpha(metrics, alphas):
    return max(
        alphas,
        key=lambda alpha: (
            metrics["methods"][f"hybrid_{alpha:g}"]["execution_accuracy"],
            -abs(alpha - 0.5),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hybrid-alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    alphas = [float(value) for value in args.hybrid_alphas.split(",") if value.strip()]
    rows = read_jsonl(args.scored_file, args.limit)
    metrics, details = evaluate(rows, alphas)
    valid_gold_rows = [row for row in rows if row.get("gold_execution_ok", True)]
    valid_gold_metrics, _ = evaluate(valid_gold_rows, alphas)
    metrics["valid_gold_subset"] = {
        "sample_count": len(valid_gold_rows),
        "oracle_ex_available": valid_gold_metrics["oracle_ex_available"],
        "methods": valid_gold_metrics["methods"],
    }
    calibration_rows, heldout_rows = calibration_split(
        valid_gold_rows, args.calibration_fraction, args.calibration_seed
    )
    if calibration_rows and heldout_rows:
        calibration_metrics, _ = evaluate(calibration_rows, alphas)
        selected_alpha = choose_alpha(calibration_metrics, alphas)
        heldout_metrics, heldout_details = evaluate(heldout_rows, [selected_alpha])
        metrics["calibrated_protocol"] = {
            "calibration_sample_count": len(calibration_rows),
            "heldout_sample_count": len(heldout_rows),
            "selected_alpha": selected_alpha,
            "calibration_alpha_execution_accuracy": {
                f"{alpha:g}": calibration_metrics["methods"][f"hybrid_{alpha:g}"]["execution_accuracy"]
                for alpha in alphas
            },
            "heldout_metrics": heldout_metrics,
        }
    else:
        heldout_details = []
        metrics["calibrated_protocol"] = None
    metrics["config"] = vars(args)
    metrics["protocol_note"] = (
        "Selection consumes only candidate order/logprob, parse/execution validity, and verifier scores. "
        "Execution-equivalence labels are used only after selection. Full-data alpha sweeps are descriptive; "
        "the calibrated protocol selects alpha on a deterministic calibration subset and reports it on disjoint held-out rows."
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "selection_details.jsonl").open("w", encoding="utf-8") as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    with (output / "heldout_selection_details.jsonl").open("w", encoding="utf-8") as handle:
        for detail in heldout_details:
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
