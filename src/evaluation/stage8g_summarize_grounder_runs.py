import argparse
import json
import statistics
from pathlib import Path


DEFAULT_METRICS = [
    "assembled_schema_recall@30",
    "assembled_schema_precision@30",
    "assembled_complete_coverage@30",
    "OUTPUT_TARGET_recall@20",
    "PREDICATE_COLUMN_recall@20",
    "VALUE_ANCHOR_recall@20",
    "JOIN_BRIDGE_column_recall@20",
    "FORMULA_COMPONENT_recall@20",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_metrics(payload):
    if isinstance(payload.get("metrics"), dict):
        return payload["metrics"]
    if isinstance(payload.get("dev_metrics"), dict):
        return payload["dev_metrics"]
    return payload


def add_derived_metrics(metrics):
    metrics = dict(metrics)
    count = metrics.get("assembled_sample_count")
    missing = metrics.get("assembled_missing_samples@30")
    if isinstance(count, (int, float)) and count > 0 and isinstance(missing, (int, float)):
        metrics.setdefault("assembled_missing_rate@30", missing / count)
        metrics.setdefault("assembled_complete_samples@30", count - missing)
        metrics.setdefault("assembled_complete_coverage@30", (count - missing) / count)
    return metrics


def parse_group(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Expected GROUP=FILE[,FILE,...]")
    name, file_text = spec.split("=", 1)
    name = name.strip()
    files = [Path(item.strip()) for item in file_text.split(",") if item.strip()]
    if not name or not files:
        raise argparse.ArgumentTypeError("Expected non-empty GROUP=FILE[,FILE,...]")
    return name, files


def metric_summary(values):
    return {
        "run_count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def markdown_table(groups, metrics):
    headers = ["group", "runs"] + metrics
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for group_name, group in groups.items():
        row = [group_name, str(group["run_count"])]
        for metric in metrics:
            summary = group["metric_summary"].get(metric)
            if summary is None:
                row.append("NA")
            elif summary["run_count"] == 1:
                row.append(f"{summary['mean']:.4f}")
            else:
                row.append(f"{summary['mean']:.4f} +/- {summary['std']:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate complete-dev Stage 8G evaluation runs into mean/std tables."
    )
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help="Repeatable GROUP=FILE[,FILE,...] specification.",
    )
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--output-dir", default="experiments/stage8g_grounder_run_summary")
    args = parser.parse_args()

    selected_metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    expected_sample_count = None

    for raw_group in args.group:
        group_name, paths = parse_group(raw_group)
        if group_name in groups:
            raise ValueError(f"Duplicate group name: {group_name}")
        runs = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Evaluation file not found: {path}")
            payload = read_json(path)
            metrics = add_derived_metrics(extract_metrics(payload))
            sample_count = metrics.get("assembled_sample_count")
            if expected_sample_count is None and sample_count is not None:
                expected_sample_count = sample_count
            if (
                expected_sample_count is not None
                and sample_count is not None
                and sample_count != expected_sample_count
            ):
                raise ValueError(
                    f"Incomparable sample counts: expected {expected_sample_count}, "
                    f"got {sample_count} in {path}"
                )
            runs.append(
                {
                    "path": str(path),
                    "seed": payload.get("seed"),
                    "checkpoint_dir": payload.get("checkpoint_dir"),
                    "metrics": metrics,
                }
            )

        summaries = {}
        for metric in selected_metrics:
            values = [
                float(run["metrics"][metric])
                for run in runs
                if isinstance(run["metrics"].get(metric), (int, float))
            ]
            if values:
                summaries[metric] = metric_summary(values)
        groups[group_name] = {
            "run_count": len(runs),
            "runs": runs,
            "metric_summary": summaries,
        }

    summary = {
        "expected_sample_count": expected_sample_count,
        "selected_metrics": selected_metrics,
        "groups": groups,
        "note": (
            "All groups must have the same assembled sample count. Standard deviation is the "
            "sample standard deviation and is zero for a single run."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    table = markdown_table(groups, selected_metrics)
    (output_dir / "summary.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
