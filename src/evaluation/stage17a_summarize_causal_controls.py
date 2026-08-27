"""Aggregate Stage 17-A1 trained baselines and paired checkpoint controls."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


PRIMARY_METRICS = (
    "complete_coverage@30",
    "schema_recall@30",
    "table_recall@30",
    "column_recall@30",
    "complete_coverage@50",
    "mrr",
)
CONTROL_MODES = (
    "zero_query_edges",
    "shuffled_schema_edges",
    "shuffled_node_identity",
)
PATH_CONTROL_MODES = (
    "shuffled_distance_buckets",
    "shuffled_path_signatures",
    "zero_path_features",
)
PERSISTENT_CONTROL_MODES = (
    "zero_update_gates",
)
DATA_KEYS = (
    "train_graph_file",
    "dev_graph_file",
    "train_label_file",
    "dev_label_file",
    "embedding_cache_dir",
    "train_limit",
    "dev_limit",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assignments(values, option):
    parsed = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"{option} expects KEY=PATH, got {value!r}")
        key, path = value.split("=", 1)
        key = key.strip()
        if not key or key in parsed:
            raise ValueError(f"Duplicate or empty key for {option}: {key!r}")
        parsed[key] = Path(path)
    return parsed


def trained_run(path, expected_model_type=None, expected_control=None):
    summary = read_json(path / "training_summary.json")
    metrics = summary["dev_metrics"]
    config = summary["config"]
    model_config = summary["model_config"]
    if isinstance(expected_model_type, (set, tuple, list)):
        expected_model_types = set(expected_model_type)
    elif expected_model_type:
        expected_model_types = {expected_model_type}
    else:
        expected_model_types = set()
    if expected_model_types and model_config.get("model_type") not in expected_model_types:
        raise ValueError(
            f"{path} has model_type={model_config.get('model_type')!r}, "
            f"expected one of {sorted(expected_model_types)!r}"
        )
    if expected_control and model_config.get("control_mode") != expected_control:
        raise ValueError(
            f"{path} has control_mode={model_config.get('control_mode')!r}, "
            f"expected {expected_control!r}"
        )
    return {
        "path": str(path),
        "metrics": {metric: float(metrics[metric]) for metric in PRIMARY_METRICS},
        "sample_count": int(metrics["sample_count"]),
        "data_config": {key: config.get(key) for key in DATA_KEYS},
        "model_type": model_config.get("model_type"),
        "control_mode": model_config.get("control_mode"),
        "persistent_diagnostics": metrics.get("persistent_diagnostics", {}),
        "best_epoch": summary.get("best_epoch"),
        "checkpoint_sha256": file_sha256(path / "best.pt")
        if (path / "best.pt").exists()
        else None,
    }


def metric_summary(runs):
    output = {}
    for metric in PRIMARY_METRICS:
        values = [run["metrics"][metric] for run in runs.values()]
        output[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "values": {
                str(seed): runs[seed]["metrics"][metric] for seed in sorted(runs)
            },
        }
    return output


def assert_matched_runs(normal, mlp):
    if set(normal) != set(mlp):
        raise ValueError(
            f"Normal/MLP seeds differ: normal={sorted(normal)} mlp={sorted(mlp)}"
        )
    reference = None
    for family, runs in (("normal", normal), ("mlp", mlp)):
        for seed, run in runs.items():
            fingerprint = run["data_config"]
            if reference is None:
                reference = fingerprint
            elif fingerprint != reference:
                raise ValueError(
                    f"Data configuration mismatch for {family} seed={seed}: "
                    f"expected={reference}, observed={fingerprint}"
                )
    sample_counts = {run["sample_count"] for run in list(normal.values()) + list(mlp.values())}
    if len(sample_counts) != 1:
        raise ValueError(f"Sample counts differ across trained runs: {sorted(sample_counts)}")
    return reference


def load_interventions(paths, normal):
    if set(paths) != set(normal):
        raise ValueError(
            f"Normal/intervention seeds differ: normal={sorted(normal)} "
            f"intervention={sorted(paths)}"
        )
    output = {}
    for seed, path in paths.items():
        summary = read_json(path / "intervention_summary.json")
        if not summary.get("parameters_unchanged"):
            raise ValueError(f"Intervention parameters changed for seed={seed}: {path}")
        missing = sorted(set(CONTROL_MODES) - set(summary.get("metrics", {})))
        if missing:
            raise ValueError(f"Intervention seed={seed} misses controls: {missing}")
        if not summary.get("reference_normal_reproduced"):
            raise ValueError(
                f"Intervention seed={seed} did not verify reference normal predictions"
            )
        expected_sha = normal[seed].get("checkpoint_sha256")
        if expected_sha and summary.get("checkpoint_sha256") != expected_sha:
            raise ValueError(
                f"Intervention seed={seed} used a different checkpoint: "
                f"expected={expected_sha} observed={summary.get('checkpoint_sha256')}"
            )
        expected_dev = {
            "dev_graph_file": normal[seed]["data_config"]["dev_graph_file"],
            "dev_label_file": normal[seed]["data_config"]["dev_label_file"],
            "embedding_cache_dir": normal[seed]["data_config"]["embedding_cache_dir"],
            "dev_limit": normal[seed]["data_config"]["dev_limit"],
        }
        if summary.get("data_config") != expected_dev:
            raise ValueError(
                f"Intervention seed={seed} uses a different dev data configuration: "
                f"expected={expected_dev} observed={summary.get('data_config')}"
            )
        normal_metrics = summary["metrics"]["normal"]
        for metric in PRIMARY_METRICS:
            if abs(float(normal_metrics[metric]) - normal[seed]["metrics"][metric]) > 1e-12:
                raise ValueError(
                    f"Intervention normal does not reproduce trained normal for seed={seed}, "
                    f"metric={metric}"
                )
        output[seed] = {
            "path": str(path),
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "metrics": {
                mode: {
                    metric: float(summary["metrics"][mode][metric])
                    for metric in PRIMARY_METRICS
                }
                for mode in CONTROL_MODES + PATH_CONTROL_MODES + PERSISTENT_CONTROL_MODES
                if mode in summary["metrics"]
            },
        }
    return output


def paired_deltas(left, right):
    output = {}
    for metric in PRIMARY_METRICS:
        values = {
            str(seed): left[seed]["metrics"][metric] - right[seed]["metrics"][metric]
            for seed in sorted(left)
        }
        output[metric] = {
            "mean": statistics.fmean(values.values()),
            "std": statistics.pstdev(values.values()),
            "values": values,
        }
    return output


def intervention_deltas(normal, interventions, modes=CONTROL_MODES):
    output = {}
    for mode in modes:
        output[mode] = {}
        for metric in PRIMARY_METRICS:
            values = {
                str(seed): normal[seed]["metrics"][metric]
                - interventions[seed]["metrics"][mode][metric]
                for seed in sorted(normal)
            }
            output[mode][metric] = {
                "mean": statistics.fmean(values.values()),
                "std": statistics.pstdev(values.values()),
                "values": values,
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-run", action="append", required=True, help="SEED=DIR")
    parser.add_argument("--mlp-run", action="append", required=True, help="SEED=DIR")
    parser.add_argument(
        "--intervention-run", action="append", required=True, help="SEED=DIR"
    )
    parser.add_argument(
        "--retrained-run", action="append", default=[], help="CONTROL=DIR"
    )
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--identity-near-largest-tolerance", type=float, default=0.005)
    args = parser.parse_args()

    normal_paths = assignments(args.normal_run, "--normal-run")
    mlp_paths = assignments(args.mlp_run, "--mlp-run")
    intervention_paths = assignments(args.intervention_run, "--intervention-run")
    retrained_paths = assignments(args.retrained_run, "--retrained-run")
    normal = {
        int(seed): trained_run(path, {"qrgta", "path_qrgta", "persistent_path_qrgta"}, "normal")
        for seed, path in normal_paths.items()
    }
    mlp = {
        int(seed): trained_run(path, "mlp_residual", "normal")
        for seed, path in mlp_paths.items()
    }
    data_config = assert_matched_runs(normal, mlp)
    interventions = load_interventions(
        {int(seed): path for seed, path in intervention_paths.items()}, normal
    )
    known_retrained_modes = set(CONTROL_MODES) | set(PATH_CONTROL_MODES) | set(PERSISTENT_CONTROL_MODES)
    unknown_retrained = sorted(set(retrained_paths) - known_retrained_modes)
    if unknown_retrained:
        raise ValueError(f"Unknown retrained controls: {unknown_retrained}")
    retrained = {
        mode: trained_run(
            path,
            {run["model_type"] for run in normal.values()},
            mode,
        )
        for mode, path in retrained_paths.items()
    }
    for mode, run in retrained.items():
        if run["data_config"] != data_config:
            raise ValueError(f"Retrained control {mode} uses a different data configuration")

    qrgta_minus_mlp = paired_deltas(normal, mlp)
    control_deltas = intervention_deltas(normal, interventions)
    normal_is_path_qrgta = all(run.get("model_type") == "path_qrgta" for run in normal.values())
    normal_is_persistent_path_qrgta = all(
        run.get("model_type") == "persistent_path_qrgta" for run in normal.values()
    )
    available_path_controls = [
        mode
        for mode in PATH_CONTROL_MODES
        if all(mode in interventions[seed]["metrics"] for seed in normal)
    ] if normal_is_path_qrgta or normal_is_persistent_path_qrgta else []
    path_control_deltas = intervention_deltas(
        normal, interventions, tuple(available_path_controls)
    ) if available_path_controls else {}
    available_persistent_controls = [
        mode
        for mode in PERSISTENT_CONTROL_MODES
        if all(mode in interventions[seed]["metrics"] for seed in normal)
    ] if normal_is_persistent_path_qrgta else []
    persistent_control_deltas = intervention_deltas(
        normal, interventions, tuple(available_persistent_controls)
    ) if available_persistent_controls else {}
    complete_drops = {
        mode: control_deltas[mode]["complete_coverage@30"]["mean"]
        for mode in CONTROL_MODES
    }
    identity_drop = complete_drops["shuffled_node_identity"]
    largest_drop = max(complete_drops.values())
    checks = {
        "qrgta_mean_complete_coverage@30_gt_mlp": qrgta_minus_mlp[
            "complete_coverage@30"
        ]["mean"]
        > 0,
        "all_controls_drop_complete_coverage@30_for_every_seed": all(
            value > 0
            for mode in CONTROL_MODES
            for value in control_deltas[mode]["complete_coverage@30"]["values"].values()
        ),
        "shuffled_node_identity_near_largest_mean_drop": identity_drop
        >= largest_drop - args.identity_near_largest_tolerance,
    }
    if available_path_controls:
        checks["shuffled_path_signatures_drop_complete_coverage@30_every_seed"] = all(
            value > 0
            for value in path_control_deltas["shuffled_path_signatures"][
                "complete_coverage@30"
            ]["values"].values()
        ) if "shuffled_path_signatures" in path_control_deltas else False
        distance_or_zero = []
        for mode in ("shuffled_distance_buckets", "zero_path_features"):
            if mode in path_control_deltas:
                distance_or_zero.extend(
                    value > 0
                    for value in path_control_deltas[mode]["complete_coverage@30"][
                        "values"
                    ].values()
                )
        checks["distance_or_zero_path_features_drop_complete_coverage@30_at_least_2_of_3"] = (
            sum(distance_or_zero) >= 2
        )
    if available_persistent_controls:
        checks["zero_update_gates_drop_complete_coverage@30_every_seed"] = all(
            value > 0
            for value in persistent_control_deltas["zero_update_gates"][
                "complete_coverage@30"
            ]["values"].values()
        )
    persistent_diagnostics = {
        str(seed): run.get("persistent_diagnostics", {})
        for seed, run in sorted(normal.items())
        if run.get("persistent_diagnostics")
    }
    output = {
        "seeds": sorted(normal),
        "data_config": data_config,
        "normal_qrgta": metric_summary(normal),
        "depth_matched_mlp": metric_summary(mlp),
        "qrgta_minus_mlp": qrgta_minus_mlp,
        "checkpoint_interventions_normal_minus_control": control_deltas,
        "path_checkpoint_interventions_normal_minus_control": path_control_deltas,
        "persistent_checkpoint_interventions_normal_minus_control": persistent_control_deltas,
        "persistent_diagnostics": persistent_diagnostics,
        "retrained_seed42_controls": retrained,
        "decision_checks": checks,
        "decision_passed": all(checks.values()),
        "identity_near_largest_tolerance": args.identity_near_largest_tolerance,
        "interpretation_policy": {
            "checkpoint_intervention": "Measures whether the trained normal QRGTA depends on the intervened information.",
            "retrained_control": "Measures how much performance can be relearned after the information is destroyed during training.",
        },
    }
    write_json(args.output_file, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
