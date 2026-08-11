import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_fold_ids(value, available):
    if not value or value.strip().lower() == "all":
        return list(available)
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    unknown = sorted(set(result) - set(available))
    if unknown:
        raise ValueError(f"Unknown fold ids: {unknown}; available={list(available)}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train strict fixed-epoch Stage 8G models and predict database-disjoint "
            "held-out folds for Stage 10-B."
        )
    )
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--relation-file", required=True)
    parser.add_argument("--graph-file", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-ids", default="all")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--output-top-k", type=int, default=30)
    parser.add_argument("--encoder-type", choices=["rgcn", "rgta"], default="rgta")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.fold_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = {int(fold["fold_id"]): fold for fold in manifest["folds"]}
    selected = parse_fold_ids(args.fold_ids, sorted(folds))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = ROOT / "src" / "training" / "stage8g_train_dense_relation_grounder.py"

    run_summary = {
        "config": vars(args),
        "strict_oof": True,
        "heldout_used_for_checkpoint_selection": False,
        "folds": [],
    }
    for fold_id in selected:
        fold = folds[fold_id]
        fold_output = output_dir / f"fold_{fold_id}"
        command = [
            sys.executable,
            str(trainer),
            "--train-relation-file",
            args.relation_file,
            "--dev-relation-file",
            args.relation_file,
            "--train-graph-file",
            args.graph_file,
            "--dev-graph-file",
            args.graph_file,
            "--embedding-cache-dir",
            args.embedding_cache_dir,
            "--train-cache-split",
            "train",
            "--dev-cache-split",
            "train",
            "--train-record-index-file",
            fold["train_index_file"],
            "--dev-record-index-file",
            fold["heldout_index_file"],
            "--output-dir",
            str(fold_output),
            "--train-limit",
            str(fold["train_record_count"]),
            "--dev-limit",
            str(fold["heldout_record_count"]),
            "--epochs",
            str(args.epochs),
            "--hidden-dim",
            str(args.hidden_dim),
            "--num-layers",
            str(args.num_layers),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--dropout",
            str(args.dropout),
            "--pos-weight",
            str(args.pos_weight),
            "--output-top-k",
            str(args.output_top_k),
            "--encoder-type",
            args.encoder_type,
            "--device",
            args.device,
            "--seed",
            str(args.seed + fold_id),
            "--checkpoint-policy",
            "last",
            "--use-relation-conditioned-prior",
        ]
        if args.deterministic:
            command.append("--deterministic")
        record = {
            "fold_id": fold_id,
            "heldout_db_count": fold["db_count"],
            "heldout_record_count": fold["heldout_record_count"],
            "train_record_count": fold["train_record_count"],
            "output_dir": str(fold_output),
            "command": command,
        }
        run_summary["folds"].append(record)
        print("\n" + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    fold_suffix = "_".join(str(fold_id) for fold_id in selected)
    summary_file = output_dir / f"oof_run_summary_folds_{fold_suffix}.json"
    summary_file.write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRun summary written to: {summary_file}")


if __name__ == "__main__":
    main()
