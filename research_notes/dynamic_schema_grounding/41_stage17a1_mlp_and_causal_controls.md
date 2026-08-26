# Stage 17-A1: Depth-matched MLP and causal controls

## Purpose

Stage 17-A0 improved `complete_coverage@30` substantially, but cosine retrieval
does not isolate the contribution of graph propagation. Stage 17-A1 therefore
adds:

1. a depth-matched node-local MLP baseline;
2. frozen-checkpoint causal interventions;
3. separately retrained controls as an adaptation diagnostic.

The intervention and retraining results must not be conflated. A frozen
intervention asks whether the trained normal model uses the information. A
retrained control asks whether the model can compensate after that information
is destroyed throughout training.

## Implemented controls

- `zero_query_edges`: removes Query-to-Schema graph messages while retaining
  the final query-conditioned scorer.
- `shuffled_schema_edges`: permutes non-self edge destinations within each
  relation type. Sources, destination marginals, relation counts, and self
  loops remain unchanged.
- `shuffled_node_identity`: permutes dense semantic embeddings separately
  among table and column nodes. Node IDs, node types, graph edges, and labels
  remain fixed.

The official MLP baseline is `mlp_residual`. The old `mlp` mode remains only
for checkpoint compatibility.

## Shared server variables

```bash
TRAIN_GRAPH=experiments/stage17a_dsg_data_corrected_scopefix1/train_examples.jsonl
DEV_GRAPH=experiments/stage17a_dsg_data_corrected_scopefix1/dev_examples.jsonl
TRAIN_LABEL=experiments/stage1_label_extraction_corrected_scopefix1/bird_train_grounding_labels.jsonl
DEV_LABEL=experiments/stage1_label_extraction_corrected_scopefix1/bird_dev_grounding_labels.jsonl
EMBED=experiments/stage17a_embedding_cache_corrected_scopefix1_qwen3_06b
```

Do not replace these with the older uncorrected Stage 17-A0 graph, label, or
embedding-cache directories.

## 100/50 smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage17a_train_full_schema_qrgta.py \
  --train-graph-file "$TRAIN_GRAPH" \
  --dev-graph-file "$DEV_GRAPH" \
  --train-label-file "$TRAIN_LABEL" \
  --dev-label-file "$DEV_LABEL" \
  --embedding-cache-dir "$EMBED" \
  --output-dir experiments/stage17a1_mlp_residual_smoke \
  --model-type mlp_residual \
  --hidden-dim 256 \
  --num-layers 3 \
  --num-heads 8 \
  --dropout 0.1 \
  --epochs 2 \
  --lr 1e-4 \
  --gradient-accumulation-steps 16 \
  --selection-metric complete_coverage@30 \
  --train-limit 100 \
  --dev-limit 50 \
  --device cuda \
  --seed 42
```

## Main normal and MLP runs

Train missing normal QRGTA seeds. Seed 42 may be reused only if it used the
same corrected graph, labels, and embedding cache.

```bash
for SEED in 42 43 44; do
  CUDA_VISIBLE_DEVICES=0 python src/training/stage17a_train_full_schema_qrgta.py \
    --train-graph-file "$TRAIN_GRAPH" \
    --dev-graph-file "$DEV_GRAPH" \
    --train-label-file "$TRAIN_LABEL" \
    --dev-label-file "$DEV_LABEL" \
    --embedding-cache-dir "$EMBED" \
    --output-dir "experiments/stage17a_full_schema_qrgta_seed${SEED}" \
    --model-type qrgta \
    --hidden-dim 256 \
    --num-layers 3 \
    --num-heads 8 \
    --dropout 0.1 \
    --epochs 8 \
    --lr 1e-4 \
    --gradient-accumulation-steps 16 \
    --selection-metric complete_coverage@30 \
    --patience 3 \
    --control-mode normal \
    --device cuda \
    --seed "$SEED"
done
```

```bash
for SEED in 42 43 44; do
  CUDA_VISIBLE_DEVICES=0 python src/training/stage17a_train_full_schema_qrgta.py \
    --train-graph-file "$TRAIN_GRAPH" \
    --dev-graph-file "$DEV_GRAPH" \
    --train-label-file "$TRAIN_LABEL" \
    --dev-label-file "$DEV_LABEL" \
    --embedding-cache-dir "$EMBED" \
    --output-dir "experiments/stage17a1_full_schema_mlp_residual_seed${SEED}" \
    --model-type mlp_residual \
    --hidden-dim 256 \
    --num-layers 3 \
    --num-heads 8 \
    --dropout 0.1 \
    --epochs 8 \
    --lr 1e-4 \
    --gradient-accumulation-steps 16 \
    --selection-metric complete_coverage@30 \
    --patience 3 \
    --control-mode normal \
    --device cuda \
    --seed "$SEED"
done
```

## Frozen-checkpoint interventions

`--reference-normal-predictions` makes the command fail if normal inference no
longer reproduces the ranking produced during training.

```bash
for SEED in 42 43 44; do
  CUDA_VISIBLE_DEVICES=0 python src/evaluation/stage17a_run_checkpoint_controls.py \
    --checkpoint "experiments/stage17a_full_schema_qrgta_seed${SEED}/best.pt" \
    --dev-graph-file "$DEV_GRAPH" \
    --dev-label-file "$DEV_LABEL" \
    --embedding-cache-dir "$EMBED" \
    --output-dir "experiments/stage17a1_checkpoint_controls_seed${SEED}" \
    --reference-normal-predictions "experiments/stage17a_full_schema_qrgta_seed${SEED}/dev_predictions.jsonl" \
    --control-modes normal,zero_query_edges,shuffled_schema_edges,shuffled_node_identity \
    --device cuda \
    --seed "$SEED"
done
```

## Retrained controls, seed 42 only

```bash
for MODE in zero_query_edges shuffled_schema_edges shuffled_node_identity; do
  CUDA_VISIBLE_DEVICES=0 python src/training/stage17a_train_full_schema_qrgta.py \
    --train-graph-file "$TRAIN_GRAPH" \
    --dev-graph-file "$DEV_GRAPH" \
    --train-label-file "$TRAIN_LABEL" \
    --dev-label-file "$DEV_LABEL" \
    --embedding-cache-dir "$EMBED" \
    --output-dir "experiments/stage17a1_retrained_${MODE}_seed42" \
    --model-type qrgta \
    --hidden-dim 256 \
    --num-layers 3 \
    --num-heads 8 \
    --dropout 0.1 \
    --epochs 8 \
    --lr 1e-4 \
    --gradient-accumulation-steps 16 \
    --selection-metric complete_coverage@30 \
    --patience 3 \
    --control-mode "$MODE" \
    --device cuda \
    --seed 42
done
```

## Unified summary

```bash
python src/evaluation/stage17a_summarize_causal_controls.py \
  --normal-run 42=experiments/stage17a_full_schema_qrgta_seed42 \
  --normal-run 43=experiments/stage17a_full_schema_qrgta_seed43 \
  --normal-run 44=experiments/stage17a_full_schema_qrgta_seed44 \
  --mlp-run 42=experiments/stage17a1_full_schema_mlp_residual_seed42 \
  --mlp-run 43=experiments/stage17a1_full_schema_mlp_residual_seed43 \
  --mlp-run 44=experiments/stage17a1_full_schema_mlp_residual_seed44 \
  --intervention-run 42=experiments/stage17a1_checkpoint_controls_seed42 \
  --intervention-run 43=experiments/stage17a1_checkpoint_controls_seed43 \
  --intervention-run 44=experiments/stage17a1_checkpoint_controls_seed44 \
  --retrained-run zero_query_edges=experiments/stage17a1_retrained_zero_query_edges_seed42 \
  --retrained-run shuffled_schema_edges=experiments/stage17a1_retrained_shuffled_schema_edges_seed42 \
  --retrained-run shuffled_node_identity=experiments/stage17a1_retrained_shuffled_node_identity_seed42 \
  --output-file experiments/stage17a1_causal_summary.json
```

The summary rejects seed mismatches, inconsistent graph/label/cache paths,
sample-count mismatches, non-normal QRGTA checkpoints, and intervention runs
whose model parameters changed.

## Interpretation

The graph-structure claim is supported only if normal QRGTA beats
`mlp_residual` on mean `complete_coverage@30`, all three frozen interventions
cause consistent degradation, and shuffled node identity produces the largest
or near-largest mean degradation. Retrained controls are reported separately
and cannot substitute for frozen-checkpoint causal evidence.
