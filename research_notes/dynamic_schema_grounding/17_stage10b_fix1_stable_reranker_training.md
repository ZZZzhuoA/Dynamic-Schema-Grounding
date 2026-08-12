# Stage 10-B-fix1: Stable OOF Schema-RGTA Training

## Finding from Stage 10-B

Database-disjoint OOF predictions successfully matched the unseen-dev ranking
distribution, but the reranker still reached its best complete coverage after the
first epoch. One epoch contained about 9,410 batch-size-one optimizer updates, and
the node/role/pairwise loss continued to improve even when Complete Coverage@30
decreased.

Stage 10-B-fix1 isolates two questions:

1. Does schema graph propagation outperform an MLP under the same OOF inputs?
2. Was the apparent epoch-1 optimum caused by overly frequent parameter updates or
   by the loss/coverage mismatch?

No architecture or candidate-data changes are introduced in this stage.

## Training change

Heterogeneous candidate graphs still run one at a time. Gradients from `N` graphs
are averaged before one optimizer update:

```text
L_batch = (1 / N) * sum_i L(graph_i)
```

This avoids graph padding and changes an epoch from about 9,410 optimizer updates
to about 589 updates when `N=16`.

Periodic dev evaluation observes checkpoints inside an epoch. Early stopping
continues to count complete epochs, so periodic evaluation cannot stop training
prematurely.

## Controlled experiments

Common inputs:

```bash
OOF_TRAIN=experiments/stage10b_oof_candidate_graphs/train_factor_graphs.jsonl
DEV_GRAPH=experiments/stage10a_factor_graph_data/dev_factor_graphs.jsonl
EMBED_CACHE=experiments/stage8g_embedding_cache_corrected_qwen3_06b
```

### A. OOF MLP control

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage10_train_factor_graph_reranker.py \
  --train-file "$OOF_TRAIN" \
  --dev-file "$DEV_GRAPH" \
  --embedding-cache-dir "$EMBED_CACHE" \
  --output-dir experiments/stage10b_fix1_oof_mlp_seed42 \
  --model-type mlp \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 8 \
  --lr 1e-4 \
  --gradient-accumulation-steps 16 \
  --eval-every-examples 2000 \
  --patience 3 \
  --device cuda \
  --seed 42
```

### B. OOF Schema-RGTA with stable optimization

```bash
CUDA_VISIBLE_DEVICES=1 python src/training/stage10_train_factor_graph_reranker.py \
  --train-file "$OOF_TRAIN" \
  --dev-file "$DEV_GRAPH" \
  --embedding-cache-dir "$EMBED_CACHE" \
  --output-dir experiments/stage10b_fix1_oof_schema_rgta_seed42 \
  --model-type schema_rgta \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 8 \
  --lr 1e-4 \
  --gradient-accumulation-steps 16 \
  --eval-every-examples 2000 \
  --patience 3 \
  --device cuda \
  --seed 42
```

These two commands may run concurrently on separate GPUs. They use identical data,
optimizer settings, checkpoint observations, and constrained decoding. Only graph
message passing differs.

## Required comparisons

Compare these three settings:

| Setting | Purpose |
|---|---|
| Existing OOF Schema-RGTA, batch 1 / lr 5e-4 | Original optimization baseline |
| OOF MLP, batch 16 / lr 1e-4 | Controlled no-graph baseline |
| OOF Schema-RGTA, batch 16 / lr 1e-4 | Graph contribution under stable optimization |

Primary metrics:

- `constrained_complete_coverage@30`
- `constrained_complete_samples@30`
- `constrained_schema_recall@30`
- `constrained_table_recall@30`
- `constrained_column_recall@30`
- `best_checkpoint.epoch_progress`
- `global_optimizer_steps`

## Interpretation gates

1. If stable Schema-RGTA exceeds stable MLP, graph propagation remains supported
   under realistic OOF noise.
2. If the best checkpoint moves inside or beyond epoch 1, the former observation
   granularity/update frequency was part of the problem.
3. If dev loss falls while complete coverage still falls, the next stage must add
   a query-level listwise/top-K coverage objective; more optimizer tuning is not a
   sufficient algorithmic response.
4. If Schema-RGTA does not exceed MLP, do not connect this reranker to the LLM as the
   claimed graph component until its graph objective or query-conditioned message
   passing is redesigned.
