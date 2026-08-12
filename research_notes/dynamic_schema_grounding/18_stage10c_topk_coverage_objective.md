# Stage 10-C: Query-Level Top-K Coverage Objective

## Motivation

Pointwise BCE and pairwise ranking do not directly optimize the downstream event:

```text
all schema nodes required by one SQL are simultaneously inside Top-K
```

For query `q`, let `P_q` be its positive candidate nodes, `N_q` its negative
candidates, and `K=30`. Complete coverage permits at most `K-|P_q|` negatives to
outrank the weakest positive. Therefore, the first violating negative is:

```text
boundary_rank = K - |P_q| + 1
```

among negatives sorted in descending score order.

## Loss

The implementation uses:

```text
weakest_positive = -tau * logsumexp(-positive_scores / tau)
boundary_negative = kth_highest(negative_scores, K - |P| + 1)

L_coverage = tau * softplus(
    (margin + boundary_negative - weakest_positive) / tau
)
```

The total objective is:

```text
L = L_node
  + lambda_role * L_role
  + lambda_pair * L_pair
  + lambda_coverage * L_coverage
```

This is query-level and budget-aware: the position of one candidate depends on the
entire candidate list and the number of schema nodes required by that query.

## Eligibility boundary

Coverage loss is active only when:

- candidate oracle recall is exactly 1;
- at least one positive and one relevant boundary negative exist;
- the number of positives does not exceed K;
- the candidate list is larger than K.

If a gold schema node is absent from the candidate graph, it has no trainable score.
Treating the remaining gold subset as complete would provide a false objective, so
those examples keep their node/role/pairwise losses but skip coverage loss.

## Controlled experiment

Use the same OOF candidate graphs and stable optimization protocol as the current
best Schema-RGTA. The existing Stage 10-B-fix1 run is the `lambda_coverage=0`
baseline.

```bash
OOF_TRAIN=experiments/stage10b_oof_candidate_graphs/train_factor_graphs.jsonl
DEV_GRAPH=experiments/stage10a_factor_graph_data/dev_factor_graphs.jsonl
EMBED_CACHE=experiments/stage8g_embedding_cache_corrected_qwen3_06b
```

### Coverage weight 0.1

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage10_train_factor_graph_reranker.py \
  --train-file "$OOF_TRAIN" \
  --dev-file "$DEV_GRAPH" \
  --embedding-cache-dir "$EMBED_CACHE" \
  --output-dir experiments/stage10c_oof_schema_rgta_cover01_seed42 \
  --model-type schema_rgta \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 8 \
  --lr 1e-4 \
  --gradient-accumulation-steps 16 \
  --eval-every-examples 2000 \
  --coverage-loss-weight 0.1 \
  --coverage-margin 0.1 \
  --coverage-temperature 0.2 \
  --patience 3 \
  --device cuda \
  --seed 42
```

### Coverage weight 0.3

```bash
CUDA_VISIBLE_DEVICES=1 python src/training/stage10_train_factor_graph_reranker.py \
  --train-file "$OOF_TRAIN" \
  --dev-file "$DEV_GRAPH" \
  --embedding-cache-dir "$EMBED_CACHE" \
  --output-dir experiments/stage10c_oof_schema_rgta_cover03_seed42 \
  --model-type schema_rgta \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 8 \
  --lr 1e-4 \
  --gradient-accumulation-steps 16 \
  --eval-every-examples 2000 \
  --coverage-loss-weight 0.3 \
  --coverage-margin 0.1 \
  --coverage-temperature 0.2 \
  --patience 3 \
  --device cuda \
  --seed 42
```

Both runs may execute concurrently. Do not select a broad hyperparameter sweep from
BIRD dev; these two weights are a bounded mechanism check. If neither exceeds the
`0.807040` complete-coverage baseline, diagnose the objective rather than increasing
the weight indefinitely.

## Evaluation

Primary comparison:

| Setting | Coverage weight | Complete Coverage@30 |
|---|---:|---:|
| Stage 10-B-fix1 | 0.0 | 0.807040 |
| Stage 10-C | 0.1 | pending |
| Stage 10-C | 0.3 | pending |

Also inspect:

- `loss_coverage_active_count`;
- `loss_coverage_loss_per_active`;
- `loss_coverage_violation_per_active`;
- raw versus constrained complete coverage;
- table and column recall;
- best checkpoint progress.

## Interpretation

An improvement supports the claim that schema grounding is a budgeted set prediction
problem rather than independent node classification. No improvement would indicate
that raw Top-K boundary optimization is insufficient, likely because final selection
is graph-constrained or because the target should be conditioned on SQL-generation
state. That result would motivate operation-step coverage objectives in the dynamic
LLM-coupled stage rather than further static weight tuning.
