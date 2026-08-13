# Stage 10-D: Constrained Selection-Aware Structured Coverage

## Motivation

Stage 10-C improved complete coverage consistently across three seeds, but its raw
Top-K boundary loss does not model the final typed decoder. The deployed selector
applies a fixed budget, a table cap, column-owner closure, graph connectivity, and
baseline retention. Stage 10-D aligns training with that exact decoding path.

## Latent feasible target

For one query, decode two sets with the same constrained selector:

```text
S_pred = Decode(scores)
S_gold = Decode(scores, required = all candidate gold nodes)
```

`S_gold` is a latent gold-feasible completion rather than the raw gold set. It
contains every candidate gold node, adds owner tables when required, and fills the
remaining budget using current model scores under the same constraints.

The loss compares only their symmetric difference:

```text
L_struct = ReLU(
    margin * missing_gold_count
    + sum(score[S_pred - S_gold])
    - sum(score[S_gold - S_pred])
) / missing_gold_count
```

Thus gradients raise nodes needed by the feasible gold completion and lower the
intruder nodes that actually displaced them. Shared filler nodes receive no
gradient from this objective.

## Eligibility

Structured loss is active only when:

- the candidate graph contains all gold schema nodes;
- ordinary constrained decoding misses at least one gold node;
- all gold nodes plus owner closure fit inside `output_top_k` and `max_tables`;
- predicted and gold-feasible sets have a non-empty difference.

This avoids supervising impossible selections. Candidate-incomplete examples retain
the pointwise, role, and pairwise objectives.

## Controlled experiment

Use Stage 10-B OOF training graphs and the unchanged full BIRD dev graph:

```bash
OOF_TRAIN=experiments/stage10b_oof_candidate_graphs/train_factor_graphs.jsonl
DEV_GRAPH=experiments/stage10a_factor_graph_data/dev_factor_graphs.jsonl
EMBED_CACHE=experiments/stage8g_embedding_cache_corrected_qwen3_06b
```

Run a bounded mechanism check at weight `0.1` and seed 42. Keep the Stage 10-C raw
coverage weight at zero so the new structured objective is isolated.

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage10_train_factor_graph_reranker.py \
  --train-file "$OOF_TRAIN" \
  --dev-file "$DEV_GRAPH" \
  --embedding-cache-dir "$EMBED_CACHE" \
  --output-dir experiments/stage10d_oof_schema_rgta_struct01_seed42 \
  --model-type schema_rgta \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 8 \
  --lr 1e-4 \
  --gradient-accumulation-steps 16 \
  --eval-every-examples 2000 \
  --coverage-loss-weight 0.0 \
  --structured-coverage-loss-weight 0.1 \
  --structured-coverage-margin 0.1 \
  --patience 3 \
  --device cuda \
  --seed 42
```

Compare against:

- Stage 10-B Schema-RGTA seed42: complete coverage `0.807040` (`1238/1534`);
- Stage 10-C raw coverage weight 0.3 seed42: `0.809648` (`1242/1534`).

Inspect `loss_structured_active_count`,
`loss_structured_loss_per_active`, and
`loss_structured_missing_gold_per_active`. If the new loss improves seed42, repeat
with seeds 43 and 44 before accepting it. Do not combine it with Stage 10-C until
its isolated contribution is established.
