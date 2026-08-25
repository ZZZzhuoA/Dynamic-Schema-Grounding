# Stage 16-A: OOF Graph-grounded Direct-SQL SFT Data

## Decision

Stage 16-A does **not** introduce IR. The target remains the corrected executable
SQLite SQL. The purpose of this stage is to train a code LLM to consume graph
beliefs without turning Top-30 into an irreversible hard schema filter.

```text
question + evidence
        + strict-OOF Schema-RGTA semantic core
        + frozen-LLM role reserve
        + Value Index bindings
        + full-schema Steiner FK closure
        + compact full-schema fallback
                    -> SQL SFT target
```

The output contract separates `inference_inputs` from `training_targets`. Gold SQL,
gold schema IDs, labels, candidate-oracle recall, and assistant answers are never
stored in the inference side.

## Why final Schema-RGTA must also be OOF

The old Stage 10-B candidate graphs use OOF Stage 8G beliefs, but a single final
Stage 10-G reranker trained on all 9,428 training questions still produces
in-sample train scores. Therefore it is not sufficient to call those predictions
OOF. Stage 16-A adds database-fold filtering and a `last` checkpoint policy to the
Stage 10 reranker, then merges only each fold's held-out predictions.

The held-out fold may be evaluated, but it cannot choose an epoch or trigger early
stopping. The epoch count must be fixed before the OOF run.

## Data contract

Each `train_sft.jsonl` row contains:

- `inference_inputs.prompt_messages`: system and user messages only;
- `inference_inputs.grounding_state.semantic_core`: final OOF Schema-RGTA Top-K;
- `role_reserve`: high role-probability nodes outside the core;
- `value_bindings`: database-value matches and confidence;
- `join_closure`: terminal tables, FK paths, and added structural endpoints;
- `full_schema`: all legal tables, columns, types, and declared FKs;
- `training_targets.response`: corrected gold SQL only.

The semantic core is graded evidence, not a whitelist. This preserves the ability
of the LLM to recover from the remaining grounding misses.

## Server commands

Run from the repository root. These paths correspond to the retained Stage 10-G
pipeline.

### 1. Produce strict final Schema-RGTA OOF predictions

```bash
MANIFEST=experiments/stage10b_oof_folds_seed42/fold_manifest.json
GRAPH=experiments/stage10g_table_conditioned_completion/train_normal.jsonl
EMBED=experiments/stage8g_embedding_cache_corrected_qwen3_06b
OOF_RUN=experiments/stage16a_oof_schema_rgta_seed42

for FILE in "$MANIFEST" "$GRAPH" "$EMBED/train_index.json"; do
  test -f "$FILE" || { echo "missing: $FILE"; exit 1; }
done
```

Run one fold per GPU. Five epochs are fixed from the previously selected Stage
10-G configuration; do not tune this count on held-out fold metrics.

```bash
for FOLD in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$FOLD python src/training/stage10_train_factor_graph_reranker.py \
    --train-file "$GRAPH" \
    --dev-file "$GRAPH" \
    --train-record-index-file experiments/stage10b_oof_folds_seed42/fold_${FOLD}/train_indices.json \
    --dev-record-index-file experiments/stage10b_oof_folds_seed42/fold_${FOLD}/heldout_indices.json \
    --train-cache-split train \
    --dev-cache-split train \
    --embedding-cache-dir "$EMBED" \
    --output-dir "$OOF_RUN/fold_${FOLD}" \
    --model-type schema_rgta \
    --hidden-dim 256 \
    --num-layers 2 \
    --epochs 5 \
    --lr 1e-4 \
    --gradient-accumulation-steps 16 \
    --structured-coverage-loss-weight 0.1 \
    --structured-coverage-margin 0.1 \
    --output-top-k 30 \
    --checkpoint-policy last \
    --patience 0 \
    --device cuda \
    --seed 42 &
done
wait
```

Merge and certify the held-out outputs:

```bash
python src/evaluation/stage16a_merge_oof_schema_predictions.py \
  --fold-manifest "$MANIFEST" \
  --fold-output-dir "$OOF_RUN" \
  --fallback-graph-file "$GRAPH" \
  --output-dir experiments/stage16a_oof_schema_predictions_seed42
```

Some corrected BIRD train records have no trainable candidate graph and are
therefore intentionally skipped by the Stage 10 trainer. The merger represents
them explicitly using only inference-safe baseline candidates from `GRAPH` (or an
empty semantic core when none exists). They remain in SFT because the complete
schema fallback is still available; no gold schema node is injected.

### 2. Build train join closure from the merged OOF belief

```bash
python src/grounding/stage10f_steiner_join_closure.py \
  --full-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --prediction-file experiments/stage16a_oof_schema_predictions_seed42/train_oof_schema_predictions.jsonl \
  --prior-file experiments/stage10g_table_conditioned_completion/train_priors.jsonl \
  --output-file experiments/stage16a_oof_schema_predictions_seed42/train_closure.jsonl
```

### 3. Construct direct-SQL SFT train/dev files

```bash
python src/data/stage16a_build_oof_graph_grounded_sft.py \
  --train-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --train-predictions experiments/stage16a_oof_schema_predictions_seed42/train_oof_schema_predictions.jsonl \
  --train-oof-summary experiments/stage16a_oof_schema_predictions_seed42/summary.json \
  --train-priors experiments/stage10g_table_conditioned_completion/train_priors.jsonl \
  --train-closure experiments/stage16a_oof_schema_predictions_seed42/train_closure.jsonl \
  --train-value-evidence experiments/stage10b_oof_train_evidence_rgta_seed42/evidence_debug.jsonl \
  --dev-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --dev-predictions experiments/stage10g_frozen_llm_schema_rgta_seed42/dev_predictions.jsonl \
  --dev-priors experiments/stage10g_table_conditioned_completion/dev_priors.jsonl \
  --dev-closure experiments/stage10g_steiner_join_closure_seed42/dev_closure.jsonl \
  --dev-value-evidence experiments/stage9_fix1_gated_value_join_rgta_seed42_limit1534/evidence_debug.jsonl \
  --output-dir experiments/stage16a_oof_graph_grounded_sql_sft \
  --reserve-k 20 \
  --reserve-min-role-score 0.25
```

Expected files:

```text
experiments/stage16a_oof_graph_grounded_sql_sft/train_sft.jsonl
experiments/stage16a_oof_graph_grounded_sql_sft/dev_sft.jsonl
experiments/stage16a_oof_graph_grounded_sql_sft/summary.json
```

`--allow-unverified-train-grounding` exists only for local smoke tests. Its output
is marked `unverified_debug_only` and must never be used for the SFT experiment.

## Acceptance gate

Before Stage 16-B training:

1. `summary.strict_oof == true`;
2. train/dev counts are 9,428 and 1,534 after any known-clean subset policy is
   applied explicitly;
3. `gold_leakage_violations == 0`;
4. every train row has database-disjoint OOF provenance;
5. prompt messages contain no assistant answer;
6. Top-30 is described as graded evidence and full schema remains available;
7. a random sample manually verifies exact identifiers, values, and FK paths.
