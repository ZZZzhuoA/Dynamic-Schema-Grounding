# Stage 12-B: Selection-Utility Grounded Training

## Motivation

Stage 12-A-fix1 made the neural RGTA-to-LLM path behaviorally active, but the first 20-example comparison changed only equivalent SQL syntax, whitespace, and parentheses. It did not change schema selection, values, or operators, and therefore produced no net EX gain.

The failure is consistent with an objective mismatch: ordinary token cross-entropy is dominated by common SQL and formatting tokens. It does not require the grounding adapter to improve the decisions that determine execution semantics.

## Token roles

Gold SQL tokens are aligned to character spans and assigned one of four database-independent roles:

- `schema`: exact table and column identifiers from the complete test-time schema;
- `operator`: aggregation, comparison, ordering, grouping, set, join, arithmetic, and type operators;
- `value`: quoted and numeric predicate values;
- `base`: remaining SQL syntax and formatting tokens.

Schema matching uses the current database graph and therefore does not contain handcrafted aliases for a particular BIRD database.

## Objective

For token-level negative log likelihood `n_t`, semantic weighting is:

```text
base=0.5, schema=4.0, operator=2.5, value=3.0
```

The weighted SQL loss is augmented with two counterfactual terms. For key tokens (`schema`, `operator`, and `value`), the real RGTA injection must improve the gold-token log probability over the same frozen LLM with zero injection:

```text
gain_t = log P_normal(y_t) - log P_zero(y_t)
L_utility = mean ReLU(margin - gain_t)
```

For `base` tokens, a preservation loss penalizes unnecessary movement in gold-token NLL:

```text
L_preserve = mean (NLL_normal - NLL_zero)^2
```

The complete training objective is:

```text
L = L_weighted_CE + lambda_cf * L_utility + lambda_preserve * L_preserve
```

This is a residual utility objective: the adapter is rewarded for improving schema-sensitive decisions and discouraged from spending capacity on superficial SQL rewrites.

## Logged diagnostics

Each split reports:

- standard and weighted token CE;
- counterfactual utility loss;
- preservation loss;
- mean key-token log-probability gain;
- key-token improvement rate;
- token counts for all four roles;
- cross-attention and steering scales per decoder layer.

The best checkpoint is selected by `mean_objective_loss` for this objective. Epoch 0 remains an excluded identity baseline.

## Server smoke command

Run from `/data/zhuoaq/Dynamic-Schema-Grounding` after pulling the Stage 12-B commit:

```bash
CUDA_VISIBLE_DEVICES=0,1 python src/training/stage12_train_dynamic_grounded_llm.py \
  --model-path /data/wufan/models/Qwen3-Coder-30B-A3B-Instruct \
  --controller-checkpoint experiments/stage11a_independent_operation_rgta_seed42/dynamic_grounding_controller.pt \
  --controller-summary experiments/stage11a_independent_operation_rgta_seed42/training_summary.json \
  --train-trajectory-file experiments/stage11a_dynamic_trajectories/train_trajectories.jsonl \
  --dev-trajectory-file experiments/stage11a_dynamic_trajectories/dev_trajectories.jsonl \
  --train-full-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --dev-full-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-dir experiments/stage12b_selection_utility_smoke \
  --training-objective semantic_utility \
  --train-limit 50 \
  --dev-limit 20 \
  --epochs 1 \
  --max-length 8192 \
  --gradient-accumulation-steps 4 \
  --schema-token-weight 4.0 \
  --operator-token-weight 2.5 \
  --value-token-weight 3.0 \
  --base-token-weight 0.5 \
  --counterfactual-loss-weight 0.5 \
  --counterfactual-margin 0.05 \
  --preservation-loss-weight 0.1 \
  --controller-device cuda:0 \
  --device-map auto \
  --dtype bfloat16 \
  --trust-remote-code
```

Inspect the result:

```bash
cat experiments/stage12b_selection_utility_smoke/training_summary.json
```

The smoke run is functional when all conditions hold:

1. schema/operator/value token counts are non-zero;
2. trained adapter scales are non-zero;
3. `mean_key_logprob_gain` is finite and rises above the epoch-0 value of zero;
4. `key_improvement_rate` rises above zero;
5. no loss is NaN or infinite.

## Medium-scale decision experiment

If the smoke run passes, train on 1,000 examples and evaluate 100 fixed dev examples:

```bash
CUDA_VISIBLE_DEVICES=0,1 python src/training/stage12_train_dynamic_grounded_llm.py \
  --model-path /data/wufan/models/Qwen3-Coder-30B-A3B-Instruct \
  --controller-checkpoint experiments/stage11a_independent_operation_rgta_seed42/dynamic_grounding_controller.pt \
  --controller-summary experiments/stage11a_independent_operation_rgta_seed42/training_summary.json \
  --train-trajectory-file experiments/stage11a_dynamic_trajectories/train_trajectories.jsonl \
  --dev-trajectory-file experiments/stage11a_dynamic_trajectories/dev_trajectories.jsonl \
  --train-full-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --dev-full-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-dir experiments/stage12b_selection_utility_train1000_seed42 \
  --training-objective semantic_utility \
  --train-limit 1000 \
  --dev-limit 100 \
  --epochs 2 \
  --lr 2e-4 \
  --max-length 8192 \
  --gradient-accumulation-steps 8 \
  --schema-token-weight 4.0 \
  --operator-token-weight 2.5 \
  --value-token-weight 3.0 \
  --base-token-weight 0.5 \
  --counterfactual-loss-weight 0.5 \
  --counterfactual-margin 0.05 \
  --preservation-loss-weight 0.1 \
  --controller-device cuda:0 \
  --device-map auto \
  --dtype bfloat16 \
  --trust-remote-code \
  --seed 42
```

Use the unchanged Stage 12 generation script with this adapter directory to produce `none`, `zero`, and `random` files for the same first 100 dev examples. The decision criterion is not merely changed SQL count: `none` must improve paired EX and produce more schema/operator/value corrections than regressions. If it does not, the next architecture should move semantic control above token decoding into typed relational-algebra plan generation.

