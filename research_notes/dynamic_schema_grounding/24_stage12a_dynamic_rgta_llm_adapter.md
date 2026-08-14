# Stage 12-A: Dynamic RGTA–LLM Neural Adapter

## Objective

Test whether dynamic schema grounding improves final SQL execution accuracy when it is injected into the LLM hidden state, rather than serialized as a reduced-schema prompt.

The controlled comparison is:

- `zero`: complete schema prompt and frozen base LLM, with neural grounding injection disabled;
- `none`: the same prompt and base LLM, with the trained dynamic RGTA adapter enabled.

This isolates the effect of neural grounding. Both conditions retain the complete schema, so a missed GNN candidate does not make a gold column physically unavailable to the LLM.

## Architecture

```text
complete database schema ----------------------> frozen causal LLM
                                                       |
question + generated partial SQL                       | hidden states
               |                                       v
               +--> frozen independent operation-RGTA belief
                                  |
                         weighted schema tokens
                                  |
                      cross-attention adapter
                                  |
                    uncertainty/route steering
                                  |
                         selected decoder layers
                                  |
                              next SQL token
```

The operation state is inferred from the current SQL prefix (`SELECT`, `JOIN`, `WHERE`, `GROUP BY`, or `ORDER BY`). The controller recomputes a schema belief from the actual generated prefix. It does not use gold SQL or gold schema labels at inference time.

## Training

The base LLM and the independent Stage 11 controller are frozen. Only cross-attention and steering adapters are optimized with teacher-forced SQL token loss. Gold SQL is used only to form the training trajectory and target tokens.

Adapters are inserted at configurable decoder-depth fractions. Their output scales are initialized to zero, making the initial model exactly equivalent to the frozen base LLM.

## Evaluation protocol

1. Run a 20-example smoke train.
2. Generate the same 20 examples with `none` and `zero` interventions.
3. Require non-identical SQL outputs for at least some examples; otherwise the adapter has no measurable behavioral effect.
4. Compare EX and execution success.
5. If the smoke test is functional, train on all available corrected training examples and evaluate all 1,534 BIRD dev examples.

The main claim is supported only if `none` improves full-dev EX over `zero` across seeds. Conditional grounding recall is diagnostic, not the final success metric.

## Server execution commands

Run all commands from the repository root:

```bash
cd /data/zhuoaq/Dynamic-Schema-Grounding
git pull
```

The commands below use the already trained independent operation-RGTA controller and the corrected Stage 8G graph/embedding artifacts:

```bash
test -f experiments/stage11a_independent_operation_rgta_seed42/dynamic_grounding_controller.pt
test -f experiments/stage11a_independent_operation_rgta_seed42/training_summary.json
test -f experiments/stage11a_dynamic_trajectories/train_trajectories.jsonl
test -f experiments/stage11a_dynamic_trajectories/dev_trajectories.jsonl
test -f experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl
test -f experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl
test -f experiments/stage8g_embedding_cache_corrected_qwen3_06b/train_index.json
test -f experiments/stage8g_embedding_cache_corrected_qwen3_06b/dev_index.json
test -d /data/wufan/models/Qwen3-Coder-30B-A3B-Instruct
```

An empty output from all `test` commands means that every required artifact exists. Stop and reconstruct the missing upstream artifact if any command returns a non-zero status.

### 1. Twenty-example smoke training

Use two visible GPUs. `device_map auto` distributes the frozen LLM, while `cuda:0` refers to the first visible GPU and hosts the much smaller frozen RGTA controller.

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
  --output-dir experiments/stage12a_dynamic_rgta_llm_adapter_smoke \
  --train-limit 20 \
  --dev-limit 20 \
  --epochs 1 \
  --max-length 8192 \
  --gradient-accumulation-steps 4 \
  --controller-device cuda:0 \
  --device-map auto \
  --dtype bfloat16 \
  --trust-remote-code
```

The expected output directory contains:

```text
adapter_config.json
dynamic_llm_adapters.pt
training_summary.json
```

Epoch 0 is the exact identity-initialized adapter baseline. Confirm that epoch 1 has a finite loss and that at least one adapter checkpoint is written.

Epoch 0 is excluded from trained-checkpoint selection. The files have distinct roles:

```text
identity_dynamic_llm_adapters.pt  # untrained exact-identity diagnostic
dynamic_llm_adapters.pt           # best trained epoch (epoch >= 1)
last_dynamic_llm_adapters.pt      # final trained epoch
```

`training_summary.json` records raw/effective cross-attention and steering scales for every epoch. Before generation, confirm that at least one scale in `best_adapter_scales` is non-zero.

### 2. Generate the normal neural-injection condition

```bash
mkdir -p experiments/stage12a_dynamic_generation

CUDA_VISIBLE_DEVICES=0,1 python src/decoding/stage12_generate_dynamic_grounded_llm.py \
  --model-path /data/wufan/models/Qwen3-Coder-30B-A3B-Instruct \
  --adapter-dir experiments/stage12a_dynamic_rgta_llm_adapter_smoke \
  --controller-checkpoint experiments/stage11a_independent_operation_rgta_seed42/dynamic_grounding_controller.pt \
  --controller-summary experiments/stage11a_independent_operation_rgta_seed42/training_summary.json \
  --trajectory-file experiments/stage11a_dynamic_trajectories/dev_trajectories.jsonl \
  --full-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-file experiments/stage12a_dynamic_generation/normal_limit20.jsonl \
  --split dev \
  --limit 20 \
  --refresh-interval 4 \
  --max-new-tokens 512 \
  --intervention none \
  --controller-device cuda:0 \
  --device-map auto \
  --dtype bfloat16 \
  --trust-remote-code
```

### 3. Generate the exact zero-injection baseline

This uses the same trained adapter and complete-schema prompt, but sets both grounding tokens and steering state to zero.

```bash
CUDA_VISIBLE_DEVICES=0,1 python src/decoding/stage12_generate_dynamic_grounded_llm.py \
  --model-path /data/wufan/models/Qwen3-Coder-30B-A3B-Instruct \
  --adapter-dir experiments/stage12a_dynamic_rgta_llm_adapter_smoke \
  --controller-checkpoint experiments/stage11a_independent_operation_rgta_seed42/dynamic_grounding_controller.pt \
  --controller-summary experiments/stage11a_independent_operation_rgta_seed42/training_summary.json \
  --trajectory-file experiments/stage11a_dynamic_trajectories/dev_trajectories.jsonl \
  --full-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-file experiments/stage12a_dynamic_generation/zero_limit20.jsonl \
  --split dev \
  --limit 20 \
  --refresh-interval 4 \
  --max-new-tokens 512 \
  --intervention zero \
  --controller-device cuda:0 \
  --device-map auto \
  --dtype bfloat16 \
  --trust-remote-code
```

### 4. Evaluate both conditions

```bash
python src/evaluation/stage3_evaluate_sql.py \
  --generation-file experiments/stage12a_dynamic_generation/normal_limit20.jsonl

python src/evaluation/stage3_evaluate_sql.py \
  --generation-file experiments/stage12a_dynamic_generation/zero_limit20.jsonl
```

Also verify that the adapter changes at least some generated SQL:

```bash
python - <<'PY'
import json

def load(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

normal = load("experiments/stage12a_dynamic_generation/normal_limit20.jsonl")
zero = load("experiments/stage12a_dynamic_generation/zero_limit20.jsonl")
different = [
    index for index, (left, right) in enumerate(zip(normal, zero))
    if left.get("generated_sql") != right.get("generated_sql")
]
print({"sample_count": len(normal), "different_sql_count": len(different), "indices": different})
PY
```

If `different_sql_count` is zero, do not start full training: the injection path is behaviorally inactive and must be diagnosed first.

### 5. Optional random-grounding causal control

```bash
CUDA_VISIBLE_DEVICES=0,1 python src/decoding/stage12_generate_dynamic_grounded_llm.py \
  --model-path /data/wufan/models/Qwen3-Coder-30B-A3B-Instruct \
  --adapter-dir experiments/stage12a_dynamic_rgta_llm_adapter_smoke \
  --controller-checkpoint experiments/stage11a_independent_operation_rgta_seed42/dynamic_grounding_controller.pt \
  --controller-summary experiments/stage11a_independent_operation_rgta_seed42/training_summary.json \
  --trajectory-file experiments/stage11a_dynamic_trajectories/dev_trajectories.jsonl \
  --full-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-file experiments/stage12a_dynamic_generation/random_limit20.jsonl \
  --split dev \
  --limit 20 \
  --refresh-interval 4 \
  --max-new-tokens 512 \
  --intervention random \
  --controller-device cuda:0 \
  --device-map auto \
  --dtype bfloat16 \
  --trust-remote-code

python src/evaluation/stage3_evaluate_sql.py \
  --generation-file experiments/stage12a_dynamic_generation/random_limit20.jsonl
```

Random grounding should not systematically outperform the real RGTA state. If it does, any gain is likely caused by generic perturbation rather than schema grounding.

### 6. Full adapter training

Only run this after the smoke test confirms finite training, non-zero behavioral influence, and no obvious EX collapse.

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
  --output-dir experiments/stage12a_dynamic_rgta_llm_adapter_full_seed42 \
  --dev-limit 100 \
  --epochs 1 \
  --lr 2e-4 \
  --max-length 8192 \
  --gradient-accumulation-steps 8 \
  --controller-device cuda:0 \
  --device-map auto \
  --dtype bfloat16 \
  --trust-remote-code \
  --seed 42
```

### 7. Full 1,534-example generation and EX evaluation

Run the normal condition:

```bash
CUDA_VISIBLE_DEVICES=0,1 python src/decoding/stage12_generate_dynamic_grounded_llm.py \
  --model-path /data/wufan/models/Qwen3-Coder-30B-A3B-Instruct \
  --adapter-dir experiments/stage12a_dynamic_rgta_llm_adapter_full_seed42 \
  --controller-checkpoint experiments/stage11a_independent_operation_rgta_seed42/dynamic_grounding_controller.pt \
  --controller-summary experiments/stage11a_independent_operation_rgta_seed42/training_summary.json \
  --trajectory-file experiments/stage11a_dynamic_trajectories/dev_trajectories.jsonl \
  --full-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-file experiments/stage12a_dynamic_generation/normal_full1534_seed42.jsonl \
  --split dev \
  --limit 1534 \
  --refresh-interval 4 \
  --max-new-tokens 512 \
  --intervention none \
  --controller-device cuda:0 \
  --device-map auto \
  --dtype bfloat16 \
  --trust-remote-code

python src/evaluation/stage3_evaluate_sql.py \
  --generation-file experiments/stage12a_dynamic_generation/normal_full1534_seed42.jsonl
```

Then repeat the same generation command with these two substitutions:

```text
--output-file experiments/stage12a_dynamic_generation/zero_full1534_seed42.jsonl
--intervention zero
```

Evaluate the zero file:

```bash
python src/evaluation/stage3_evaluate_sql.py \
  --generation-file experiments/stage12a_dynamic_generation/zero_full1534_seed42.jsonl
```

The primary result is paired full-dev EX for `none` versus `zero`. Execution success rate and changed-SQL count are secondary diagnostics.

## Decision rule

- Continue Stage 12 if neural injection produces stable positive EX and its random/negated interventions degrade performance.
- If EX is unchanged or worse, do not return to prompt selection. Replace token-level steering with a stronger structure, such as relation-algebra plan decoding with schema recovery or a jointly trained constrained decoder.
