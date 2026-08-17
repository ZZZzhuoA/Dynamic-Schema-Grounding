# Stage 13-B1: Action-Synchronous Typed Pointer Decoder

## Objective

Stage 12 showed that a generic schema residual applied to every LLM token can corrupt operators,
values, and ordinary language tokens. Stage 13-B1 instead exposes schema grounding only at typed
relational-action boundaries.

```text
question embedding + complete schema-node embeddings
                         |
               state-conditioned RGTA
                         |
                 controller state
                         |
        +-------------+--------------+------------------+
        |             |              |                  |
    action head   operator head  typed pointers     value route head
        |         =/AVG/DESC/... table/column/FK edge    |
        +-------------+--------------+------------------+
                         |
               teacher-forced transition
                         |
                 next action boundary
```

The decoder accepts an optional `plan_hidden` vector. B1 uses the cached question embedding as the
initial plan state; a later LLM integration can supply the hidden state at the current action token
without changing the graph-pointer architecture.

## Clean trajectory policy

A record is retained only when:

```text
parse_status == supported_flat
schema_label_coverage == 1.0
join_path_connected == true
schema graph and action sequence are non-empty
```

The preparation stage removes gold SQL text and appends an explicit `STOP` action. Nested and set
queries remain audit records for the later recursive-scope decoder.

## Build clean trajectories

```bash
python src/data/stage13b_prepare_typed_trajectories.py \
  --train-file experiments/stage13a_typed_ra_typefix1/train_typed_ra.jsonl \
  --dev-file experiments/stage13a_typed_ra_typefix1/dev_typed_ra.jsonl \
  --output-dir experiments/stage13b_clean_typed_trajectories
```

## Build a matching typefix1 embedding cache

Old caches contain shifted column types and must not be reused.

```bash
CUDA_VISIBLE_DEVICES=0 python src/embedding/stage8g_build_embedding_cache.py \
  --train-examples experiments/stage13a_dsg_data_typefix1/train_examples.jsonl \
  --dev-examples experiments/stage13a_dsg_data_typefix1/dev_examples.jsonl \
  --splits train,dev \
  --output-dir experiments/stage13b_embedding_cache_typefix1_qwen3_06b \
  --model-path /data/1_pretrained_models/Qwen3-Embedding-0.6B \
  --batch-size 32 \
  --max-length 512 \
  --pooling last \
  --normalize \
  --device cuda \
  --dtype bfloat16 \
  --trust-remote-code \
  --deduplicate-node-texts
```

## Smoke training

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage13b_train_typed_ra_decoder.py \
  --train-file experiments/stage13b_clean_typed_trajectories/train_trajectories.jsonl \
  --dev-file experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl \
  --embedding-cache-dir experiments/stage13b_embedding_cache_typefix1_qwen3_06b \
  --output-dir experiments/stage13b_typed_ra_decoder_smoke \
  --train-limit 100 \
  --dev-limit 100 \
  --hidden-dim 128 \
  --num-layers 2 \
  --epochs 2 \
  --device cuda \
  --seed 42
```

## Full training

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage13b_train_typed_ra_decoder.py \
  --train-file experiments/stage13b_clean_typed_trajectories/train_trajectories.jsonl \
  --dev-file experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl \
  --embedding-cache-dir experiments/stage13b_embedding_cache_typefix1_qwen3_06b \
  --output-dir experiments/stage13b_typed_ra_decoder_rgta_seed42 \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 8 \
  --lr 1e-4 \
  --device cuda \
  --seed 42
```

Primary diagnostics are action accuracy, typed table/column recall, FK-edge recall, value-route
recall, operator recall, and complete-step rate. The preparation summary also reports what fraction
of gold join edges are present in the test-time FK candidate graph; explicit non-FK equality joins
remain supervised through their JOIN column pointers. SQL EX is deferred until the typed DAG compiler and LLM action
bridge are connected.
