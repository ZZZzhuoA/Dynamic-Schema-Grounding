# Stage 10-E: Frozen LLM Semantic Prior + Schema RGTA

## Hypothesis

The complete-coverage audit separated the remaining failures into two mechanisms:

- semantic targets are missing from the selected set;
- the selected semantic tables are not connected by a legal schema path.

A schema graph is appropriate for the second mechanism but has no reason to
decode every abbreviation, business concept, implicit metric, or question-role
mapping by itself.  Stage 10-E therefore uses a frozen LLM as an inference-only
semantic sensor and keeps RGTA as the trainable structural reasoner.

```text
question + evidence + candidate schema metadata
                    |
                    v
             frozen LLM call
                    |
        validated role-by-node prior
                    |  stop-gradient / cached JSONL
                    v
 dense node + numeric evidence + LLM prior
                    |
                    v
               Schema RGTA
                    |
                    v
      connectivity-aware Top-30 selector
```

The LLM prompt never receives gold SQL, gold schema IDs, or execution labels.
Only the GNN, fusion/scoring layers, and structured selector are trained.

## Semantic-prior feature

For every candidate schema node, the LLM assigns confidence to ten existing
operation roles:

```text
OUTPUT_TARGET, ENTITY_NAME, METRIC_TARGET, PREDICATE_COLUMN,
VALUE_ANCHOR, TEMPORAL_FILTER, ORDER_KEY, GROUP_KEY,
JOIN_BRIDGE, FORMULA_COMPONENT
```

The attachment step appends 13 numeric features:

```text
prior_present + max_role_score + mean_role_score + ten role scores
```

Unknown schema IDs are rejected. Confidence is clamped to `[0, 1]`, duplicate
predictions use the maximum score, and a source fingerprint prevents stale priors
from being attached to a changed question or candidate graph.

## Causal controls

The same best RGTA checkpoint is evaluated under three inputs:

- `normal`: correct node--prior association;
- `zero`: all 13 LLM features are zero;
- `shuffled_node_identity`: the same per-query prior vectors are permuted among
  candidate nodes, preserving their distribution while destroying semantic
  identity.

Controls are never used for checkpoint selection. Evidence for semantic use
requires `normal` to outperform both controls, especially on column recall,
semantic completeness, and final structural completeness.

## Server smoke experiment (1000 train / 100 dev)

If an OpenAI-compatible vLLM server is not already running:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve \
  /data/1_pretrained_models/Qwen2.5-Coder-32B-Instruct \
  --served-model-name qwen2.5-coder-32b \
  --tensor-parallel-size 2 \
  --port 9019 \
  --dtype bfloat16 \
  --trust-remote-code
```

In another shell:

```bash
export LLM_API_KEY=dummy
mkdir -p experiments/stage10e_llm_semantic_prior_smoke

python src/data/stage10e_generate_llm_semantic_priors.py \
  --factor-graph-file experiments/stage10b_oof_candidate_graphs/train_factor_graphs.jsonl \
  --output-file experiments/stage10e_llm_semantic_prior_smoke/train_priors.jsonl \
  --base-url http://127.0.0.1:9019/v1 \
  --model qwen2.5-coder-32b \
  --workers 16 \
  --limit 1000 \
  --disable-thinking

python src/data/stage10e_generate_llm_semantic_priors.py \
  --factor-graph-file experiments/stage10a_factor_graph_data/dev_factor_graphs.jsonl \
  --output-file experiments/stage10e_llm_semantic_prior_smoke/dev_priors.jsonl \
  --base-url http://127.0.0.1:9019/v1 \
  --model qwen2.5-coder-32b \
  --workers 16 \
  --limit 100 \
  --disable-thinking
```

Attach the normal training features and all three aligned dev variants:

```bash
python src/data/stage10e_attach_llm_semantic_priors.py \
  --factor-graph-file experiments/stage10b_oof_candidate_graphs/train_factor_graphs.jsonl \
  --prior-file experiments/stage10e_llm_semantic_prior_smoke/train_priors.jsonl \
  --output-file experiments/stage10e_llm_semantic_prior_smoke/train_normal.jsonl \
  --control-mode normal \
  --limit 1000

for MODE in normal zero shuffled_node_identity; do
  python src/data/stage10e_attach_llm_semantic_priors.py \
    --factor-graph-file experiments/stage10a_factor_graph_data/dev_factor_graphs.jsonl \
    --prior-file experiments/stage10e_llm_semantic_prior_smoke/dev_priors.jsonl \
    --output-file "experiments/stage10e_llm_semantic_prior_smoke/dev_${MODE}.jsonl" \
    --control-mode "$MODE" \
    --limit 100
done
```

Train only the downstream Schema-RGTA and evaluate both controls from its best
checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage10_train_factor_graph_reranker.py \
  --train-file experiments/stage10e_llm_semantic_prior_smoke/train_normal.jsonl \
  --dev-file experiments/stage10e_llm_semantic_prior_smoke/dev_normal.jsonl \
  --dev-control-file zero=experiments/stage10e_llm_semantic_prior_smoke/dev_zero.jsonl \
  --dev-control-file shuffled=experiments/stage10e_llm_semantic_prior_smoke/dev_shuffled_node_identity.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-dir experiments/stage10e_frozen_llm_schema_rgta_smoke_seed42 \
  --model-type schema_rgta \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 8 \
  --lr 1e-4 \
  --gradient-accumulation-steps 16 \
  --coverage-loss-weight 0.0 \
  --structured-coverage-loss-weight 0.1 \
  --structured-coverage-margin 0.1 \
  --patience 3 \
  --device cuda \
  --seed 42
```

The bounded files already contain only 1000/100 records, so separate training
limits are unnecessary. For the full experiment, rerun prior generation and
attachment without `--limit`, use a new output directory, and keep every other
hyperparameter fixed for a fair Stage 10-D comparison.

## Acceptance criteria

Do not accept the method merely because a separately trained model improves.
Require all of the following:

1. normal complete coverage exceeds the Stage 10-D baseline;
2. normal exceeds zero on the same checkpoint;
3. normal exceeds shuffled-node-identity on the same checkpoint;
4. the gain is concentrated in semantic/column failures rather than only tables;
5. join connectivity is preserved rather than traded away for semantic recall.
