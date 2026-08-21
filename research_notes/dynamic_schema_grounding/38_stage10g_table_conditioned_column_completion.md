# Stage 10-G: Table-Conditioned Column Completion

## Motivation

After scope-aware target repair, the remaining semantic misses are dominated by
columns that never enter the Stage 10 candidate graph.  At the same time, table
recall is already high.  This creates a specific algorithmic opportunity:

```text
high-confidence table belief
        -> retrieve omitted columns inside that table
        -> let Schema-RGTA rerank the enlarged graph
        -> constrained fixed Top-30 output
```

This is candidate-space completion, not a larger final Top-K and not a prompt
rewrite.  Retrieved columns are not automatically selected.

## Inference-safe algorithm

For each sample, Stage 10-G:

1. reads the current table candidates and their upstream priority;
2. computes frozen query--table cosine similarity from the Stage 8G cache;
3. ranks table anchors by equal fusion of normalized structural belief and
   semantic similarity;
4. retrieves semantically closest omitted columns only from the top anchor
   tables;
5. allocates the completion budget round-robin across tables, preventing a wide
   table from monopolizing the graph;
6. rebuilds the induced schema edges, so every added column exchanges RGTA
   messages with its owner table;
7. appends four completion features to every candidate node;
8. leaves final selection to the trained Schema-RGTA and the unchanged fixed
   Top-30 constrained selector.

Gold SQL and labels are never read by steps 1--6. They are used only to generate
training labels for appended nodes and to report the post-hoc candidate ceiling.

## Why this does not overfit BIRD dev errors

- The same fixed rule and budgets are applied to OOF train and dev.
- There are no database names, aliases, manually corrected dev mappings, or
  gold-conditioned thresholds in retrieval.
- Equal-rank fusion and round-robin allocation replace a parameter sweep on dev.
- The first decision gate is candidate recall; downstream Top-30 gains must then
  survive frozen-LLM controls and RGTA reranking.
- Scope-correct labels are used for research diagnosis, while leaderboard EX is
  still evaluated against the official benchmark protocol.

## Server commands

Set paths once and verify every input before running:

```bash
OOF_TRAIN=experiments/stage10b_oof_candidate_graphs/train_factor_graphs.jsonl
DEV_GRAPH=experiments/stage10a_factor_graph_data/dev_factor_graphs.jsonl
FULL_TRAIN=experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl
FULL_DEV=experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl
REL_TRAIN=experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl
REL_DEV=experiments/stage5j_relation_labels_corrected_llm_cards/dev_relation_labels.jsonl
SCOPE_DEV=experiments/stage1_label_extraction_scopefix1/bird_dev_grounding_labels.jsonl
EMBED_CACHE=experiments/stage8g_embedding_cache_corrected_qwen3_06b
OUT=experiments/stage10g_table_conditioned_completion

for FILE in "$OOF_TRAIN" "$DEV_GRAPH" "$FULL_TRAIN" "$FULL_DEV" \
  "$REL_TRAIN" "$REL_DEV" \
  "$SCOPE_DEV" "$EMBED_CACHE/train_index.json" "$EMBED_CACHE/dev_index.json"; do
  test -f "$FILE" || { echo "missing: $FILE"; exit 1; }
done

mkdir -p "$OUT"
```

Build candidate-completed train and dev graphs with the identical rule:

```bash
python src/data/stage10g_table_conditioned_completion.py \
  --factor-graph-file "$OOF_TRAIN" \
  --full-graph-file "$FULL_TRAIN" \
  --relation-file "$REL_TRAIN" \
  --embedding-cache-dir "$EMBED_CACHE" \
  --split train \
  --output-file "$OUT/train_completed.jsonl" \
  --max-anchor-tables 4 \
  --columns-per-table 8 \
  --max-additions 24

python src/data/stage10g_table_conditioned_completion.py \
  --factor-graph-file "$DEV_GRAPH" \
  --full-graph-file "$FULL_DEV" \
  --relation-file "$REL_DEV" \
  --evaluation-label-file "$SCOPE_DEV" \
  --embedding-cache-dir "$EMBED_CACHE" \
  --split dev \
  --output-file "$OUT/dev_completed.jsonl" \
  --max-anchor-tables 4 \
  --columns-per-table 8 \
  --max-additions 24

cat "$OUT/train_completed_summary.json"
cat "$OUT/dev_completed_summary.json"
```

The candidate-only gate is:

- candidate oracle recall increases;
- complete candidate coverage increases;
- recovered samples are primarily missing columns from already-correct tables;
- average candidate growth stays bounded by 24.

If this gate fails, do not call the LLM and do not retrain RGTA.

If it passes, regenerate frozen LLM priors because the candidate identity has
changed:

```bash
export LLM_API_KEY=dummy

python src/data/stage10e_generate_llm_semantic_priors.py \
  --factor-graph-file "$OUT/train_completed.jsonl" \
  --output-file "$OUT/train_priors.jsonl" \
  --base-url http://127.0.0.1:9019/v1 \
  --model qwen2.5-coder-32b \
  --workers 16 \
  --disable-thinking

python src/data/stage10e_generate_llm_semantic_priors.py \
  --factor-graph-file "$OUT/dev_completed.jsonl" \
  --output-file "$OUT/dev_priors.jsonl" \
  --base-url http://127.0.0.1:9019/v1 \
  --model qwen2.5-coder-32b \
  --workers 16 \
  --disable-thinking

python src/data/stage10e_attach_llm_semantic_priors.py \
  --factor-graph-file "$OUT/train_completed.jsonl" \
  --prior-file "$OUT/train_priors.jsonl" \
  --output-file "$OUT/train_normal.jsonl" \
  --control-mode normal

for MODE in normal zero shuffled_node_identity; do
  python src/data/stage10e_attach_llm_semantic_priors.py \
    --factor-graph-file "$OUT/dev_completed.jsonl" \
    --prior-file "$OUT/dev_priors.jsonl" \
    --output-file "$OUT/dev_${MODE}.jsonl" \
    --control-mode "$MODE"
done
```

Train the graph reranker; output remains fixed Top-30:

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage10_train_factor_graph_reranker.py \
  --train-file "$OUT/train_normal.jsonl" \
  --dev-file "$OUT/dev_normal.jsonl" \
  --dev-control-file zero="$OUT/dev_zero.jsonl" \
  --dev-control-file shuffled="$OUT/dev_shuffled_node_identity.jsonl" \
  --embedding-cache-dir "$EMBED_CACHE" \
  --output-dir experiments/stage10g_frozen_llm_schema_rgta_seed42 \
  --model-type schema_rgta \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 8 \
  --lr 1e-4 \
  --gradient-accumulation-steps 16 \
  --coverage-loss-weight 0.0 \
  --structured-coverage-loss-weight 0.1 \
  --structured-coverage-margin 0.1 \
  --output-top-k 30 \
  --patience 3 \
  --device cuda \
  --seed 42
```

## Acceptance criteria

Stage 10-G is retained only if:

1. candidate complete coverage improves before training;
2. constrained Top-30 complete coverage improves after training;
3. normal beats zero and shuffled-node-identity on the same checkpoint;
4. semantic/column recall improves without reducing join connectivity;
5. final official BIRD execution accuracy is reported separately from corrected
   diagnostic labels.
