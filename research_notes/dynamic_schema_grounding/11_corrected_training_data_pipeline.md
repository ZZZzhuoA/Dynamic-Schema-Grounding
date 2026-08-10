# Corrected BIRD Training Data Pipeline

## Purpose

`Data/BIRD/processed_final_data.json` is a partial correction set, not a replacement for the
9,428-record BIRD training split. Stage 0 overlays confidently aligned corrections onto
`bird-schema/train_question_answer.json` while preserving the original file.

The correction set must be applied before schema-label, clause-label, relation-label, graph, and
embedding generation. Replacing SQL after those artifacts have been built leaves stale supervision.

## Stage 0: audited correction merge

```bash
python src/data/stage0_merge_train_corrections.py \
  --train-question-answer Data/BIRD/bird-schema/train_question_answer.json \
  --corrections Data/BIRD/processed_final_data.json \
  --output-dir experiments/stage0_train_correction_merge
```

Outputs:

- `merged_train_question_answer.json`: corrected 9,428-record training source.
- `correction_manifest.jsonl`: original/corrected values, matching method, and change flags.
- `unresolved_corrections.jsonl`: corrections that were not safe to apply automatically.
- `review_map_template.json`: template for explicit manual mappings.
- `summary.json`: merge counts and configuration.

Changed SQL invalidates the old `hit_info`. The merger therefore clears `hit_info` only when SQL
changes; Stage 1 then derives labels from the corrected SQL and schema graph. The unsafe legacy
behavior can be requested explicitly with `--retain-changed-hit-info`.

The current correction file contains 2,375 records. With the default conservative thresholds,
2,360 are applied and 15 remain unresolved. The merged output always keeps the original training
length.

## Resolving rewritten or ambiguous questions

Review `unresolved_corrections.jsonl`, then fill `original_index` in a copy of
`review_map_template.json`. Null entries are ignored.

```bash
python src/data/stage0_merge_train_corrections.py \
  --review-map experiments/stage0_train_correction_merge/review_map_reviewed.json \
  --output-dir experiments/stage0_train_correction_merge_reviewed
```

Manual mappings are rejected if they cross database boundaries or reuse an original record.

## Stage 1: rebuild schema grounding labels

```bash
python src/data/stage1_extract_bird_labels.py \
  --train-question-answer experiments/stage0_train_correction_merge/merged_train_question_answer.json \
  --output-dir experiments/stage1_label_extraction_corrected \
  --splits train,dev
```

The input override is recorded in `bird_label_statistics.json`, making corrected and uncorrected
runs distinguishable.

## Rebuild dependent supervision

Clause labels:

```bash
python src/data/stage5g_build_clause_labels.py \
  --train-labels experiments/stage1_label_extraction_corrected/bird_train_grounding_labels.jsonl \
  --dev-labels experiments/stage1_label_extraction_corrected/bird_dev_grounding_labels.jsonl \
  --output-dir experiments/stage5g_clause_labels_corrected
```

Relation labels:

```bash
python src/data/stage5j_build_relation_labels.py \
  --train-clause-labels experiments/stage5g_clause_labels_corrected/train_clause_labels.jsonl \
  --dev-clause-labels experiments/stage5g_clause_labels_corrected/dev_clause_labels.jsonl \
  --output-dir experiments/stage5j_relation_labels_corrected
```

Question semantic cards must also be regenerated for records whose question or evidence changed.
Schema cards can be reused because the database schemas are unchanged. After card regeneration,
rebuild graph examples and dense embedding caches before retraining the relation-conditioned
grounder.

## Parallel and incremental Stage 8F generation

Stage 8F supports concurrent requests for vLLM continuous batching. It can also seed a new output
directory from one or more old Stage 8F directories. Cached question cards are reused only when the
normalized question and evidence still match; changed records are regenerated.

```bash
python src/data/stage8f_llm_card_generation.py \
  --train-labels experiments/stage1_label_extraction_corrected/bird_train_grounding_labels.jsonl \
  --dev-labels experiments/stage1_label_extraction_corrected/bird_dev_grounding_labels.jsonl \
  --output-dir experiments/stage8f_llm_cards_corrected_incremental \
  --reuse-card-dir experiments/OLD_TRAIN_CARD_DIR \
  --reuse-card-dir experiments/OLD_DEV_CARD_DIR \
  --splits train,dev \
  --card-types both \
  --workers 8 \
  --question-max-tokens 512 \
  --disable-thinking \
  --resume \
  --retry-errors
```

Every completed result is appended by the main thread, so workers never write the same file
concurrently. Final files are canonicalized into source order. Start with 8 workers and increase to
16 only if vLLM has spare GPU capacity and stable latency.

Stage 8F now records `schema_fallback_card_count`, `schema_fallback_rate`, and
`schema_chunk_error_count`. The command fails by default if more than 25% of schema cards use the
fallback representation. `--retry-errors` also regenerates an old database cache when its cached
fallback rate exceeds this threshold, even if an older status file incorrectly recorded
`error: null`. Do not use `--allow-excessive-schema-fallback` for model training.

Stage 5J also requires `METRIC_TARGET`, `TEMPORAL_FILTER`, and `FORMULA_COMPONENT` to be non-empty
on the full dev split. This prevents a syntactically successful run from silently collapsing the
relation-label space because of low-quality semantic cards.

## Full corrected LLM-card training pipeline

After a two-GPU vLLM server is available at `http://127.0.0.1:9019/v1`, regenerate failed schema
card caches and changed question cards:

```bash
export LLM_API_KEY=dummy

python src/data/stage8f_llm_card_generation.py \
  --train-labels experiments/stage1_label_extraction_corrected/bird_train_grounding_labels.jsonl \
  --dev-labels experiments/stage1_label_extraction_corrected/bird_dev_grounding_labels.jsonl \
  --output-dir experiments/stage8f_llm_cards_corrected_incremental \
  --reuse-card-dir experiments/stage8f_llm_cards_qwen25_train1000 \
  --reuse-card-dir experiments/stage8f_llm_cards_qwen25_dev \
  --splits train,dev \
  --card-types both \
  --base-url http://127.0.0.1:9019/v1 \
  --model qwen2.5-coder-32b \
  --workers 16 \
  --schema-card-mode table \
  --max-tokens 8192 \
  --question-max-tokens 1024 \
  --disable-thinking \
  --resume \
  --retry-errors \
  --max-schema-fallback-rate 0.05
```

Compact cards, then rebuild relation supervision and graph examples from the same card source:

```bash
python src/data/stage8f_compact_llm_cards.py \
  --train-schema-cards experiments/stage8f_llm_cards_corrected_incremental/train_schema_semantic_cards.jsonl \
  --train-question-cards experiments/stage8f_llm_cards_corrected_incremental/train_question_cards.jsonl \
  --dev-schema-cards experiments/stage8f_llm_cards_corrected_incremental/dev_schema_semantic_cards.jsonl \
  --dev-question-cards experiments/stage8f_llm_cards_corrected_incremental/dev_question_cards.jsonl \
  --output-dir experiments/stage8f_compact_llm_cards_corrected

python src/data/stage5j_build_relation_labels.py \
  --train-clause-labels experiments/stage5g_clause_labels_corrected/train_clause_labels.jsonl \
  --dev-clause-labels experiments/stage5g_clause_labels_corrected/dev_clause_labels.jsonl \
  --train-schema-semantic-cards experiments/stage8f_compact_llm_cards_corrected/train_schema_semantic_cards.jsonl \
  --dev-schema-semantic-cards experiments/stage8f_compact_llm_cards_corrected/dev_schema_semantic_cards.jsonl \
  --output-dir experiments/stage5j_relation_labels_corrected_llm_cards

python src/data/stage5_build_dsg_data.py \
  --train-labels experiments/stage1_label_extraction_corrected/bird_train_grounding_labels.jsonl \
  --dev-labels experiments/stage1_label_extraction_corrected/bird_dev_grounding_labels.jsonl \
  --train-tables Data/BIRD/train_databases/train_databases/train_tables.json \
  --dev-tables Data/BIRD/dev_tables.json \
  --train-schema-semantic-cards experiments/stage8f_compact_llm_cards_corrected/train_schema_semantic_cards.jsonl \
  --dev-schema-semantic-cards experiments/stage8f_compact_llm_cards_corrected/dev_schema_semantic_cards.jsonl \
  --train-question-cards experiments/stage8f_compact_llm_cards_corrected/train_question_cards.jsonl \
  --dev-question-cards experiments/stage8f_compact_llm_cards_corrected/dev_question_cards.jsonl \
  --output-dir experiments/stage8g_dsg_data_corrected_llm_cards
```

Build the dense embedding cache once:

```bash
CUDA_VISIBLE_DEVICES=0 python src/embedding/stage8g_build_embedding_cache.py \
  --train-examples experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --dev-examples experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --splits train,dev \
  --output-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
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

Train matched full-data RGCN and RGTA runs. The only architectural difference between the two
commands is `--encoder-type`:

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage8g_train_dense_relation_grounder.py \
  --train-relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl \
  --dev-relation-file experiments/stage5j_relation_labels_corrected_llm_cards/dev_relation_labels.jsonl \
  --train-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --dev-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-dir experiments/stage8g_corrected_llm_cards_rgcn_seed42 \
  --train-limit 9428 \
  --dev-limit 1534 \
  --feature-mode dense \
  --encoder-type rgcn \
  --use-relation-conditioned-prior \
  --epochs 8 \
  --hidden-dim 96 \
  --num-layers 2 \
  --lr 1e-4 \
  --patience 3 \
  --seed 42 \
  --device cuda \
  --output-top-k 30

CUDA_VISIBLE_DEVICES=1 python src/training/stage8g_train_dense_relation_grounder.py \
  --train-relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl \
  --dev-relation-file experiments/stage5j_relation_labels_corrected_llm_cards/dev_relation_labels.jsonl \
  --train-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --dev-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-dir experiments/stage8g_corrected_llm_cards_rgta_seed42 \
  --train-limit 9428 \
  --dev-limit 1534 \
  --feature-mode dense \
  --encoder-type rgta \
  --use-relation-conditioned-prior \
  --epochs 8 \
  --hidden-dim 96 \
  --num-layers 2 \
  --lr 1e-4 \
  --patience 3 \
  --seed 42 \
  --device cuda \
  --output-top-k 30
```

## Experimental control

Keep both pipelines:

- original labels: data-noise baseline;
- corrected labels: primary training setting.

This provides a direct test of whether improved supervision changes schema recall, complete schema
coverage, relation-specific recall, and final execution accuracy.
