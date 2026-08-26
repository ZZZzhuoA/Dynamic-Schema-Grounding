# Stage 17-A0: Full-Schema Binary QRGTA

## Purpose

Stage 17-A0 removes the pre-QRGTA candidate bottleneck.  Every database table and
column enters a query-conditioned relational graph transformer, and the first
training objective is deliberately restricted to one binary event:

\[
P(v_i \in S_{gold} \mid Q, G_{full}).
\]

The first version does not predict operation roles or join edges and does not use
Value Index, old operation scores, candidate completion, or query-conditioned LLM
role priors.  Static schema/question cards remain part of the frozen dense
embedding text.

Implementation commit:

```text
bdf5ff5 feat: add full-schema binary QRGTA grounder
```

Main files:

```text
src/modeling/full_schema_qrgta.py
src/training/stage17a_train_full_schema_qrgta.py
src/evaluation/stage17a_evaluate_full_schema_qrgta.py
tests/test_stage17a_full_schema_qrgta.py
```

## Critical data-version invariant

The following four artifacts must be built from the same question/evidence
version and must never be mixed:

1. scope-aware train/dev grounding labels;
2. question semantic cards;
3. full-schema graph examples;
4. query/node embedding cache.

Question correction changes not only the label file.  It changes the query text,
therefore the Question Card, graph `inference_inputs.question`, and query
embedding must also be regenerated.

The failure that exposed this invariant was:

```text
record_index=1577
graph: State the emails of the top 10 Sales Representatives ...
label: state 10 emails of uk sales rep ...
```

Disabling the alignment check would incorrectly train:

```text
old question embedding -> corrected question gold schema
```

Stage 17 must fail loudly on any question, schema ID, schema name, node count, or
cache-index mismatch.

## Naming distinction

These directories have different meanings:

```text
stage1_label_extraction_scopefix1
```

was the immediate **dev-only** audit output.  It normally does not contain a
train label file.

```text
stage1_label_extraction_corrected_scopefix1
```

is the authoritative **train+dev** output based on the corrected training merge.
Stage 17 training must use the latter.

## Reproducible data rebuild

### 1. Build corrected scope-aware labels

Verify the corrected source before starting:

```bash
test -f experiments/stage0_train_correction_merge/merged_train_question_answer.json
```

Build both splits:

```bash
python src/data/stage1_extract_bird_labels.py \
  --bird-dir Data/BIRD \
  --train-question-answer experiments/stage0_train_correction_merge/merged_train_question_answer.json \
  --output-dir experiments/stage1_label_extraction_corrected_scopefix1 \
  --splits train,dev \
  --fk-label-mode explicit_sql
```

Expected line counts:

```bash
wc -l \
  experiments/stage1_label_extraction_corrected_scopefix1/bird_train_grounding_labels.jsonl \
  experiments/stage1_label_extraction_corrected_scopefix1/bird_dev_grounding_labels.jsonl
```

```text
9428 train
1534 dev
```

### 2. Incrementally refresh Question Cards

Schema metadata is unchanged, so existing schema cards may be reused.  Question
cards must be checked against the corrected question/evidence content.  With the
two-GPU vLLM endpoint running at port 9019:

```bash
export LLM_API_KEY=dummy

python src/data/stage8f_llm_card_generation.py \
  --train-labels experiments/stage1_label_extraction_corrected_scopefix1/bird_train_grounding_labels.jsonl \
  --dev-labels experiments/stage1_label_extraction_corrected_scopefix1/bird_dev_grounding_labels.jsonl \
  --output-dir experiments/stage8f_question_cards_corrected_scopefix1 \
  --reuse-card-dir experiments/stage8f_llm_cards_corrected_incremental \
  --splits train,dev \
  --card-types question \
  --base-url http://127.0.0.1:9019/v1 \
  --model qwen2.5-coder-32b \
  --workers 16 \
  --question-max-tokens 1024 \
  --disable-thinking \
  --refresh-mismatched-question-cards \
  --resume \
  --retry-errors
```

Only cards whose normalized question/evidence changed should be regenerated.

Compact the refreshed question cards:

```bash
python src/data/stage8f_compact_llm_cards.py \
  --train-question-cards experiments/stage8f_question_cards_corrected_scopefix1/train_question_cards.jsonl \
  --dev-question-cards experiments/stage8f_question_cards_corrected_scopefix1/dev_question_cards.jsonl \
  --output-dir experiments/stage8f_compact_question_cards_corrected_scopefix1
```

### 3. Rebuild full-schema graphs

Use existing compact schema cards and the refreshed compact question cards:

```bash
python src/data/stage5_build_dsg_data.py \
  --train-labels experiments/stage1_label_extraction_corrected_scopefix1/bird_train_grounding_labels.jsonl \
  --dev-labels experiments/stage1_label_extraction_corrected_scopefix1/bird_dev_grounding_labels.jsonl \
  --train-tables Data/BIRD/train_databases/train_databases/train_tables.json \
  --dev-tables Data/BIRD/dev_tables.json \
  --train-schema-semantic-cards experiments/stage8f_compact_llm_cards_corrected/train_schema_semantic_cards.jsonl \
  --dev-schema-semantic-cards experiments/stage8f_compact_llm_cards_corrected/dev_schema_semantic_cards.jsonl \
  --train-question-cards experiments/stage8f_compact_question_cards_corrected_scopefix1/train_question_cards.jsonl \
  --dev-question-cards experiments/stage8f_compact_question_cards_corrected_scopefix1/dev_question_cards.jsonl \
  --output-dir experiments/stage17a_dsg_data_corrected_scopefix1
```

### 4. Rebuild embeddings

Do not reuse `stage8g_embedding_cache_corrected_qwen3_06b`: its query vectors may
encode the stale question text.

```bash
CUDA_VISIBLE_DEVICES=0 python src/embedding/stage8g_build_embedding_cache.py \
  --train-examples experiments/stage17a_dsg_data_corrected_scopefix1/train_examples.jsonl \
  --dev-examples experiments/stage17a_dsg_data_corrected_scopefix1/dev_examples.jsonl \
  --splits train,dev \
  --output-dir experiments/stage17a_embedding_cache_corrected_scopefix1_qwen3_06b \
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

## Training commands

### Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage17a_train_full_schema_qrgta.py \
  --train-graph-file experiments/stage17a_dsg_data_corrected_scopefix1/train_examples.jsonl \
  --dev-graph-file experiments/stage17a_dsg_data_corrected_scopefix1/dev_examples.jsonl \
  --train-label-file experiments/stage1_label_extraction_corrected_scopefix1/bird_train_grounding_labels.jsonl \
  --dev-label-file experiments/stage1_label_extraction_corrected_scopefix1/bird_dev_grounding_labels.jsonl \
  --embedding-cache-dir experiments/stage17a_embedding_cache_corrected_scopefix1_qwen3_06b \
  --output-dir experiments/stage17a_full_schema_qrgta_smoke \
  --model-type qrgta \
  --hidden-dim 256 \
  --num-layers 3 \
  --num-heads 8 \
  --epochs 2 \
  --gradient-accumulation-steps 16 \
  --selection-metric complete_coverage@30 \
  --train-limit 100 \
  --dev-limit 50 \
  --device cuda \
  --seed 42
```

### Full run

Use the same command without `--train-limit` and `--dev-limit`, set
`--epochs 8`, `--patience 3`, and write to:

```text
experiments/stage17a_full_schema_qrgta_seed42
```

The MLP baseline must use the same graph, labels, and embedding cache, changing
only `--model-type mlp` and the output directory.

## Evaluation

```bash
python src/evaluation/stage17a_evaluate_full_schema_qrgta.py \
  --prediction-file experiments/stage17a_full_schema_qrgta_seed42/dev_predictions.jsonl \
  --label-file experiments/stage1_label_extraction_corrected_scopefix1/bird_dev_grounding_labels.jsonl \
  --output-file experiments/stage17a_full_schema_qrgta_seed42/dev_metrics_recomputed.json
```

Primary checkpoint metric:

```text
complete_coverage@30
```

Tie breakers are `schema_recall@30`, then lower dev loss.

## Do-not-repeat checklist

- Do not use dev-only `stage1_label_extraction_scopefix1` as a train source.
- Do not pair corrected labels with an older graph just because schema IDs match.
- Do not reuse an embedding cache after question/evidence text changes.
- Do not suppress the Stage 17 question/schema/cache alignment errors.
- Do not overwrite old graph/cache directories; preserve them for prior results.
- Do not compare models built from different label/question/card versions.
