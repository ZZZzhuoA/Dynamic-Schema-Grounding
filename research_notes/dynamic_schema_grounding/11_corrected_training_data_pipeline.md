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

## Experimental control

Keep both pipelines:

- original labels: data-noise baseline;
- corrected labels: primary training setting.

This provides a direct test of whether improved supervision changes schema recall, complete schema
coverage, relation-specific recall, and final execution accuracy.
