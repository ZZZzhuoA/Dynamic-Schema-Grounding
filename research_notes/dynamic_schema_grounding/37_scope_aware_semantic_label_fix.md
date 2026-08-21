# Scope-Aware Semantic Label Fix

## Finding

The Stage 10-F casebook exposed a second source of gold-label inflation after the
FK-closure fix.  Clause extraction resolved an unqualified normalized column name
against every SQL-used table.  This ignored alias ownership and nested-query scope.

Examples include:

- `T1.id` owned by `cards` also labeling `legalities.id` and `rulings.id`;
- `T1.amount` owned by `loan` also labeling `order.amount`;
- `frpm.\`Charter School (Y/N)\`` also labeling `schools.Charter`;
- `Examination.\`aCL IgA\`` also labeling `Laboratory.IGA`;
- outer and inner unqualified `id` references being assigned across both query
  blocks.

On the 115 Stage 10-F semantic-missing cases, the corrected parser removed 56 of
173 missing targets and made 34 samples no longer semantic misses.  This means the
previous `0.9250` semantic-complete result was a conservative underestimate.

## Corrected policy

The SQL label extractor now recursively separates query blocks and applies:

1. exact alias-qualified ownership (`T1.column -> owner table`);
2. exact matching for backtick/bracket/double-quoted identifiers;
3. no short-name matching inside a longer quoted identifier;
4. unqualified-column acceptance only when one table in the local query scope
   owns the name;
5. nested subquery isolation with inherited aliases for correlated references;
6. no fallback that treats a schema table named `order` as used merely because
   the SQL contains `ORDER BY`.

The policy intentionally prefers an unassigned ambiguous reference over a false
positive gold node.

## Immediate dev-only rebuild

This does not require the corrected training merge file:

```bash
python src/data/stage1_extract_bird_labels.py \
  --bird-dir Data/BIRD \
  --output-dir experiments/stage1_label_extraction_scopefix1 \
  --splits dev \
  --fk-label-mode explicit_sql
```

Re-evaluate the frozen Stage 10-F output against the corrected labels:

```bash
python src/evaluation/stage10f_evaluate_join_closure.py \
  --full-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --closure-file experiments/stage10f_steiner_join_closure_seed42/dev_closure.jsonl \
  --exact-label-file experiments/stage1_label_extraction_scopefix1/bird_dev_grounding_labels.jsonl \
  --output-dir experiments/stage10f_steiner_join_closure_seed42/evaluation_scopefix1
```

Regenerate the semantic-miss casebook:

```bash
python src/diagnosis/stage10f_organize_semantic_misses.py \
  --full-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --factor-graph-file experiments/stage10a_factor_graph_data/dev_factor_graphs.jsonl \
  --prediction-file experiments/stage10e_frozen_llm_schema_rgta_full_seed42/dev_predictions.jsonl \
  --prior-file experiments/stage10e_llm_semantic_prior/dev_priors.jsonl \
  --exact-label-file experiments/stage1_label_extraction_scopefix1/bird_dev_grounding_labels.jsonl \
  --output-dir experiments/stage10f_semantic_missing_audit_scopefix1
```

## Full train rebuild gate

Do not rebuild training graphs or retrain RGTA until the frozen dev evaluation is
checked.  If the new labels pass the audit, rebuild train and dev with the existing
corrected train-question merge:

```bash
python src/data/stage1_extract_bird_labels.py \
  --bird-dir Data/BIRD \
  --train-question-answer experiments/stage0_train_correction_merge/merged_train_question_answer.json \
  --output-dir experiments/stage1_label_extraction_corrected_scopefix1 \
  --splits train,dev \
  --fk-label-mode explicit_sql
```

Verify the merge path with `test -f` before running this full command.

