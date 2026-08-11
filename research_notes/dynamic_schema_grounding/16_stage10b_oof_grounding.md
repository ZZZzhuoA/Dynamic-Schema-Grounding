# Stage 10-B: Database-Disjoint OOF Grounding

## Objective

Stage 10-A used first-stage predictions produced by a Stage 8G model trained on the
same 9,428 BIRD training questions. This is in-sample stacking: the reranker sees
cleaner upstream beliefs during training than during cross-database inference.

Stage 10-B replaces these inputs with strict out-of-fold (OOF) predictions:

```text
g_i = Stage8G trained without database(db_i), evaluated on i
```

The split unit is `db_id`, not question. Every database appears in exactly one
held-out fold.

## Leakage boundary

The held-out fold must not select a checkpoint. Each fold therefore uses:

```text
checkpoint_policy = last
fixed epochs = 6
```

The fold is evaluated only after all fixed training epochs complete. The fixed
epoch count is inherited from the previously selected Stage 8G configuration.

## 1. Build five database-disjoint folds

```bash
python src/data/stage10b_build_oof_folds.py \
  --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl \
  --output-dir experiments/stage10b_oof_folds_seed42 \
  --fold-count 5 \
  --seed 42
```

Expected integrity fields:

```text
heldout_record_count = 9428
heldout_record_coverage = 1.0
database_count = 69
database_disjoint = true
```

## 2. Train the fold models

The wrapper reuses the full training relation/graph files and the original train
embedding cache. Index manifests filter records while retaining original
`record_index`, so no embedding arrays are copied.

Sequential execution:

```bash
python src/training/stage10b_run_oof_grounding.py \
  --fold-manifest experiments/stage10b_oof_folds_seed42/fold_manifest.json \
  --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl \
  --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-dir experiments/stage10b_oof_stage8g_rgta_seed42 \
  --fold-ids all \
  --epochs 6 \
  --encoder-type rgta \
  --hidden-dim 96 \
  --num-layers 2 \
  --lr 1e-4 \
  --device cuda
```

Parallel execution on five GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage10b_run_oof_grounding.py --fold-manifest experiments/stage10b_oof_folds_seed42/fold_manifest.json --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b --output-dir experiments/stage10b_oof_stage8g_rgta_seed42 --fold-ids 0 --epochs 6 --device cuda
CUDA_VISIBLE_DEVICES=1 python src/training/stage10b_run_oof_grounding.py --fold-manifest experiments/stage10b_oof_folds_seed42/fold_manifest.json --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b --output-dir experiments/stage10b_oof_stage8g_rgta_seed42 --fold-ids 1 --epochs 6 --device cuda
CUDA_VISIBLE_DEVICES=2 python src/training/stage10b_run_oof_grounding.py --fold-manifest experiments/stage10b_oof_folds_seed42/fold_manifest.json --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b --output-dir experiments/stage10b_oof_stage8g_rgta_seed42 --fold-ids 2 --epochs 6 --device cuda
CUDA_VISIBLE_DEVICES=3 python src/training/stage10b_run_oof_grounding.py --fold-manifest experiments/stage10b_oof_folds_seed42/fold_manifest.json --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b --output-dir experiments/stage10b_oof_stage8g_rgta_seed42 --fold-ids 3 --epochs 6 --device cuda
CUDA_VISIBLE_DEVICES=4 python src/training/stage10b_run_oof_grounding.py --fold-manifest experiments/stage10b_oof_folds_seed42/fold_manifest.json --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b --output-dir experiments/stage10b_oof_stage8g_rgta_seed42 --fold-ids 4 --epochs 6 --device cuda
```

Each process writes a separate run summary, so parallel processes do not overwrite
one another.

## 3. Merge and validate OOF predictions

```bash
python src/evaluation/stage10b_merge_oof_predictions.py \
  --fold-manifest experiments/stage10b_oof_folds_seed42/fold_manifest.json \
  --fold-output-dir experiments/stage10b_oof_stage8g_rgta_seed42 \
  --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl \
  --output-dir experiments/stage10b_oof_merged_rgta_seed42
```

Primary output:

```text
experiments/stage10b_oof_merged_rgta_seed42/train_oof_relation_predictions.jsonl
```

The merger rejects missing folds, duplicate `(record_index, relation_type)` keys,
foreign-fold predictions, and incomplete base-record coverage.

Compare the OOF distribution with the old in-sample train predictions and unseen
dev predictions:

```bash
python src/diagnosis/stage10b_grounding_shift_diagnosis.py \
  --oof-predictions experiments/stage10b_oof_merged_rgta_seed42/train_oof_relation_predictions.jsonl \
  --in-sample-predictions experiments/stage10a_first_stage_train_rgta_seed42/train_relation_predictions.jsonl \
  --dev-predictions experiments/stage8g_corrected_llm_cards_rgta_seed42/dev_relation_predictions.jsonl \
  --output-file experiments/stage10b_oof_merged_rgta_seed42/grounding_shift_diagnosis.json
```

The report compares gold recall/MRR and confidence-shape statistics. It is an
offline diagnostic and is never consumed by training or inference.

## 4. Rebuild train evidence and candidate graphs

The existing training value index can be reused:

```bash
python src/grounding/stage9_value_join_completion.py \
  --relation-predictions experiments/stage10b_oof_merged_rgta_seed42/train_oof_relation_predictions.jsonl \
  --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl \
  --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --value-index experiments/stage10a_value_index/train_value_index.sqlite \
  --output-dir experiments/stage10b_oof_train_evidence_rgta_seed42 \
  --enable-value-index \
  --enable-join-path \
  --value-fusion-mode gated

python src/data/stage10_build_factor_graph_data.py \
  --relation-predictions experiments/stage10b_oof_merged_rgta_seed42/train_oof_relation_predictions.jsonl \
  --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl \
  --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --evidence-debug experiments/stage10b_oof_train_evidence_rgta_seed42/evidence_debug.jsonl \
  --output-file experiments/stage10b_oof_candidate_graphs/train_factor_graphs.jsonl \
  --relation-top-m 20 \
  --max-candidates 80
```

Dev candidate graphs remain the unchanged Stage 10-A graphs because dev predictions
already come from a model trained only on BIRD train.

## 5. Controlled Schema-RGTA comparison

```bash
python src/training/stage10_train_factor_graph_reranker.py \
  --train-file experiments/stage10b_oof_candidate_graphs/train_factor_graphs.jsonl \
  --dev-file experiments/stage10a_factor_graph_data/dev_factor_graphs.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-dir experiments/stage10b_schema_rgta_oof_seed42 \
  --model-type schema_rgta \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 8 \
  --device cuda \
  --seed 42
```

The first comparison keeps the Stage 10 reranker hyperparameters unchanged. Only
the train grounding source changes from in-sample predictions to OOF predictions.
Learning-rate/batch changes should be tested afterward as a separate ablation.

## Success criteria

- OOF merge covers all 9,428 training records exactly once.
- Candidate oracle recall remains high enough that candidate construction is not the bottleneck.
- Schema-RGTA retains a clear gain over the MLP reranker.
- Best dev epoch becomes later or the dev curve becomes flatter.
- Complete Coverage@30 improves beyond the current 0.797914, or becomes more stable across seeds.
