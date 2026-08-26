# Dynamic Schema Grounding 复现手册

本手册是当前权威复现路径：从外部 BIRD 数据重建到 Stage 17-A1。历史路线只作为档案，不应与
当前数据链混用。

## 1. 外部依赖与复现限制

下列内容不在 Git 中，必须外部提供：

```text
Data/BIRD/
Data/BIRD/processed_final_data.json
/data/1_pretrained_models/Qwen3-Embedding-0.6B
/data/1_pretrained_models/Qwen2.5-Coder-32B-Instruct
```

仓库目前没有锁定的 `requirements.txt`、Conda YAML 或 lockfile。复现时应另行记录 Python、
PyTorch、Transformers、CUDA 和 vLLM 版本。vLLM 只用于 LLM cards/SQL candidates；Stage 17
QRGTA 训练不加载在线 LLM。

代码基线为 `9117c17` 或包含它的后续提交。

## 2. 权威数据链

| Artifact | 路径 | 期望规模 |
|---|---|---:|
| corrected merge v1 | `experiments/stage0_train_correction_merge/merged_train_question_answer.json` | 9428 |
| labels | `experiments/stage1_label_extraction_corrected_scopefix1/` | 9428/1534 |
| compact schema cards | `experiments/stage8f_compact_llm_cards_corrected/` | 4061/873（历史期望） |
| compact question cards | `experiments/stage8f_compact_question_cards_corrected_scopefix1/` | 9428/1534 |
| full graph | `experiments/stage17a_dsg_data_corrected_scopefix1/` | 9428/1534 |
| embedding cache | `experiments/stage17a_embedding_cache_corrected_scopefix1_qwen3_06b/` | 9428/1534 |

一次 question/evidence 修改会同时使 question card、graph question text 和 query embedding 失效。
禁止将早期 `stage1_label_extraction_scopefix1`、旧 graph 或旧 cache 混入这条链。

## 3. BIRD 完整性检查

```bash
python src/data/stage0_bird_check.py \
  --bird-dir Data/BIRD \
  --output-dir experiments/stage0_bird_check \
  --sample-limit 50 \
  --splits train,dev
```

期望：train 9428 条/69 DB；dev 1534 条/11 DB；无 missing DB；抽样 Gold SQL 可执行。关键 schema
文件是：

```text
Data/BIRD/train_databases/train_databases/train_tables.json
Data/BIRD/dev_tables.json
```

## 4. Corrected merge v1

`processed_final_data.json` 是部分修正集，不是完整训练集：

```bash
python src/data/stage0_merge_train_corrections.py \
  --train-question-answer Data/BIRD/bird-schema/train_question_answer.json \
  --corrections Data/BIRD/processed_final_data.json \
  --output-dir experiments/stage0_train_correction_merge
```

必须得到 2375 correction records、2360 applied、15 unresolved、9428 merged。核对：

```bash
sha256sum experiments/stage0_train_correction_merge/merged_train_question_answer.json
```

期望 SHA256：

```text
71776394f5f4075f993c23f153d9ff130c864196b59ddd59861c5cb6cdb0625f
```

PowerShell 使用：

```powershell
Get-FileHash experiments/stage0_train_correction_merge/merged_train_question_answer.json -Algorithm SHA256
```

2366-correction reviewed merge（SHA256
`1CD74A02D4DB7864AB9FF2E258A505D0035D7B347F46652505EEC6AEEC04516A`）是未来 v2。
切换 v2 必须重建后续全部 artifact 并使用新实验目录，不能继续引用现有 Stage 17 指标。

## 5. Scope-aware labels

```bash
python src/data/stage1_extract_bird_labels.py \
  --bird-dir Data/BIRD \
  --train-question-answer experiments/stage0_train_correction_merge/merged_train_question_answer.json \
  --output-dir experiments/stage1_label_extraction_corrected_scopefix1 \
  --splits train,dev \
  --fk-label-mode explicit_sql

wc -l \
  experiments/stage1_label_extraction_corrected_scopefix1/bird_train_grounding_labels.jsonl \
  experiments/stage1_label_extraction_corrected_scopefix1/bird_dev_grounding_labels.jsonl
```

期望 9428/1534。`explicit_sql` 只将 SQL 实际引用的 FK endpoints 纳入 Gold；改变该策略会改变
coverage 定义。

## 6. LLM schema/question cards

先启动 OpenAI-compatible vLLM endpoint。以下假设端口 9019：

```bash
export LLM_API_KEY=dummy

python src/data/stage8f_llm_card_generation.py \
  --train-tables Data/BIRD/train_databases/train_databases/train_tables.json \
  --dev-tables Data/BIRD/dev_tables.json \
  --train-labels experiments/stage1_label_extraction_corrected_scopefix1/bird_train_grounding_labels.jsonl \
  --dev-labels experiments/stage1_label_extraction_corrected_scopefix1/bird_dev_grounding_labels.jsonl \
  --output-dir experiments/stage8f_llm_cards_corrected_scopefix1 \
  --splits train,dev --card-types both \
  --base-url http://127.0.0.1:9019/v1 --model qwen2.5-coder-32b \
  --workers 16 --schema-card-mode table --schema-chunk-max-items 24 \
  --max-tokens 8192 --question-max-tokens 1024 \
  --disable-thinking --resume --retry-errors --max-schema-fallback-rate 0.05
```

检查 `summary.json`：question cards 应为 9428/1534，error count 应为 0 或逐条审计，fallback rate
不得超过门限。可以用重复的 `--reuse-card-dir` 复用已验证缓存，但不能跨版本使用
`--no-refresh-mismatched-question-cards`。LLM 输出不保证 bitwise determinism；严格复现必须归档
cards、model revision、vLLM 配置和代码提交。

压缩完整 cards：

```bash
python src/data/stage8f_compact_llm_cards.py \
  --train-schema-cards experiments/stage8f_llm_cards_corrected_scopefix1/train_schema_semantic_cards.jsonl \
  --train-question-cards experiments/stage8f_llm_cards_corrected_scopefix1/train_question_cards.jsonl \
  --dev-schema-cards experiments/stage8f_llm_cards_corrected_scopefix1/dev_schema_semantic_cards.jsonl \
  --dev-question-cards experiments/stage8f_llm_cards_corrected_scopefix1/dev_question_cards.jsonl \
  --output-dir experiments/stage8f_compact_llm_cards_corrected_scopefix1
```

现有 Stage 17 artifact 复用了 `stage8f_compact_llm_cards_corrected` 的 schema cards，只刷新了
question cards。若从本手册第 6 步的完整输出继续，执行：

```bash
python src/data/stage8f_compact_llm_cards.py \
  --train-question-cards experiments/stage8f_llm_cards_corrected_scopefix1/train_question_cards.jsonl \
  --dev-question-cards experiments/stage8f_llm_cards_corrected_scopefix1/dev_question_cards.jsonl \
  --output-dir experiments/stage8f_compact_question_cards_corrected_scopefix1
```

记录最终 graph 实际使用的是哪组 cards，不能在目录中混合覆盖。如果历史
`stage8f_compact_llm_cards_corrected` 不存在，则使用本节生成的
`stage8f_compact_llm_cards_corrected_scopefix1` 中 schema cards，并将 graph/cache/训练输出全部写入
新的版本目录；这种重建不能声明为与现有 Stage 17 artifact bitwise 相同。

## 7. Full-schema graph

重建与现有 Stage 17 一致的数据：

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

检查 `summary.json`：9428/1534 records、question-card attachment rate 1.0。每个样本的节点必须是
该 DB 的全部 table+column，不是旧 candidate graph。

## 8. Qwen3 embedding cache

```bash
CUDA_VISIBLE_DEVICES=0 python src/embedding/stage8g_build_embedding_cache.py \
  --train-examples experiments/stage17a_dsg_data_corrected_scopefix1/train_examples.jsonl \
  --dev-examples experiments/stage17a_dsg_data_corrected_scopefix1/dev_examples.jsonl \
  --splits train,dev \
  --output-dir experiments/stage17a_embedding_cache_corrected_scopefix1_qwen3_06b \
  --model-path /data/1_pretrained_models/Qwen3-Embedding-0.6B \
  --batch-size 32 --max-length 512 --pooling last --normalize \
  --device cuda --dtype bfloat16 --trust-remote-code --deduplicate-node-texts
```

检查 `summary.json`、两个 `*_index.json` 和 `.npy` shape。query vectors 应为
`[9428,1024]`/`[1534,1024]`。node texts 可去重，不能以 node `.npy` 第一维代替总节点数。

## 9. Stage 17-A1 实验矩阵

```bash
TRAIN_GRAPH=experiments/stage17a_dsg_data_corrected_scopefix1/train_examples.jsonl
DEV_GRAPH=experiments/stage17a_dsg_data_corrected_scopefix1/dev_examples.jsonl
TRAIN_LABEL=experiments/stage1_label_extraction_corrected_scopefix1/bird_train_grounding_labels.jsonl
DEV_LABEL=experiments/stage1_label_extraction_corrected_scopefix1/bird_dev_grounding_labels.jsonl
EMBED=experiments/stage17a_embedding_cache_corrected_scopefix1_qwen3_06b
```

先验证测试和 smoke：

```bash
python -m unittest tests.test_stage17a_full_schema_qrgta -v

CUDA_VISIBLE_DEVICES=0 python src/training/stage17a_train_full_schema_qrgta.py \
  --train-graph-file "$TRAIN_GRAPH" --dev-graph-file "$DEV_GRAPH" \
  --train-label-file "$TRAIN_LABEL" --dev-label-file "$DEV_LABEL" \
  --embedding-cache-dir "$EMBED" --output-dir experiments/stage17a1_mlp_residual_smoke \
  --model-type mlp_residual --hidden-dim 256 --num-layers 3 --num-heads 8 \
  --dropout 0.1 --epochs 2 --lr 1e-4 --gradient-accumulation-steps 16 \
  --selection-metric complete_coverage@30 --train-limit 100 --dev-limit 50 \
  --device cuda --seed 42
```

Normal QRGTA 三 seeds：

```bash
for SEED in 42 43 44; do
  CUDA_VISIBLE_DEVICES=0 python src/training/stage17a_train_full_schema_qrgta.py \
    --train-graph-file "$TRAIN_GRAPH" --dev-graph-file "$DEV_GRAPH" \
    --train-label-file "$TRAIN_LABEL" --dev-label-file "$DEV_LABEL" \
    --embedding-cache-dir "$EMBED" \
    --output-dir "experiments/stage17a_full_schema_qrgta_seed${SEED}" \
    --model-type qrgta --hidden-dim 256 --num-layers 3 --num-heads 8 \
    --dropout 0.1 --epochs 8 --lr 1e-4 --gradient-accumulation-steps 16 \
    --selection-metric complete_coverage@30 --patience 3 \
    --control-mode normal --device cuda --seed "$SEED"
done
```

Depth-matched MLP 三 seeds：

```bash
for SEED in 42 43 44; do
  CUDA_VISIBLE_DEVICES=0 python src/training/stage17a_train_full_schema_qrgta.py \
    --train-graph-file "$TRAIN_GRAPH" --dev-graph-file "$DEV_GRAPH" \
    --train-label-file "$TRAIN_LABEL" --dev-label-file "$DEV_LABEL" \
    --embedding-cache-dir "$EMBED" \
    --output-dir "experiments/stage17a1_full_schema_mlp_residual_seed${SEED}" \
    --model-type mlp_residual --hidden-dim 256 --num-layers 3 --num-heads 8 \
    --dropout 0.1 --epochs 8 --lr 1e-4 --gradient-accumulation-steps 16 \
    --selection-metric complete_coverage@30 --patience 3 \
    --control-mode normal --device cuda --seed "$SEED"
done
```

同 checkpoint 推理干预：

```bash
for SEED in 42 43 44; do
  CUDA_VISIBLE_DEVICES=0 python src/evaluation/stage17a_run_checkpoint_controls.py \
    --checkpoint "experiments/stage17a_full_schema_qrgta_seed${SEED}/best.pt" \
    --dev-graph-file "$DEV_GRAPH" --dev-label-file "$DEV_LABEL" \
    --embedding-cache-dir "$EMBED" \
    --output-dir "experiments/stage17a1_checkpoint_controls_seed${SEED}" \
    --reference-normal-predictions "experiments/stage17a_full_schema_qrgta_seed${SEED}/dev_predictions.jsonl" \
    --control-modes normal,zero_query_edges,shuffled_schema_edges,shuffled_node_identity \
    --device cuda --seed "$SEED"
done
```

Seed42 控制重训：

```bash
for MODE in zero_query_edges shuffled_schema_edges shuffled_node_identity; do
  CUDA_VISIBLE_DEVICES=0 python src/training/stage17a_train_full_schema_qrgta.py \
    --train-graph-file "$TRAIN_GRAPH" --dev-graph-file "$DEV_GRAPH" \
    --train-label-file "$TRAIN_LABEL" --dev-label-file "$DEV_LABEL" \
    --embedding-cache-dir "$EMBED" \
    --output-dir "experiments/stage17a1_retrained_${MODE}_seed42" \
    --model-type qrgta --hidden-dim 256 --num-layers 3 --num-heads 8 \
    --dropout 0.1 --epochs 8 --lr 1e-4 --gradient-accumulation-steps 16 \
    --selection-metric complete_coverage@30 --patience 3 \
    --control-mode "$MODE" --device cuda --seed 42
done
```

统一汇总：

```bash
python src/evaluation/stage17a_summarize_causal_controls.py \
  --normal-run 42=experiments/stage17a_full_schema_qrgta_seed42 \
  --normal-run 43=experiments/stage17a_full_schema_qrgta_seed43 \
  --normal-run 44=experiments/stage17a_full_schema_qrgta_seed44 \
  --mlp-run 42=experiments/stage17a1_full_schema_mlp_residual_seed42 \
  --mlp-run 43=experiments/stage17a1_full_schema_mlp_residual_seed43 \
  --mlp-run 44=experiments/stage17a1_full_schema_mlp_residual_seed44 \
  --intervention-run 42=experiments/stage17a1_checkpoint_controls_seed42 \
  --intervention-run 43=experiments/stage17a1_checkpoint_controls_seed43 \
  --intervention-run 44=experiments/stage17a1_checkpoint_controls_seed44 \
  --retrained-run zero_query_edges=experiments/stage17a1_retrained_zero_query_edges_seed42 \
  --retrained-run shuffled_schema_edges=experiments/stage17a1_retrained_shuffled_schema_edges_seed42 \
  --retrained-run shuffled_node_identity=experiments/stage17a1_retrained_shuffled_node_identity_seed42 \
  --output-file experiments/stage17a1_causal_summary.json
```

训练目录应包含 `best.pt`、`last.pt`、`training_history.jsonl`、`training_summary.json`、
`best_metrics.json`、`dev_predictions.jsonl`、`model_config.json`。干预实验另存 checkpoint SHA；
汇总器拒绝混用 graph、label、cache、seed 或 sample count。

## 10. 常见失败

1. 路径不存在：先核对本文件 manifest 和源码 `--help`，不要猜旧目录。
2. Question mismatch：混用了 corrected/uncorrected 数据，重建 cards、graph、cache。
3. `record_index=1577` mismatch：已知旧 graph + corrected label 症状，不能关闭校验。
4. Schema ID/name/order mismatch：tables、labels、graph、cache 不同源，不能静默截断。
5. schema fallback 过多：修复 vLLM 后 `--resume --retry-errors`，不要放宽门限训练。
6. Stage 17 OOM：它不加载 32B LLM；检查残留 vLLM/LLM 进程和 cache device。
7. 指标不可比：先记录 merge SHA、artifact 路径、commit、seed 和 sample count。
8. normal reference 仅有尾部节点互换：检查 `reference_normal_check`。脚本允许不影响
   Top-10/20/30/50 与 MRR、且 logit 漂移不超过 `1e-5` 的 CUDA numerical tie；其他差异仍报错。

更详细说明见
[`40_stage17a_full_schema_binary_qrgta.md`](research_notes/dynamic_schema_grounding/40_stage17a_full_schema_binary_qrgta.md)
和
[`41_stage17a1_mlp_and_causal_controls.md`](research_notes/dynamic_schema_grounding/41_stage17a1_mlp_and_causal_controls.md)。
