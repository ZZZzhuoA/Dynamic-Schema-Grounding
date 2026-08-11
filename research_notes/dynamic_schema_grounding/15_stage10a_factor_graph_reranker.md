# Stage 10-A：Heterogeneous Factor-Graph Reranker

记录日期：2026-08-11

## 研究问题

Stage 9 的最终 Top-30 是 relation 内排序与固定预算拼接的结果，没有模型比较“新增候选”和
“即将被淘汰候选”的全局效用。Value evidence 因此能提高 Column Recall，同时仍可能挤掉
正确 table/column。

Stage 10-A 将最终 Schema 选择改写为：

> 在查询条件化的异质证据因子图上推断节点效用，再进行带类型与 owner closure 约束的集合解码。

## 创新点与原理

### 1. 检索图与重排序图分工

- 第一层 RGTA 在完整数据库图上进行 relation-conditioned 高召回检索；
- 第二层只处理约 60–100 个候选节点，联合校准 relation、value 和 join-path 证据。

两个网络不重复：第一层优化 candidate ceiling，第二层优化固定预算下的全局组合。

### 2. 异质因子图

变量节点是候选 table/column；因子节点分为：

- relation factor；
- database value factor；
- join-path factor。

Schema graph 的 table-column/FK 边保留在变量节点之间。每层先从 Schema 更新 factor，
再从 factor 反向更新 Schema，并通过独立 gate 融合原 Schema graph message 与 evidence
factor message。

### 3. 非孤立列解码

最终 selector 不是简单 `argsort[:30]`。若选择 column 而 owner table 尚未选择，二者作为
一个 package 共同参与效用/成本比较，因此满足：

```text
selected(column) => selected(owner table)
```

同时设置 table partition 上限和由 baseline 推导的最小 table 数，阻止 Value 列跨类型无条件
挤掉结构骨架。

### 4. 多任务训练

训练目标包括：

- whole-SQL schema node BCE；
- relation-role auxiliary BCE；
- gold node 与 hard negative 的 pairwise margin loss。

候选生成阶段禁止 gold injection，并显式报告 candidate oracle recall 与 complete coverage。

## 文件

- `src/data/stage10_build_factor_graph_data.py`
- `src/modeling/factor_graph_reranker.py`
- `src/grounding/stage10_constrained_selector.py`
- `src/training/stage10_train_factor_graph_reranker.py`
- `tests/test_stage10_factor_graph_reranker.py`

## 服务器数据流程

以下路径沿用当前最佳 RGTA。Embedding cache 路径直接从训练配置读取，避免手写错误路径。

```bash
RGTA_DIR=experiments/stage8g_corrected_llm_cards_rgta_seed42
EMBED_CACHE=$(python -c "import json; print(json.load(open('$RGTA_DIR/train_config.json'))['embedding_cache_dir'])")
```

### 1. 生成训练集第一层 relation predictions

```bash
python src/evaluation/stage8g_evaluate_relation_grounder.py \
  --checkpoint-dir "$RGTA_DIR" \
  --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl \
  --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --embedding-cache-dir "$EMBED_CACHE" \
  --split train \
  --output-dir experiments/stage10a_first_stage_train_rgta_seed42 \
  --device cuda
```

输出：

```text
experiments/stage10a_first_stage_train_rgta_seed42/train_relation_predictions.jsonl
```

### 2. 建立训练数据库 Value Index

```bash
python src/data/stage9_build_value_index.py \
  --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --db-root Data/BIRD/train_databases \
  --output-file experiments/stage10a_value_index/train_value_index.sqlite \
  --max-values-per-column 20000 \
  --max-value-chars 128 \
  --rebuild
```

### 3. 生成训练 evidence debug

```bash
python src/grounding/stage9_value_join_completion.py \
  --relation-predictions experiments/stage10a_first_stage_train_rgta_seed42/train_relation_predictions.jsonl \
  --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl \
  --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --value-index experiments/stage10a_value_index/train_value_index.sqlite \
  --output-dir experiments/stage10a_train_evidence_rgta_seed42 \
  --enable-value-index \
  --enable-join-path \
  --value-fusion-mode gated
```

### 4. 构造 train/dev 候选因子图

```bash
python src/data/stage10_build_factor_graph_data.py \
  --relation-predictions experiments/stage10a_first_stage_train_rgta_seed42/train_relation_predictions.jsonl \
  --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/train_relation_labels.jsonl \
  --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --evidence-debug experiments/stage10a_train_evidence_rgta_seed42/evidence_debug.jsonl \
  --output-file experiments/stage10a_factor_graph_data/train_factor_graphs.jsonl \
  --relation-top-m 20 \
  --max-candidates 80

python src/data/stage10_build_factor_graph_data.py \
  --relation-predictions experiments/stage8g_corrected_llm_cards_rgta_seed42/dev_relation_predictions.jsonl \
  --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/dev_relation_labels.jsonl \
  --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --evidence-debug experiments/stage9_fix1_gated_value_join_rgta_seed42_limit1534/evidence_debug.jsonl \
  --output-file experiments/stage10a_factor_graph_data/dev_factor_graphs.jsonl \
  --relation-top-m 20 \
  --max-candidates 80
```

先检查两个 summary 中的 `candidate_oracle_recall` 和 `complete_candidate_coverage`。候选上界若低于
Stage 9 enhanced recall，则不能开始解释 reranker 性能。

### 5. 完整 Factor-RGTA 训练

```bash
python src/training/stage10_train_factor_graph_reranker.py \
  --train-file experiments/stage10a_factor_graph_data/train_factor_graphs.jsonl \
  --dev-file experiments/stage10a_factor_graph_data/dev_factor_graphs.jsonl \
  --embedding-cache-dir "$EMBED_CACHE" \
  --output-dir experiments/stage10a_factor_rgta_seed42 \
  --model-type factor_rgta \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 8 \
  --device cuda \
  --seed 42
```

### 6. 结构消融

仅把 `--model-type` 和输出目录替换为：

```text
mlp          # 无 Schema 图、无 Factor 图
schema_rgta  # 只有候选 Schema 子图
factor_rgta  # Schema 图 + Relation/Value/Join factors
```

每次运行同时报告：

- `baseline_*`：原固定预算 Top-30；
- `reranker_raw_*`：模型直接 Top-30；
- `constrained_*`：类型/owner-closure 约束 Top-30；
- candidate oracle ceiling。

## 严格实验边界

直接使用完整训练集训练出的第一层 RGTA 再生成同一训练集 predictions，属于 in-sample stacking，
适合 Stage 10-A 功能验证，但会低估第一层错误。论文主实验应生成 K-fold out-of-fold 第一层
predictions：每折只用其他折训练 RGTA，再为保留折生成候选。第二层 reranker 只使用拼接后的
OOF train candidates 训练，dev/test 始终使用完整训练集训练出的第一层 checkpoint。
