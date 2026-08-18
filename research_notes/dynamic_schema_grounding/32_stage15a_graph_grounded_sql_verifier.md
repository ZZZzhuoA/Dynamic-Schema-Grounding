# 32. Stage 15A: Graph-Grounded SQL Hypothesis Verifier

记录日期：2026-08-17

## 1. 研究转向

Stage 14B-fix1 证明 RGTA 能改善结构和类型驱动的绝对定位性能，但没有证明 semantic slot identity
被可靠使用。Stage 15 不再要求图网络从问题中独立恢复完整 SQL 语义，而采用后验验证：

```text
Question + Full Schema
          ↓
LLM proposes K SQL hypotheses
          ↓
SQL → typed relational-algebra candidate graph
          ↓
Candidate graph ↔ database schema graph matching
          ↓
RGTA compatibility energy + execution evidence
          ↓
rerank / verify / repair
```

概率上从前置硬选择

```text
P(schema grounding | question, schema) → SQL
```

改成候选条件化的后验验证：

```text
P(SQL | question, schema) · P(graph consistency | question, schema, SQL)
```

这保留 LLM 的代码生成能力，同时让图网络专注于 schema binding、join path、operator/type 和
value-column compatibility，不再引入 Top-30 的硬召回上限。

## 2. Stage 15A 的边界

Stage 15A 是不调用 LLM 的离线可识别性实验。以 gold typed trajectory 为正样本，并生成结构受控的
困难负样本：

1. 同表、同类型列替换；
2. 跨表、同类型列替换；
3. SCAN table 替换；
4. FK/join edge 替换；
5. operator 替换；
6. value route 替换；
7. 不同 clause role 的 column binding 交换。

模型看不到 `corruption_type`。该字段只用于训练标签分组和诊断。

## 3. 模型

数据库 schema graph 先经过 query-conditioned RGTA：

```text
H_S = RGTA(S, q)
```

候选 SQL 被表示为 typed plan graph。每个 plan node 包含 action、operators、value routes，且通过
显式 pointer 与 `H_S` 中的表、列和 FK endpoint 对齐。候选 plan graph 再进行关系感知传播：

```text
h_u = Fuse(action_u, operator_u, value_route_u, Bind(H_S, pointers_u))
H_C = PlanRGTA(C, q)
```

最终能量同时包含：

- node binding compatibility；
- plan-edge consistency；
- FK endpoint consistency；
- query-conditioned global plan compatibility。

训练采用 query-level listwise cross entropy 与 positive-negative margin loss。它学习的是完整 SQL
假设排序，而不是单个 schema node 的 Top-K 排序。

## 4. Stage 15A 决策门

在进入真实 LLM 多候选重排前，要求：

1. overall Hits@1 ≥ 0.80；
2. overall pairwise accuracy ≥ 0.80；
3. same-table column corruption accuracy ≥ 0.70；
4. join corruption accuracy ≥ 0.90；
5. operator 与 value-route corruption accuracy ≥ 0.80；
6. shuffled/corrupted schema graph 显著降低正样本分数。

如果同表列替换仍不可辨别，说明当前 query/schema embedding 缺少细粒度语义，不能靠扩大模型解决；
下一步应引入 LLM candidate rationale、database value evidence 或 execution verifier。

## 5. 实现文件

- `src/data/stage15_build_sql_hypothesis_data.py`
- `src/modeling/sql_hypothesis_verifier.py`
- `src/training/stage15_train_sql_hypothesis_verifier.py`
- `src/evaluation/stage15_evaluate_sql_hypothesis_verifier.py`
- `tests/test_stage15_sql_hypothesis_verifier.py`

## 6. 服务器执行流程

### 6.1 构造候选组

```bash
python src/data/stage15_build_sql_hypothesis_data.py \
  --train-trajectories experiments/stage13b_clean_typed_trajectories/train_trajectories.jsonl \
  --dev-trajectories experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl \
  --output-dir experiments/stage15a_sql_hypothesis_data \
  --negatives-per-example 6 \
  --seed 42
```

### 6.2 训练 RGTA verifier

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage15_train_sql_hypothesis_verifier.py \
  --train-file experiments/stage15a_sql_hypothesis_data/train_hypotheses.jsonl \
  --dev-file experiments/stage15a_sql_hypothesis_data/dev_hypotheses.jsonl \
  --embedding-cache-dir experiments/stage13b_embedding_cache_typefix1_qwen3_06b \
  --init-checkpoint experiments/stage13b_typed_ra_decoder_rgta_seed42/typed_ra_decoder.pt \
  --init-summary experiments/stage13b_typed_ra_decoder_rgta_seed42/training_summary.json \
  --output-dir experiments/stage15a_sql_hypothesis_verifier_rgta_seed42 \
  --hidden-dim 256 \
  --num-schema-layers 2 \
  --num-plan-layers 2 \
  --epochs 8 \
  --lr 1e-4 \
  --margin-weight 0.5 \
  --margin 0.5 \
  --device cuda \
  --seed 42
```

### 6.3 独立重算候选排序指标

```bash
python src/evaluation/stage15_evaluate_sql_hypothesis_verifier.py \
  --prediction-file experiments/stage15a_sql_hypothesis_verifier_rgta_seed42/dev_predictions.jsonl \
  --output-file experiments/stage15a_sql_hypothesis_verifier_rgta_seed42/dev_metrics_recomputed.json
```
