# 33. Stage 15B: Real LLM SQL Candidate Reranking

## 1. 目标

Stage 15A-fix1 已证明 verifier 能区分 gold typed plan 与受控结构扰动，但这些仍是合成负样本。
Stage 15B 将检验对象替换成 Qwen 实际生成的 SQL 候选，回答更关键的问题：

> 当 LLM 的 top-1 SQL 错误、但候选集合中存在执行正确的 SQL 时，图结构 verifier 能否把正确候选排到第一？

这一步不再修改 LLM hidden states，也不要求 RGTA 单独生成 SQL。职责划分是：

1. LLM 提出完整 SQL 假设；
2. SQLite 仅提供可执行性信号；
3. 每条候选 SQL 独立解析为 typed relational-algebra plan；
4. Stage 15A-fix1 verifier 计算 plan、question 与 schema graph 的一致性能量；
5. verifier 分数与 LLM log-prob/rank prior 做候选级融合。

## 2. 防止标签泄漏

候选 SQL 的 schema pointers 必须从该候选 SQL 自身解析，不能复用 gold SQL 的 schema labels。
执行结果等价标签只在完成候选选择后用于评估。选择器可使用：

- LLM rank 或 mean token log-prob；
- SQL 是否可执行；
- candidate typed plan；
- question/schema embeddings；
- schema graph 与 verifier score。

选择器不可使用 `execution_correct`、gold SQL 或 gold result。

## 3. 评价指标

- `LLM top-1 EX`：原始第一候选；
- `execution-filter EX`：第一个可执行候选；
- `verifier EX`：verifier 分数最高候选；
- `verifier + execution-filter EX`：可执行候选中 verifier 最高者；
- `hybrid EX`：组内标准化后的 verifier/LLM prior 加权；
- `Oracle EX@K`：前 K 个候选中是否至少存在一个执行正确候选；
- `recovery rate`：LLM top-1 错误但 Oracle 可恢复的样本中，被重排器纠正的比例；
- `regressed count`：LLM top-1 原本正确却被重排器改错的数量。

同一 dev 集上的 alpha sweep 只报告敏感性，不能把最优 alpha 当作无偏最终结果。正式选择 alpha
应使用独立 calibration split 或训练数据库上的 held-out queries。

## 4. 服务器运行指令

以下命令假设已经启动 OpenAI-compatible vLLM 服务，并已有 Stage 15A-fix1 checkpoint、
corrected Qwen3 embedding cache、full-schema prompt 和 clean typed trajectories。

### 4.1 生成 K=8 个真实候选

```bash
export LLM_API_KEY=EMPTY

python src/generation/stage15b_generate_sql_candidates.py \
  --prompt-file experiments/stage3_prompt_sql_generation/prompts_full_schema_dev.jsonl \
  --record-index-file experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl \
  --output-file experiments/stage15b_real_candidates/qwen25_k8_dev.jsonl \
  --base-url http://127.0.0.1:9009/v1 \
  --model qwen2.5-coder-32b \
  --candidate-count 8 \
  --max-rounds 3 \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-tokens 768 \
  --request-logprobs \
  --disable-thinking \
  --resume
```

先做 100 条 smoke test 时增加 `--limit 100`，并改用单独输出文件，避免 resume 文件混入不同配置。

### 4.2 执行判等并解析 candidate-owned typed plans

```bash
python src/data/stage15b_prepare_real_sql_candidates.py \
  --generation-file experiments/stage15b_real_candidates/qwen25_k8_dev.jsonl \
  --graph-file experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl \
  --db-root Data/BIRD/dev_databases \
  --output-file experiments/stage15b_real_candidates/qwen25_k8_dev_prepared.jsonl
```

### 4.3 使用 Stage 15A-fix1 verifier 打分

```bash
CUDA_VISIBLE_DEVICES=0 python src/grounding/stage15b_score_real_sql_candidates.py \
  --candidate-file experiments/stage15b_real_candidates/qwen25_k8_dev_prepared.jsonl \
  --checkpoint experiments/stage15a_fix1_consistency_rgta_seed42/sql_hypothesis_verifier.pt \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-file experiments/stage15b_real_candidates/qwen25_k8_dev_scored.jsonl \
  --device cuda
```

首轮允许 `partial_nested` 候选被评分，同时在结果中保留 parse status。补充消融可增加
`--flat-only`，检查简化 parser 是否是主要瓶颈。

### 4.4 评估重排

```bash
python src/evaluation/stage15b_evaluate_real_sql_reranking.py \
  --scored-file experiments/stage15b_real_candidates/qwen25_k8_dev_scored.jsonl \
  --output-dir experiments/stage15b_real_candidates/evaluation_qwen25_k8 \
  --hybrid-alphas 0,0.25,0.5,0.75,1
```

## 5. 决策门

Stage 15B 是否值得继续训练真实候选 verifier，依次看：

1. `Oracle EX@8 - LLM top-1 EX` 是否足够大；若很小，瓶颈是候选生成而非重排；
2. verifier 是否有正的 net gain，且 recovery 大于 regression；
3. `flat-only` 与全部候选差异是否显示 parser 为瓶颈；
4. 若 synthetic Hits@1 很高但 real-candidate verifier 无增益，说明合成 corruption 与 LLM 真实错误分布不匹配，下一步应使用 OOF/真实候选训练，而不是继续调 Stage 15A 合成负样本。

## 6. Stage 15B 首轮结果

在 1393 条 clean dev queries、5686 个去重候选上：

- first sampled candidate EX：`0.5327`；
- execution-filter EX：`0.5406`；
- verifier + execution-filter EX：`0.5477`；
- descriptive hybrid alpha=0.75 EX：`0.5528`；
- Oracle EX：`0.6432`。

alpha=0.75 相比第一随机样本恢复 82 条、回退 54 条，净增益 28 条。结果证明真实候选后验
验证有价值，但同时暴露两个实验缺陷：第一候选来自 temperature=0.7，不能称为标准 greedy
baseline；alpha 也在同一 dev 上观察，不能作为无偏结果。

## 7. Stage 15B-fix1：因果与校准协议

### 7.1 创新点与原理

Stage 15B-fix1 不增加手工 schema 规则，而把后验验证写成可证伪的结构实验：

1. **Greedy-anchored hypothesis set**：候选 0 由独立 temperature=0 请求产生，其余候选负责
   探索多样性。这样重排增益相对于确定的 LLM policy，而不是随机样本顺序。
2. **Graph counterfactual controls**：正确图与 shuffled-FK、shuffled-node-identity 使用完全相同的
   LLM candidates。若只有正确图能提升，才能把收益归因于关系结构和节点语义身份。
3. **Calibration/held-out separation**：只在 20% calibration queries 上选择融合 alpha，在不相交的
   80% 上报告 EX，避免融合权重对同一评估集过拟合。
4. **Oracle-reachable diagnosis**：对“正确 SQL 已在候选集但仍未选中”的问题比较 typed-plan
   因子，区分 table、column、clause binding、operator、join edge、value route 和 parser failure。

### 7.2 clean-1393 候选生成

使用新输出目录，不能 `--resume` 到 Stage 15B 的旧随机首候选文件：

```bash
python src/generation/stage15b_generate_sql_candidates.py \
  --prompt-file experiments/stage3_prompt_sql_generation/prompts_full_schema_dev.jsonl \
  --record-index-file experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl \
  --output-file experiments/stage15b_fix1_real_candidates/qwen25_greedy_k8_clean1393.jsonl \
  --base-url http://127.0.0.1:9009/v1 \
  --model qwen2.5-coder-32b \
  --candidate-count 8 \
  --greedy-anchor \
  --max-rounds 3 \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-tokens 768 \
  --request-logprobs \
  --disable-thinking \
  --resume
```

完整 1534 条实验删除 `--record-index-file`，输出到另一个文件；prepare 阶段将 graph file 改为：

```text
experiments/stage13a_typed_ra_typefix1/dev_typed_ra.jsonl
```

### 7.3 prepare、正确图与反事实图打分

```bash
python src/data/stage15b_prepare_real_sql_candidates.py \
  --generation-file experiments/stage15b_fix1_real_candidates/qwen25_greedy_k8_clean1393.jsonl \
  --graph-file experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl \
  --db-root Data/BIRD/dev_databases \
  --output-file experiments/stage15b_fix1_real_candidates/qwen25_greedy_k8_clean1393_prepared.jsonl

CUDA_VISIBLE_DEVICES=7 python src/grounding/stage15b_score_real_sql_candidates.py \
  --candidate-file experiments/stage15b_fix1_real_candidates/qwen25_greedy_k8_clean1393_prepared.jsonl \
  --checkpoint experiments/stage15a_fix1_consistency_rgta_seed42/sql_hypothesis_verifier.pt \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-file experiments/stage15b_fix1_real_candidates/qwen25_greedy_k8_clean1393_scored.jsonl \
  --control-modes shuffled_fk,shuffled_node_identity \
  --control-seed 42 \
  --device cuda
```

### 7.4 calibration/held-out 评估与错误诊断

```bash
python src/evaluation/stage15b_evaluate_real_sql_reranking.py \
  --scored-file experiments/stage15b_fix1_real_candidates/qwen25_greedy_k8_clean1393_scored.jsonl \
  --output-dir experiments/stage15b_fix1_real_candidates/evaluation_clean1393 \
  --hybrid-alphas 0,0.25,0.5,0.75,1 \
  --calibration-fraction 0.2 \
  --calibration-seed 42

python src/diagnosis/stage15b_oracle_reranking_diagnosis.py \
  --scored-file experiments/stage15b_fix1_real_candidates/qwen25_greedy_k8_clean1393_scored.jsonl \
  --output-dir experiments/stage15b_fix1_real_candidates/diagnosis_clean1393 \
  --alpha 0.75
```

诊断命令中的 alpha 应替换为 `metrics.json -> calibrated_protocol.selected_alpha`。正确图的 held-out
结果必须同时高于 greedy baseline 和两种 counterfactual control，才算通过图结构因果门。
