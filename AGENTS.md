# Dynamic-Schema-Grounding Agent Entry

本文件是新 session 的最小入口。不要从聊天记忆或实验目录名猜测项目状态。

## 必读顺序

1. [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)：当前问题、架构、已验证结论和下一步。
2. [RESULTS_LEDGER.md](RESULTS_LEDGER.md)：指标、证据等级和可比性边界。
3. [REPRODUCTION.md](REPRODUCTION.md)：从 BIRD 到 Stage 17-A1 的权威命令。
4. 只有需要历史细节时，才进入
   [research_notes/dynamic_schema_grounding/README.md](research_notes/dynamic_schema_grounding/README.md)。

## 当前主线

当前研究对象是 **Stage 17 Full-Schema Binary QRGTA**：完整数据库中的全部 table/column
节点进入 query-conditioned relational graph transformer，模型预测每个节点被 Gold SQL 引用的
概率。Stage 17-A1 正在用 depth-matched MLP 和三个 frozen-checkpoint control 验证性能是否真的
来自图结构，而不是监督式 embedding reranking。

当前代码基线：

```text
9117c17 feat: add Stage 17 causal graph controls
```

尚未完成的正式服务器实验：

- `mlp_residual` seeds 42/43/44；
- normal QRGTA seeds 42/43/44（可复用的 seed42 必须先通过数据版本核验）；
- 对每个 normal checkpoint 的 `zero_query_edges`、`shuffled_schema_edges`、
  `shuffled_node_identity` 推理干预；
- 三个 control 的 seed42 从头重训；
- `stage17a1_causal_summary.json` 统一汇总。

## 权威数据版本

当前 Stage 17 结果对应 **corrected merge v1**：

```text
applied corrections: 2360
merged records: 9428
SHA256: 71776394F5F4075F993C23F153D9FF130C864196B59DDD59861C5CB6CDB0625F
```

Stage 17 主线必须成套使用：

```text
experiments/stage1_label_extraction_corrected_scopefix1/
experiments/stage17a_dsg_data_corrected_scopefix1/
experiments/stage17a_embedding_cache_corrected_scopefix1_qwen3_06b/
```

Git 中的 2366-correction reviewed merge 是未来 `v2`，不是现有 Stage 17 的直接替代品。切换
merge、question/evidence 或 cards 后，必须重建 labels、question cards、graph 和 embedding cache，
并使用全新的实验目录。

## 不可违反的规则

- Gold SQL/schema labels 只能用于训练目标、checkpoint 选择和事后评估，不能写入 inference input
  或无 Gold 的 prediction 文件。
- Question、schema ID/name/order、graph node order 和 embedding-cache index 必须严格一致；发现
  mismatch 立即报错，不能截断或按数组位置强行配对。
- 不把不同数据版本、样本规模、prompt、LLM 或 checkpoint 的指标放在同一无说明对照表。
- `Complete Coverage@30` 是整份 Gold schema 都进入 Top-30 的样本比例，不等于平均 Recall@30，
  更不等于 SQL Execution Accuracy。
- 先核对 [REPRODUCTION.md](REPRODUCTION.md) 的 artifact manifest；路径不存在时不要发明旧目录。
- 不修改或删除用户已有的 `experiments/` 产物；新实验使用新目录。
- 除非用户明确要求，不执行 commit 或 push。即使要求提交，也只暂存当前任务相关文件。
- 不提交 checkpoint、embedding、数据库和大 JSONL；它们通常被 `.gitignore` 排除。

## 快速状态检查

```bash
git log -3 --oneline
git status --short
python -m unittest tests.test_stage17a_full_schema_qrgta -v
```

如果任务是继续 Stage 17-A1，直接执行
[REPRODUCTION.md](REPRODUCTION.md) 的“Stage 17-A1 实验矩阵”，不要回到 prompt、logits bias、
hidden-state steering 或 dynamic recurrence 调参。
