# Stage 9-fix1: Confidence-Gated Value Fusion

记录日期：2026-08-11

## 修正动机

完整 1,534 条消融显示，直接 Value 注入虽然净增加 20 个完整覆盖样本，但同时破坏
14 个原本完整的样本；Join-only 增加 22 个、只破坏 1 个。问题来自两个位置：

1. 低精度 value candidates 会直接占用 relation Top-K；
2. 所有 value matched tables 都能成为 join terminals，进而诱导额外路径。

因此不能继续通过调大或调小注入预算解决问题，而应显式建模 Value 证据的不确定性。

## 条件证据模型

修正版将 value-column 候选置信度定义为：

```text
confidence(value, column)
  = value_likelihood
  × [ambiguity prior + semantic support + competitor margin]
```

- `value_likelihood`：短语匹配或 token coverage；
- `ambiguity prior`：同一规范化值所命中列数的逆平方根；
- `semantic support`：表级 relation rank prior 与
  `PREDICATE_COLUMN` / `VALUE_ANCHOR` 列级 prior 的组合；
- `competitor margin`：当前列相对包含同一值的其他列的归一化优势。

候选被划分为三种状态：

- `inject`：高置信，可引入 relation list 中不存在的新列；
- `rerank`：中置信，只能重排 relation list 中已经存在的列；
- `reject`：不改变 schema belief。

Join terminal 使用比列注入更高的置信阈值，并额外要求最小 semantic support。
唯一值可以补充 Predicate，但在没有额外语义支持时不能创建新的 Join terminal。
这样可以隔离 Value 噪声对 FK 路径搜索的级联影响。

## 可复现性

默认 `--value-fusion-mode gated` 启用修正版；
`--value-fusion-mode direct` 保留原始 Stage 9 直接注入方式，作为严格消融对照。

完整 dev 运行命令：

```bash
python src/grounding/stage9_value_join_completion.py \
  --relation-predictions experiments/stage8g_corrected_llm_cards_rgta_seed42/dev_relation_predictions.jsonl \
  --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/dev_relation_labels.jsonl \
  --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --value-index experiments/stage9_value_index/dev_value_index.sqlite \
  --output-dir experiments/stage9_fix1_gated_value_join_rgta_seed42_limit1534 \
  --enable-value-index \
  --enable-join-path \
  --value-fusion-mode gated \
  --limit 1534
```

直接注入对照只需将输出目录更换，并加入：

```bash
--value-fusion-mode direct
```

## 新增诊断字段

`evidence_debug.jsonl` 为每个 Value 候选保存：

- `value_confidence`；
- `value_ambiguity_count` 与 `value_ambiguity_score`；
- `value_margin_score`；
- `value_semantic_support`；
- `value_gate`；
- `eligible_for_terminal`。

`summary.json` 额外统计门控后候选数量、门控候选 gold precision 和 Value terminal 数量。
Baseline 与 enhanced metrics 还分别报告最终混合 schema Top-K 内的
`assembled_table_recall@K` 和 `assembled_column_recall@K`。两者按问题做 macro average；
没有相应类型 gold 标签的问题不参与该类型均值，并通过独立 sample count 标明分母。
`coverage_transition_diagnostics.jsonl` 只保存 complete coverage 状态发生变化的样本，列出
恢复或淘汰的 gold IDs 以及 Top-K 中新增/移除的 schema IDs。该文件只在推理完成后由 gold
标签生成，不会被模型输入读取。
主要目标不是盲目减少候选，而是在保持 complete-coverage gains 的同时显著降低
complete-coverage losses。
