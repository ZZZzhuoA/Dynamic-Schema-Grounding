# Value Index and Join-Path Completion

记录日期：2026-08-11

## 目标

当前最佳 RGTA 的主要薄弱关系是 `PREDICATE_COLUMN` 和 `VALUE_ANCHOR`。此外，SQL 所需的
中间 JOIN 表和 FK endpoint 往往不会在问题中显式出现。本阶段增加两个独立、可消融、无需
重新训练 RGTA 的推理证据通道：

```text
RGTA relation predictions
  ├─ Value Index → PREDICATE_COLUMN / VALUE_ANCHOR
  └─ semantic/value terminals → FK graph → join-path completion → JOIN_BRIDGE
                                      ↓
                         existing relation-budget assembly
```

## Value Index

`src/data/stage9_build_value_index.py` 从测试时可访问的 SQLite 数据库内容建立持久化索引：

```text
normalized value
  → value tokens
  → db_id
  → schema_item_id
  → table.column
```

实现特性：

- Unicode NFKC、大小写、空白和分隔符归一化；
- 简单英文词形归一化；
- 每张表只扫描一次，同时收集所有可搜索列；
- SQLite 中建立 `(db_id, token)` 倒排索引；
- 默认不索引数值列，避免把任意阈值误当作数据库实体值；
- 不读取 gold SQL 或 schema labels；
- 支持按数据库断点续建。

问题级查询先通过 token index 找到候选值，再计算短语匹配和 token coverage。多个列包含同一值时，
不是直接选择最高频列，而是联合：

```text
value evidence
+ PREDICATE_COLUMN / VALUE_ANCHOR rank support
+ semantic terminal-table support
```

因此 `Alameda` 同时出现在 `County`、`City`、`MailCity` 等列时，value index 只提供候选，
relation belief 和 table context 负责消歧。

## Join-Path Completion

`src/grounding/stage9_value_join_completion.py` 从非 JOIN relations 和 value matches 中抽取
高置信 terminal tables，在 FK table graph 上执行：

1. 计算 terminal 两两之间的最短路径；
2. 构造 terminal metric closure；
3. 在 metric closure 上求 MST；
4. 将 MST 边展开回原 FK graph；
5. 注入路径中的中间表和 FK endpoint columns 到 `JOIN_BRIDGE` belief。

这是 Steiner tree 的 metric-closure MST 近似，而不是简单的一跳 FK closure。路径边成本还会被
原始 `JOIN_BRIDGE` rank support 调低，使 RGTA 已经支持的 FK 边更容易进入连接子图。

## 保守注入

增强层默认保护每个 relation 的前两个原始候选，只在其后注入有限数量的 value/path candidates。
这避免 evidence channel 直接覆盖 RGTA 最强预测。输出保持原 Top-K 大小，并同时保存：

- 原始 baseline assembly；
- enhanced assembly；
- 每个问题的 value matches；
- terminal tables 和补全路径；
- complete coverage gained/lost samples；
- value/path 候选对 gold missing schema 的可恢复上界。

## Dev value index

```bash
python src/data/stage9_build_value_index.py \
  --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --db-root Data/BIRD/dev_databases \
  --output-file experiments/stage9_value_index/dev_value_index.sqlite \
  --max-values-per-column 20000 \
  --max-value-chars 128 \
  --rebuild
```

## 三组严格消融

### Value only

```bash
python src/grounding/stage9_value_join_completion.py \
  --relation-predictions experiments/stage8g_corrected_llm_cards_rgta_seed42/dev_relation_predictions.jsonl \
  --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/dev_relation_labels.jsonl \
  --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --value-index experiments/stage9_value_index/dev_value_index.sqlite \
  --output-dir experiments/stage9_value_only_rgta_seed42 \
  --enable-value-index
```

### Join path only

```bash
python src/grounding/stage9_value_join_completion.py \
  --relation-predictions experiments/stage8g_corrected_llm_cards_rgta_seed42/dev_relation_predictions.jsonl \
  --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/dev_relation_labels.jsonl \
  --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --output-dir experiments/stage9_join_path_only_rgta_seed42 \
  --enable-join-path
```

### Value + join path

```bash
python src/grounding/stage9_value_join_completion.py \
  --relation-predictions experiments/stage8g_corrected_llm_cards_rgta_seed42/dev_relation_predictions.jsonl \
  --relation-file experiments/stage5j_relation_labels_corrected_llm_cards/dev_relation_labels.jsonl \
  --graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --value-index experiments/stage9_value_index/dev_value_index.sqlite \
  --output-dir experiments/stage9_value_join_rgta_seed42 \
  --enable-value-index \
  --enable-join-path
```

每个输出目录中的 `summary.json` 同时包含 baseline、enhanced 和 delta，因此无需重新运行基线。
先用 `--limit 100` 验证格式，再去掉 limit 评估完整 1,534 条。

## 实验解释边界

Value index 和 join completion 的推理本身不使用 gold labels；gold 只在完成推理后计算 recall 和
complete coverage。但当前 Stage 8G prediction 文件仍只包含训练/评估流程激活的 relation rows。
在完全未知测试集上，还需要独立 operation/relation controller 判断哪些 relation 应被激活。
因此本阶段用于验证两个 evidence channels 是否能够恢复缺失 schema，不声称已经完成最终动态
LLM decoding 架构。
