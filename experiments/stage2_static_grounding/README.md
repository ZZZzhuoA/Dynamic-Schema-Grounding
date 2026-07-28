# Stage 2A: Lexical Static Grounding Baseline

本阶段目标是建立一个不训练的 static grounding baseline：

```text
question + evidence + schema items → ranked schema elements
```

它用于验证 Stage 1A 的 labels 是否可用，并为后续 neural static grounder 提供下限对照。

## 输入文件

默认使用：

```text
experiments/stage1_label_extraction/bird_dev_grounding_labels.jsonl
```

## 执行命令

先小样本：

```powershell
python src/grounding/stage2_lexical_grounding.py --input experiments/stage1_label_extraction/bird_dev_grounding_labels.jsonl --split dev --limit 100
```

全量 dev：

```powershell
python src/grounding/stage2_lexical_grounding.py --input experiments/stage1_label_extraction/bird_dev_grounding_labels.jsonl --split dev
```

如果想评估 train：

```powershell
python src/grounding/stage2_lexical_grounding.py --input experiments/stage1_label_extraction/bird_train_grounding_labels.jsonl --split train
```

## 输出文件

```text
experiments/stage2_static_grounding/
  lexical_dev_metrics.json
  lexical_dev_predictions.jsonl
  lexical_dev_topk_examples.md
```

## 指标

| Metric | 含义 |
|---|---|
| `schema_recall@10` | top-10 schema items 覆盖 gold labels 的比例 |
| `schema_recall@20` | top-20 schema items 覆盖 gold labels 的比例 |
| `schema_precision@10` | top-10 中 gold labels 的比例 |
| `schema_mrr` | 第一个 gold schema item 的倒数排名 |
| `table_recall@5` | top-5 tables 覆盖 gold tables 的比例 |
| `column_recall@10` | top-10 columns 覆盖 gold columns 的比例 |

## 验收标准

Stage 2A 是 lexical baseline，不要求特别强，但至少需要有基本信号：

| Metric | Minimum target |
|---|---:|
| `table_recall@5` | >= 0.70 |
| `column_recall@10` | >= 0.40 |
| `schema_recall@20` | >= 0.50 |
| dev sample count | 1534 |

如果指标明显低于这个水平，优先检查：

- schema item 文本构造是否合理；
- evidence 是否被纳入 query；
- Stage 1A labels 是否过宽；
- BIRD 的列名和自然语言之间是否存在强语义 gap。

## 你跑完后发给我什么

请贴出：

```text
lexical_dev_metrics.json
```

如果 Stage 2A 达到基本验收标准，我们进入 **Stage 2B：Neural Static Grounder 计划确认**。

