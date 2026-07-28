# Stage 1A: BIRD Whole-SQL Grounding Label Extraction

本阶段目标是抽取 whole-SQL schema grounding labels，用于下一阶段训练 static grounding baseline。

## 为什么不用单独的 `hit_info`

`hit_info` 不包含外键信息，通常只能覆盖自然语言显式命中的表/列。Text-to-SQL 还需要 SQL 实际使用的表列和 join keys。

因此 Stage 1A 使用三类来源联合构造 label：

| 来源 | 作用 |
|---|---|
| `hit_info` | question-mentioned semantic labels |
| gold SQL parse | SQL-used tables / columns |
| schema foreign keys | join keys between SQL-used tables |

最终：

```text
whole_sql_labels = hit_info labels ∪ sql_parse labels ∪ foreign_key labels
```

## 输入文件

```text
Data/BIRD/bird-schema/train_question_answer.json
Data/BIRD/train_databases/train_databases/train_tables.json
Data/BIRD/dev.json
Data/BIRD/dev_tables.json
```

## 执行命令

先小样本检查：

```powershell
python src/data/stage1_extract_bird_labels.py --bird-dir Data/BIRD --splits train,dev --limit 100
```

如果输出正常，再跑全量：

```powershell
python src/data/stage1_extract_bird_labels.py --bird-dir Data/BIRD --splits train,dev
```

如果只跑 train：

```powershell
python src/data/stage1_extract_bird_labels.py --bird-dir Data/BIRD --splits train
```

## 输出文件

```text
experiments/stage1_label_extraction/
  bird_train_grounding_labels.jsonl
  bird_train_failed_label_cases.jsonl
  bird_dev_grounding_labels.jsonl
  bird_dev_failed_label_cases.jsonl
  bird_label_statistics.json
```

## 输出样例

```json
{
  "split": "train",
  "db_id": "movie_platform",
  "question": "Name movie titles released in year 1945.",
  "sql": "SELECT movie_title FROM movies WHERE movie_release_year = 1945 ...",
  "schema_items": [
    {"id": 0, "type": "table", "name": "movies"},
    {"id": 1, "type": "column", "name": "movies.movie_title"}
  ],
  "whole_sql_labels": [0, 1],
  "label_names": ["movies", "movies.movie_title"],
  "label_sources": {
    "foreign_key": [],
    "hit_info": ["movies.movie_release_year"],
    "sql_parse": ["movies", "movies.movie_title", "movies.movie_release_year"]
  }
}
```

## 验收标准

| 指标 | 目标 |
|---|---:|
| train processed_count | 9428 |
| dev processed_count | 1534 |
| schema_db_count.train | 69 |
| schema_db_count.dev | 11 |
| non_empty_label_rate | >= 0.95 |
| avg_labels_per_sample | 合理，通常大于 2 |
| failed_count | 越低越好 |
| source_label_counts.sql_parse | 应该非零且为主要来源 |
| source_label_counts.foreign_key | 对多表 SQL 应有贡献 |

## 你跑完后发给我什么

请贴出 `bird_label_statistics.json` 的内容，重点是：

```text
train.processed_count:
train.non_empty_label_rate:
train.avg_labels_per_sample:
train.failed_count:
train.source_label_counts:

dev.processed_count:
dev.non_empty_label_rate:
dev.avg_labels_per_sample:
dev.failed_count:
dev.source_label_counts:
```

如果 Stage 1A 结果合格，我们再进入 **Stage 2：Static Grounding Baseline 计划确认**。

