# Stage 0A: BIRD Data Check

本阶段目标是确认 BIRD 数据是否可以作为第一轮实验数据使用。

## 你需要准备什么

当前项目已经检测到：

```text
Data/BIRD/
  train.json
  dev.json
  dev.sql
  dev_tables.json
  train_databases/
  dev_databases/
  bird-schema/
```

你暂时不需要准备模型，也不需要训练。

## 执行命令

在项目根目录执行：

```powershell
python src/data/stage0_bird_check.py --bird-dir Data/BIRD --sample-limit 50
```

如果想先快速检查 dev：

```powershell
python src/data/stage0_bird_check.py --bird-dir Data/BIRD --splits dev --sample-limit 50
```

如果 train SQL 执行较慢，可以先小样本检查：

```powershell
python src/data/stage0_bird_check.py --bird-dir Data/BIRD --splits train --sample-limit 5
```

如果想检查更多 SQL 可执行性：

```powershell
python src/data/stage0_bird_check.py --bird-dir Data/BIRD --sample-limit 200
```

## 输出文件

脚本会生成：

```text
experiments/stage0_data_check/bird_stage0_report_dev.json
experiments/stage0_data_check/bird_stage0_report_train.json
experiments/stage0_data_check/bird_stage0_report_dev_train.json
```

具体文件名取决于你传入的 `--splits`。

## 本阶段需要关注的结果

打开报告后重点看：

```json
{
  "train": {
    "record_count": 9428,
    "unique_db_count": 69,
    "sqlite_file_count": 69,
    "missing_db_ids": [],
    "sample_execution": {
      "ok": ...
    }
  },
  "dev": {
    "record_count": 1534,
    "unique_db_count": 11,
    "sqlite_file_count": 13,
    "missing_db_ids": [],
    "sample_execution": {
      "ok": ...
    }
  }
}
```

## 跑通标准

满足以下条件即可认为 Stage 0A 跑通：

| 检查项 | 标准 |
|---|---|
| train/dev JSON 可读 | 是 |
| train/dev sqlite 可找到 | 是 |
| missing_db_ids | 空列表 |
| dev gold SQL 抽样执行 | 大部分成功 |
| train gold SQL 抽样执行 | 大部分成功 |

## 你跑完后发给我什么

把下面几项贴给我即可：

```text
train record_count:
train unique_db_count:
train sqlite_file_count:
train missing_db_ids:
train sample_execution ok/attempted:

dev record_count:
dev unique_db_count:
dev sqlite_file_count:
dev missing_db_ids:
dev sample_execution ok/attempted:

failed_examples 是否为空:
```

如果这些都正常，我们进入 **Stage 1：Gold SQL → Schema Grounding Label 抽取**。
