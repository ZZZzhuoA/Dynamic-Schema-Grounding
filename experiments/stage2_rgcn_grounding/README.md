# Stage 2B: R-GCN Static Schema Grounder

本阶段目标是构造 database schema graph，并用 R-GCN-style message passing 生成 schema node representations，再训练 static grounding scorer。

## 当前实现说明

当前环境未检测到 PyTorch / sklearn，因此第一版实现为：

```text
NumPy fixed-projection R-GCN-style encoder + trainable logistic scorer
```

它包含真实的 schema graph 构造与 relation-specific message passing，但 R-GCN 投影矩阵暂时固定，训练的是最终 grounding scorer。

如果后续安装 PyTorch，可以升级为完整 end-to-end trainable R-GCN。

## 输入

```text
experiments/stage1_label_extraction/bird_train_grounding_labels.jsonl
experiments/stage1_label_extraction/bird_dev_grounding_labels.jsonl
Data/BIRD/train_databases/train_databases/train_tables.json
Data/BIRD/dev_tables.json
```

## 图结构

节点：

- table node
- column node

边类型：

- `self_loop`
- `table_to_column`
- `column_to_table`
- `foreign_key_forward`
- `foreign_key_backward`

`same_table_column` 是可选 dense edge，默认关闭，因为 BIRD 部分表列较多，列两两连接会显著变慢。需要开启时使用：

```powershell
python src/grounding/stage2_rgcn_grounding.py --include-same-table-edges
```

## 执行命令

先跑小样本 smoke test：

```powershell
python src/grounding/stage2_rgcn_grounding.py --train-limit 500 --dev-limit 200 --epochs 2
```

全量训练：

```powershell
python src/grounding/stage2_rgcn_grounding.py --epochs 3
```

如果太慢，可以先用：

```powershell
python src/grounding/stage2_rgcn_grounding.py --train-limit 3000 --epochs 3
```

## 输出

```text
experiments/stage2_rgcn_grounding/
  rgcn_train_config.json
  rgcn_train_log.jsonl
  rgcn_dev_metrics.json
  rgcn_dev_predictions.jsonl
  rgcn_dev_topk_examples.md
```

## 对比基线

Stage 2A lexical dev 指标：

```json
{
  "schema_recall@20": 0.7754,
  "table_recall@5": 0.8174,
  "column_recall@10": 0.6871
}
```

## 验收标准

由于当前是无 PyTorch 的轻量版本，验收分两层：

### 工程验收

- 可以构造 schema graph；
- train loss 下降；
- 能输出 dev ranking metrics；
- predictions/top-k examples 可读。

### 效果参考

如果接近 Stage 2A lexical baseline，说明图结构信号可用。若低于 lexical，也不直接失败，因为当前 R-GCN 投影还不是端到端训练版本。

重点看：

- `schema_recall@20`
- `table_recall@5`
- `column_recall@10`
- train loss 是否下降
