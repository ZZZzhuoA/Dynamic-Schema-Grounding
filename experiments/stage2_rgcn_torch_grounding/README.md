# Stage 2B 修正版：PyTorch Trainable R-GCN Static Grounder

本阶段使用 `huaweicup` 环境中的 PyTorch 实现端到端可训练的 R-GCN schema graph encoder。

## 环境检查

```powershell
conda run -n huaweicup python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

当前已确认：

```text
torch 2.12.0+cpu
cuda False
```

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

默认边：

- `self_loop`
- `table_to_column`
- `column_to_table`
- `foreign_key_forward`
- `foreign_key_backward`

可选边：

- `same_table_column`

开启方式：

```powershell
--include-same-table-edges
```

## 模型

```text
schema node hash features
   ↓
Linear projection
   ↓
R-GCN layers
   ↓
schema node embeddings

question + evidence hash features
   ↓
query MLP
   ↓
query embedding

[q; z_i; q*z_i; |q-z_i|]
   ↓
MLP scorer
   ↓
schema relevance logit
```

## Encoder types

脚本支持两种 schema graph encoder：

| 参数 | 含义 |
|---|---|
| `--encoder-type rgcn` | Relational GCN，按关系类型做均值消息聚合 |
| `--encoder-type rgta` | Relational Graph Transformer Attention，节点通过 typed edges 做关系感知注意力 |

R-GTA 中每条边 `j -> i` 带 relation type `r`，目标节点 `i` 对源节点 `j` 做 attention，并加入 relation-specific key/value bias。

## Smoke test

```powershell
conda run -n huaweicup python src/grounding/stage2_rgcn_torch_grounding.py --train-limit 500 --dev-limit 200 --epochs 2
```

推荐的 Graph+Lexical 版本：

```powershell
conda run -n huaweicup python src/grounding/stage2_rgcn_torch_grounding.py --train-limit 1000 --epochs 3 --use-lexical-features
```

R-GTA smoke test：

```powershell
conda run -n huaweicup python src/grounding/stage2_rgcn_torch_grounding.py --encoder-type rgta --train-limit 500 --dev-limit 200 --epochs 2 --use-lexical-features --output-dir experiments/stage2_rgta_torch_grounding_smoke
```

R-GTA 可比实验：

```powershell
conda run -n huaweicup python src/grounding/stage2_rgcn_torch_grounding.py --encoder-type rgta --train-limit 500 --epochs 2 --use-lexical-features --output-dir experiments/stage2_rgta_torch_grounding_hybrid_500
```

## 全量训练

CPU 版可先跑：

```powershell
conda run -n huaweicup python src/grounding/stage2_rgcn_torch_grounding.py --train-limit 1000 --epochs 3
```

如果速度可以，再扩大：

```powershell
conda run -n huaweicup python src/grounding/stage2_rgcn_torch_grounding.py --train-limit 3000 --epochs 3
```

全量 CPU 训练可能超过 10 分钟，不建议第一轮直接跑。

## 输出

```text
experiments/stage2_rgcn_torch_grounding/
  rgcn_torch_train_config.json
  rgcn_torch_train_log.jsonl
  rgcn_torch_dev_metrics.json
  rgcn_torch_dev_predictions.jsonl
  rgcn_torch_dev_topk_examples.md
  rgcn_torch_model.pt
```

## Lexical features

`--use-lexical-features` 会把以下轻量特征加入 scorer：

- query-schema token overlap
- question-schema token overlap
- evidence-schema token overlap
- table phrase bonus
- column phrase bonus
- table-node indicator

这不是替代图网络，而是让 scorer 同时利用文本匹配信号与 R-GCN 结构表示。

## 对比基线

Stage 2A lexical：

```json
{
  "schema_recall@20": 0.7754,
  "table_recall@5": 0.8174,
  "column_recall@10": 0.6871,
  "column_recall@20": 0.7812
}
```

NumPy fixed R-GCN：

```json
{
  "schema_recall@20": 0.5944,
  "table_recall@5": 0.7738,
  "column_recall@10": 0.3939
}
```

PyTorch trainable R-GCN 的目标：

| Metric | Minimum target |
|---|---:|
| `schema_recall@20` | >= 0.75 |
| `column_recall@10` | >= 0.65 |
| `table_recall@5` | >= 0.80 |
