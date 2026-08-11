# Best Full-Dev Schema Grounder Record

记录日期：2026-08-11

## 当前结论

截至本记录，完整 BIRD dev（1,534 个问题）上的最佳 schema grounder 是：

> Corrected BIRD supervision + valid compact LLM cards + Qwen3-Embedding-0.6B dense features + relation-conditioned prior + RGTA

对应实验目录：

```text
experiments/stage8g_corrected_llm_cards_rgta_seed42
```

当前最佳 checkpoint 为 epoch 6：

```text
assembled_schema_recall@30       = 0.9333072570940697
assembled_schema_precision@30    = 0.23538260956774648
assembled_complete_coverage@30   = 0.7105606258148631
assembled_complete_samples@30    = 1090 / 1534
assembled_missing_samples@30     = 444 / 1534
dev loss                         = 0.25127058482355963
```

## 完整覆盖率定义

对第 i 个问题，令 gold SQL 所需的全部表和列为 `G_i`，模型选出的 Top-30 schema 集合为
`P_i^30`。当且仅当：

```text
G_i is a subset of P_i^30
```

该问题才计为完整覆盖。漏掉任意一个必要表或列，都记为不完整。

因此：

```text
Complete Coverage@30
= number of samples whose complete gold schema is in Top-30 / all samples
```

该指标比平均 Recall@30 更严格。平均 recall 较高时，仍可能有很多问题各自漏掉一个关键列；
完整覆盖率能够直接统计还有多少问题具备生成正确 SQL 所需的完整 schema 前提。

当前 RGTA 在 1,534 个问题中完整覆盖 1,090 个，完整覆盖率为 71.056%。该指标只说明
schema 未被遗漏，不保证聚合、过滤值、排序方向、嵌套结构或最终 SQL 一定正确。

## 同设置 RGCN 对照

RGCN 对照实验目录：

```text
experiments/stage8g_corrected_llm_cards_rgcn_seed42
```

| Encoder | Best epoch | Recall@30 | Precision@30 | Complete coverage@30 | Complete samples | Dev loss |
|---|---:|---:|---:|---:|---:|---:|
| RGCN | 6 | 0.924075 | 0.232493 | 0.683181 | 1048 | 0.264473 |
| RGTA | 6 | **0.933307** | **0.235383** | **0.710561** | **1090** | **0.251271** |

在其他设置相同的情况下，RGTA 相对 RGCN：

- Recall@30 提升 0.923 个百分点；
- Precision@30 提升 0.289 个百分点；
- Complete Coverage@30 提升 2.738 个百分点；
- 完整覆盖样本增加 42 个；
- missing samples 从 486 降至 444。

因此，目前只对编码器结构作判断时，RGTA 是主模型，RGCN 是结构消融基线。

## RGTA 最佳 checkpoint 的关系级结果

| Relation | Examples | Recall@5 | Recall@10 | Recall@20 | MRR |
|---|---:|---:|---:|---:|---:|
| OUTPUT_TARGET | 1518 | 0.779831 | 0.892512 | 0.952108 | 0.685554 |
| ENTITY_NAME | 1301 | 0.777325 | 0.892621 | 0.955419 | 0.668500 |
| METRIC_TARGET | 1036 | 0.830817 | 0.893213 | 0.942958 | 0.730395 |
| PREDICATE_COLUMN | 1364 | 0.620152 | 0.765530 | 0.872769 | 0.527093 |
| VALUE_ANCHOR | 1020 | 0.641895 | 0.794601 | 0.902862 | 0.494879 |
| TEMPORAL_FILTER | 349 | 0.906877 | 0.952722 | 0.988539 | 0.726898 |
| ORDER_KEY | 271 | 0.759533 | 0.876999 | 0.943419 | 0.631964 |
| GROUP_KEY | 95 | 0.792105 | 0.883333 | 0.949123 | 0.577572 |
| JOIN_BRIDGE | 1534 | 0.688047 | 0.867025 | 0.960346 | 0.835553 |
| FORMULA_COMPONENT | 255 | 0.929739 | 0.982026 | 0.998693 | 0.822228 |

当前最明显的弱项是 `PREDICATE_COLUMN` 和 `VALUE_ANCHOR`；公式、时间和指标关系已经具有
较高召回。该观察只记录为后续研究依据，本阶段不据此修改算法。

## Learned similarity-prior scales

RGTA 最佳 checkpoint 学到的关系特定相似度权重：

| Relation | Scale |
|---|---:|
| OUTPUT_TARGET | 2.341594 |
| ENTITY_NAME | 2.284576 |
| METRIC_TARGET | 2.452800 |
| PREDICATE_COLUMN | 1.376606 |
| VALUE_ANCHOR | 1.199133 |
| TEMPORAL_FILTER | 1.573314 |
| ORDER_KEY | 1.856989 |
| GROUP_KEY | 1.680597 |
| JOIN_BRIDGE | 1.532974 |
| FORMULA_COMPONENT | 2.111687 |

这些权重表明 dense semantic similarity 对 metric、output、entity 和 formula 的贡献较强，
而 predicate/value 对纯文本相似度的依赖较弱。该结果是模型行为描述，不等同于因果结论。

## Checkpoint 行为

两个模型都在 epoch 6 达到最佳：

- RGTA：epoch 6 为 0.933307，epoch 8 降至 0.928114；
- RGCN：epoch 6 为 0.924075，epoch 8 为 0.923727。

RGTA 在 epoch 6 后出现更明显的回落，因此必须使用 best checkpoint，而不是最后一个 epoch。
训练总计 8 epochs，epoch 6 后只有两个未提升 epoch，尚未满足 `patience=3`，所以
`stopped_early=false` 是正常现象。

## 实验有效性边界

本记录只支持以下结论：

> 在相同 corrected data、valid LLM cards、dense embeddings、relation-conditioned prior、
> seed 和训练配置下，RGTA 优于 RGCN。

以下旧结果不能直接加入同一比较表：

- 只评估 100 条 dev 的早期 RGTA/RGCN 结果；
- 使用 100% fallback schema cards、导致关系标签塌缩的 corrected 结果；
- 使用不同 schema-card 来源、不同训练样本量或不同 dev-limit 的结果。

此外，当前 BIRD dev 同时用于 best-epoch 选择和指标报告，因此这是开发阶段最佳记录，不是严格
隔离的最终测试结果。后续若用于论文主结果，应从 train databases 中构造 database-disjoint
validation split，只用它选择 checkpoint，并将完整 BIRD dev 保留为一次性最终评估。

## 当前状态

本阶段停留在 schema grounder 结果记录，不进入 LLM cross-attention、hidden-state steering、
下游 SQL generation 或新的算法修改阶段。
