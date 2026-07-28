# 05. Experiments

## 实验目标

实验不应只证明 Execution Accuracy 提升，还要证明：

1. dynamic grounding 优于 static grounding；
2. cross-attention 与 hidden-state steering 解决不同问题；
3. grounding state 真的影响 SQL generation；
4. grounding trajectory 与 SQL 生成阶段存在合理对应关系。

## Datasets

### Spider

用途：

- 基础 Text-to-SQL 能力验证；
- 与经典方法比较；
- 分析复杂 SQL 类型。

指标：

- Exact Match；
- Execution Accuracy。

### BIRD

用途：

- 更复杂、更接近真实业务场景；
- schema 较大；
- question 更长，含外部知识需求。

指标：

- Execution Accuracy；
- Valid Efficiency Score，如需要。

### 长 Schema 扩展测试

为了突出 grounding 能力，可以构造扩展 schema：

```text
Original schema + distractor tables / columns
```

设置：

- 1x schema；
- 3x schema；
- 5x schema；
- 10x schema。

观察模型在长 schema 和干扰列下的鲁棒性。

## Baselines

### Classical Text-to-SQL

- RAT-SQL；
- LGESQL；
- PICARD。

### LLM-based Text-to-SQL

可根据可复现性选择：

- GPT-style prompting；
- DIN-SQL；
- C3；
- MAC-SQL；
- RAS-SQL；
- DAIL-SQL；
- CodeS；
- 本地开源 LLM + prompt。

### Retrieval / Linking Baselines

需要专门比较：

- static schema linking + LLM；
- schema retrieval + prompt；
- RAG-style relevant schema prompt；
- oracle schema linking + LLM。

其中 oracle schema linking 是上界参考：

```text
只把 gold SQL 用到的 schema elements 提供给模型
```

## Main Results

主结果表：

| Method | Spider EM | Spider EX | BIRD EX |
|---|---:|---:|---:|
| LLM Prompting | - | - | - |
| Static Schema Linking + LLM | - | - | - |
| RAG Schema Retrieval + LLM | - | - | - |
| Ours | - | - | - |

注意：

如果暂时没有资源跑 GPT-4 级别模型，可以主打开源 LLM 上的可控实验。

## Ablation Study

核心消融表：

| Model | Dynamic Grounding | Cross-Attn | Steering | Gate | EX |
|---|---|---|---|---|---:|
| Base LLM | × | × | × | × | - |
| + Static Grounding Prompt | × | × | × | × | - |
| + Cross-Attn | × | ✓ | × | × | - |
| + Steering | × | × | ✓ | ✓ | - |
| + Cross-Attn + Steering | × | ✓ | ✓ | ✓ | - |
| Full Dynamic Model | ✓ | ✓ | ✓ | ✓ | - |

关键预期：

- Cross-attention 单独提升 schema selection；
- Steering 单独可能提升但不稳定；
- Cross-attention + gate steering 更稳；
- dynamic grounding 在复杂 SQL 和长 schema 下优势最大。

## Static vs Dynamic Grounding

专门比较：

```text
Static:  g = f(Q, S)
Dynamic: g_t = f(Q, S, SQL_<t)
```

按 SQL 类型分组：

| SQL Type | Static EX | Dynamic EX | Gain |
|---|---:|---:|---:|
| Single table | - | - | - |
| Multi-table join | - | - | - |
| Aggregation | - | - | - |
| Nested query | - | - | - |
| Group by + order by | - | - | - |

预期：

dynamic 对复杂 SQL 提升更明显。

## Grounding Intervention Experiment

这是论文亮点之一，用来证明 grounding state 不是装饰。

### Intervention 1：Random Grounding Replacement

将正常 grounding state：

```text
g_t
```

替换为随机 schema elements：

```text
g'_t
```

观察 EX 下降。

如果下降明显，说明模型确实依赖 grounding。

### Intervention 2：Adversarial Column Swap

把高相关列替换成语义相近但错误的列：

```text
orders.create_time → customers.create_time
orders.amount → products.price
```

观察模型是否生成错误 SQL。

该实验可以证明模型对 grounding state 敏感。

### Intervention 3：Oracle Grounding

用 gold SQL 抽取出的 schema elements 替换模型预测 grounding。

如果 oracle grounding 提升明显，说明瓶颈仍在 grounding quality。

## Grounding Trajectory Visualization

选择典型样例展示 grounding 随 SQL 生成阶段变化。

示例问题：

```text
Find the customer who spent the most money.
```

可能轨迹：

| Stage | Top grounded schema elements |
|---|---|
| Initial | `customers`, `orders`, `orders.amount` |
| SELECT | `customers.name`, `customers.id` |
| FROM | `customers`, `orders` |
| JOIN | `customers.id`, `orders.customer_id` |
| ORDER BY | `orders.amount` |

这类图比只报分数更能体现方法思想。

## Long Schema Robustness

构造 distractor schema：

- 添加无关 table；
- 添加同名或近义 column；
- 添加相似 value；
- 添加多条 plausible join path。

评估：

| Schema Scale | Prompt | Static Grounding | Dynamic Grounding |
|---|---:|---:|---:|
| 1x | - | - | - |
| 3x | - | - | - |
| 5x | - | - | - |
| 10x | - | - | - |

预期：

dynamic grounding 随 schema 变长下降更慢。

## Error Analysis

建议错误类型：

1. wrong table grounding；
2. wrong column grounding；
3. wrong join path；
4. missing aggregation；
5. wrong filter condition；
6. wrong nested query；
7. grounding correct but SQL syntax wrong；
8. SQL executable but semantically wrong。

这能帮助证明方法主要解决了 schema grounding，而不是所有 Text-to-SQL 问题。

