# 01. Motivation

## 为什么 Text-to-SQL 不再只卷网络结构

早期 Text-to-SQL 的主要问题是模型表示能力不足，因此大量工作集中在模型结构上：

- Seq2Seq decoder；
- grammar-based decoder；
- sketch-based decoder；
- schema-aware encoder；
- graph neural network；
- relation-aware Transformer；
- pretrained encoder-decoder。

这些方法主要解决：

```text
Question + Schema → SQL
```

中的编码和解码能力问题。

但在 LLM 时代，通用模型已经具备较强的语言理解、代码生成和 SQL 模板组合能力。新的瓶颈变成：

1. schema 很复杂；
2. 表列名称语义模糊；
3. join path 难以选择；
4. 问题表达与数据库字段并非字面匹配；
5. SQL 生成过程需要阶段性推理；
6. 执行结果和真实意图之间存在差距。

因此，研究重点从“设计一个更强 encoder”逐渐转向：

- schema grounding；
- SQL planning；
- execution feedback；
- self-correction；
- reward learning；
- agentic SQL generation；
- verification。

## 静态 Schema Linking 的局限

传统 schema linking 通常假设：

```text
g = f(Q, S)
```

也就是说，根据问题和 schema 一次性确定相关表列。

但 SQL 生成不是一次性决策，而是逐步构造。不同 partial SQL 会改变后续需要关注的 schema。

例如问题：

```text
Find the customers with the highest total purchase amount.
```

初始可能相关：

```text
customers.name
orders.customer_id
orders.amount
```

当模型已经生成：

```sql
SELECT customers.name
FROM customers
```

下一步 grounding 应重点转向：

```text
orders.customer_id
orders.amount
```

因为需要 join 和 aggregation。

如果 grounding 始终静态，模型无法利用 partial SQL 提供的新约束。

## 为什么 Prompt 注入不够

最简单的做法是把 schema linking 结果写进 prompt：

```text
Relevant columns:
- customers.name
- orders.amount
- orders.customer_id

Generate SQL:
```

这种方法有几个问题：

1. grounding 是离散文本，不是可学习状态；
2. grounding 进入 prompt 后难以在 decoding 过程中动态更新；
3. 模型是否真正使用这些 schema evidence 难以验证；
4. 对长 schema 场景不稳定；
5. 很容易被审稿人归为 schema retrieval / RAG 工程改进。

因此，论文级创新需要超越 prompt：

> 将 grounding state 注入 LLM 的 hidden representation，而不是只写入上下文文本。

## 为什么 Cross-Attention 合理

Cross-attention 解决的是读取问题：

> LLM 在当前生成阶段应该读取哪些 schema evidence？

形式上：

```text
LLM hidden state → query
Schema grounding state → key / value
```

于是模型可以在不同 SQL 阶段关注不同表列。

例如：

- 生成 `SELECT` 时关注输出列；
- 生成 `FROM` 时关注表；
- 生成 `JOIN` 时关注 foreign key；
- 生成 `WHERE` 时关注过滤列和值；
- 生成 `ORDER BY` 时关注聚合指标。

## 为什么 Hidden-State Steering 合理

Cross-attention 提供信息，但不一定能强制改变模型生成方向。

Hidden-state steering 进一步解决：

> 当前 grounding state 如何调控 LLM 的内部推理状态？

简单形式：

```text
h'_t = h_t + alpha_t c_t
```

其中：

- `h_t` 是 LLM 原隐藏状态；
- `c_t` 是 cross-attention 得到的 schema context；
- `alpha_t` 是 gate，控制 grounding 注入强度。

这样 grounding state 不只是被“看见”，而是能实际改变下一 token 的概率分布。

## 为什么二合一合理

Cross-attention 和 hidden-state steering 不是重复模块。

它们分别对应：

- **Cross-attention**：读取 schema evidence；
- **Hidden-state steering**：调控 LLM decoding。

可以统一成一个机制：

```text
Dynamic Grounding State
        ↓
Cross-Attention extracts schema context
        ↓
Gate controls injection strength
        ↓
Hidden-State Steering changes SQL decoding
```

所以贡献不是“加两个模块”，而是：

> 使用同一个动态 grounding state 同时完成 schema evidence retrieval 与 decoding control。

