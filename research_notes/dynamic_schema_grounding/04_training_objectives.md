# 04. Training Objectives

## Training Overview

推荐三阶段训练：

1. Schema grounding pretraining；
2. SQL generation alignment；
3. Dynamic grounding joint training。

这种训练方式比较稳，不会一开始就让 LLM、schema encoder、grounding controller 全部混在一起训练。

## Stage 1：Schema Grounding Pretraining

目标：

```text
Question + Schema → Relevant Tables / Columns
```

训练模块：

- schema graph encoder；
- initial grounding scorer。

可冻结：

- LLM 主体。

### Grounding Labels

可以从 gold SQL 中自动抽取：

- used tables；
- used columns；
- join columns；
- aggregation columns；
- filter columns；
- order columns。

例如 gold SQL：

```sql
SELECT customers.name
FROM customers
JOIN orders ON customers.id = orders.customer_id
GROUP BY customers.id
ORDER BY SUM(orders.amount) DESC
LIMIT 1
```

positive schema elements：

```text
customers.name
customers.id
orders.customer_id
orders.amount
customers
orders
```

### Loss

多标签分类：

```text
L_g = - Σ_i [y_i log p_i + (1 - y_i) log(1 - p_i)]
```

其中 `y_i` 表示 schema element 是否出现在 gold SQL 中。

## Stage 2：SQL Generation Alignment

目标：

让 LLM 学会使用 grounding state 生成 SQL。

训练模块：

- cross-attention；
- steering gate；
- LoRA / adapter；
- SQL decoder head。

可冻结：

- LLM 主体大部分参数；
- schema encoder 可先冻结。

### Loss

标准 SQL token loss：

```text
L_sql = - Σ_t log P(y_t | Q, S, Y_<t, g_t)
```

## Stage 3：Dynamic Grounding Joint Training

目标：

让 grounding state 随 partial SQL 更新，并与最终 SQL 使用的 schema 保持一致。

训练模块：

- dynamic grounding controller；
- cross-attention；
- steering gate；
- adapter / LoRA；
- optionally schema encoder。

### Dynamic Grounding Labels

可以根据 SQL clause 构造阶段性标签。

例如：

| Stage | Positive schema elements |
|---|---|
| SELECT | selected columns, aggregation target |
| FROM | main tables |
| JOIN | join keys |
| WHERE | filter columns, matched values |
| GROUP BY | grouping columns |
| ORDER BY | ranking columns, aggregation target |

这样可以监督：

```text
g_SELECT, g_FROM, g_JOIN, g_WHERE, ...
```

而不是只监督最终使用了哪些 column。

### Consistency Loss

如果当前 SQL clause 使用了某个 schema element，则该阶段 grounding score 应该较高：

```text
L_c = - Σ_t Σ_i y_i^t log p_i^t
```

其中：

- `y_i^t` 是第 `t` 阶段 schema element `i` 是否应该被关注；
- `p_i^t` 是模型在第 `t` 阶段的 grounding score。

### Transition Smoothness Loss

为了避免 grounding state 在相邻阶段剧烈跳动，可以加入平滑约束：

```text
L_smooth = Σ_t ||p^t - p^{t-1}||_2
```

但该 loss 不能太强，因为 SQL 阶段切换时 grounding 本来就应变化。

建议作为可选项，不作为主贡献。

### Final Objective

总 loss：

```text
L = L_sql + λ L_g + β L_c + γ L_smooth
```

推荐默认：

```text
γ = 0
```

先验证核心机制，再考虑 smoothness。

## 可选：Execution Feedback Training

如果后续扩展，可以加入 execution feedback：

```text
Generated SQL → Execute → Error / Result → Repair
```

对应 reward：

```text
R = execution_success + result_correctness
```

但第一篇论文不建议把 RL、execution repair 和 dynamic grounding 全部塞进去。容易显得主题分散。

更稳的策略：

第一篇只讲：

> dynamic schema grounding during SQL generation

后续工作再讲：

> execution-aware grounding correction

## 参数高效微调

由于 LLM 较大，建议使用：

- LoRA；
- adapter；
- prefix / soft prompt，仅作为参数高效机制，不作为 grounding 注入主方法。

推荐训练参数：

- schema graph encoder；
- grounding controller；
- cross-attention；
- steering gate；
- LLM LoRA。

冻结：

- LLM backbone 大部分参数。

## 训练风险

### 风险 1：Grounding label 噪声

从 gold SQL 抽取 schema label 虽然方便，但不能完全代表自然语言里的真实 grounding。

应对：

- 区分 table-level、column-level、clause-level label；
- 使用 SQL AST 解析提高标签质量；
- 在分析中承认这是 weak supervision。

### 风险 2：Steering 干扰 LLM 原能力

如果 gate 过强，模型可能生成语法不自然或非法 SQL。

应对：

- 使用 residual injection；
- 使用 gate；
- 只在中高层注入；
- 对比 no-gate ablation。

### 风险 3：动态更新计算量较大

应对：

- 使用 clause-level update；
- schema encoder 缓存；
- 只更新 relevance score，不重新编码整个 schema graph。

