# 03. Method

## Problem Formulation

给定自然语言问题 `Q` 和数据库 schema `S`，目标是生成 SQL query `Y`：

```text
Y = {y_1, y_2, ..., y_T}
```

传统方法直接建模：

```text
P(Y | Q, S)
```

本方法引入动态 schema grounding state：

```text
G = {g_0, g_1, ..., g_T}
```

并建模：

```text
P(Y | Q, S) = ∏_t P(y_t | Q, S, Y_<t, g_t)
```

其中：

```text
g_t = f_phi(Q, S, Y_<t)
```

## 1. Schema Graph Encoder

将 schema 表示为图：

```text
S_graph = (V, E)
```

节点 `V` 包括 table 与 column。

每个节点的初始表示：

```text
x_i = [e_name; e_table; e_type; e_value; e_desc]
```

其中：

- `e_name`：表列名称 embedding；
- `e_table`：所属表 embedding；
- `e_type`：数据类型 embedding；
- `e_value`：sample values embedding；
- `e_desc`：可选的 column description embedding。

通过 graph encoder 得到：

```text
z_i = GraphEncoder(x_i, E)
```

最终得到 schema memory：

```text
Z = {z_1, ..., z_n}
```

## 2. Initial Grounding State

初始 grounding state 只依赖 question 与 schema：

```text
g_0 = f_init(Q, Z)
```

可以计算每个 schema element 的 relevance score：

```text
p_i^0 = sigmoid(q^T W z_i)
```

形成初始 belief：

```text
G_0 = {p_i^0 z_i}_{i=1}^n
```

## 3. Dynamic Grounding Update

在第 `t` 步，根据 partial SQL 更新 grounding state：

```text
g_t = f_update(Q, Z, Y_<t, g_{t-1})
```

一种简单实现：

```text
u_t = SQLContextEncoder(Y_<t)
```

然后：

```text
p_i^t = sigmoid(MLP([q; u_t; z_i; p_i^{t-1}]))
```

更新 schema grounding memory：

```text
G_t = {p_i^t z_i}_{i=1}^n
```

其中 `p_i^t` 表示第 `t` 步 schema element `i` 的 relevance belief。

## 4. Recurrent Cross-Attention

LLM 当前隐藏状态为 `h_t`。

使用 `h_t` 查询当前 grounding memory：

```text
c_t = CrossAttn(h_t, G_t)
```

展开为：

```text
Q_t = h_t W_Q
K_t = G_t W_K
V_t = G_t W_V
```

```text
c_t = softmax(Q_t K_t^T / sqrt(d)) V_t
```

`c_t` 是当前 SQL 生成步骤读取到的 schema context。

## 5. Hidden-State Steering

为了避免 grounding 过度干扰 LLM 原有语言和 SQL 能力，引入 gate：

```text
alpha_t = sigmoid(W_g [h_t; c_t])
```

然后执行 steering：

```text
h'_t = h_t + alpha_t c_t
```

下一 token 分布：

```text
P(y_t | Q, S, Y_<t) = softmax(W_o h'_t)
```

如果使用多层注入，可以只在 LLM 的中高层加入该模块。低层保留语言表示，高层负责 SQL 推理与 schema selection。

## 6. Clause-level Variant

Token-level update 计算量较大。更自然的版本是在 clause boundary 更新 grounding：

```text
g_SELECT = f(Q, S)
g_FROM = f(Q, S, SELECT_clause)
g_WHERE = f(Q, S, SELECT_clause, FROM_clause)
g_GROUP = f(Q, S, previous_clauses)
g_ORDER = f(Q, S, previous_clauses)
```

这与 SQL 结构更匹配。

例如：

```sql
SELECT customers.name
FROM customers
JOIN orders ON customers.id = orders.customer_id
GROUP BY customers.id
ORDER BY SUM(orders.amount) DESC
LIMIT 1
```

不同阶段的 grounding 应该不同：

| SQL stage | 主要 grounding |
|---|---|
| SELECT | `customers.name` |
| FROM | `customers`, `orders` |
| JOIN | `customers.id`, `orders.customer_id` |
| GROUP BY | `customers.id` |
| ORDER BY | `orders.amount` |

## 7. 与 Prompt-based Grounding 的区别

Prompt-based 方法：

```text
Relevant columns are: ...
LLM generates SQL
```

本方法：

```text
Schema graph → grounding state
partial SQL → update grounding
grounding state → cross-attention
schema context → hidden-state steering
LLM generates next SQL step
```

关键区别：

1. grounding 是可学习 hidden representation；
2. grounding 随 partial SQL 动态更新；
3. grounding 直接影响 LLM hidden state；
4. 可以通过 intervention 实验证明 grounding 的因果作用。

