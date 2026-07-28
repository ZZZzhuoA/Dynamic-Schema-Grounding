# 00. Overview

## 研究题目

可选题目：

1. **Dynamic Schema Grounding via Recurrent Cross-Attention for LLM-based Text-to-SQL**
2. **Grounding-in-the-Loop: Dynamic Schema Grounding for Faithful Text-to-SQL Generation**
3. **Schema Grounding as a Dynamic Latent State in LLM-based Text-to-SQL**

推荐第三个，概念更干净，论文故事也更集中。

## 核心问题

当前 LLM-based Text-to-SQL 的瓶颈不再只是“模型会不会写 SQL”，而是：

> 模型是否知道当前 SQL 生成步骤应该依赖哪些表、列、值和 join path，并能否在生成过程中持续修正这种判断。

传统 schema linking 往往是一次性的：

```text
Question + Schema → Relevant Tables / Columns → SQL
```

但真实 SQL 生成是逐步展开的。不同阶段需要的 schema evidence 不同：

- `SELECT` 阶段关注输出列或聚合列；
- `FROM` 阶段关注核心表和 join path；
- `WHERE` 阶段关注过滤条件与值匹配；
- `GROUP BY` 阶段关注维度列；
- `ORDER BY` 阶段关注排序指标；
- nested query 阶段还需要重新建立局部 schema context。

因此 schema grounding 不应被视为静态预处理，而应被视为动态推理状态。

## 核心假设

本文方向的核心假设是：

> Schema grounding is a dynamic latent state that should be updated according to the partial SQL generation process.

形式化为：

```text
g_t = f_phi(Q, S, Y_<t)
```

其中：

- `Q`：自然语言问题；
- `S`：数据库 schema；
- `Y_<t`：已经生成的 partial SQL；
- `g_t`：第 `t` 步的动态 schema grounding state。

然后 LLM 根据该状态生成下一步 SQL：

```text
y_t = f_theta(Q, Y_<t, g_t)
```

## 方法概览

整体方法由四个部分组成：

1. **Schema Graph Encoder**

   将数据库 schema 编码成结构化图表示，包括 table、column、foreign key、datatype、sample value、semantic similarity 等信息。

2. **Dynamic Grounding Controller**

   根据 question、schema graph 和 partial SQL 更新 grounding state。

3. **Recurrent Cross-Attention**

   让 LLM 在每个生成阶段动态读取当前需要的 schema evidence。

4. **Hidden-State Steering**

   通过门控机制将 grounding context 注入 LLM hidden state，改变 SQL decoding 的方向。

## 预期贡献

可以写成四点：

1. 提出将 schema grounding 建模为动态隐变量，而不是一次性 schema linking。
2. 设计 recurrent grounding controller，根据 partial SQL 更新 schema grounding state。
3. 通过 cross-attention 与 hidden-state steering 实现 grounding state 与 LLM decoding 的深度交互。
4. 设计 grounding trajectory 与 grounding intervention 实验，验证 grounding 对 SQL 生成的实际作用。

## 与 Visual Grounding / VividMed 的类比

VividMed 的缺陷在于：

```text
Image → Text → Grounding
```

即先生成医学结论，再给结论找视觉区域。

更理想的方式是：

```text
Image → Candidate Claim → Ground Evidence → Verify → Continue / Revise
```

迁移到 Text-to-SQL：

传统方法类似：

```text
Question → SQL → explain which schema was used
```

更理想的方式是：

```text
Question → Ground Schema Evidence → Generate Partial SQL → Update Evidence → Continue / Revise
```

因此，本方案本质上是把 grounding 从“生成后的解释”提升为“生成过程中的动态约束”。

