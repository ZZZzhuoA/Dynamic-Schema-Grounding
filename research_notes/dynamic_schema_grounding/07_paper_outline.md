# 07. Paper Outline

## Title

推荐：

**Schema Grounding as a Dynamic Latent State for LLM-based Text-to-SQL**

备选：

**Grounding-in-the-Loop: Dynamic Schema Grounding for Faithful Text-to-SQL Generation**

## Abstract Draft

Large language models have significantly improved text-to-SQL generation, yet they still struggle with complex schemas, ambiguous column semantics, and multi-step SQL reasoning. Existing schema linking or retrieval-based methods usually treat schema grounding as a static preprocessing step, producing a fixed set of relevant tables and columns before SQL generation. However, SQL generation is inherently sequential and structured: different clauses require different schema evidence, and partial SQL decisions should influence subsequent grounding. In this work, we formulate schema grounding as a dynamic latent state that evolves with partial SQL generation. We propose a recurrent grounding controller that updates schema grounding conditioned on the question, schema graph, and generated SQL context. The grounding state interacts with the LLM decoder through recurrent cross-attention and gated hidden-state steering, allowing the model to both retrieve schema evidence and adjust its decoding trajectory. Experiments on standard and long-schema text-to-SQL benchmarks demonstrate that dynamic grounding improves execution accuracy, especially on complex queries involving joins, aggregation, and nested structures. Further grounding intervention and trajectory analyses show that the proposed grounding state has a causal influence on SQL generation.

## 1. Introduction

要讲清楚四点：

1. LLM 已经提升 Text-to-SQL，但复杂 schema 下仍然不稳定；
2. 现有 schema linking / retrieval 大多是静态的；
3. SQL 生成是动态结构化过程，不同阶段需要不同 schema evidence；
4. 本文提出 dynamic schema grounding state。

建议关键句：

> We argue that schema grounding should not be treated as a one-shot preprocessing result, but as a recurrent latent state that evolves throughout SQL generation.

## 2. Related Work

### 2.1 Text-to-SQL

包括：

- grammar-based methods；
- sketch-based methods；
- schema-aware neural parsers；
- pretrained language model methods；
- LLM-based prompting and agentic methods。

### 2.2 Schema Linking and Schema Grounding

强调：

- 传统 schema linking；
- schema retrieval；
- value linking；
- relation-aware schema encoding。

指出不足：

> Most existing methods produce static schema relevance signals before decoding.

### 2.3 Grounding and Tool-augmented LLMs

可以简短类比：

- visual grounding；
- evidence-grounded generation；
- retrieval-augmented generation；
- hidden-state control / adapter / cross-attention。

## 3. Method

### 3.1 Problem Formulation

定义：

```text
Q, S, Y, g_t
```

核心公式：

```text
g_t = f_phi(Q, S, Y_<t)
```

```text
P(Y | Q, S) = ∏_t P(y_t | Q, S, Y_<t, g_t)
```

### 3.2 Schema Graph Encoder

讲：

- schema graph；
- node features；
- edge types；
- graph encoder output。

### 3.3 Dynamic Grounding Controller

讲：

- initial grounding；
- recurrent update；
- token-level 或 clause-level update；
- grounding score。

### 3.4 Recurrent Cross-Attention

讲：

- LLM hidden state queries grounding memory；
- output schema context。

### 3.5 Gated Hidden-State Steering

讲：

- gate；
- residual steering；
- final token prediction。

## 4. Training

### 4.1 Schema Grounding Pretraining

从 gold SQL 抽取 schema labels。

### 4.2 SQL Generation Alignment

训练 cross-attention、steering gate、adapter。

### 4.3 Dynamic Grounding Joint Training

加入 clause-level consistency loss。

总 loss：

```text
L = L_sql + λ L_g + β L_c
```

## 5. Experiments

### 5.1 Setup

Datasets：

- Spider；
- BIRD；
- long-schema extension。

Metrics：

- Exact Match；
- Execution Accuracy；
- grounding precision / recall；
- intervention degradation。

### 5.2 Main Results

比较整体性能。

### 5.3 Ablation Study

比较：

- no grounding；
- static grounding；
- cross-attention only；
- steering only；
- full model。

### 5.4 Static vs Dynamic Grounding

按 SQL 类型分析。

### 5.5 Grounding Intervention

这是亮点：

- random grounding replacement；
- adversarial column swap；
- oracle grounding；
- top-k grounding removal。

### 5.6 Grounding Trajectory Analysis

展示 SQL clause 与 grounding state 的动态对应。

## 6. Conclusion

重申：

> Schema grounding should be a dynamic reasoning state rather than a static linking result.

## Contribution Statement

可以最终写成：

1. We formulate schema grounding as a dynamic latent state in LLM-based text-to-SQL generation.
2. We propose a recurrent grounding controller that updates schema relevance conditioned on partial SQL.
3. We design a dual interaction mechanism with recurrent cross-attention and gated hidden-state steering.
4. We provide intervention and trajectory analyses to verify the causal role of grounding in SQL generation.

## 可能的短版中文总结

本文提出一种面向大语言模型 Text-to-SQL 的动态 Schema Grounding 框架。不同于传统方法将 schema linking 作为 SQL 生成前的一次性预处理，本文将 grounding 建模为随 partial SQL 生成不断更新的隐状态。模型通过 Schema Graph Encoder 表示数据库结构，通过 Dynamic Grounding Controller 根据问题、数据库模式和当前 SQL 片段更新 grounding state，并利用 cross-attention 和 gated hidden-state steering 将该状态注入 LLM decoding。实验将从整体执行准确率、静态与动态 grounding 对比、模块消融、长 schema 鲁棒性、grounding intervention 和 grounding trajectory 可视化等方面验证方法有效性。

