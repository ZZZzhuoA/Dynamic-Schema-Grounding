# 06. Risks and Rebuttals

## 质疑 1：这是不是只是加了 cross-attention？

### 风险

审稿人可能认为：

> Cross-attention 是常规模块，方法创新不足。

### 回应

论文重点不能放在“我们加了 cross-attention”，而应放在：

> schema grounding is modeled as a dynamic latent state conditioned on partial SQL generation.

Cross-attention 只是实现 grounding state 与 LLM hidden state 交互的机制之一。

需要用实验支撑：

- static vs dynamic grounding；
- grounding trajectory；
- intervention experiment。

这些实验能证明核心贡献是动态 grounding，而不是 attention 模块堆叠。

## 质疑 2：Hidden-state steering 会不会破坏 LLM 能力？

### 风险

直接修改 hidden state 可能导致：

- SQL 语法变差；
- LLM 原有 reasoning 能力下降；
- 训练不稳定。

### 回应

方法中使用 gate：

```text
h'_t = h_t + alpha_t c_t
```

其中 `alpha_t` 自动控制注入强度。

同时做消融：

- no steering；
- steering without gate；
- gated steering；
- layer-wise steering。

如果 gated steering 最稳，就能说明该设计必要。

## 质疑 3：为什么不用 prompt / RAG 就够了？

### 风险

审稿人可能认为：

> 直接检索相关 schema 写进 prompt 就可以，没必要改 hidden state。

### 回应

Prompt / RAG 的 grounding 是静态文本上下文：

```text
g = f(Q, S)
```

本方法的 grounding 是动态神经状态：

```text
g_t = f(Q, S, SQL_<t)
```

优势主要体现在：

- 长 schema；
- 多表 join；
- nested query；
- ambiguous columns；
- SQL clause 之间依赖强的场景。

必须在这些场景中做 targeted evaluation。

## 质疑 4：动态 grounding 的监督标签从 gold SQL 抽取，是否可靠？

### 风险

Gold SQL 中出现的列不一定等于自然语言中真实提到的 schema grounding。

### 回应

承认这是 weak supervision，并说明：

- Text-to-SQL 缺乏人工 phrase-to-column grounding 标注；
- SQL-derived labels 是可扩展监督；
- clause-level extraction 比 whole-SQL extraction 更细；
- 通过 execution accuracy 与 intervention 验证 grounding 的实用性。

可补充小规模人工标注评估：

抽取 100-300 个样例，人工判断 question phrase 与 schema element 对应关系，评估 grounding precision / recall。

## 质疑 5：方法计算量太大

### 风险

如果每个 token 都更新 schema graph，开销会很高。

### 回应

采用：

- schema graph encoder caching；
- clause-level update；
- lightweight grounding score update；
- only top-k schema elements participate in cross-attention。

复杂度控制为：

```text
Encode schema once
Update relevance score per clause
Cross-attend to top-k schema states
```

## 质疑 6：性能提升是否来自更多参数？

### 风险

新增模块带来更多参数，提升可能只是参数量增加。

### 回应

需要做参数量控制实验：

1. same-size adapter baseline；
2. random schema memory baseline；
3. static grounding with same number of parameters；
4. dynamic grounding full model。

如果 full model 显著优于 same-size adapter 和 random memory，说明提升来自 grounding 机制。

## 质疑 7：能否泛化到新数据库？

### 风险

Text-to-SQL 的核心是跨数据库泛化。

### 回应

Schema graph encoder 不依赖固定数据库 ID，而是依赖：

- table / column name；
- relation type；
- datatype；
- sample values；
- schema structure。

实验上使用 Spider / BIRD 的 database split，并在 unseen database 上报告结果。

## 质疑 8：Grounding trajectory 是否只是可视化，不代表真实因果？

### 风险

Attention 可视化常被质疑不能说明因果。

### 回应

所以不能只做 attention map。

必须加入 grounding intervention：

- random replacement；
- adversarial swap；
- oracle grounding；
- remove top-k grounded elements。

如果改变 grounding state 会系统性改变 SQL 输出，才能说明 grounding 有因果作用。

## 最重要的写作策略

论文不要写成：

> We add cross-attention and hidden-state steering to LLM.

而要写成：

> We formulate schema grounding as a recurrent latent state that evolves with partial SQL generation, and instantiate its interaction with LLM decoding through cross-attention and gated hidden-state steering.

也就是说：

- dynamic grounding 是思想；
- controller 是核心模块；
- cross-attention 是读取机制；
- steering 是控制机制；
- intervention 是证据。

