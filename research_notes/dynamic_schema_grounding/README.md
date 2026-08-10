# Dynamic Schema Grounding for LLM-based Text-to-SQL

本目录整理一个面向 Text-to-SQL 的研究方案：

> 将 Schema Grounding 从一次性的 schema linking，改造成 SQL 生成过程中的动态隐变量，并通过 recurrent cross-attention 与 hidden-state steering 注入 LLM decoding。

核心直觉是：

```text
Question
   ↓
Initial Schema Grounding
   ↓
Partial SQL Generation
   ↓
Update Grounding State
   ↓
Cross-Attention + Hidden-State Steering
   ↓
Next SQL Token / Clause
```

这不是简单把相关表列写进 prompt，而是让 schema grounding 成为 LLM 推理过程中的可更新神经状态。

## 文件结构

- [00_overview.md](./00_overview.md)：研究问题、核心假设与整体方案。
- [01_motivation.md](./01_motivation.md)：为什么 Text-to-SQL 不再只卷网络结构，以及为什么动态 grounding 有价值。
- [02_architecture.md](./02_architecture.md)：整体流程架构与模块关系。
- [03_method.md](./03_method.md)：方法细节，包括 schema graph encoder、dynamic grounding controller、cross-attention、hidden-state steering。
- [04_training_objectives.md](./04_training_objectives.md)：训练阶段、监督信号与 loss 设计。
- [05_experiments.md](./05_experiments.md)：实验方案、baseline、消融、干预实验与可视化分析。
- [06_risks_and_rebuttals.md](./06_risks_and_rebuttals.md)：潜在审稿质疑、方法风险与应对策略。
- [07_paper_outline.md](./07_paper_outline.md)：可直接扩写的论文大纲。
- [08_model_architecture_diagrams.md](./08_model_architecture_diagrams.md)：可直接放进论文草稿的模型架构图与流程图。
- [09_experiment_design_tables.md](./09_experiment_design_tables.md)：实验设计总表、消融矩阵、干预实验表与预期分析表。
- [10_staged_experiment_roadmap.md](./10_staged_experiment_roadmap.md)：按阶段推进实验的执行路线图，每阶段包含准备材料、流程、验收标准和进入下一阶段条件。
- [11_corrected_training_data_pipeline.md](./11_corrected_training_data_pipeline.md)：局部修正训练集的安全合并、审计与监督重建流程。

## 一句话版本

传统 Text-to-SQL 通常先做静态 schema linking：

```text
g = f(Q, S)
```

本文方向将其改为随 SQL 生成过程变化的动态状态：

```text
g_t = f(Q, S, SQL_<t)
```

其中：

- `Q` 是自然语言问题；
- `S` 是数据库 schema；
- `SQL_<t` 是当前已生成的 partial SQL；
- `g_t` 是当前生成步骤所需的 schema grounding state。

最终目标不是证明“加了一个模块更强”，而是证明：

> Schema grounding 对 SQL 推理具有动态、阶段性、可干预的因果作用。
