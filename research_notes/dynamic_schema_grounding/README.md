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

> 当前实验状态：静态 relation-conditioned RGTA 与 structured Top-K selection 已得到完整 dev
> 证据；直接 prompt 化、logits bias 和连续 hidden-state steering 未得到端到端 EX 支持。
> Stage 13C 静态 Graph Adapter 未通过图因果门，当前转入 Stage 14 显式 Typed RGTA Schema Tool。
> 完整证据与负结果见
> [29_research_path_retrospective.md](./29_research_path_retrospective.md)。

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
- [12_best_full_dev_grounder_record.md](./12_best_full_dev_grounder_record.md)：完整 BIRD dev 上 RGCN/RGTA 的最佳静态 grounding 记录。
- [13_value_index_and_join_path_completion.md](./13_value_index_and_join_path_completion.md)：Value Index 与 Steiner 近似 Join-Path Completion。
- [14_confidence_gated_value_fusion.md](./14_confidence_gated_value_fusion.md)：置信门控 Value 证据融合。
- [15_stage10a_factor_graph_reranker.md](./15_stage10a_factor_graph_reranker.md)：异质因子图候选重排序器。
- [16_stage10b_oof_grounding.md](./16_stage10b_oof_grounding.md)：按数据库隔离的 OOF grounding 流程。
- [17_stage10b_fix1_stable_reranker_training.md](./17_stage10b_fix1_stable_reranker_training.md)：稳定的 OOF Schema-RGTA 训练。
- [18_stage10c_topk_coverage_objective.md](./18_stage10c_topk_coverage_objective.md)：Query-level Top-K coverage objective。
- [19_stage10d_constrained_structured_coverage.md](./19_stage10d_constrained_structured_coverage.md)：Selection-aware structured coverage。
- [20_stage11a_dynamic_grounding_controller.md](./20_stage11a_dynamic_grounding_controller.md)：Partial-SQL recurrent grounding controller。
- [21_stage11b_uncertainty_residual_history.md](./21_stage11b_uncertainty_residual_history.md)：不确定性门控的残差历史。
- [22_stage11b_fix1_counterfactual_utility_gate.md](./22_stage11b_fix1_counterfactual_utility_gate.md)：反事实 utility gate。
- [23_stage11c_selection_regret_gate.md](./23_stage11c_selection_regret_gate.md)：Selection-regret dynamic gate。
- [24_stage12a_dynamic_rgta_llm_adapter.md](./24_stage12a_dynamic_rgta_llm_adapter.md)：动态图状态到 LLM 的 cross-attention/steering 实验。
- [25_stage12b_selection_utility_training.md](./25_stage12b_selection_utility_training.md)：Schema/operator/value-aware utility objective。
- [26_stage13a_typed_ra_supervision.md](./26_stage13a_typed_ra_supervision.md)：Typed relational-algebra 监督与数据审计。
- [27_stage13b_typed_action_pointer_decoder.md](./27_stage13b_typed_action_pointer_decoder.md)：Typed action/pointer RGTA decoder。
- [28_stage13c_static_graph_adapter_alignment.md](./28_stage13c_static_graph_adapter_alignment.md)：冻结 RGTA 与冻结 Code LLM 之间的静态 Graph Adapter 对齐实验、训练命令和验收标准。
- [29_research_path_retrospective.md](./29_research_path_retrospective.md)：从 Stage 0 到 Stage 13C 的完整研究复盘、创新分级、负结果和下一步决策门。
- [30_stage14_typed_graph_tool_interface.md](./30_stage14_typed_graph_tool_interface.md)：显式 Typed RGTA Schema Tool、约束组装、诊断协议与服务器命令。
- [31_stage14b_semantic_slot_graph_binder.md](./31_stage14b_semantic_slot_graph_binder.md)：语义槽位条件化 RGTA、表—列层次先验、同表 hard negatives，以及 correct/action-only/shuffled 因果实验。
- [32_stage15a_graph_grounded_sql_verifier.md](./32_stage15a_graph_grounded_sql_verifier.md)：LLM SQL 假设之后的 typed plan–schema graph 后验验证器、困难结构负样本及决策门。

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
