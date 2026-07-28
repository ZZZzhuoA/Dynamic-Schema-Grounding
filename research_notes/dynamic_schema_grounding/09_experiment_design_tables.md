# 09. Experiment Design Tables

本文件整理可直接放入 proposal 或论文实验部分的表格。重点不是只列 benchmark，而是围绕本文核心命题设计实验：

> Dynamic grounding is useful, necessary, and causally involved in SQL generation.

## Table 1：Overall Experimental Plan

| Experiment | Research question | Dataset | Compared methods | Main metrics | Expected evidence |
|---|---|---|---|---|---|
| Main performance | 方法整体是否提升 Text-to-SQL 性能？ | Spider, BIRD | LLM prompting, static linking, RAG schema retrieval, ours | EM, EX | Ours 在复杂 query 上提升 |
| Static vs dynamic grounding | 动态 grounding 是否优于一次性 schema linking？ | Spider, BIRD | static `g=f(Q,S)` vs dynamic `g_t=f(Q,S,Y_<t)` | EX by SQL type | dynamic 在 join / nested / aggregation 上更强 |
| Module ablation | cross-attention、steering、gate 分别是否必要？ | Spider dev, BIRD dev | remove each module | EX, valid SQL rate | full model 最稳 |
| Grounding intervention | grounding state 是否真的影响 SQL？ | Spider subset, BIRD subset | normal vs random / adversarial / oracle grounding | EX drop, SQL change rate | 替换 grounding 后性能显著下降 |
| Long schema robustness | 长 schema 和干扰列下是否更鲁棒？ | augmented Spider / BIRD | prompt, static, dynamic | EX under schema scale | dynamic 下降更慢 |
| Grounding trajectory | grounding 是否随 SQL 阶段合理变化？ | selected cases | static vs dynamic | qualitative + grounding precision | grounding focus 与 clause 对齐 |
| Error analysis | 方法主要解决哪些错误？ | failed cases | base vs ours | error type ratio | wrong column / join path 错误减少 |

## Table 2：Dataset Usage

| Dataset | Role | Why needed | Evaluation focus |
|---|---|---|---|
| Spider | Standard benchmark | 跨数据库 Text-to-SQL 经典数据集，方便与传统方法比较 | EM, EX, SQL difficulty split |
| BIRD | Main realistic benchmark | schema 更复杂，更接近真实数据库场景 | EX, robustness, complex schema reasoning |
| Augmented Spider | Long-schema stress test | 人工加入 distractor tables / columns，专门测试 grounding 鲁棒性 | EX under increasing schema size |
| Manually annotated subset | Grounding validation | 小规模人工标注 question phrase 到 schema element 的对应关系 | grounding precision / recall |

## Table 3：Main Result Template

| Method | Type | Spider EM | Spider EX | BIRD EX | Notes |
|---|---|---:|---:|---:|---|
| Base LLM prompting | Prompt-only | - | - | - | Full schema in prompt |
| Schema retrieval + prompt | RAG-style | - | - | - | Retrieves top-k schema elements |
| Static schema linking + LLM | Static grounding | - | - | - | `g=f(Q,S)` before decoding |
| Cross-attn static grounding | Neural grounding | - | - | - | Uses schema context but not dynamic |
| Ours | Dynamic grounding | - | - | - | `g_t=f(Q,S,Y_<t)` |
| Oracle schema grounding + LLM | Upper bound | - | - | - | Gold SQL schema elements provided |

## Table 4：Ablation Study

| Variant | Dynamic update | Cross-attention | Hidden steering | Gate | Clause-level labels | Expected behavior |
|---|---|---|---|---|---|---|
| Base LLM | × | × | × | × | × | 依赖 prompt 与模型先验 |
| Static prompt grounding | × | × | × | × | × | 比 full schema prompt 稳，但无法动态修正 |
| Static neural grounding | × | ✓ | × | × | × | 能读取 schema memory，但 grounding 不随 SQL 变化 |
| Dynamic grounding only | ✓ | × | × | × | ✓ | grounding 可变，但与 decoder 交互弱 |
| Cross-attention only | ✓ | ✓ | × | × | ✓ | 能动态读取 schema，生成方向不一定强约束 |
| Steering without gate | ✓ | ✓ | ✓ | × | ✓ | 可能提升，也可能破坏 SQL fluency |
| Full model | ✓ | ✓ | ✓ | ✓ | ✓ | 预期最佳且最稳定 |

## Table 5：Static vs Dynamic Grounding by SQL Type

| SQL type | Static grounding EX | Dynamic grounding EX | Expected gain source |
|---|---:|---:|---|
| Single table | - | - | 提升可能较小，因为 grounding 简单 |
| Multi-table join | - | - | partial SQL 帮助确定 join path |
| Aggregation | - | - | grounding 随聚合目标更新 |
| Group by + order by | - | - | 区分 grouping column 与 ranking metric |
| Nested query | - | - | 子查询阶段需要重新建立局部 grounding |
| Ambiguous columns | - | - | partial SQL context 帮助消歧 |

## Table 6：Grounding Intervention Design

| Intervention | Operation | What it tests | Expected result |
|---|---|---|---|
| Random grounding replacement | 用随机 schema elements 替换 `g_t` | 模型是否依赖 grounding state | EX 明显下降 |
| Adversarial column swap | 将正确列替换为语义相近错误列 | 模型是否对 schema evidence 敏感 | SQL 中更容易出现被替换列 |
| Top-k grounding removal | 移除 grounding score 最高的 schema elements | top grounded elements 是否关键 | EX 下降，join / column 错误增加 |
| Oracle grounding | 用 gold SQL schema elements 替换预测 grounding | grounding quality 是否是瓶颈 | EX 上升，作为上界 |
| Frozen initial grounding | 始终使用 `g_0`，不更新 | 动态更新是否必要 | 复杂 SQL 上低于 full model |

## Table 7：Long Schema Robustness

| Schema scale | Prompt-only EX | Static grounding EX | Dynamic grounding EX | Expected observation |
|---|---:|---:|---:|---|
| 1x original | - | - | - | 三者差距可能较小 |
| 3x distractors | - | - | - | prompt-only 开始下降 |
| 5x distractors | - | - | - | static grounding 受干扰列影响 |
| 10x distractors | - | - | - | dynamic grounding 应下降最慢 |

### Distractor construction

建议加入四类干扰：

| Distractor type | Example | Purpose |
|---|---|---|
| Same-name columns | `created_at` in many tables | 测试上下文消歧 |
| Semantically similar columns | `price`, `amount`, `cost`, `revenue` | 测试指标 grounding |
| Plausible join tables | 多个表都可连接到目标表 | 测试 join path reasoning |
| Similar sample values | 多列都含有相似枚举值 | 测试 value linking 鲁棒性 |

## Table 8：Grounding Metrics

| Metric | Definition | Why useful |
|---|---|---|
| Table grounding precision | predicted relevant tables 中 gold table 比例 | 评估表级 grounding |
| Table grounding recall | gold tables 被覆盖比例 | 防止漏表 |
| Column grounding precision | predicted relevant columns 中 gold column 比例 | 评估列级选择准确性 |
| Column grounding recall | gold columns 被覆盖比例 | 防止漏列 |
| Clause-level grounding accuracy | 每个 SQL clause 的 grounded columns 是否匹配 gold clause labels | 验证动态 grounding |
| Grounding transition accuracy | grounding state 是否在 clause boundary 正确迁移 | 验证 recurrent update |
| Intervention degradation | intervention 后 EX 下降幅度 | 验证因果作用 |

## Table 9：Error Analysis Template

| Error type | Base LLM count | Static grounding count | Full model count | Expected change |
|---|---:|---:|---:|---|
| Wrong table | - | - | - | 明显减少 |
| Wrong column | - | - | - | 明显减少 |
| Wrong join path | - | - | - | 明显减少 |
| Missing aggregation | - | - | - | 中等减少 |
| Wrong grouping | - | - | - | 中等减少 |
| Wrong ordering metric | - | - | - | 中等减少 |
| Wrong nested query | - | - | - | 可能减少，但仍困难 |
| SQL syntax error | - | - | - | 不一定显著变化 |
| Executable but semantically wrong | - | - | - | 应减少 |

## Table 10：Case Study Template

| Step | Partial SQL | Top grounded schema elements | Interpretation |
|---|---|---|---|
| Initial | empty | `customers`, `orders`, `orders.amount` | 初始识别客户和消费金额相关 |
| SELECT | `SELECT customers.name` | `customers.name`, `customers.id` | 输出客户名称，同时保留客户 ID 用于 join/group |
| FROM/JOIN | `FROM customers JOIN orders` | `customers.id`, `orders.customer_id` | 确定 join path |
| GROUP BY | `GROUP BY customers.id` | `customers.id` | 按客户聚合 |
| ORDER BY | `ORDER BY SUM(orders.amount) DESC` | `orders.amount` | 按总消费金额排序 |

## Recommended Experiment Priority

如果资源有限，建议按优先级推进：

| Priority | Experiment | Reason |
|---:|---|---|
| 1 | Main results on Spider / BIRD | 必须证明基本有效 |
| 2 | Ablation study | 证明模块必要性 |
| 3 | Static vs dynamic grounding | 证明核心思想 |
| 4 | Grounding intervention | 证明 grounding 不是装饰 |
| 5 | Long schema robustness | 突出方法优势 |
| 6 | Grounding trajectory visualization | 增强论文说服力 |
| 7 | Manual grounding annotation | 如果时间允许，可增强 grounding 评价可信度 |

## 最推荐放进论文正文的表

正文建议保留：

1. Main result table；
2. Ablation table；
3. Static vs dynamic by SQL type；
4. Grounding intervention table；
5. Long schema robustness table。

Case study 和完整 error analysis 可以放 appendix。

