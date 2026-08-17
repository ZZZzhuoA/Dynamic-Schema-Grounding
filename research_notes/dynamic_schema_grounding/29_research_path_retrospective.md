# 29. Research Path Retrospective

记录日期：2026-08-17

## 1. 当前研究结论

本项目最初的核心假设是：

> Schema grounding 不应只是 SQL 生成前的一次性检索结果，而应成为能够影响 LLM SQL 推理的神经状态。

截至 Stage 13C，实验将这个宽泛假设拆成了三个可独立检验的问题：

1. 图网络能否在跨数据库条件下学习可靠的 schema 结构表示？
2. 固定预算下，能否把高召回候选转化为完整、连通且不互相挤占的 schema 子图？
3. 图结构信息能否在不破坏 LLM 原有代码生成能力的情况下，对 SQL 生成产生有益且可干预的影响？

当前证据支持前两个问题，但第三个问题尚未解决：

- relation-conditioned RGTA 在完整 BIRD dev 上取得了可靠的静态 grounding 提升；
- OOF candidate reranking 与 constrained structured coverage 将 Complete Coverage@30 从
  `0.710561` 提升到 `0.808996`；
- prompt 裁剪、logits bias 和 Stage 12 连续 hidden-state steering 均未把 schema 指标稳定转化为 SQL EX；
- Stage 13C 正在检验更保守的方案：冻结 RGTA 和 Code LLM，只训练一个保留语义与图拓扑的静态 Graph Adapter。

因此，当前论文主张必须保持边界：

> 已有实验支持“结构化图建模能提高 schema grounding 与固定预算覆盖率”，但尚不能声称“动态图状态已经提高端到端 SQL generation”。

## 2. 术语与评价标准

| 术语 | 本项目中的含义 |
|---|---|
| Schema Grounding | 将问题语义映射到表、列和结构关系的概率状态 |
| RGTA | 本项目实现的关系感知图 Transformer/Attention 编码器，当前主 schema encoder |
| RGCN | 关系类型控制的图卷积编码器，作为结构消融基线 |
| Complete Coverage@30 | 一个样本所需的全部 gold 表列均包含在最终 Top-30 中的比例 |
| OOF Grounding | 按数据库划分、由未见过该数据库的模型产生训练集第一层预测 |
| Structured Coverage | 在固定预算内，显式优化完整 gold schema 集合可被共同选中的目标 |
| Neural Injection | 图状态通过 cross-attention 或 residual steering 进入 LLM hidden states |
| Typed RA | 将 SQL 分解成有类型的关系代数 DAG、指针、算子和值路由 |
| Graph Adapter | 将冻结图编码器输出映射并注入冻结 LLM 的可训练小网络 |

不能混淆以下指标：

- Schema Recall@30 衡量平均找回比例；
- Complete Coverage@30 衡量单个问题是否具备全部必要 schema；
- Execution Success 只表示 SQL 可执行；
- Execution Accuracy（EX）才表示执行结果与 gold 一致。

高 schema recall 是 SQL 正确的必要条件之一，但不是充分条件。

## 3. 研究路径总览

| 阶段 | 核心尝试 | 主要结论 | 状态 |
|---|---|---|---|
| Stage 0–1 | BIRD 审计、SQL/命中/FK 标签融合 | 建立可执行、可追踪的监督基础 | 必要且有效 |
| Stage 2 | RGCN/RGTA 静态 schema grounder | 图结构能够提高表列 grounding；RGTA 最终优于 RGCN | 已支持 |
| Stage 3–6 | Top-K schema prompt、子图 prompt、role/value prompt | 过滤后的 prompt 多数弱于 full schema，不能体现图网络的神经结构优势 | 负结果 |
| Stage 7 | token logits bias 与 operation gate | 能改变少数输出，但几乎无稳定 EX 增益，容易产生无约束扰动 | 负结果 |
| Stage 8 | relation-conditioned beliefs、LLM cards、dense embeddings | 显著改善跨数据库 schema 表示，形成最佳第一层 grounder | 已支持 |
| Stage 9 | Value Index、置信门控、Steiner 近似 Join completion | Value/Join 是独立必要证据；Join completion 尤其有意义，直接 value 注入有破坏性 | 部分支持 |
| Stage 10 | OOF reranker、Top-K coverage、constrained structured coverage | 固定预算应按集合效用而非独立分数选择；Complete Coverage@30 显著提高 | 已支持 |
| Stage 11 | partial-SQL recurrent controller、历史残差与 utility gates | 动态状态可训练，但历史更新未稳定优于独立 operation-RGTA | 未支持动态优势 |
| Stage 12 | RGTA–LLM cross-attention + hidden-state steering | 注入路径可激活，但端到端 EX 明显下降；单一连续残差职责过宽 | 明确负结果 |
| Stage 13A | Typed relational-algebra supervision | 分离 schema、operator、value 与 join，修复隐式监督污染 | 有意义的重构 |
| Stage 13B | Typed action/pointer decoder | 证明图网络可学习结构化动作和指针，但尚未形成最终 SQL EX | 中间结果有效 |
| Stage 13C | Frozen-GNN → Graph Adapter → Frozen-LLM | CE 下降但 correct/corrupted graph 因果门失败 | 已停止 |
| Stage 14 | Typed RGTA schema tool + constrained assembly | 用显式 schema IDs/FK edges/value routes 替代连续 hidden-state 注入 | 已实现首版 |

## 4. 各阶段尝试与证据

### 4.1 Stage 0–1：数据、监督与审计

完成的工作包括：

- 检查 BIRD train/dev 问题、SQLite 数据库和 gold SQL 可执行性；
- 将 `hit_info`、SQL parse 和 FK closure 合并为 schema labels；
- 发现 `hit_info` 不含外键，不能独立构成完整监督；
- 审计并合并 `processed_final_data` 中的训练问题与 gold SQL 修正；
- 在 Stage 13A-fix1 中修复 `column_types` 的 wildcard 索引偏移；
- 修复短数值错误匹配到长数字片段、PROJECT/SORT 顺序和 join path 标签问题；
- 明确分离 `inference_inputs` 与 `training_targets`，防止 gold SQL 泄漏。

意义：这些工作不是模型创新，但决定了后续结论是否可信。尤其是类型索引偏移和问题—SQL
错配会系统性污染训练，不能把修正后的提升全部归因于网络结构。

### 4.2 Stage 2：静态图 Grounding

数据库被表示为 text-attributed relational schema graph：

- 节点：table 和 column；
- 节点属性：名称、类型、所属表、语义卡、样例/问题相关特征；
- 边：table-to-column、column-to-table、FK forward/backward 和 self-loop；
- query 作为条件状态参与节点打分，而不是数据库图中的普通持久节点。

先后实现了 lexical baseline、RGCN 和 RGTA。最终同设置完整 BIRD dev 对照为：

| Encoder | Recall@30 | Precision@30 | Complete Coverage@30 | Complete samples |
|---|---:|---:|---:|---:|
| RGCN | 0.924075 | 0.232493 | 0.683181 | 1048 / 1534 |
| RGTA | **0.933307** | **0.235383** | **0.710561** | **1090 / 1534** |

有意义的结论：关系控制的全局注意力比局部固定聚合更适合长 schema 中的 query-conditioned
选择，RGTA 应作为主编码器，RGCN 保留为结构消融。

### 4.3 Stage 3–6：Prompt-based Schema Selection

尝试包括：

- lexical Top-30、RGTA Top-30/50/80；
- FK endpoint closure 和预算化候选子图；
- clause-aware、TCCE、value-aware、relation-fusion 和 role-aware prompt；
- 更严格的 SQL-only 输出约束和 schema 格式修复。

代表性 100 条在线 Qwen3-32B 结果：

| Setting | EX | Execution success |
|---|---:|---:|
| Full schema | **0.38** | 0.96 |
| RGTA Top-30 | 0.23 | 0.87（对应 relation-fusion 版本） |
| RGTA Top-50 | 0.20 | 0.86 |
| RGTA Top-80 | 0.28 | 0.93 |
| Lexical Top-30 | 0.18 | 0.70 |
| Subgraph budget 50 / 80 | 0.20 / 0.24 | 0.86 / 0.94 |
| Role-aware relation fusion | 0.15 | 0.92 |

注：不同 prompt 版本的 RGTA Top-30 曾出现 `0.17–0.23`，不能把它们当成完全相同设置。

负结论：增加 Top-K 只能减少“schema 不可见”，不能修复候选排序逻辑；把图网络结果翻译成
文本列表也丢失了图拓扑、概率状态和关系类型。Prompt pruning 不应再作为主算法路线，只保留为
外部可比基线。

### 4.4 Stage 7：解码期 Logits Bias

尝试将 schema 名称分词后，通过 logits processor 提升对应 token 概率，并加入 operation-aware gate。

观测：

- 初版 normal/random/zero/reverse 产生完全相同的 20 条 SQL，暴露 token map 没有真正作用；
- 修复后强 bias 能改变输出，但 20 条 EX 从 zero 的 `0.55` 降至 normal 的 `0.50`；
- operation gate 的 20 条实验只改变 1 条 SQL，normal 与 zero 均为 `0.55` EX；
- 较温和设置出现 6/20 SQL 改变，但没有可靠准确率收益。

负结论：identifier 是多 token 序列，局部 token bias 不等于结构选择；它无法约束 table-column
所有权、join path、算子和值，且容易在错误位置提升同名子词。该路线不再作为核心模型。

### 4.5 Stage 8：Relation-conditioned Grounding

本阶段从“统一 schema relevance”转向多关系 belief：

```text
OUTPUT_TARGET, ENTITY_NAME, METRIC_TARGET,
PREDICATE_COLUMN, VALUE_ANCHOR, TEMPORAL_FILTER,
ORDER_KEY, GROUP_KEY, JOIN_BRIDGE, FORMULA_COMPONENT
```

关键实现包括：

- clause/relation-specific supervision；
- 本地 heuristic schema cards，随后升级为 LLM schema/question cards；
- 将长自由文本卡压缩为结构化短字段；
- 使用 Qwen3-Embedding-0.6B 构造 query/node dense embedding cache；
- relation-conditioned similarity prior 与 RGTA 图传播联合训练；
- 修正训练集并在全 1,534 dev 上比较 RGCN/RGTA。

这是目前最可靠的静态模型创新。关系级结果还揭示了明确瓶颈：

- `FORMULA_COMPONENT Recall@20 = 0.998693`；
- `TEMPORAL_FILTER Recall@20 = 0.988539`；
- `PREDICATE_COLUMN Recall@20 = 0.872769`；
- `VALUE_ANCHOR Recall@20 = 0.902862`。

意义：问题与 schema 的关系不是单一相似度。输出列、谓词列、公式组件和 join bridge 应被视为
不同潜在角色。这个分解同时为后续 action/pointer decoder 提供了接口。

### 4.6 Stage 9：Value 与 Join 的外部证据

实现了：

- SQLite Value Index：规范化问题中的实体值并检索可能的 predicate/value columns；
- FK 图上的 join-path completion：使用 terminal metric closure + MST 的 Steiner tree 近似补全中间表和 FK endpoints；
- confidence-gated value fusion：根据歧义度、semantic support 和 competitor margin 将候选分为
  `inject`、`rerank`、`reject`。

实验观察：直接 value 注入虽净增加完整覆盖，但同时破坏多个原本完整样本；Join-only 的收益更稳定。

有意义的结论：

- 条件值可以提供 predicate column 的互补证据，但不能直接当作确定链接；
- join bridge 经常不在自然语言中显式出现，必须由图连通性机制补全；
- value candidate 不能无条件成为 join terminal，否则会产生级联路径错误。

### 4.7 Stage 10：候选集合重排序

第一层 RGTA 负责高召回检索，第二层候选图负责固定预算选择。尝试包括：

- heterogeneous factor graph：schema variables + relation/value/join factors；
- MLP、Schema-RGTA、Factor-RGTA 对照；
- database-disjoint OOF predictions，避免 in-sample stacking；
- gradient accumulation 与 epoch 内 checkpoint；
- query-level Top-K coverage objective；
- constrained selection-aware structured coverage loss；
- owner closure、table partition 和固定预算 selector。

Factor-RGTA 没有形成足够稳定的额外收益，因此被暂停；真正有效的是 OOF Schema-RGTA 与集合级
目标。最佳 Stage 10-D 结果为：

| Metric | First-stage baseline | Constrained reranker |
|---|---:|---:|
| Schema Recall@30 | 0.933307 | **0.956051** |
| Complete Coverage@30 | 0.710561 | **0.808996** |
| Complete samples | 1090 | **1241** |
| Table Recall@30 | 0.977694 | **0.986484** |
| Column Recall@30 | 0.918674 | **0.946117** |

候选上界为 `candidate_oracle_recall = 0.982602`、`candidate_complete_coverage = 0.905476`，说明
后续仍有选择空间，但不能突破候选构造上限。

多 seed 中结构损失的收益较小但并非完全稳定；例如 seed 42/44 达到 `0.808996`，seed 43 为
`0.799218`。因此可以声称集合级重排序有效，不能声称当前 structured loss 在所有 seed 上都显著优于
普通 coverage objective。

### 4.8 Stage 11：Dynamic Grounding Controller

将 gold SQL 转成 operation trajectory，并测试：

- independent operation-RGTA；
- recurrent partial-SQL state；
- residual uncertainty-gated history；
- identity-preserving counterfactual utility gate；
- selection-regret gate。

独立 operation-RGTA 的 step Schema Recall@10 约为 `0.9122`。加入残差历史后约为 `0.9065`，
selection-regret 版本约为 `0.9101`，均未稳定超过无历史模型。

负结论：partial SQL 历史“理论上相关”不等于“当前 recurrence 有效”。历史状态可能重复已有
operation 信息，也可能把早期错误传播到后续步骤。当前独立 operation-RGTA 应作为动态控制器基线；
在静态 GNN–LLM 对齐尚未解决前，不继续增加 recurrence 复杂度。

### 4.9 Stage 12：Cross-Attention + Hidden-State Steering

本阶段首次真正把图状态连接进冻结 LLM hidden states：

- 完整 schema 仍保留在输入中，避免 Top-K 遗漏造成硬上限；
- operation-RGTA 根据生成中的 SQL prefix 产生 belief；
- cross-attention 让 LLM 读取 schema memory；
- steering residual 修改选定 decoder layers；
- zero/random/normal 干预用于检测神经路径是否真实生效。

Stage 12-A 修复后，20 条中 normal/zero 有 3 条不同，但两者 EX 都为 `0.50`。Stage 12-B 加入
schema/operator/value token weighting 和 counterfactual utility objective 后，在训练 1000 条、评估
同一 100 条时：

| Condition | EX | Execution success |
|---|---:|---:|
| Zero injection | **0.35** | 0.92 |
| Normal injection | 0.20 | 0.92 |

配对分析为 2 gains、17 losses。大量损失来自大小写敏感值被修改，例如 `Kacey → kacey`、
`00D4 → 00d4` 和 `San Joaquin → san joaquin`。

明确负结论：单一 schema residual 不应同时控制 schema identifier、operator、literal value 和
格式 token。更强的 loss 无法补足输入状态本身不包含的值规范信息。Stage 12 停止，不再通过加大
steering scale 或延长训练挽救。

### 4.10 Stage 13A：Typed Relational-Algebra Supervision

Stage 13 将 SQL 决策分解为：

- typed relational actions：`SCAN/JOIN/FILTER/AGGREGATE/HAVING_FILTER/SORT/LIMIT/PROJECT`；
- table/column/FK pointers；
- operators/functions；
- exact value routes；
- relational DAG 与 deterministic compiler 边界。

修正后的完整数据质量为：

| Split | Supported flat | Avg schema coverage | Full schema coverage | Join connected | Direct copy exact |
|---|---:|---:|---:|---:|---:|
| Train | 0.889690 | 0.975518 | 0.947285 | 0.958548 | 0.985626 |
| Dev | 0.917862 | 0.995693 | 0.986310 | 0.984655 | 0.977660 |

有意义的创新：将图网络限制在它擅长的 schema/edge pointer，将大小写敏感值交给 exact-copy 或
value-index route，将 SQL 结构交给有类型 action。它解决的是 Stage 12 暴露的职责混淆。

边界：当前 v1 只可靠支持 flat SQL；nested/set queries 被明确标为 partial，不能静默作为完整标签训练。

### 4.11 Stage 13B：Typed Action Pointer Decoder

冻结 dense embedding 输入，以 query-conditioned RGTA 预测 action、table/column pointers、FK edges、
operator 和 value route。完整训练最佳 epoch 7：

| Metric | Value |
|---|---:|
| Action accuracy | 0.896008 |
| Table Recall@3 | 0.846559 |
| Column Recall@5 | 0.576625 |
| Join-edge Recall@3 | 0.706306 |
| Value-route Recall@2 | 0.823192 |
| Operator Recall@3 | 0.772877 |
| Step complete rate | 0.584815 |
| Mean target recall | 0.745112 |

意义：图状态能够学习结构化 action/pointer 任务，且最佳点出现在 epoch 7，而不是所有模型都只在
epoch 1 有效。它也暴露出 column pointer 和 complete-step 仍是主要瓶颈。

边界：这不是端到端 SQL EX，不能与 Qwen 的执行准确率直接比较。Typed decoder 暂作为结构教师和
冻结 graph encoder 来源，不取代 LLM 代码生成。

### 4.12 Stage 13C：Static Graph Adapter Alignment

当前方案刻意先不使用 partial SQL 动态更新：

```text
complete text-attributed schema graph + question
                    |
        frozen typed RGTA encoder
                    |
      semantic path + topology residual
                    |
      trainable low-rank Graph Adapter
                    |
          frozen Code LLM layers
                    |
                   SQL
```

冻结 RGTA 和 Qwen2.5-Coder-32B，只训练：

- graph-memory projector；
- low-rank decoder cross-attention；
- token-wise residual gate；
- residual/structure scales。

训练目标包括 SQL teacher-forcing、node-name contrastive alignment，以及只对多表 JOIN 生效的
FK-topology counterfactual margin。

早期 Fix1 训练出现明显 train/dev 断裂：epoch 1 train `mean_schema_logprob_gain = 0.01055`，但 dev
为 `-0.10696`，dev improvement rate 仅 `0.0843`。这说明旧 projector 能记忆训练数据库语义，却未
跨数据库泛化。随后增加了语义保留双路径、FP32 adapter、归一化低秩残差、低初始化 scale、JOIN-only
topology counterfactual、SQL-suffix logits 计算和 identity/causal checkpoint gate。

Fix1-v3 修复后完成了 1000 train/100 dev、3 epochs 的实验。Identity dev weighted CE 为
`0.345273`，epoch 3 降至 `0.251722`；但最佳正 dev schema log-probability gain 仅为 epoch 1 的
`0.001442`，improvement rate 始终低于 `0.32`，FK corruption counterfactual loss 约等于
`0.05` margin，且 gate 接近全开。Train alignment Recall@1 达到 `1.0` 时 dev 仅为
`0.163371`。因此它只证明了 generic schema-semantic residual 能降低 teacher-forcing CE，没有证明
RGTA topology 对跨数据库生成有独立贡献。Stage 13C 按预设门槛停止，不继续调 scale/gate。

### 4.13 Stage 14：Typed RGTA Schema Tool Interface

Stage 14 不再修改 LLM hidden states。LLM 或诊断计划只提交 `SCAN/FILTER/PROJECT/...` typed
requests；冻结 Stage 13B RGTA 返回 table/column IDs、FK edges、operators、value routes 和分数。
确定性 assembler 再执行 column owner closure、FK path connectivity、literal surface preservation
和 budget feasibility 检查。

首版已实现，但尚未获得服务器指标。第一个实验使用 oracle action skeleton，同时严格删除 gold
pointers/operators/value routes，用于把 planner error 与 schema-tool error 分离。

## 5. 创新贡献分级

### 5.1 已有实验支持、可进入论文主方法的贡献

1. **Relation-conditioned text-attributed schema graph**：把 schema relevance 分解为十类 SQL 角色，
   联合 dense semantic prior 与 RGTA 结构传播。
2. **Database-disjoint OOF grounding protocol**：第二层 selector 训练只使用未见当前数据库的第一层
   predictions，避免 in-sample stacking 泄漏。
3. **Selection-aware structured coverage**：从独立节点分类转向固定预算下的集合覆盖，并加入
   owner/table 约束；Complete Coverage@30 从 `0.710561` 提升到 `0.808996`。
4. **Typed SQL decision decomposition**：显式分离 relational action、schema pointer、join edge、
   operator 与 exact value route，避免一个连续 residual 控制所有语义类型。

### 5.2 有理论意义，但仍缺端到端证据的贡献

1. **Value likelihood + graph prior 的置信融合**：方向合理，但需在最终 selector/SQL EX 中验证。
2. **Steiner-approximate join-path completion**：适合补全问题未显式提到的桥接表，已有 grounding
   收益，但还缺 SQL join 正确率的独立验证。
3. **Typed action pointer decoder**：中间任务表现明确，但尚未接入可靠 compiler/LLM 形成 EX。
4. **Static semantic-preserving Graph Adapter**：连接设计比 Stage 12 更有边界感，但当前仍在实验中。
5. **Dynamic grounding as latent state**：仍是长期假设；现有 recurrence 没有证明动态历史优于独立
   operation-conditioned state。

### 5.3 有价值但不应包装为算法创新的基础工作

- BIRD 数据可执行性检查与 gold SQL 修正合并；
- label extraction、FK closure 和类型索引修复；
- LLM card 并行、缓存、压缩与复用；
- embedding cache 去重与索引一致性；
- best-checkpoint、OOF、seed、干预和完整覆盖诊断；
- CUDA 显存优化、dtype 对齐、SQL suffix logits 和单元测试。

这些工作支撑可复现性，应写入实现与数据处理部分，不应占据论文核心 novelty。

## 6. 已验证无效或当前不值得继续的路线

| 路线 | 负证据 | 决策 |
|---|---|---|
| Top-K/子图 prompt 替代 full schema | 100 条 EX 多数为 0.15–0.28，full schema 为 0.38 | 只保留 baseline |
| 通过增大 Top-K 提升性能 | Top-80 改善可见性但没有解决选择逻辑 | 不作为创新 |
| Direct value injection | gains 与 losses 同时出现，并能诱导错误 join terminals | 只保留 gated evidence |
| Token-level schema logits bias | 初版无行为效果；修复后改变输出但无 EX 增益甚至下降 | 停止 |
| Operation-gated logits processor | 20 条仅改变 1 条，EX 不变 | 停止 |
| Factor-RGTA 继续堆叠 | 复杂度增加但没有稳定超过 Schema-RGTA | 暂停 |
| 仅靠 optimizer 调参解决 reranker epoch-1 | loss 与 Complete Coverage 目标不一致 | 已转向集合级 loss |
| Recurrent history 默认优于 independent controller | step Recall@10 没有超过 independent operation-RGTA | 暂停 recurrence |
| 全局 cross-attention + hidden steering | 100 条 EX 0.20，zero 为 0.35；2 gains/17 losses | 明确停止 Stage 12 |
| 用更大 steering scale 强迫生效 | 已证明行为变化不等于语义收益 | 禁止作为补救 |

“无效”表示在当前假设、数据与实现下出现负证据，不表示该技术在所有 Text-to-SQL 设置中普遍无效。

## 7. 从失败中得到的算法认识

1. **可见性不等于利用性。** Schema 出现在 prompt 或 Top-K 中，不代表 LLM 会在正确 SQL 位置使用它。
2. **节点相关性不等于集合可行性。** Top-K schema 必须共同满足 owner、join connectivity 和预算约束。
3. **结构信息与值信息不能共用同一控制通道。** 图网络适合表、列和边；精确 literal 应由 copy/value
   route 保真处理。
4. **行为变化不等于因果收益。** normal/zero 产生不同 SQL 只是最低功能检查，必须进一步比较 paired EX、
   correct/corrupted graph 和 schema-token gain。
5. **动态性不是免费提升。** 历史状态会累积错误；只有在静态连接已经被证明有益后，才应加入 partial-SQL
   更新。
6. **训练目标必须对应最终选择操作。** 节点 BCE 下降时 Complete Coverage 仍可能下降；固定 Top-K 的任务
   需要 listwise/coverage/structured objective。
7. **跨数据库泛化是主要边界。** 训练 schema 上的 alignment gain 可以与 dev gain 方向相反，因此必须保留
   database-disjoint validation、OOF 和 causal checkpoint gate。

## 8. 当前最终架构定位

当前不再把完整系统描述成“一个动态 controller 直接 steering LLM”。更准确的分层是：

```text
Layer 1: semantic schema representation
  LLM/heuristic cards + dense embeddings + text-attributed schema graph

Layer 2: structural grounding
  relation-conditioned RGTA + value evidence + join-path completion

Layer 3: budgeted schema selection
  OOF-trained Schema-RGTA reranker + constrained structured coverage

Layer 4: explicit typed graph tool (current experiment)
  typed requests -> frozen RGTA pointers/FK edges -> constrained assembly

Layer 5: optional typed/dynamic control (only after Layer 4 succeeds)
  typed action/pointer interface, then uncertainty-triggered partial-SQL updates
```

Stage 13C 已经表明 loss 下降不足以证明图结构传递。当前 Stage 14 的研究问题是：

> 在不连续扰动 LLM hidden states 的条件下，typed RGTA tool 能否稳定返回完整、连通且保真的结构化
> schema evidence，并由 LLM 计划与代码生成能力消费？

## 9. 下一步决策门

先完成 Stage 14 oracle-action-skeleton 诊断，不接入 LLM planner。只有满足以下条件才进入下一阶段：

1. pointer recall 接近 Stage 13B teacher-forced 上界；
2. constrained assembly 提高 complete schema coverage；
3. owner/FK closure 的预算溢出率可控；
4. literal surface 保持完全一致；
5. predicted-state rollout 没有相对 teacher forcing 明显崩溃。

通过后再训练/调用 LLM planner，并以 plan accuracy、tool-call pointer recall、SQL EX 和 execution
repair success 分层评估。若 rollout 崩溃，应修改 typed transition 或采用独立 per-request pointer，
不能返回 hidden-state steering。

## 10. 论文叙事建议

当前最稳妥的论文主线不是“我们从一开始就成功实现了动态 grounding”，而是：

> Text-to-SQL 中的 schema grounding 需要同时解决跨数据库语义表示、图结构补全和固定预算集合选择；
> 将其简单文本化或作为无类型连续残差注入 LLM 会造成信息损失与语义干扰。我们通过
> relation-conditioned graph grounding、OOF structured selection 和 typed/causal neural interface，逐步建立
> 一个可验证的结构—生成连接框架。

主结果应优先报告 Stage 8/10 的完整 dev grounding 证据。Stage 12 应作为有解释力的负结果或方法动机，
Stage 13C 只有在通过 correct/zero/corrupted graph 与 EX 验证后，才能成为端到端主贡献。
