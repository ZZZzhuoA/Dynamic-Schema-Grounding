# Dynamic Schema Grounding 结果台账

本文件只回答“得到了什么证据、能否比较、结论边界是什么”。运行命令见
[`REPRODUCTION.md`](REPRODUCTION.md)，研究脉络见 [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md)。

## 1. 证据等级

- **A**：仓库内阶段笔记或可定位 artifact/summary，可直接核验。
- **B**：对话中报告了完整指标，但 artifact 尚未归档到当前工作区。
- **C**：小样本、prompt 开发或存在协议缺陷，只能用于工程诊断。
- **D**：明确负结果或已停止路线。

证据等级不是模型质量排序。不同 data version、sample count、seed、LLM、prompt 或 candidate set 的
结果不得在无说明情况下横向比较。

指标：`Recall@K` 是样本平均 schema-item recall；`Complete@K` 是全部 Gold schema items 均进入
Top-K 的样本比例；`EX` 是 execution accuracy；`ExecOK` 仅表示预测 SQL 可执行。

## 2. 当前主结果：Stage 17

### Stage 17-A0（B 级，会话报告，seed42）

数据：corrected merge v1、scopefix1 labels、full-schema graph、Qwen3-Embedding-0.6B cache；dev 1534。

| Setting | Schema R@30 | Table R@30 | Column R@30 | Complete@30 | Complete@50 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| Embedding cosine | 0.793449 | 0.783518 | 0.798262 | 0.481095 | — | 0.877257 |
| Full-Schema QRGTA | 0.950572 | 0.989244 | 0.935191 | 0.766623 | 0.915254 | 0.939676 |

QRGTA best epoch 为 1；train/dev records 为 9428/1534，空标签后可训练 train 样本 9389。该结果证明
监督式 Full-Schema QRGTA 显著优于未训练 cosine，但**尚不能证明提升来自图传播**：缺少
depth-matched MLP 与同 checkpoint causal controls。

### Stage 17-A1（待运行）

| Setting | Seeds | 状态 |
|---|---:|---|
| normal QRGTA | 42/43/44 | 待完成/核验 |
| `mlp_residual` | 42/43/44 | 待完成 |
| frozen checkpoint interventions | 42/43/44 | 待完成 |
| retrained controls | 42 | 待完成 |

在 `stage17a1_causal_summary.json` 产生前，不得声称 QRGTA 的 topology 或 Query edges 具有因果贡献。

## 3. Grounding 与集合选择结果

### Stage 8 full-dev RGCN/RGTA（A 级）

来源：[`12_best_full_dev_grounder_record.md`](research_notes/dynamic_schema_grounding/12_best_full_dev_grounder_record.md)。

| Setting | N | Schema R@30 | Precision@30 | Complete@30 |
|---|---:|---:|---:|---:|
| dense relation RGCN | 1534 | 0.924075 | 0.232493 | 0.683181（1048） |
| dense relation RGTA | 1534 | 0.933307 | 0.235383 | 0.710561（1090） |

同一实验家族中 RGTA 优于 RGCN，但这些模型包含 relation-conditioned features，不能直接解释为
纯 schema topology 的收益，也不能与 Stage 17-A0 无条件比较。

### Stage 10-D constrained structured coverage（A 级）

来源：项目阶段记录与 `29_research_path_retrospective.md`。

| N | Schema R@30 | Table R@30 | Column R@30 | Complete@30 | Candidate oracle complete |
|---:|---:|---:|---:|---:|---:|
| 1534 | 0.956051 | 0.986484 | 0.946117 | 0.808996（1241） | 0.905476 |

结论：selection-aware structured objective 可以改善固定预算集合完整性；但最终结果仍受进入 Stage 10
candidate graph 的上限约束，且包含 operation/value/closure 特征，不是 Stage 17 的架构对照。

### Stage 10-G semantic completion + Steiner closure（B 级）

数据：scopefix1 dev，1534。

| Metric | Before | After |
|---|---:|---:|
| semantic complete | 0.949153 | 不因 closure 改变 |
| join connected | 0.958931 | 0.970013 |
| grounding complete | 0.925033 | 0.934159 |

Steiner closure 恢复 14 个样本、0 regression。结论仅限于 schema set/graph connectivity，不代表 SQL
EX；semantic Top-K core 不因追加 closure nodes 获得 recall credit。

## 4. SQL 生成、连续注入与负结果

### Prompt baselines（C 级，online Qwen3-32B，前 100 条）

| Setting | EX | ExecOK |
|---|---:|---:|
| full schema | 0.38 | 0.96 |
| relation-fusion RGTA Top30 | 0.23 | 0.87 |
| RGTA Top50 | 0.20 | 0.86 |
| RGTA Top80 | 0.28 | 0.93 |
| lexical Top30 | 0.18 | 0.70 |
| subgraph budget50 / budget80 | 0.20 / 0.24 | 0.86 / 0.94 |
| role-aware relation fusion Top30 | 0.15 | 0.92 |

这些结果的 prompt、模型服务和样本顺序不构成稳定论文协议，只说明简单 Top-K prompt 在该开发设置
下没有体现图网络优势，不能与 Stage 17 grounding 指标直接比较。

### Stage 12 continuous hidden-state injection（D 级）

来源：[`29_research_path_retrospective.md`](research_notes/dynamic_schema_grounding/29_research_path_retrospective.md)。

| Setting | N | EX | ExecOK |
|---|---:|---:|---:|
| normal graph injection | 100 | 0.20 | 0.92 |
| zero control | 100 | 0.35 | 0.92 |

normal 相对 zero 仅 2 gains、17 losses。结论：当前全局 cross-attention/steering 接口破坏了 LLM
已有生成能力；不再通过增大 scale 或延长训练继续该路线。这不等于所有 GNN–LLM interface 均被
普遍证伪。

### Stage 13C static graph adapter（D 级）

Frozen GNN→Adapter→Frozen LLM 路线经历 cache、dtype、role alignment、OOM 修复后仍未通过图因果门，
停止继续调 residual scale。后续结构信息优先通过 typed pointer/tool、constraints 或 verifier 使用。

## 5. Typed plan/tool 与 verifier

### Stage 13-A typed relational algebra data（A 级）

| Split | Supported flat | Avg schema coverage | Full schema coverage | Join connected | Direct-copy exact |
|---|---:|---:|---:|---:|---:|
| train | 0.889690 | 0.975518 | 0.947285 | 0.958548 | 0.985626 |
| dev | 0.917862 | 0.995693 | 0.986310 | 0.984655 | 0.977660 |

这证明 typed RA 数据转换覆盖大部分 BIRD flat SQL，但它是数据质量/可表示性指标，不是模型 EX。

### Stage 13-B typed action decoder（A 级）

| Action acc. | Table R@3 | Column R@5 | Join R@3 | Value R@2 | Operator R@3 | Step complete |
|---:|---:|---:|---:|---:|---:|---:|
| 0.896008 | 0.846559 | 0.576625 | 0.706306 | 0.823192 | 0.772877 | 0.584815 |

结构动作可学，但 column pointer 和整步 completeness 仍是瓶颈。

### Stage 14-A typed schema tool（B/C 级，100 条）

| Table recall | Column recall | Join recall | Operator recall | Value-route recall | Step complete | Assembled schema recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.919881 | 0.384858 | 0.841463 | 0.704878 | 0.706186 | 0.591398 | 0.537803 |

表和 join 较强，column/PROJECT/FILTER 明显不足。Stage 14B-fix1 被记录为“绝对性能有效、语义身份
因果验证失败”，因此不能据此声称 tool 真正利用了节点语义身份。

### Stage 15-A SQL hypothesis verifier（A 级，1393 clean dev）

| MRR | Hits@1 | Pairwise acc. | Schema-control gain | Schema-control win rate |
|---:|---:|---:|---:|---:|
| 0.629289 | 0.363963 | 0.834530 | 14.087188 | 0.867193 |

细分：same-table column 0.781919、join edge 0.780761、operator 0.933812、value route 0.932520。
它证明 verifier 能识别人工 corruption；Stage 15A-fix1 MRR 约 0.811019 仅有会话附件，记 B 级，待
完整 artifact 归档后再进入主表。

### Stage 15-B real SQL candidates（A/C 级，1393）

第一版：5686 candidates，candidate execution rate 0.9509，oracle EX@K 0.6432。

| Selection | EX | Recovered | Regressed | Net |
|---|---:|---:|---:|---:|
| LLM top1 | 0.5327 | 0 | 0 | 0 |
| execution filter | 0.5406 | 11 | 0 | +11 |
| verifier + execution filter | 0.5477 | 75 | 54 | +21 |
| hybrid alpha=0.75 | 0.5528 | 82 | 54 | +28 |
| oracle | 0.6432 | — | — | — |

Hybrid alpha 在同一 dev 上扫描，且旧候选协议存在 random-first 问题，因此 0.5528 是描述性开发结果，
不是隔离测试主结果。greedy clean1393 后续候选为 5449、oracle 0.638191，诊断为 recovered 67、
regressed 56、unresolved 67；尚无独立 held-out 选择指标。

## 6. 可比性矩阵

| 结果族 | 可与 Stage 17 直接比较？ | 原因 |
|---|---|---|
| Stage 17 MLP/QRGTA/controls 同数据三 seed | 是 | 相同 graph、label、cache、目标和指标 |
| Stage 8 RGCN/RGTA | 否，仅背景 | relation-conditioned features 与候选/组装流程不同 |
| Stage 10-D/G | 否，仅上限/补全参考 | candidate graph、额外 prior、closure 与 loss 不同 |
| Prompt/Stage 12 EX | 否 | 指标、LLM、样本规模和接口不同 |
| Stage 15 verifier EX | 否 | 候选生成上限和 reranking 任务不同 |

## 7. 下一次结果归档模板

每个新结果至少记录：

```text
stage / setting:
evidence grade:
git commit:
data merge SHA256:
graph / label / card / cache paths:
model / checkpoint SHA:
sample count / seed:
primary and secondary metrics:
artifact path:
directly comparable baselines:
known protocol limitations:
conclusion boundary:
```

Stage 17-A1 完成后，应先将三 seed mean/std、每个 frozen intervention drop 和 seed42 retraining
写入本台账，再决定是否进入 role/value/join prediction 或 SQL SFT。
