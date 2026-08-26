# Dynamic Schema Grounding 项目交接

更新日期：2026-08-26

代码基线：`9117c17`

## 1. 项目现在研究什么

本项目研究 BIRD Text-to-SQL 中的 schema grounding：给定自然语言问题和一个未在训练中出现过的
数据库，找出生成 SQL 所需的表、列和结构关系。

最初假设是：

> Schema grounding 应随 partial SQL 动态更新，并通过 cross-attention/hidden-state steering
> 进入 LLM。

Stage 0–16 的实验说明，这个目标拆得不够干净：图网络确实能改善 schema grounding 和结构选择，
但直接修改 LLM hidden states 会破坏它原有的 SQL、operator 和大小写敏感 literal 能力。当前主线
因此退回一个更可证伪的问题：

> 在不裁剪候选的完整 Schema 图上，Query-conditioned relational graph propagation 是否比
> 同深度的独立节点打分更能找齐 Gold SQL 使用的全部表列？

这就是 Stage 17 Full-Schema Binary QRGTA。动态更新、role prediction、Value Index 和 SQL 生成均
暂时不进入当前模型，先证明静态图结构贡献。

## 2. 当前模型：Full-Schema Binary QRGTA

### 2.1 输入

每个问题对应一个可变大小的完整 schema 图。BIRD 中节点数 `N` 不固定，大库约可达到 520 个节点。

节点：

- table node；
- column node。

Schema 边：

- `table_to_column`；
- `column_to_table`；
- `foreign_key_forward`；
- `foreign_key_backward`；
- `self_loop`。

运行时加入一个固定 Query source node，并向每个 schema node 添加：

- `query_to_table`；
- `query_to_column`。

没有 Schema→Query 边，Query 不参与最终节点排名。

Dense 输入由 `/data/1_pretrained_models/Qwen3-Embedding-0.6B` 预计算：

```text
query embedding:  [1, 1024]
schema embedding: [N, 1024]
```

Schema embedding 文本已包含 schema card；query embedding 文本包含 question/evidence/question
card。当前 Stage 17 不再加入旧 operation score、Frozen LLM role prior、Value Index 或手工 numeric
features。

### 2.2 网络

```text
Query embedding [1,1024] ── projection ──> q [1,256]

Schema embeddings [N,1024]
        + table/column type embedding
        └───────────────────────────────> H0 [N,256]

H0 + fixed q + typed sparse edges
        └── 3 × Sparse QRGTA layer
              8 heads
              relation-specific key/value/bias
              query-edge cosine attention bias
              residual + LayerNorm + FFN
                                            ↓
                                       H3 [N,256]

ui = [hi ; q ; hi⊙q ; |hi-q|] [N,1024]
                         ↓ MLP scorer
                    logits/probabilities [N]
```

输出：

```python
{
    "logits": Tensor[N],
    "probabilities": Tensor[N],
    "schema_states": Tensor[N, 256]
}
```

预测文件保存完整 `N` 节点排名以及 Top-10/20/30/50 IDs，不写入 Gold labels。

### 2.3 目标与指标

训练目标仅为节点是否被 Gold SQL 实际引用的 class-balanced BCE：

```text
0.5 * mean positive softplus(-logit)
+ 0.5 * mean negative softplus(logit)
```

空 Gold 样本跳过并计数。best checkpoint 按 `complete_coverage@30` 选择，并列时依次比较
`schema_recall@30` 和较低 dev loss。

核心指标：

- `schema/table/column recall@10/20/30/50`；
- `schema precision@10/20/30`；
- MRR；
- `complete_coverage@10/20/30/50`。

其中 Complete Coverage@30 要求该样本的全部 Gold 表列都出现在 Top-30；它只说明正确 SQL 仍然
可由当前 schema 集合构造，不保证 operator、literal、join direction 或最终 EX 正确。

## 3. Stage 0–17 研究脉络

| 研究块 | Stage | 做了什么 | 当前判断 |
|---|---|---|---|
| 数据与监督 | 0–1 | BIRD 审计、corrected SQL merge、scope-aware labels、FK policy/type 修复 | 必要基础；版本必须锁定 |
| 静态图 grounding | 2、5、8 | lexical/RGCN/RGTA、LLM cards、dense embedding、十类 relation beliefs | RGTA 和关系分解有效 |
| Prompt/解码偏置 | 3–7 | Top-K prompt、子图 prompt、role/value prompt、token logits bias | 多数弱于 full schema；停止主线 |
| 集合选择 | 9–10G | Value Index、Steiner join completion、OOF reranker、coverage/structured loss | 集合选择和 join closure 有效，但受候选上限限制 |
| 动态 grounding | 11 | partial-SQL recurrent controller、uncertainty/history/utility gates | 没有稳定超过 independent operation-RGTA |
| GNN→LLM 连续注入 | 12、13C | cross-attention、hidden steering、static graph adapter | 路径会改变输出，但端到端 EX/图因果门失败；停止 |
| Typed graph interface | 13A–14B | typed RA、action/pointer decoder、schema tool、semantic slot binder | 分工合理；column pointer 仍弱，未形成最终 EX |
| SQL hypothesis verifier | 15 | LLM 生成候选，图网络做 typed-plan/schema 后验验证 | 能恢复部分错误，也会回退；值得保留为后验模块 |
| OOF SQL SFT | 16 | 构造无 in-sample grounding 泄漏的 Graph-grounded direct-SQL SFT | 数据接口已实现，尚非当前优先级 |
| Full-Schema 主线 | 17 | 所有表列进入 QRGTA，取消候选入口瓶颈 | A0 有强会话结果；A1 因果验证待完成 |

详细的每阶段命令和实验解释保存在
[research_notes/dynamic_schema_grounding/README.md](research_notes/dynamic_schema_grounding/README.md)。

## 4. 已支持、未支持与负结果

### 已有证据支持

- text-attributed relational schema graph 可以提高跨数据库 schema grounding；
- 相同 corrected data/dense features 下，RGTA 优于 RGCN；
- question–schema 关系不能只用统一 relevance，OUTPUT/PREDICATE/JOIN/FORMULA 等角色具有不同难度；
- 固定 Top-K 应优化集合完整性，而不是只优化独立节点 BCE；
- owner closure 和 FK/Steiner join completion 能补充自然语言未显式表达的结构路径；
- SQL candidate verifier 能利用 typed plan 与 schema graph 后验恢复一部分 LLM 错误；
- Full-Schema QRGTA A0 明显高于 cosine retrieval，但尚未排除“监督式 MLP 即可获得同样收益”。

### 有意义但没有端到端主结果

- Value Index 是 predicate column 的互补证据，但直接注入会引入错误；
- Typed RA/action pointer 将 schema、operator、value、join 分工清楚，但还不是最终 SQL EX；
- OOF Graph-grounded SQL SFT 数据已经构造接口，但尚未形成隔离测试结果；
- 动态 grounding 仍是长期假设，现有 recurrence 不构成正证据。

### 已停止或只保留 baseline

- 用 Top-K/schema 子图 prompt 替代 full schema；
- 单纯增大 Top-K；
- schema token logits bias 与 operation gate；
- 全局 cross-attention + hidden-state steering；
- 继续放大 steering scale 或延长该路线训练；
- Stage 13C Frozen-GNN→Adapter→Frozen-LLM；
- 在 Factor-RGTA 上继续堆叠复杂模块。

“停止”只表示当前数据、接口和目标下已有负证据，不表示技术在所有 Text-to-SQL 场景中普遍无效。

## 5. 当前 Stage 17-A1 决策门

正式实验包括：

1. normal QRGTA seeds 42/43/44；
2. depth-matched `mlp_residual` seeds 42/43/44；
3. 对每个 normal checkpoint 的三个冻结推理干预；
4. 三个 control 的 seed42 从头重训；
5. 统一 summary。

结构贡献至少满足：

- QRGTA 三 seed 平均 `complete_coverage@30` 高于 `mlp_residual`；
- `shuffled_schema_edges` 在同 checkpoint 上稳定下降；
- `zero_query_edges` 在同 checkpoint 上稳定下降；
- `shuffled_node_identity` 产生最大或接近最大的下降。

解释：

- QRGTA≈MLP：提升主要来自监督式 dense reranking；
- shuffled edges 不下降：模型没有有效利用 schema topology；
- zero query edges 不下降：Query graph messages 冗余，Query 主要从最终 scorer 起作用；
- 推理干预下降但重训恢复：正常模型使用该信息，但它可被其他通路补偿；
- 推理干预与重训都下降：该信息具有更强的不可替代性。

只有通过该门，才进入 role/value/join-edge prediction 或 Stage 17-B SQL SFT。否则先诊断 Full-Schema
QRGTA 为什么没有利用图，而不是重新加入动态 controller。

## 6. 关键代码与产物流

```text
Data/BIRD + processed_final_data
  -> src/data/stage0_merge_train_corrections.py
  -> src/data/stage1_extract_bird_labels.py
  -> src/data/stage8f_llm_card_generation.py
  -> src/data/stage8f_compact_llm_cards.py
  -> src/data/stage5_build_dsg_data.py
  -> src/embedding/stage8g_build_embedding_cache.py
  -> src/modeling/full_schema_qrgta.py
  -> src/training/stage17a_train_full_schema_qrgta.py
  -> src/evaluation/stage17a_run_checkpoint_controls.py
  -> src/evaluation/stage17a_summarize_causal_controls.py
```

详细输入、输出和命令见 [REPRODUCTION.md](REPRODUCTION.md)。结果与证据等级见
[RESULTS_LEDGER.md](RESULTS_LEDGER.md)。

## 7. 新 session 交接检查

开始工作前应能回答：

- 当前研究问题是不是 Full-Schema 图传播的独立贡献？
- 当前数据是不是 corrected merge v1（2360 corrections）？
- labels、cards、graph、cache 是否来自相同 question/evidence 版本？
- 当前结果是 A/B/C/D 哪个证据等级？
- 本次任务是训练、评估、诊断还是只解释结果？
- 是否会误把 complete coverage 当作 SQL EX？
- 是否会误把 frozen intervention 与 control retraining 当成同一实验？

如果任何答案不明确，先检查 [REPRODUCTION.md](REPRODUCTION.md) 和输出目录中的
`training_summary.json`/`model_config.json`，不要直接启动新实验。
