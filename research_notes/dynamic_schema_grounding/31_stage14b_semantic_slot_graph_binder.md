# 31. Stage 14B: Semantic Slot-Conditioned Graph Binder

记录日期：2026-08-17

## 1. 进入本阶段的直接证据

Stage 14A-fix1 在 BIRD dev 前 100 条上的主要结果为：

| Metric | Stage 14A-fix1 |
|---|---:|
| Table recall | 0.919881 |
| Column recall | 0.384858 |
| Join-edge recall | 0.841463 |
| Semantic-step complete | 0.308943 |
| Assembled schema recall | 0.537803 |
| Assembled complete coverage | 0.0 |

代码审计发现，Stage 14A 请求中的 `role` 和 `value_surface` 没有进入 pointer query。
`TypedRAPointerDecoder` 实际只用 controller state 与 action embedding 构造 schema pointer。因此，同一问题中的多个
`FILTER` 或 `PROJECT` 缺少独立语义身份，只能依赖全局问题和易累积误差的 recurrent state。

## 2. 核心创新

Stage 14B 将一次 SQL 决策表示为 semantic slot：

```text
z_s = Embed(action, role description, semantic focus, question, evidence,
            literal surfaces, expected value type)
```

每个 slot 独立条件化 RGTA：

```text
u_s = LN(u_question + Gate(u_question, q, z_s) * W_slot z_s)
H_s = RGTA(G_schema, condition=u_s)
```

随后通过 typed table/column pointer 打分。Column posterior 额外接收：

- 同一 embedding 空间中的 slot-node cosine evidence；
- owner-table posterior；
- 同表 hard negative 与 owner consistency supervision。

这不是把 GNN 输出转成 prompt，也不是修改 LLM 所有 token hidden states。RGTA 的图传播本身随当前语义槽位变化，
因而属于 slot-synchronous dynamic schema grounding。

## 3. 第一轮因果边界

第一轮默认冻结 Stage 13B 主干，仅训练：

- `slot_input`；
- `slot_gate`；
- `slot_norm`；
- semantic similarity scale；
- owner-table prior scale。

对照组：

- `correct`：正确 semantic slot embedding；
- `action_only`：slot embedding 置零；
- `shuffled`：同一问题内部循环置换 slot embedding。

这能区分“槽位接口真正有效”与“重新训练整个 RGTA 后整体漂移”。

当前数据构造仍属于 oracle-plan diagnostic：没有 schema identity 或 gold SQL 泄露进 `inference_inputs`，但 action、arity、
plan-derived literal semantics 来自 teacher plan。通过本阶段后，必须用仅消费 question/evidence 的 LLM planner 替换它。

## 4. 实现文件

- `src/data/stage14b_build_semantic_slots.py`
- `src/embedding/stage14b_build_slot_embedding_cache.py`
- `src/modeling/semantic_slot_binder.py`
- `src/training/stage14b_train_semantic_slot_binder.py`
- `src/grounding/stage14b_semantic_slot_tool.py`
- `src/evaluation/stage14b_compare_slot_ablation.py`
- `tests/test_stage14b_semantic_slot_binder.py`

## 5. 服务器实验：先跑 1000/100

### 5.1 构造 semantic slots

```bash
python src/data/stage14b_build_semantic_slots.py \
  --train-trajectories experiments/stage13b_clean_typed_trajectories/train_trajectories.jsonl \
  --dev-trajectories experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl \
  --output-dir experiments/stage14b_semantic_slots_1000_100 \
  --train-limit 1000 \
  --dev-limit 100
```

### 5.2 生成 slot embedding

```bash
CUDA_VISIBLE_DEVICES=0 python src/embedding/stage14b_build_slot_embedding_cache.py \
  --train-slots experiments/stage14b_semantic_slots_1000_100/train_semantic_slots.jsonl \
  --dev-slots experiments/stage14b_semantic_slots_1000_100/dev_semantic_slots.jsonl \
  --splits train,dev \
  --model-path /data/1_pretrained_models/Qwen3-Embedding-0.6B \
  --output-dir experiments/stage14b_slot_embeddings_qwen3_06b_1000_100 \
  --train-limit 1000 \
  --dev-limit 100 \
  --batch-size 64 \
  --max-length 512 \
  --pooling last \
  --normalize \
  --device cuda \
  --dtype bfloat16 \
  --trust-remote-code \
  --write-texts
```

### 5.3 Warm-start 并训练 semantic slot interface

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage14b_train_semantic_slot_binder.py \
  --train-graph-file experiments/stage13b_clean_typed_trajectories/train_trajectories.jsonl \
  --dev-graph-file experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl \
  --train-slot-file experiments/stage14b_semantic_slots_1000_100/train_semantic_slots.jsonl \
  --dev-slot-file experiments/stage14b_semantic_slots_1000_100/dev_semantic_slots.jsonl \
  --embedding-cache-dir experiments/stage13b_embedding_cache_typefix1_qwen3_06b \
  --slot-embedding-cache-dir experiments/stage14b_slot_embeddings_qwen3_06b_1000_100 \
  --init-checkpoint experiments/stage13b_typed_ra_decoder_rgta_seed42/typed_ra_decoder.pt \
  --init-summary experiments/stage13b_typed_ra_decoder_rgta_seed42/training_summary.json \
  --output-dir experiments/stage14b_semantic_slot_binder_1000_100_seed42 \
  --train-limit 1000 \
  --dev-limit 100 \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 5 \
  --lr 1e-4 \
  --contrastive-weight 0.3 \
  --contrastive-temperature 0.1 \
  --semantic-dropout 0.2 \
  --freeze-pretrained-backbone \
  --device cuda \
  --seed 42
```

### 5.4 运行三组因果对照

```bash
for MODE in correct action_only shuffled same_action_shuffled; do
  CUDA_VISIBLE_DEVICES=0 python src/grounding/stage14b_semantic_slot_tool.py \
    --checkpoint experiments/stage14b_semantic_slot_binder_1000_100_seed42/semantic_slot_binder.pt \
    --graph-file experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl \
    --slot-file experiments/stage14b_semantic_slots_1000_100/dev_semantic_slots.jsonl \
    --embedding-cache-dir experiments/stage13b_embedding_cache_typefix1_qwen3_06b \
    --slot-embedding-cache-dir experiments/stage14b_slot_embeddings_qwen3_06b_1000_100 \
    --split dev \
    --slot-mode "$MODE" \
    --output-file "experiments/stage14b_semantic_slot_tool_100/${MODE}.jsonl" \
    --limit 100 \
    --table-top-k 3 \
    --column-top-k 5 \
    --join-top-k 3 \
    --operator-top-k 3 \
    --value-route-top-k 2 \
    --max-schema-items 30 \
    --device cuda
done
```

### 5.5 汇总因果差异

```bash
python src/evaluation/stage14b_compare_slot_ablation.py \
  --correct-output experiments/stage14b_semantic_slot_tool_100/correct.jsonl \
  --action-only-output experiments/stage14b_semantic_slot_tool_100/action_only.jsonl \
  --shuffled-output experiments/stage14b_semantic_slot_tool_100/shuffled.jsonl \
  --same-action-shuffled-output experiments/stage14b_semantic_slot_tool_100/same_action_shuffled.jsonl \
  --target-trajectories experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl \
  --output-file experiments/stage14b_semantic_slot_tool_100/ablation_summary.json
```

## 6. 预注册决策门

相对 Stage 14A-fix1，目标为：

1. Column recall 从 `0.384858` 提高到至少 `0.53`；
2. FILTER 和 PROJECT column recall 各提高至少 0.15；
3. Semantic-step complete 达到至少 `0.42`；
4. Table/FK recall 下降不超过 0.02；
5. `correct` 同时优于 `action_only` 和 `shuffled` 的 column recall；
6. correct-vs-action-only candidate ranking change rate 至少 10%。

如果第 1--4 条提高但第 5--6 条失败，不能声称 semantic slot interface 有效，只能说明重新训练或损失函数改变了排序。
若本阶段通过，再进入 Stage 14C joint slot-schema assignment；若失败，转向 LLM SQL candidates + RGTA graph verifier。

## 7. 首轮结果与 Fix1 决策

100-example 首轮结果：

| Variant | Column recall | Semantic complete | Assembly recall | Complete coverage |
|---|---:|---:|---:|---:|
| correct | 0.422713 | 0.308943 | 0.572040 | 0.01 |
| action-only zero | 0.129338 | 0.086721 | 0.460770 | 0.00 |
| within-question shuffled | 0.405363 | 0.289973 | 0.567760 | 0.01 |

`correct` 相对 action-only 提升很大，但相对 shuffled 的 column recall 只高 `0.01735`。这说明模型依赖非零
slot input，却没有充分学习正确 slot 的语义身份。首轮没有通过性能门，不能进入 Stage 14C。

Stage 14B-fix1 因而作出以下修改：

1. global question/evidence 只进入 query encoder，不再重复写入每个 slot；
2. focus 与 literal value 分别 embedding；action 与 expected type 使用独立 learned embedding；
3. 增加 Slot–Schema InfoNCE；
4. 训练时使用 semantic dropout，使 action-only 成为分布内控制；
5. 新增跨样本、同 action 的 `same_action_shuffled`，并要求 correct column recall 至少高 0.05、semantic
   complete 至少高 0.03。

Fix1 必须重新执行 5.1--5.5；旧 slot embedding cache 与旧 Stage 14B checkpoint 不兼容，不能复用。

## 8. Fix1 最终结论：绝对性能有效、语义身份因果验证失败

Fix1 在 BIRD dev 前 100 条上的结果为：

| Variant | Column recall | Semantic complete | Assembly recall | Complete coverage |
|---|---:|---:|---:|---:|
| correct | 0.468454 | 0.363144 | 0.580599 | 0.01 |
| action-only | 0.462145 | 0.349593 | 0.564907 | 0.00 |
| shuffled | 0.454259 | 0.357724 | 0.573467 | 0.02 |
| same-action shuffled | 0.465300 | 0.357724 | 0.574893 | 0.01 |

相对 Stage 14A-fix1，`correct` 的 column recall、semantic-step complete 和 assembled schema
recall 分别提高 `8.36pp`、`5.42pp` 和 `4.28pp`。其中 FILTER、PROJECT、SORT column recall
分别提高 `8.15pp`、`9.20pp` 和 `14.00pp`，说明局部 focus/value 表示、type/action 融合及
对比损失对绝对性能有帮助。

但是，`correct` 相对更严格的 `same-action shuffled` 仅获得：

- column recall：`+0.32pp`；
- semantic complete：`+0.54pp`；
- assembly recall：`+0.57pp`。

虽然 candidate ranking change rate 为 `14.29%`，变化主要发生在候选内部排序，没有转化为目标列
进入候选集。与此同时，table recall 与 join-edge recall 相对 Stage 14A 分别下降 `3.86pp` 和
`3.66pp`。因此预注册的绝对性能门和语义身份因果门均未通过。

本阶段的正式结论是：

> **绝对性能有效、语义身份因果验证失败。** 模型主要利用 action、schema type、全局 question
> embedding 和图结构先验，没有可靠学习 semantic slot 的具体身份。

因此不进入 Stage 14C joint slot-schema assignment，也不继续调 semantic residual scale。后续转向
Stage 15：让 LLM 负责提出完整 SQL 假设，让 RGTA 计算候选 SQL 与数据库图之间的后验结构一致性能量。
