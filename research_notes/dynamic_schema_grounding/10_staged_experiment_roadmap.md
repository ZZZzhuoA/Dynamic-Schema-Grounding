# 10. Staged Experiment Roadmap

本文件用于指导后续实验推进。我们不一次性实现完整模型，而是按阶段推进：

```text
Stage 0: 环境与数据可用性检查
Stage 1: Gold SQL → schema grounding label 抽取
Stage 2: Static grounding baseline
Stage 3: Static grounding + prompt-based SQL generation
Stage 4: Clause-level dynamic grounding
Stage 5: Dynamic grounding + cross-attention
Stage 6: Gated hidden-state steering
Stage 7: Grounding intervention + long-schema robustness
```

每个阶段都包含：

- 需要准备什么；
- 具体流程；
- 产出文件；
- 跑通标准；
- 效果可以的判断；
- 进入下一阶段条件。

原则：

> 只有当前阶段跑通且指标合理，才进入下一阶段。否则先修当前阶段，不急着堆模型。

---

# Stage 0：环境与数据可用性检查

## 目标

确认当前项目可以稳定读取数据、解析 schema、访问模型和运行基础评估。

这个阶段不训练模型，只做工程地基。

## 需要准备

### 数据

优先使用 Spider。如果当前机器已有 BIRD，也可以保留，但第一阶段建议先从 Spider 开始。

需要确认：

```text
dataset/
  spider/
    train_spider.json
    dev.json
    tables.json
    database/
```

如果目前只有 BIRD 目录，则先确认 BIRD 结构：

```text
BIRD/
  train/
  dev/
  database/
```

### 工具

需要：

- Python 环境；
- SQL parser；
- SQLite 执行环境；
- 一个基础 LLM 或 encoder；
- 后续如果微调 LLM，需要 transformers / peft / accelerate。

### 建议新增目录

```text
src/
  data/
  schema/
  grounding/
  model/
  training/
  evaluation/

experiments/
  stage0_data_check/
```

## 流程

1. 读取数据集样例。
2. 打印 question、gold SQL、database id。
3. 根据 database id 读取 schema。
4. 将 schema 标准化为统一格式：

   ```json
   {
     "db_id": "...",
     "tables": [...],
     "columns": [...],
     "foreign_keys": [...],
     "primary_keys": [...]
   }
   ```

5. 尝试执行 gold SQL，确认数据库连接和执行器可用。

## 产出

建议产出：

```text
experiments/stage0_data_check/
  sample_records.jsonl
  schema_preview.json
  execution_check.json
```

## 跑通标准

满足：

- 能读取至少 10 条样例；
- 能读取对应 schema；
- gold SQL 至少大部分可执行；
- schema 中 table / column / foreign key 能标准化输出。

## 效果可以的判断

这个阶段没有模型指标，只看数据链路。

进入下一阶段条件：

```text
数据读取、schema 读取、gold SQL 执行三件事都跑通。
```

---

# Stage 1：Gold SQL 到 Schema Grounding Label 抽取

## 目标

从 gold SQL、`hit_info` 和 schema foreign keys 联合抽取 schema grounding labels。

这是后面 static grounding 和 dynamic grounding 的监督来源。

## 需要准备

### 输入

- question；
- gold SQL；
- schema；
- database id。

### Label sources

Stage 1A 不单独依赖 `hit_info`，因为 `hit_info` 通常不包含 join 所需的外键信息。

联合使用三类来源：

- `hit_info`：自然语言显式命中的 semantic schema labels；
- gold SQL parse：SQL 实际使用的 tables / columns；
- schema foreign keys：已使用表之间的 join keys。

### SQL parser

需要能从 SQL 中抽出：

- used tables；
- used columns；
- join columns；
- select columns；
- where columns；
- group by columns；
- order by columns；
- aggregation target columns。

第一版可以用规则 + SQL parser 混合，不追求完美，但要可统计。

## 流程

对每条样例：

1. 读取 `hit_info` 并抽取 question-mentioned schema labels；
2. parse gold SQL 并抽取 SQL-used schema labels；
3. 根据 schema foreign keys 补充已使用表之间的 join keys；
4. 合并三类 label；
5. 将 label 对齐到 schema element id；
6. 保存为 jsonl。

## 输出格式

建议每条样例如下：

```json
{
  "db_id": "shop",
  "question": "Find the customer who spent the most money.",
  "sql": "SELECT ...",
  "schema_items": [
    {"id": 0, "type": "table", "name": "customers"},
    {"id": 1, "type": "column", "name": "customers.id"},
    {"id": 2, "type": "column", "name": "customers.name"}
  ],
  "whole_sql_labels": [0, 1, 2, 5, 8],
  "label_sources": {
    "hit_info": ["customers.name"],
    "sql_parse": ["customers", "customers.name", "orders"],
    "foreign_key": ["customers.id", "orders.customer_id"]
  },
  "clause_labels": {
    "select": [2],
    "from": [0, 4],
    "join": [1, 5],
    "where": [],
    "group_by": [1],
    "order_by": [8]
  }
}
```

## 产出

```text
experiments/stage1_label_extraction/
  train_grounding_labels.jsonl
  dev_grounding_labels.jsonl
  label_statistics.json
  failed_parse_cases.jsonl
```

## 跑通标准

满足：

- 至少 90% 样例能完成 SQL parse 或规则抽取；
- 每条样例能得到 whole-SQL labels；
- clause-level labels 覆盖 SELECT / FROM / WHERE / GROUP BY / ORDER BY 中出现的主要表列；
- failed cases 被保存，方便后续修。

## 效果可以的判断

重点看统计：

| Check | Reasonable target |
|---|---:|
| SQL parse success rate | >= 90% |
| samples with non-empty whole labels | >= 95% |
| samples with table labels | >= 95% |
| samples with column labels | >= 85% |
| clause label extraction success | >= 80% |

如果低于这个水平，先不要训练模型，先修 label extractor。

## 进入下一阶段条件

```text
whole-SQL labels 基本可靠，clause-level labels 可用于弱监督。
```

---

# Stage 2：Static Grounding Baseline

## 目标

训练一个静态 schema grounder：

```text
g = f(Q, S)
```

它根据 question 和 schema 给每个 table / column 一个 relevance score。

这是后续 dynamic grounding 的基础。

## 需要准备

### 输入

来自 Stage 1：

```text
train_grounding_labels.jsonl
dev_grounding_labels.jsonl
```

### 模型

第一版尽量简单：

```text
question encoder
schema item encoder
MLP scorer
```

schema item 文本格式：

```text
table.column : datatype
```

例如：

```text
orders.amount : number
customers.name : text
```

## 流程

1. 编码 question 得到 `q`。
2. 编码每个 schema item 得到 `z_i`。
3. 计算 relevance：

   ```text
   p_i = sigmoid(MLP([q; z_i; q*z_i; |q-z_i|]))
   ```

4. 使用 whole-SQL labels 训练多标签分类。

## Loss

```text
L_g = BCE(p_i, y_i)
```

建议处理正负样本不均衡：

- positive class weight；
- top-k negative sampling；
- table 和 column 分开算指标。

## 产出

```text
experiments/stage2_static_grounding/
  checkpoints/
  dev_predictions.jsonl
  metrics.json
  topk_examples.md
```

## 评估指标

| Metric | 含义 |
|---|---|
| table recall@k | gold tables 是否在 top-k tables 中 |
| column recall@k | gold columns 是否在 top-k columns 中 |
| schema recall@k | gold schema elements 是否被 top-k 覆盖 |
| precision@k | top-k 中有多少是 gold schema |
| MRR | gold schema 排名是否靠前 |

## 跑通标准

满足：

- 模型能训练，loss 下降；
- dev predictions 能输出 top-k schema；
- 指标脚本稳定可复现。

## 效果可以的判断

粗略目标：

| Metric | Minimum target |
|---|---:|
| table recall@5 | >= 85% |
| column recall@10 | >= 65% |
| schema recall@20 | >= 75% |

如果达不到，先检查：

- schema item 文本构造；
- gold label 抽取；
- question/schema encoder；
- 正负样本比例。

## 进入下一阶段条件

```text
static grounder 能稳定召回大部分 gold tables 和关键 columns。
```

---

# Stage 3：Static Grounding + Prompt-based SQL Generation

## 目标

建立一个强 baseline：

```text
Question + top-k grounded schema → LLM → SQL
```

虽然这不是最终创新，但它是必须的对照组。

## 需要准备

### 输入

- Stage 2 的 top-k schema predictions；
- question；
- database schema；
- gold SQL。

### 模型

可以选：

- 本地开源 Code LLM；
- API LLM；
- 如果要可训练，使用 LoRA。

第一版建议先 inference-only 跑通，再考虑 LoRA。

## Prompt 模板

```text
You are an expert SQL generator.

Database schema:
{top_k_schema}

Question:
{question}

Generate the SQL query.
```

可加约束：

```text
Return only SQL.
```

## 流程

1. 对 dev set 每条样例获取 top-k schema。
2. 构造 prompt。
3. 调用 LLM 生成 SQL。
4. 执行 SQL。
5. 计算 Execution Accuracy。
6. 保存错误样例。

## 产出

```text
experiments/stage3_static_prompt_sql/
  generated_sql.jsonl
  execution_results.jsonl
  metrics.json
  error_cases.jsonl
```

## 跑通标准

满足：

- 能批量生成 SQL；
- 能批量执行；
- 能计算 EX；
- 错误样例可追踪到 question、schema、gold SQL、pred SQL。

## 效果可以的判断

不要求超过 SOTA，但至少要证明 schema grounding 有用：

| Comparison | Expected |
|---|---|
| full schema prompt vs top-k schema prompt | top-k 不应显著更差，最好更好 |
| random schema prompt vs top-k schema prompt | top-k 明显更好 |
| oracle schema prompt vs predicted top-k | oracle 明显更好 |

如果 top-k schema prompt 明显弱于 full schema prompt，说明 Stage 2 grounder 召回不够。

## 进入下一阶段条件

```text
static grounding + prompt baseline 已可运行，并且 top-k schema 对 SQL 生成有正向作用。
```

---

# Stage 4：Clause-level Dynamic Grounding

## 目标

实现动态 grounding controller：

```text
g_t = f(Q, S, SQL_<t)
```

但这个阶段先不接 LLM hidden state，只训练并评估 grounding 本身。

## 需要准备

### 输入

来自 Stage 1 的 clause-level labels：

```text
select / from / join / where / group_by / order_by
```

### Partial SQL 表示

第一版使用 gold partial SQL。

例如：

```text
step=select, context=""
step=from, context="SELECT customers.name"
step=join, context="SELECT customers.name FROM customers"
```

注意：

第一版训练时用 gold partial SQL 是合理的，因为先验证 controller 是否能学会阶段性 grounding。

## 流程

1. 读取每条 gold SQL。
2. 拆成 clause-level partial SQL。
3. 对每个 clause 构造训练样例：

   ```text
   input: question + schema + previous clauses
   target: current clause schema labels
   ```

4. 训练 dynamic controller。
5. 评估每个 clause 的 grounding recall / precision。

## 模型

基于 Stage 2 static grounder 增加 partial SQL encoder：

```text
q = QuestionEncoder(Q)
u_t = SQLContextEncoder(Y_<t)
z_i = SchemaEncoder(s_i)

p_i^t = sigmoid(MLP([q; u_t; z_i; p_i^{t-1}]))
```

## 产出

```text
experiments/stage4_dynamic_grounding/
  clause_train_examples.jsonl
  checkpoints/
  dev_clause_predictions.jsonl
  clause_metrics.json
  trajectory_examples.md
```

## 跑通标准

满足：

- 能构造 clause-level training examples；
- dynamic controller loss 下降；
- 每个 clause 都能输出 top-k schema；
- trajectory examples 可读。

## 效果可以的判断

重点比较：

```text
static grounding vs dynamic grounding
```

按 clause 看：

| Clause | Minimum useful signal |
|---|---|
| SELECT | selected column recall 明显高于 static |
| FROM | table recall 高 |
| JOIN | join key recall 有提升 |
| WHERE | filter column recall 有提升 |
| GROUP BY | grouping column recall 有提升 |
| ORDER BY | ranking metric recall 有提升 |

如果 dynamic grounding 没有超过 static，先不要接 cross-attention。

要回头检查：

- clause labels 是否准确；
- partial SQL encoder 是否有效；
- `p_i^{t-1}` 是否导致错误累积；
- 是否需要按 clause type 加 embedding。

## 进入下一阶段条件

```text
dynamic controller 在 clause-level grounding 上优于 static grounder，至少在复杂 clause 上有明显收益。
```

---

# Stage 5：Dynamic Grounding + Cross-Attention

## 目标

让 LLM hidden state 通过 cross-attention 读取 dynamic grounding memory：

```text
c_t = Attention(h_t, G_t)
```

这个阶段先不加 steering gate，只验证 cross-attention 是否能把 grounding 信息用于 SQL generation。

## 需要准备

### 模型基础

- LLM backbone；
- Stage 4 dynamic grounding controller；
- cross-attention module；
- LoRA 或 adapter。

### Grounding memory

```text
G_t = {p_i^t z_i}
```

只取 top-k schema elements 进入 cross-attention，降低计算量：

```text
top-k = 20 or 30
```

## 流程

1. LLM 生成 SQL。
2. 在 clause boundary 更新 `g_t`。
3. 根据 `g_t` 构造 schema memory `G_t`。
4. LLM hidden state 对 `G_t` 做 cross-attention。
5. 融合：

   ```text
   h'_t = h_t + c_t
   ```

6. 预测下一 token / clause。

## 训练

建议冻结：

- LLM backbone。

训练：

- LoRA；
- cross-attention；
- optionally dynamic controller。

Loss：

```text
L = L_sql + λ L_c
```

## 产出

```text
experiments/stage5_dynamic_cross_attention/
  checkpoints/
  generated_sql.jsonl
  execution_results.jsonl
  metrics.json
  attention_maps/
```

## 跑通标准

满足：

- forward 能跑通；
- loss 下降；
- 能生成 SQL；
- 能执行评估；
- cross-attention weights 可导出。

## 效果可以的判断

比较：

| Model | Expected |
|---|---|
| static prompt baseline | 基线 |
| static cross-attention | 比 prompt 更稳定 |
| dynamic cross-attention | 在复杂 SQL 上优于 static cross-attention |

如果 dynamic cross-attention 不提升：

- 检查 dynamic grounding 是否准确；
- 检查 cross-attention 注入层；
- 检查 top-k schema 是否覆盖 gold；
- 检查 LLM 是否忽略 cross-attention。

## 进入下一阶段条件

```text
dynamic cross-attention 至少在复杂 SQL 或长 schema 场景中优于 static cross-attention / prompt baseline。
```

---

# Stage 6：Gated Hidden-State Steering

## 目标

在 cross-attention 基础上加入 gate，让 grounding context 直接调控 hidden state：

```text
alpha_t = sigmoid(W[h_t; c_t])
h'_t = h_t + alpha_t c_t
```

这是完整模型。

## 需要准备

来自 Stage 5 的可运行模型。

新增：

- steering gate；
- layer-wise injection 配置；
- no-gate ablation。

## 流程

1. 在 LLM 中高层注入 cross-attention output。
2. 使用 gate 控制注入强度。
3. 训练 LoRA + cross-attention + gate。
4. 比较 no-gate 与 gated steering。

## 产出

```text
experiments/stage6_gated_steering/
  checkpoints/
  generated_sql.jsonl
  execution_results.jsonl
  metrics.json
  gate_statistics.json
  ablation_results.md
```

## 跑通标准

满足：

- gated steering forward/backward 正常；
- SQL 语法合法率不下降；
- gate value 有合理分布，不全是 0 或 1；
- 可以导出不同 clause 的 gate 强度。

## 效果可以的判断

至少满足一个：

1. overall EX 提升；
2. complex SQL EX 提升；
3. long schema 下更稳；
4. no-gate 容易不稳定，而 gated steering 更稳。

重点指标：

| Metric | Expected |
|---|---|
| EX | full >= cross-attention only |
| valid SQL rate | full 不应下降 |
| complex SQL EX | full 应提升 |
| gate distribution | clause-dependent，而不是常数 |

## 进入下一阶段条件

```text
gated steering 相比 no-gate 或 cross-attention only 有稳定收益，或者至少证明它提升复杂场景鲁棒性。
```

---

# Stage 7：Grounding Intervention 与 Long-schema Robustness

## 目标

证明 grounding state 不是装饰，而是真的影响 SQL generation。

这是论文说服力的关键。

## 需要准备

来自 Stage 6 的 full model。

需要实现 intervention hooks：

- replace grounding；
- remove top-k grounding；
- adversarial swap；
- oracle grounding。

## Intervention 设计

### 1. Random replacement

```text
g_t → random schema elements
```

预期：

```text
EX drops significantly
```

### 2. Adversarial column swap

例如：

```text
orders.amount → products.price
customers.created_at → orders.created_at
```

预期：

模型更容易生成被替换的错误列，说明 grounding state 对 decoding 有影响。

### 3. Top-k removal

移除 grounding score 最高的 schema elements。

预期：

join、where、order by 错误增加。

### 4. Oracle grounding

使用 gold SQL schema labels 替换预测 grounding。

预期：

EX 上升，说明 grounding quality 是瓶颈之一。

## Long-schema robustness

构造 distractor schema：

- same-name columns；
- semantically similar columns；
- plausible join distractors；
- similar sample values。

测试：

```text
1x / 3x / 5x / 10x schema scale
```

## 产出

```text
experiments/stage7_intervention_robustness/
  intervention_results.json
  long_schema_results.json
  adversarial_cases.jsonl
  trajectory_visualization.md
```

## 跑通标准

满足：

- intervention 能自动执行；
- 每种 intervention 都能输出 EX；
- long-schema 数据可构造；
- 能定位错误类型变化。

## 效果可以的判断

| Experiment | Expected result |
|---|---|
| random grounding | EX 明显下降 |
| adversarial swap | 错误列使用率上升 |
| top-k removal | EX 下降，schema-related error 增加 |
| oracle grounding | EX 上升 |
| long schema | dynamic full model 下降最慢 |

如果这些现象成立，论文故事就很完整。

---

# 推荐执行顺序

实际推进时按下面顺序：

| Order | Stage | Decision gate |
|---:|---|---|
| 1 | Stage 0 数据检查 | 数据链路全部跑通 |
| 2 | Stage 1 label 抽取 | label 覆盖率足够 |
| 3 | Stage 2 static grounding | recall@k 合理 |
| 4 | Stage 3 static prompt SQL | top-k schema 对 SQL 有帮助 |
| 5 | Stage 4 dynamic grounding | dynamic > static |
| 6 | Stage 5 cross-attention | dynamic CA > static CA |
| 7 | Stage 6 gated steering | gate 有稳定收益 |
| 8 | Stage 7 intervention | grounding 有因果证据 |

---

# 当前最应该开始的阶段

下一步应该从 **Stage 0** 开始。

你需要先确认：

1. 当前要先用 Spider 还是 BIRD；
2. 数据集目录是否完整；
3. gold SQL 能否执行；
4. schema 能否统一解析。

由于当前项目根目录已有：

```text
BIRD/
```

如果暂时没有 Spider，建议先用 BIRD 做 Stage 0 数据检查。但从研究开发稳定性看，Spider 更适合作为第一版原型。

推荐选择：

```text
Stage 0A: 先检查当前 BIRD 数据是否完整
Stage 0B: 如果 BIRD 太复杂，再补 Spider 作为原型数据
```

---

# 每阶段向我反馈什么

你每跑完一个阶段，可以把下面这些信息发给我：

## Stage 0 反馈

```text
数据集：
样例数：
数据库数：
gold SQL 可执行比例：
schema 解析是否成功：
报错样例：
```

## Stage 1 反馈

```text
SQL parse success rate：
whole label coverage：
clause label coverage：
failed parse cases 数量：
```

## Stage 2 反馈

```text
table recall@5：
column recall@10：
schema recall@20：
loss 曲线是否下降：
top-k examples：
```

## Stage 3 反馈

```text
full schema prompt EX：
top-k schema prompt EX：
random schema prompt EX：
oracle schema prompt EX：
主要错误类型：
```

后续阶段同理。每次你把结果发回来，我再判断是否进入下一阶段，或者应该先修哪里。
