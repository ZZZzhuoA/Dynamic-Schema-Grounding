# 02. Architecture

## Overall Architecture

```text
                         Database Schema
                                |
                         Schema Graph
                                |
                       Schema Graph Encoder
                                |
                           Schema Memory Z
                                |
                                v
Question ------------> Dynamic Grounding Controller
                                |
                          Grounding State g_t
                                |
               +----------------+----------------+
               |                                 |
               v                                 v
        Cross-Attention                  Steering Gate
               |                                 |
               +----------------+----------------+
                                |
                         LLM Hidden State
                                |
                         SQL Decoder Head
                                |
                           Next SQL Step
                                |
                           Partial SQL
                                |
                                +---- feedback to controller
```

## 输入与输出

输入：

- natural language question `Q`;
- database schema `S`;
- optional database content evidence, such as sample values or column descriptions;
- partial SQL `Y_<t` during decoding。

输出：

- final SQL query `Y`;
- optional grounding trajectory `{g_0, g_1, ..., g_T}`;
- optional schema attention maps for interpretability。

## 模块 1：Schema Graph Construction

将数据库 schema 构造成异构图。

### 节点

可以包括：

- table node；
- column node；
- value node，可选；
- foreign key node，可选；
- SQL operation node，可选。

最小版本只需要 table node 与 column node。

### 边

建议包含：

| Edge type | 含义 |
|---|---|
| table-column | 表和列的归属关系 |
| same-table | 同一张表内列之间的关系 |
| primary-foreign-key | 主外键关系 |
| semantic-similarity | 表列名或描述语义相似 |
| value-match | question mention 与 sample value 匹配 |

## 模块 2：Schema Graph Encoder

Schema graph encoder 将 schema graph 编码成 schema memory：

```text
Z = {z_1, z_2, ..., z_n}
```

其中每个 `z_i` 表示一个 schema element。

可选结构：

1. Graph Transformer；
2. Relation-aware Transformer；
3. Heterogeneous GNN；
4. lightweight MLP + relation bias。

推荐从 Graph Transformer 或 relation-aware Transformer 起步，因为它和 Text-to-SQL 领域已有工作衔接自然。

## 模块 3：Dynamic Grounding Controller

Grounding controller 是核心模块。

它根据三类信息更新 grounding state：

```text
g_t = Controller(Q, Z, Y_<t)
```

其中：

- `Q` 提供用户意图；
- `Z` 提供 schema memory；
- `Y_<t` 提供当前 SQL 生成上下文。

### 更新粒度

有两种设计：

#### Token-level update

每生成一个 token 更新一次：

```text
g_t = f(Q, Z, y_1, ..., y_{t-1})
```

优点是精细，缺点是计算代价高。

#### Clause-level / AST-level update

在 SQL clause 边界更新：

```text
SELECT finished → update grounding
FROM finished → update grounding
WHERE finished → update grounding
GROUP BY finished → update grounding
```

推荐论文主方法使用 clause-level update，因为 SQL 本身是结构化语言，这样更自然，也更省计算。

## 模块 4：Recurrent Cross-Attention

Cross-attention 负责让 LLM hidden state 读取当前 grounding state。

设 LLM 当前 hidden state 为：

```text
h_t
```

grounding state 中的 schema representations 为：

```text
G_t = {g_{t,1}, g_{t,2}, ..., g_{t,n}}
```

则：

```text
c_t = Attention(Q_h, K_g, V_g)
```

其中：

```text
Q_h = h_t W_Q
K_g = G_t W_K
V_g = G_t W_V
```

`c_t` 表示当前 SQL 生成步骤需要读取的 schema context。

## 模块 5：Hidden-State Steering

Cross-attention 生成 schema context 后，通过 gate 控制注入强度：

```text
alpha_t = sigmoid(W [h_t; c_t])
```

然后：

```text
h'_t = h_t + alpha_t * c_t
```

最终使用 `h'_t` 预测下一个 SQL token 或 clause：

```text
P(y_t | Q, S, Y_<t) = softmax(W_o h'_t)
```

## 模块 6：SQL Decoder

可以有两种实现路线：

### 路线 A：Token Decoder

直接生成 SQL token。

优点：

- 实现简单；
- 可以直接基于开源 LLM 改。

缺点：

- grounding 与 SQL clause 的对应关系不够清晰。

### 路线 B：Structure-aware Decoder

先生成 SQL sketch / AST，再填充表列。

优点：

- 更适合 clause-level grounding；
- 更容易做 consistency loss；
- 可解释性更强。

缺点：

- 工程更复杂。

推荐：

第一版使用 token decoder，实验中增加 clause-level analysis；后续版本再升级到 structure-aware decoder。

