# 08. Model Architecture Diagrams

本文件整理 Dynamic Schema Grounding 方法的图示版本。所有图均使用 Mermaid，可直接放入 Markdown、论文草稿或开题材料中继续修改。

## Figure 1：Overall Framework

这张图用于论文方法总览。核心表达是：

> schema grounding 不是一次性检索结果，而是在 SQL 生成过程中被 partial SQL 反复更新的动态状态。

```mermaid
flowchart TD
    Q["Natural Language Question Q"]
    S["Database Schema S"]
    SG["Schema Graph Construction"]
    SE["Schema Graph Encoder"]
    Z["Schema Memory Z"]

    LLM["LLM SQL Decoder"]
    PS["Partial SQL Y_<t"]
    DGC["Dynamic Grounding Controller"]
    GT["Dynamic Grounding State g_t"]

    CA["Recurrent Cross-Attention"]
    CTX["Schema Context c_t"]
    Gate["Steering Gate α_t"]
    HS["Hidden-State Steering"]
    YT["Next SQL Token / Clause y_t"]
    SQL["Final SQL"]

    S --> SG --> SE --> Z
    Q --> DGC
    Z --> DGC
    PS --> DGC
    DGC --> GT

    Q --> LLM
    PS --> LLM
    LLM --> CA
    GT --> CA
    CA --> CTX
    CTX --> Gate
    LLM --> Gate
    Gate --> HS
    CTX --> HS
    LLM --> HS
    HS --> YT
    YT --> PS
    YT --> SQL
```

## Figure 2：Static Schema Linking vs Dynamic Schema Grounding

这张图适合放在 Introduction 或 Motivation，突出本文与传统 schema linking 的区别。

```mermaid
flowchart LR
    subgraph Static["Static schema linking"]
        Q1["Question"]
        S1["Schema"]
        G1["One-shot grounding g = f(Q,S)"]
        SQL1["SQL generation"]

        Q1 --> G1
        S1 --> G1
        G1 --> SQL1
    end

    subgraph Dynamic["Dynamic schema grounding"]
        Q2["Question"]
        S2["Schema"]
        G20["Initial grounding g_0"]
        SQL20["Generate partial SQL"]
        G2T["Update grounding g_t = f(Q,S,Y_<t)"]
        SQL2T["Generate next SQL step"]

        Q2 --> G20
        S2 --> G20
        G20 --> SQL20
        SQL20 --> G2T
        Q2 --> G2T
        S2 --> G2T
        G2T --> SQL2T
        SQL2T --> G2T
    end
```

## Figure 3：Detailed Module View

这张图用于 Method 部分，展示每个模块的输入输出。

```mermaid
flowchart TD
    subgraph SchemaSide["Schema side"]
        T["Tables"]
        C["Columns"]
        FK["Foreign keys"]
        V["Sample values / descriptions"]
        Graph["Heterogeneous schema graph"]
        Encoder["Graph / Relation-aware Transformer"]
        Memory["Schema memory Z = {z_i}"]

        T --> Graph
        C --> Graph
        FK --> Graph
        V --> Graph
        Graph --> Encoder --> Memory
    end

    subgraph GenerationSide["Generation side"]
        Q["Question encoding"]
        PrevSQL["Partial SQL encoding"]
        Controller["Dynamic grounding controller"]
        Scores["Schema relevance scores p_i^t"]
        GState["Grounding memory G_t = {p_i^t z_i}"]

        Q --> Controller
        PrevSQL --> Controller
        Memory --> Controller
        Controller --> Scores
        Scores --> GState
        Memory --> GState
    end

    subgraph Interaction["LLM interaction"]
        H["LLM hidden state h_t"]
        CrossAttn["Cross-attention: Attn(h_t, G_t)"]
        Context["Schema context c_t"]
        Alpha["Gate α_t = σ(W[h_t;c_t])"]
        Steer["h'_t = h_t + α_t c_t"]
        Decode["SQL token distribution"]

        H --> CrossAttn
        GState --> CrossAttn
        CrossAttn --> Context
        H --> Alpha
        Context --> Alpha
        H --> Steer
        Context --> Steer
        Alpha --> Steer
        Steer --> Decode
    end
```

## Figure 4：Clause-level Grounding Update

如果论文采用 clause-level update，这张图很关键。它能说明为什么动态 grounding 符合 SQL 结构。

```mermaid
stateDiagram-v2
    [*] --> InitialGrounding
    InitialGrounding --> SelectGrounding: generate SELECT
    SelectGrounding --> FromGrounding: SELECT clause fixed
    FromGrounding --> JoinGrounding: FROM tables selected
    JoinGrounding --> WhereGrounding: join path fixed
    WhereGrounding --> GroupGrounding: filters fixed
    GroupGrounding --> OrderGrounding: grouping fixed
    OrderGrounding --> FinalSQL: ranking / limit fixed
    FinalSQL --> [*]

    note right of SelectGrounding
        focus:
        output columns
        aggregation targets
    end note

    note right of FromGrounding
        focus:
        main tables
        candidate join tables
    end note

    note right of JoinGrounding
        focus:
        primary keys
        foreign keys
        join path
    end note

    note right of WhereGrounding
        focus:
        filter columns
        matched values
    end note

    note right of OrderGrounding
        focus:
        ranking column
        aggregation metric
    end note
```

## Figure 5：Cross-Attention and Steering as Two Roles of One Grounding State

这张图用于解释“二合一为什么合理”：cross-attention 负责读取，steering 负责调控。

```mermaid
flowchart LR
    GT["Dynamic Grounding State g_t"]

    subgraph Read["Read schema evidence"]
        CA["Cross-Attention"]
        C["Schema Context c_t"]
    end

    subgraph Control["Control decoding"]
        Gate["Gate α_t"]
        HS["Hidden-State Steering"]
        HPrime["Steered hidden state h'_t"]
    end

    H["LLM hidden state h_t"]
    Y["Next SQL decision"]

    GT --> CA
    H --> CA
    CA --> C
    C --> Gate
    H --> Gate
    C --> HS
    H --> HS
    Gate --> HS
    HS --> HPrime
    HPrime --> Y
```

## Figure 6：Example Grounding Trajectory

示例问题：

```text
Find the customer who spent the most money.
```

对应 SQL：

```sql
SELECT customers.name
FROM customers
JOIN orders ON customers.id = orders.customer_id
GROUP BY customers.id
ORDER BY SUM(orders.amount) DESC
LIMIT 1
```

```mermaid
flowchart TD
    Q["Question: Find the customer who spent the most money"]

    G0["g_0: customers, orders, orders.amount"]
    S1["SELECT customers.name"]
    G1["g_SELECT: customers.name, customers.id"]
    S2["FROM customers JOIN orders"]
    G2["g_FROM/JOIN: customers.id, orders.customer_id"]
    S3["GROUP BY customers.id"]
    G3["g_GROUP: customers.id"]
    S4["ORDER BY SUM(orders.amount) DESC"]
    G4["g_ORDER: orders.amount"]
    Final["Final SQL"]

    Q --> G0 --> S1 --> G1 --> S2 --> G2 --> S3 --> G3 --> S4 --> G4 --> Final
```

## Figure 7：Training Pipeline

```mermaid
flowchart TD
    Data["Text-to-SQL datasets: Spider / BIRD"]
    SQLParse["Parse gold SQL AST"]
    Labels["Extract schema labels and clause-level labels"]

    Stage1["Stage 1: Schema grounding pretraining"]
    Stage2["Stage 2: SQL generation alignment"]
    Stage3["Stage 3: Dynamic grounding joint training"]

    Eval["Evaluation: EX / EM / grounding / intervention"]

    Data --> SQLParse --> Labels
    Labels --> Stage1
    Data --> Stage2
    Stage1 --> Stage2
    Labels --> Stage3
    Stage2 --> Stage3
    Stage3 --> Eval
```

## 论文插图建议

建议最终论文保留 3 张主图：

1. **Figure 1 Overall Framework**：放在 Introduction 或 Method 开头。
2. **Figure 2 Static vs Dynamic**：放在 Motivation。
3. **Figure 4 Clause-level Grounding Update** 或 **Figure 6 Grounding Trajectory**：放在实验分析或方法解释。

如果篇幅有限，Figure 3 可以作为 supplementary。

