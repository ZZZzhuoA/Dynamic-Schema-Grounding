# Stage 11-A: Recurrent Partial-SQL Grounding Controller

## Transition from static to dynamic grounding

Stages 8--10 learn one schema ranking per question. Stage 11 changes the random
variable being estimated:

```text
static:   B = p(schema | question, database)
dynamic:  B_t = p(schema | question, database, partial_SQL_<t, B_<t)
```

The controller is trained on causal teacher-forcing trajectories. For each top-level
SQL clause, the input prefix stops immediately after its clause keyword. Therefore,
the identifier to be predicted in that clause is not present in the input.

## Causal trajectory

Example:

```text
SELECT                         -> projection schema
SELECT ... FROM                -> source/join schema
SELECT ... FROM ... WHERE      -> predicate schema
... GROUP BY                   -> grouping schema
... ORDER BY                   -> ordering schema
```

Nested-query keywords are ignored by the first implementation so that every event
has an unambiguous top-level target. Later work can represent nested scopes as a
stack of recurrent states.

The observed-schema mask contains only schema elements targeted by earlier events.
It never includes the current or future clause target.

## Architecture

At event `t`:

```text
partial SQL structural features ----+
operation embedding ----------------+
question embedding -----------------+--> GRU --> controller state g_t
previous belief schema summary ------+
observed schema summary -------------+

g_t + schema graph --> state-conditioned RGTA --> schema states H_t

(H_t, g_t, question) --> logits --> belief B_t
```

The state enters the RGTA query. Putting one shared state vector only in every edge
key would cancel under each destination softmax and would not dynamically change
neighbor attention.

The controller exposes two neural interfaces for the future LLM connection:

- `grounding_tokens`: one dynamic vector per schema node for cross-attention;
- `steering_state`: a belief-weighted summary for gated hidden-state steering.

`GroundingLLMBridge` projects these vectors to an arbitrary decoder hidden size.

## Build trajectories

Use OOF candidate graphs for training and the unchanged dev candidate graphs:

```bash
python src/data/stage11_build_dynamic_grounding_trajectories.py \
  --graph-file experiments/stage10b_oof_candidate_graphs/train_factor_graphs.jsonl \
  --label-file experiments/stage5g_clause_labels/train_clause_labels.jsonl \
  --output-file experiments/stage11a_dynamic_trajectories/train_trajectories.jsonl

python src/data/stage11_build_dynamic_grounding_trajectories.py \
  --graph-file experiments/stage10a_factor_graph_data/dev_factor_graphs.jsonl \
  --label-file experiments/stage5g_clause_labels/dev_clause_labels.jsonl \
  --output-file experiments/stage11a_dynamic_trajectories/dev_trajectories.jsonl
```

## Train the recurrent controller

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage11_train_dynamic_grounding_controller.py \
  --train-file experiments/stage11a_dynamic_trajectories/train_trajectories.jsonl \
  --dev-file experiments/stage11a_dynamic_trajectories/dev_trajectories.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-dir experiments/stage11a_dynamic_controller_rgta_seed42 \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 5 \
  --lr 1e-4 \
  --output-top-k 10 \
  --device cuda \
  --seed 42
```

## Required recurrence ablation

Use the same architecture and operation conditioning, but reset state and belief at
every event:

```bash
CUDA_VISIBLE_DEVICES=1 python src/training/stage11_train_dynamic_grounding_controller.py \
  --train-file experiments/stage11a_dynamic_trajectories/train_trajectories.jsonl \
  --dev-file experiments/stage11a_dynamic_trajectories/dev_trajectories.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-dir experiments/stage11a_independent_operation_rgta_seed42 \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 5 \
  --lr 1e-4 \
  --output-top-k 10 \
  --disable-recurrence \
  --device cuda \
  --seed 42
```

Primary metrics are step schema recall@10 and MRR. The mean belief total variation
is diagnostic only: a nonzero trajectory is necessary but does not prove useful
dynamics. The recurrent controller must outperform the independent operation-RGTA
under the same data and optimization before it is connected to the 30B LLM.
