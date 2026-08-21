# Stage 10-F: Full-Schema Steiner Join Closure

## Motivation

Stage 10-E reached `0.884615` structural grounding completeness, but 62 samples
had a complete semantic core and disconnected required tables.  The candidate
audit showed that most of these failures cannot be repaired by raising a
connectivity weight: the necessary bridge nodes are absent from the candidate
subgraph itself.

Stage 10-F removes structural bridge nodes from competition with the semantic
Top-30:

```text
immutable semantic Top-30
          |
          v
LLM-supported terminal tables
          |
          v
full declared FK table graph
          |
          v
metric closure + MST approximation to Steiner tree
          |
          v
intermediate tables + FK endpoint columns
```

The result is represented as two typed sets:

```text
semantic_core_ids       # unchanged Stage 10-E Top-30
structural_closure_ids  # outside the semantic budget
grounded_schema_ids     # their union
```

An added bridge node can improve connectivity but receives no semantic-recall
credit. Consequently, the method cannot hide semantic mistakes by accidentally
adding a gold column as a path endpoint.

## Inference policy

1. Read the Stage 10-E normal prediction and frozen-LLM prior.
2. A selected node proposes its owner table as a terminal only when a non-
   structural LLM role has confidence at least `0.5`.
3. `JOIN_BRIDGE` alone never creates a semantic terminal.
4. Build a table graph from the complete database schema, not the Stage 10
   candidate graph.
5. Compute pairwise shortest FK paths between terminals and an MST over the
   metric closure.
6. Expand the MST paths into intermediate tables and FK endpoint columns.
7. Do not invent undeclared edges. Disconnected schemas remain explicitly
   classified for later implicit-join or query-block analysis.

The maximum terminal count and path length are safety guards against connecting
every weakly supported table in a large schema.

## Full BIRD dev command

```bash
cd /data/zhuoaq/Dynamic-Schema-Grounding

python src/grounding/stage10f_steiner_join_closure.py \
  --full-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --prediction-file experiments/stage10e_frozen_llm_schema_rgta_full_seed42/dev_predictions.jsonl \
  --prior-file experiments/stage10e_llm_semantic_prior/dev_priors.jsonl \
  --output-file experiments/stage10f_steiner_join_closure_seed42/dev_closure.jsonl \
  --minimum-terminal-score 0.5 \
  --max-terminal-tables 6 \
  --support-weight 0.25 \
  --max-path-hops 6
```

Evaluate without changing semantic-core credit:

```bash
python src/evaluation/stage10f_evaluate_join_closure.py \
  --full-graph-file experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --closure-file experiments/stage10f_steiner_join_closure_seed42/dev_closure.jsonl \
  --exact-label-file experiments/stage1_label_extraction_exact_fk_v1/bird_dev_grounding_labels.jsonl \
  --output-dir experiments/stage10f_steiner_join_closure_seed42/evaluation
```

Inspect:

```bash
cat experiments/stage10f_steiner_join_closure_seed42/dev_closure_summary.json
cat experiments/stage10f_steiner_join_closure_seed42/evaluation/summary.json
```

## Acceptance criteria

- `semantic_complete_coverage` must remain exactly the Stage 10-E value;
- `regressed_samples` must be zero because closure is a set union;
- `after_join_connected_coverage` must exceed `before_join_connected_coverage`;
- `after_grounding_complete_coverage` must exceed `0.884615`;
- average closure size and path length must remain small enough for an LLM or SQL
  planner to consume explicitly;
- unresolved cases must be separated into semantic misses, missing terminal
  detection, no declared/explicit path, and closure algorithm failures.

If most unresolved cases have no declared FK path, the next step is not a larger
Steiner budget. It is a typed soft-edge model based on key compatibility and value
overlap, followed by execution verification. If failures are mainly terminal
detection misses, the LLM-to-terminal calibration must be improved instead.

