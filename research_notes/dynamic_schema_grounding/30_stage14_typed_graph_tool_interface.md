# 30. Stage 14: Typed RGTA Schema Tool Interface

记录日期：2026-08-17

## Why Stage 13C is closed

Stage 13C Fix1-v3 lowered dev weighted teacher-forcing CE from the frozen-LLM identity value
`0.345273` to `0.251722` by epoch 3, but it failed the causal graph criteria:

- best positive dev schema log-probability gain was only `0.001442` at epoch 1;
- dev schema improvement rate remained below `0.32`, versus the required `0.5`;
- FK-corruption counterfactual loss stayed near its `0.05` margin;
- token gates saturated near one;
- alignment Recall@1 reached `1.0` on train but only `0.163371` on dev.

The adapter therefore learned a useful generic schema-semantic residual but did not demonstrate that
RGTA topology improved cross-database generation. Per the pre-registered decision rule, Stage 13C is
closed rather than extended by more residual-scale or gate tuning.

## Stage 14 architecture

```text
LLM or oracle diagnostic action plan
                 |
       typed requests (SCAN/FILTER/...)
                 |
       frozen Stage 13-B RGTA tool
       +---------+----------+------------+
       |         |          |            |
   schema IDs  FK edges   operators   value routes
       +---------+----------+------------+
                 |
       deterministic constraint assembly
       - column owner closure
       - FK path connectivity
       - literal surface preservation
       - explicit budget feasibility
                 |
        structured JSON tool payload
                 |
       later: LLM SQL generator + verifier
```

The graph state is no longer injected into every LLM hidden token. The LLM/planner requests a typed
decision, while RGTA returns only the decisions for which it was trained: table/column IDs, FK edges,
operators, value routes, and uncalibrated scores.

## Implemented files

- `src/grounding/stage14_typed_schema_tool.py`
- `src/evaluation/stage14_evaluate_typed_schema_tool.py`
- `src/modeling/typed_ra_decoder.py`: parameter-free `forced_action` inference path
- `tests/test_stage14_typed_schema_tool.py`

## First experiment: oracle action skeleton

The first experiment deliberately uses the gold action names but strips all gold pointers,
operators, value routes, and values before RGTA inference. This isolates the schema tool from the
future LLM planner.

```bash
CUDA_VISIBLE_DEVICES=0 python src/grounding/stage14_typed_schema_tool.py \
  --graph-checkpoint experiments/stage13b_typed_ra_decoder_rgta_seed42/typed_ra_decoder.pt \
  --graph-summary experiments/stage13b_typed_ra_decoder_rgta_seed42/training_summary.json \
  --graph-file experiments/stage13a_dsg_data_typefix1/dev_examples.jsonl \
  --plan-file experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl \
  --embedding-cache-dir experiments/stage13b_embedding_cache_typefix1_qwen3_06b \
  --split dev \
  --output-file experiments/stage14a_typed_schema_tool/oracle_action_limit100.jsonl \
  --limit 100 \
  --table-top-k 3 \
  --column-top-k 5 \
  --join-top-k 3 \
  --operator-top-k 3 \
  --value-route-top-k 2 \
  --max-schema-items 30 \
  --device cuda
```

Evaluate the tool without invoking an LLM:

```bash
python src/evaluation/stage14_evaluate_typed_schema_tool.py \
  --tool-output experiments/stage14a_typed_schema_tool/oracle_action_limit100.jsonl \
  --target-trajectories experiments/stage13b_clean_typed_trajectories/dev_trajectories.jsonl \
  --output-dir experiments/stage14a_typed_schema_tool/evaluation_limit100
```

If the 100-example run is structurally valid, remove `--limit 100` and write to a new full-dev file.
The clean Stage 13B dev trajectory contains only supported flat SQL records; this is not the complete
1,534-example BIRD dev set.

## External planner request format

A later LLM planner should produce JSONL without schema answers:

```json
{
  "record_index": 0,
  "plan_source": "llm_typed_plan",
  "requests": [
    {"request_id": "scan_0", "action": "SCAN"},
    {"request_id": "filter_0", "action": "FILTER", "value_surface": "Alameda"},
    {"request_id": "project_0", "action": "PROJECT", "cardinality": 2},
    {"request_id": "sort_0", "action": "SORT"},
    {"request_id": "limit_0", "action": "LIMIT", "value_surface": "1"},
    {"request_id": "stop", "action": "STOP"}
  ]
}
```

`value_surface` is copied byte-for-byte into `literal_surfaces`; the schema tool never lowercases or
normalizes it. Database canonicalization remains a separate Value Index operation.

## Output boundary

The tool output contains:

- alternatives for each typed request;
- uncalibrated sigmoid confidence and raw logits;
- one selected assignment per request;
- owner-table additions;
- FK paths connecting selected terminal tables;
- budget/connectivity diagnostics;
- an `llm_tool_payload` containing exact schema names, IDs, join paths, and literal surfaces.

The first version does not claim calibrated probabilities, optimal global assignment, nested-query
support, or end-to-end SQL accuracy.

## Decision gate

Proceed to an LLM planner only if the oracle-action diagnostic shows:

1. table/column/FK recall remains reasonably close to Stage 13B teacher-forced metrics;
2. assembled complete coverage is materially higher than independent pointer recall;
3. owner closure and FK completion do not create frequent budget overflow;
4. exact literal surfaces are unchanged;
5. predicted-state rollout does not collapse relative to teacher-forced Stage 13B.

If predicted-state rollout is much worse, the next fix belongs in the typed pointer transition
(scheduled sampling or non-recurrent per-request inference), not in the LLM hidden states.

