# Stage 10 Complete-Coverage Label Audit

## Finding

The previous BIRD schema target applied an ``all_used_tables`` FK closure: once two
tables appeared anywhere in the SQL, every declared FK between those tables was
added to the gold schema set. This is broader than the SQL hypothesis itself and
penalizes a selector for omitting unused parallel FKs.

Full BIRD dev audit:

| Audit item | Value |
|---|---:|
| Samples | 1,534 |
| Samples with legacy-only FK labels | 221 (14.41%) |
| Legacy-only FK nodes | 536 |
| Mean legacy gold size | 7.1102 |
| Mean corrected gold size | 6.7608 |
| Legacy targets larger than Top-30 | 4 |
| Corrected targets larger than Top-30 | 0 |
| Corrected maximum target size | 16 |

The four formally impossible Top-30 targets were all caused by the legacy closure,
not true SQL complexity. For example, a query joining Player and Match could receive
all ``home_player_1...11`` and ``away_player_1...11`` columns as gold.

## Corrected policy

Stage 1 now defaults to:

```text
--fk-label-mode explicit_sql
```

An FK pair is recorded only when both endpoints were recovered from the SQL text.
Because these endpoints already belong to ``sql_parse`` labels, FK classification
does not expand the gold set. Reproduction of old experiments remains possible with:

```text
--fk-label-mode all_used_tables
```

Each corrected label row stores ``label_policy.legacy_fk_extra_*`` for audit only;
those fields never enter training targets.

## Connectivity-aware primary metric

``explicit_sql`` is a reference-path label, not the final grounding criterion. A
different declared FK path may still connect the same required semantic tables. The
diagnosis therefore separates:

```text
semantic target = required SQL tables
                + SELECT/WHERE/GROUP/HAVING/ORDER semantic columns

reference join  = endpoint columns used by the reference SQL

join feasible   = required tables are connected in the selected induced graph
                  through selected owner/FK endpoints
```

The primary structural event is:

```text
grounding_complete = semantic_complete AND join_connected
```

``reference_join_complete`` remains a diagnostic metric. A sample for which
``grounding_complete=true`` and ``reference_join_complete=false`` is reported as
``alternate_join_path_accepted`` rather than a grounding failure. Explicit SQL
column-equality edges are accepted alongside declared schema FK edges, while simple
table co-occurrence is never enough to establish connectivity.

## Loss decomposition

For the recorded Stage 10-D result, legacy metrics decomposed as:

| Layer | Complete samples | Coverage |
|---|---:|---:|
| Candidate pool | 1,389 | 0.905476 |
| Constrained Top-30 | 1,241 | 0.808996 |

Thus 145 failures occurred before reranking and another 148 occurred after the
candidate pool was already complete. These counts must be recomputed under the exact
labels before changing the architecture.

## Server commands

Rebuild exact dev labels without touching the historical directory:

```bash
python src/data/stage1_extract_bird_labels.py \
  --bird-dir Data/BIRD \
  --output-dir experiments/stage1_label_extraction_exact_fk_v1 \
  --splits dev \
  --fk-label-mode explicit_sql
```

Compare the exact and legacy targets on the same frozen candidate graph and frozen
Stage 10-D predictions:

```bash
python src/diagnosis/stage10_complete_coverage_diagnosis.py \
  --factor-graph-file experiments/stage10a_factor_graph_data/dev_factor_graphs.jsonl \
  --prediction-file experiments/stage10d_oof_schema_rgta_struct01_seed42/dev_predictions.jsonl \
  --legacy-label-file experiments/stage1_label_extraction_corrected_typefix1/bird_dev_grounding_labels.jsonl \
  --exact-label-file experiments/stage1_label_extraction_exact_fk_v1/bird_dev_grounding_labels.jsonl \
  --output-dir experiments/stage10_complete_coverage_exact_fk_audit \
  --top-k 30
```

Before running the second command, use ``test -f`` on all four inputs. If the actual
best Stage 10-D output directory differs, read it from the experiment's
``training_summary.json`` rather than substituting a different checkpoint.

The diagnosis reports, for both policies:

- candidate ceiling;
- raw reranker coverage;
- constrained selector coverage;
- baseline coverage;
- candidate-missing failures;
- raw-ranking failures;
- cases lost only by constrained decoding;
- old-missing to exact-complete transitions;
- the most frequently missing schema names.

It additionally reports under ``structural``:

- semantic complete coverage;
- reference Join coverage;
- selected-subgraph Join connectivity;
- connectivity-aware grounding complete coverage;
- alternate legal paths accepted;
- reference endpoints present but disconnected;
- semantic-missing versus Join-disconnected failures.

## Decision gate

Do not retrain Stage 10 before this audit. After recomputation:

1. optimize candidate retrieval only if exact-label candidate misses remain material;
2. optimize listwise/structured selection only on candidate-complete failures;
3. do not increase Top-K merely to accommodate legacy FK over-labeling;
4. keep the old metric solely as a reproducibility column, not the primary target.
