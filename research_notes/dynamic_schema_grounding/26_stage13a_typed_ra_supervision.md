# Stage 13-A: Typed Relational-Algebra Supervision

## Why Stage 12 is stopped

On the same 100 BIRD dev examples, the frozen Qwen2.5-Coder-32B baseline achieved `0.35` EX while Stage 12-B dynamic hidden-state injection achieved `0.20`. Both had `0.92` execution success, so the regression was semantic rather than syntactic.

Paired analysis found 2 gains and 17 losses. Nearly every loss corrupted a case-sensitive database value (`Kacey` to `kacey`, `00D4` to `00d4`, `San Joaquin` to `san joaquin`, and similar changes). The schema-only RGTA state was being asked to steer schema, operators, values, and formatting through one continuous residual. It did not possess the information required to control all of those decision types.

Stage 13 therefore replaces global hidden-state steering with typed, explicit decisions.

## Target architecture

```text
question + complete schema graph
              |
             RGTA
              |
      LLM plan hidden state
              |
     +--------+----------+----------------+
     |                   |                |
operator head      schema pointer    value-copy pointer
     |                   |                |
relational action   table/column IDs  exact source/value ID
     +-------------------+----------------+
                         |
                typed relational DAG
                         |
          FK join solver + type constraint solver
                         |
              deterministic SQL compiler
```

RGTA is used only where its representation is semantically compatible: table, column, and join-edge selection. Values preserve their exact surface form through a separate copy/value-index mechanism.

## Stage 13-A output

Each training record contains two strictly separated blocks:

- `inference_inputs`: question, evidence, complete schema items, and schema graph edges;
- `training_targets`: gold SQL, typed relational-algebra DAG, action sequence, join path, and value-copy targets.

The v1 action vocabulary is:

```text
SCAN
JOIN
FILTER
AGGREGATE
HAVING_FILTER
PROJECT
SORT
LIMIT
```

Every action stores separate targets for:

- table pointers;
- column pointers;
- relational/functions/operators;
- exact values and their source;
- column types, numeric-expression requirements, explicit casts, and real division;
- FK join edges.

Each table receives its own `SCAN` node. Multi-table queries produce a `JOIN` node whose inputs are the scan nodes, making the representation a real DAG rather than a clause list.

## Parser boundary

No untracked parser dependency is introduced. Stage 13-A uses the repository's quote- and parenthesis-aware SQL scanner.

- Flat SQL is marked `supported_flat` and is eligible for v1 training.
- Nested SQL is marked `partial_nested`.
- UNION/INTERSECT/EXCEPT SQL is marked `partial_set_query`.

Partial records remain in the audit output but must be excluded from the first decoder training run. Their explicit status prevents silent label corruption.

## Value targets

String literals retain exact case and quoting. Each target is classified as:

- exact copy from question;
- exact copy from evidence;
- case-fold-only source match;
- database value lookup required.

Case-fold-only matches are not safe exact-copy targets. They should be canonicalized through the value index before SQL compilation.

## Server smoke build

Run from the repository root after pulling the Stage 13-A commit:

```bash
python src/data/stage13_build_typed_ra_data.py \
  --train-labels experiments/stage1_label_extraction_corrected/bird_train_grounding_labels.jsonl \
  --dev-labels experiments/stage1_label_extraction_corrected/bird_dev_grounding_labels.jsonl \
  --train-graphs experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --dev-graphs experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --output-dir experiments/stage13a_typed_ra_smoke \
  --train-limit 100 \
  --dev-limit 100
```

Inspect:

```bash
cat experiments/stage13a_typed_ra_smoke/summary.json
head -n 1 experiments/stage13a_typed_ra_smoke/train_typed_ra.jsonl
head -n 20 experiments/stage13a_typed_ra_smoke/train_typed_ra_audit_issues.jsonl
```

Smoke acceptance criteria:

1. `graph_attachment_rate = 1.0`;
2. `avg_schema_label_coverage` is close to 1.0;
3. `join_path_connected_rate` is high for multi-table examples;
4. all action types and value-source counts are non-empty where expected;
5. partial nested/set queries are explicitly counted, not silently treated as flat;
6. no gold SQL or target pointers occur under `inference_inputs`.

## Full build

After the smoke audit passes:

```bash
python src/data/stage13_build_typed_ra_data.py \
  --train-labels experiments/stage1_label_extraction_corrected/bird_train_grounding_labels.jsonl \
  --dev-labels experiments/stage1_label_extraction_corrected/bird_dev_grounding_labels.jsonl \
  --train-graphs experiments/stage8g_dsg_data_corrected_llm_cards/train_examples.jsonl \
  --dev-graphs experiments/stage8g_dsg_data_corrected_llm_cards/dev_examples.jsonl \
  --output-dir experiments/stage13a_typed_ra_full
```

Expected files:

```text
train_typed_ra.jsonl
dev_typed_ra.jsonl
train_typed_ra_audit_issues.jsonl
dev_typed_ra_audit_issues.jsonl
summary.json
```

Stage 13-B should not begin until full-data coverage and join-path failures have been audited by database and SQL feature.

