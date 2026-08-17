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
- database value lookup required;
- semantic inference from a superlative (`highest`, `most`, and similar) for `LIMIT 1`;
- operator inference required for an otherwise unexpressed LIMIT constant;
- expression constant for a literal introduced inside a SELECT expression.

Case-fold-only matches are not safe exact-copy targets. They should be canonicalized through the value index before SQL compilation.
Numeric copies use complete-value boundaries, so `LIMIT 1` cannot be aligned to the first digit of `1945`.

## Stage 13-A-fix1 correctness repairs

The first full audit exposed three supervision bugs that must be fixed before decoder training:

1. BIRD's `column_names_original` and `column_types` both retain the wildcard at index zero. Stage 1 previously subtracted one from the column index and shifted every real column type onto its predecessor.
2. A raw substring search aligned short numeric constants to fragments of longer values.
3. PROJECT preceded SORT, which removed non-output ordering keys from the logical relation.

Fix1 therefore:

- reads `column_types[column_index]`;
- uses boundary-aware numeric and string source matching;
- places final PROJECT after SORT and LIMIT;
- fuses SQL-explicit qualified equality edges with the target-table-induced FK subgraph;
- distinguishes copied values, inferred operator constants, expression constants, and database lookup values.

On the corrected local 100/100 smoke run, join-path connectivity changed from `0.956/0.901`
to `1.000/0.986` for train/dev. The lower all-target exact-copy rate is intentional: false
substring matches are no longer counted as successful copies. Use `direct_copy_exact_rate` to
measure fidelity only among targets actually assigned to question/evidence copy sources.

Because the Stage 1 type-index bug propagated into graph node features and dense node text, old
labels, graphs, embedding caches, and models are not compatible with fix1. Preserve them as
ablations and rebuild into new output directories.

## Mandatory fix1 rebuild

Schema and question cards can be reused because their source schema/question content did not
change. Rebuild labels first:

```bash
python src/data/stage1_extract_bird_labels.py \
  --bird-dir Data/BIRD \
  --train-question-answer experiments/stage0_train_correction_merge/merged_train_question_answer.json \
  --output-dir experiments/stage1_label_extraction_corrected_typefix1 \
  --splits train,dev
```

Rebuild graph examples with the existing compact cards:

```bash
python src/data/stage5_build_dsg_data.py \
  --train-labels experiments/stage1_label_extraction_corrected_typefix1/bird_train_grounding_labels.jsonl \
  --dev-labels experiments/stage1_label_extraction_corrected_typefix1/bird_dev_grounding_labels.jsonl \
  --train-tables Data/BIRD/train_databases/train_databases/train_tables.json \
  --dev-tables Data/BIRD/dev_tables.json \
  --train-schema-semantic-cards experiments/stage8f_compact_llm_cards_corrected/train_schema_semantic_cards.jsonl \
  --dev-schema-semantic-cards experiments/stage8f_compact_llm_cards_corrected/dev_schema_semantic_cards.jsonl \
  --train-question-cards experiments/stage8f_compact_llm_cards_corrected/train_question_cards.jsonl \
  --dev-question-cards experiments/stage8f_compact_llm_cards_corrected/dev_question_cards.jsonl \
  --output-dir experiments/stage13a_dsg_data_typefix1
```

Build fix1 typed RA supervision:

```bash
python src/data/stage13_build_typed_ra_data.py \
  --train-labels experiments/stage1_label_extraction_corrected_typefix1/bird_train_grounding_labels.jsonl \
  --dev-labels experiments/stage1_label_extraction_corrected_typefix1/bird_dev_grounding_labels.jsonl \
  --train-graphs experiments/stage13a_dsg_data_typefix1/train_examples.jsonl \
  --dev-graphs experiments/stage13a_dsg_data_typefix1/dev_examples.jsonl \
  --output-dir experiments/stage13a_typed_ra_typefix1
```

Only rebuild a dense embedding cache when a subsequent Stage 13 model consumes cached node
embeddings. Do not reuse `stage8g_embedding_cache_corrected_qwen3_06b`, because its node text
contains the shifted data types.

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
