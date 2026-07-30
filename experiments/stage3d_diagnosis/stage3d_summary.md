# Stage 3D-A Grounding Bottleneck Diagnosis

## Generation results

| Setting | EX | Exec success | Correct | Pred exec ok |
|---|---:|---:|---:|---:|
| full_schema | 0.3100 | 0.8900 | 31 | 89 |
| rgta_top30 | 0.1700 | 0.8100 | 17 | 81 |
| lexical_top30 | 0.1800 | 0.7000 | 18 | 70 |

## Grounding recall

### rgta

Max available k: 30

| k | available | schema recall | table recall | column recall | FK endpoint recall |
|---:|:---:|---:|---:|---:|---:|
| 10 | yes | 0.6474 | 0.9560 | 0.5209 | 0.5736 |
| 20 | yes | 0.7541 | 0.9830 | 0.6440 | 0.7435 |
| 30 | yes | 0.8163 | 0.9880 | 0.7320 | 0.8548 |
| 50 | no | - | - | - | - |
| 80 | no | - | - | - | - |

### lexical

Max available k: 30

| k | available | schema recall | table recall | column recall | FK endpoint recall |
|---:|:---:|---:|---:|---:|---:|
| 10 | yes | 0.4199 | 0.8540 | 0.3940 | 0.1263 |
| 20 | yes | 0.5596 | 0.8900 | 0.4970 | 0.3490 |
| 30 | yes | 0.5973 | 0.8900 | 0.5403 | 0.4167 |
| 50 | no | - | - | - | - |
| 80 | no | - | - | - | - |

## Case-level comparisons

- full_correct_rgta_wrong: 20
- rgta_correct_full_wrong: 6
- rgta_correct_lexical_wrong: 3
- lexical_correct_rgta_wrong: 4
- full_correct_rgta_wrong_missing_column_count: 17
- full_correct_rgta_wrong_missing_table_count: 1

## R-GTA error buckets

- correct: 17
- semantic_mismatch: 64
- no_such_column: 17
- syntax_error: 1
- execution_error: 1

## Preliminary diagnosis

Among full_schema-correct but rgta_top30-wrong cases, 17/20 (85.0%) miss at least one gold column in R-GTA top30, and 1/20 (5.0%) miss at least one gold table. If missing-column rate is high, top-k recall is the primary bottleneck; otherwise prompt/reasoning/value grounding is likely dominant.
