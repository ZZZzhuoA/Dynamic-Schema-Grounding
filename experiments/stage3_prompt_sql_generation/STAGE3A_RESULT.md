# Stage 3A Result: Prompt Construction

Stage 3A generates prompt files for different schema selection settings without calling an LLM.

## Command

```powershell
python src/generation/stage3_build_prompts.py
```

## Outputs

| File | Count |
|---|---:|
| `prompts_full_schema_dev.jsonl` | 1534 |
| `prompts_oracle_schema_dev.jsonl` | 1534 |
| `prompts_lexical_top20_dev.jsonl` | 1534 |
| `prompts_lexical_top30_dev.jsonl` | 1534 |
| `prompts_rgcn_top20_dev.jsonl` | 1534 |
| `prompts_rgcn_top30_dev.jsonl` | 1534 |
| `prompts_rgta_top20_dev.jsonl` | 1534 |
| `prompts_rgta_top30_dev.jsonl` | 1534 |

## Prompt statistics

| Setting | Avg prompt chars | Max prompt chars | Avg selected schema items |
|---|---:|---:|---:|
| full_schema | 2406.81 | 6834 | 82.63 |
| oracle_schema | 534.68 | 2315 | 7.02 |
| lexical_top20 | 1108.19 | 2546 | 19.53 |
| lexical_top30 | 1322.90 | 2874 | 28.42 |
| rgcn_top20 | 1121.60 | 2595 | 19.53 |
| rgcn_top30 | 1326.62 | 2855 | 28.42 |
| rgta_top20 | 1088.19 | 2605 | 19.53 |
| rgta_top30 | 1308.16 | 2893 | 28.42 |

## Judgment

Stage 3A passes.

All prompt settings have 1534 dev examples. Prompts include question, evidence, table-grouped schema text, selected foreign keys, and gold SQL for later evaluation.

