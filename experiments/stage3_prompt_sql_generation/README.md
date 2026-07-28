# Stage 3A: Graph-grounded Prompt Construction

本阶段只生成 prompt 文件，不调用 LLM。

## Goal

比较不同 schema setting：

- full schema
- oracle schema
- lexical top-k
- R-GCN top-k
- R-GTA top-k

## Inputs

```text
experiments/stage1_label_extraction/bird_dev_grounding_labels.jsonl
experiments/stage2_static_grounding/lexical_dev_predictions.jsonl
experiments/stage2_rgcn_torch_grounding_hybrid_500/rgcn_torch_dev_predictions.jsonl
experiments/stage2_rgta_torch_grounding_hybrid_500/rgcn_torch_dev_predictions.jsonl
Data/BIRD/dev_tables.json
```

## Command

Smoke test:

```powershell
python src/generation/stage3_build_prompts.py --limit 20
```

Full dev:

```powershell
python src/generation/stage3_build_prompts.py
```

## Outputs

```text
experiments/stage3_prompt_sql_generation/
  prompts_full_schema_dev.jsonl
  prompts_oracle_schema_dev.jsonl
  prompts_lexical_top20_dev.jsonl
  prompts_lexical_top30_dev.jsonl
  prompts_rgcn_top20_dev.jsonl
  prompts_rgcn_top30_dev.jsonl
  prompts_rgta_top20_dev.jsonl
  prompts_rgta_top30_dev.jsonl
  prompt_statistics.json
  prompt_examples.md
```

## Acceptance criteria

| Check | Target |
|---|---:|
| prompt count per setting | 1534 |
| each prompt has question | yes |
| each prompt has schema text | yes |
| selected schema grouped by table | yes |
| foreign keys included when selected | yes |
| prompt statistics generated | yes |

