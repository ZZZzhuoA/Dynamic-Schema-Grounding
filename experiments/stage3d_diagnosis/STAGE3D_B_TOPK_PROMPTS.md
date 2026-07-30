# Stage 3D-B: Export larger R-GTA top-k prompts

Stage 3D-A showed that the main bottleneck is column-level recall:

```text
full_schema-correct but rgta_top30-wrong cases: 20
missing at least one gold column in R-GTA top30: 17/20
missing at least one gold table in R-GTA top30: 1/20
```

So Stage 3D-B prepares larger R-GTA schema prompts:

```text
rgta_top50
rgta_top80
```

This stage does not call an LLM. It only exports larger R-GTA rankings and builds prompt files.

## 1. Export R-GTA top80 predictions

Run this on the server environment that has PyTorch installed.

```bash
cd /root/autodl-tmp/Dynamic-Schema-Grounding

python src/grounding/stage2_rgcn_torch_grounding.py \
  --encoder-type rgta \
  --train-limit 500 \
  --epochs 2 \
  --use-lexical-features \
  --eval-only-model-path experiments/stage2_rgta_torch_grounding_hybrid_500/rgcn_torch_model.pt \
  --output-top-k 80 \
  --output-dir experiments/stage2_rgta_torch_grounding_hybrid_500_top80
```

Expected output:

```text
experiments/stage2_rgta_torch_grounding_hybrid_500_top80/rgcn_torch_dev_predictions.jsonl
```

Each prediction record should contain:

```text
top_30
top_80
```

## 2. Build R-GTA top50/top80 prompts

```bash
python src/generation/stage3_build_prompts.py \
  --rgta-predictions experiments/stage2_rgta_torch_grounding_hybrid_500_top80/rgcn_torch_dev_predictions.jsonl \
  --prediction-methods rgta \
  --prediction-top-ks 30,50,80 \
  --output-dir experiments/stage3_prompt_sql_generation_v3_topk
```

Expected output:

```text
experiments/stage3_prompt_sql_generation_v3_topk/prompts_rgta_top30_dev.jsonl
experiments/stage3_prompt_sql_generation_v3_topk/prompts_rgta_top50_dev.jsonl
experiments/stage3_prompt_sql_generation_v3_topk/prompts_rgta_top80_dev.jsonl
experiments/stage3_prompt_sql_generation_v3_topk/prompt_statistics.json
```

## 3. Optional local diagnosis after top80 export

```bash
python src/analysis/stage3d_grounding_diagnosis.py \
  --rgta-predictions experiments/stage2_rgta_torch_grounding_hybrid_500_top80/rgcn_torch_dev_predictions.jsonl \
  --output-dir experiments/stage3d_diagnosis_top80 \
  --limit 100
```

This should make `recall@50` and `recall@80` available.

## 4. Acceptance criteria

| Check | Target |
|---|---|
| R-GTA prediction file has `top_80` | yes |
| Prompt files for top50/top80 exist | yes |
| No LLM/API call in this stage | yes |
| Prompt statistics generated | yes |

