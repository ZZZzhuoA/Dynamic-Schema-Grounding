# Stage 3B-server: vLLM Local 32B SQL Generation

This stage runs local SQL generation on the server with vLLM and the local 32B model.

Known server paths:

```text
project: /root/autodl-tmp/Dynamic-Schema-Grounding
model: /root/autodl-tmp/qwen_coder_32B
```

## 1. Server preparation

```bash
cd /root/autodl-tmp/Dynamic-Schema-Grounding
git pull
```

Check GPU and vLLM:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import vllm; print(vllm.__version__)"
```

Check the local model directory:

```bash
ls -lh /root/autodl-tmp/qwen_coder_32B | head
```

## 2. Regenerate v2 prompts if needed

This requires `Data/BIRD` on the server.

```bash
python src/data/stage1_extract_bird_labels.py --bird-dir Data/BIRD --splits train,dev

python src/grounding/stage2_lexical_grounding.py \
  --input experiments/stage1_label_extraction/bird_dev_grounding_labels.jsonl \
  --split dev

python src/grounding/stage2_rgcn_torch_grounding.py \
  --encoder-type rgta \
  --train-limit 500 \
  --epochs 2 \
  --use-lexical-features \
  --output-dir experiments/stage2_rgta_torch_grounding_hybrid_500

python src/generation/stage3_build_prompts.py \
  --output-dir experiments/stage3_prompt_sql_generation_v2
```

If the generated prompt file already exists, you can skip this step:

```text
experiments/stage3_prompt_sql_generation_v2/prompts_rgta_top30_dev.jsonl
```

## 3. Smoke test: R-GTA top30 v2, limit=5

```bash
python src/generation/stage3_vllm_generate.py \
  --model-path /root/autodl-tmp/qwen_coder_32B \
  --prompt-file experiments/stage3_prompt_sql_generation_v2/prompts_rgta_top30_dev.jsonl \
  --output-file experiments/stage3_vllm_generation/generations_rgta_top30_v2_limit5.jsonl \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-tokens 512 \
  --limit 5
```

Evaluate:

```bash
python src/evaluation/stage3_evaluate_sql.py \
  --generation-file experiments/stage3_vllm_generation/generations_rgta_top30_v2_limit5.jsonl
```

## 4. Limit=100

```bash
python src/generation/stage3_vllm_generate.py \
  --model-path /root/autodl-tmp/qwen_coder_32B \
  --prompt-file experiments/stage3_prompt_sql_generation_v2/prompts_rgta_top30_dev.jsonl \
  --output-file experiments/stage3_vllm_generation/generations_rgta_top30_v2_limit100.jsonl \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-tokens 512 \
  --limit 100
```

Evaluate:

```bash
python src/evaluation/stage3_evaluate_sql.py \
  --generation-file experiments/stage3_vllm_generation/generations_rgta_top30_v2_limit100.jsonl
```

## 5. Fixed generation parameters

For fair comparison with online LLM results and later trainable model results, keep:

```text
temperature = 0
top_p = 1
max_tokens = 512
```

## 6. Acceptance criteria

| Check | Target |
|---|---|
| vLLM import | success |
| model load | success |
| generated_sql non-empty | yes |
| SQLite evaluation runs | yes |
| no OOM | yes |

