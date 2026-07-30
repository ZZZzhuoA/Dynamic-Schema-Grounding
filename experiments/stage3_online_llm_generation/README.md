# Stage 3B-online: DashScope / OpenAI-compatible SQL Generation

本阶段使用在线 OpenAI-compatible 接口生成 SQL，并执行 SQLite 评估。

当前配置：

```text
base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
model = qwen3-32b
```

注意：DashScope Qwen3 非流式调用要求关闭 thinking，因此脚本默认：

```text
enable_thinking = false
```

如需开启可显式传入 `--enable-thinking`，但非流式接口可能报错。

API key 不写入代码，通过环境变量读取。

## 1. 准备环境变量

PowerShell:

```powershell
$env:LLM_API_KEY="你的 DashScope API Key"
$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL="qwen3-32b"
```

## 2. Smoke test：先跑 5 条 R-GTA top30

```powershell
python src/generation/stage3_online_llm_generate.py `
  --prompt-file experiments/stage3_prompt_sql_generation/prompts_rgta_top30_dev.jsonl `
  --output-file experiments/stage3_online_llm_generation/generations_rgta_top30_limit5.jsonl `
  --limit 5 `
  --temperature 0 `
  --top-p 1 `
  --max-tokens 512
```

评估：

```powershell
python src/evaluation/stage3_evaluate_sql.py `
  --generation-file experiments/stage3_online_llm_generation/generations_rgta_top30_limit5.jsonl
```

## 3. 小规模对比：lexical top30 vs R-GTA top30

建议先跑 50 条：

```powershell
python src/generation/stage3_online_llm_generate.py `
  --prompt-file experiments/stage3_prompt_sql_generation/prompts_lexical_top30_dev.jsonl `
  --output-file experiments/stage3_online_llm_generation/generations_lexical_top30_limit50.jsonl `
  --limit 50 `
  --temperature 0 `
  --top-p 1 `
  --max-tokens 512

python src/generation/stage3_online_llm_generate.py `
  --prompt-file experiments/stage3_prompt_sql_generation/prompts_rgta_top30_dev.jsonl `
  --output-file experiments/stage3_online_llm_generation/generations_rgta_top30_limit50.jsonl `
  --limit 50 `
  --temperature 0 `
  --top-p 1 `
  --max-tokens 512
```

评估：

```powershell
python src/evaluation/stage3_evaluate_sql.py `
  --generation-file experiments/stage3_online_llm_generation/generations_lexical_top30_limit50.jsonl

python src/evaluation/stage3_evaluate_sql.py `
  --generation-file experiments/stage3_online_llm_generation/generations_rgta_top30_limit50.jsonl
```

## 4. 固定 decoding 参数

所有 setting 必须固定：

```text
temperature = 0
top_p = 1
max_tokens = 512
```

唯一变化只能是 schema prompt setting。

## 5. 输出

生成文件每行包含：

```json
{
  "question_id": 0,
  "db_id": "california_schools",
  "setting": "rgta_top30",
  "raw_output": "...",
  "generated_sql": "...",
  "gold_sql": "...",
  "generation_config": {...},
  "error": null
}
```

评估文件：

```text
experiments/stage3_online_llm_generation/evaluation/
  *_metrics.json
  *_execution_details.jsonl
```

## 6. 验收标准

| Check | Target |
|---|---|
| API call succeeds | yes |
| generated_sql non-empty | yes |
| SQLite evaluation runs | yes |
| lexical_top30 and rgta_top30 metrics are comparable | yes |
