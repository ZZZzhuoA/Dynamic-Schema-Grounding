# Stage 3B-fix1 Result: Prompt Schema Formatting Repair

## Changes

- Removed fake placeholder column `[no selected columns]`.
- Added FK endpoint closure: selected foreign-key endpoints are inserted into table columns.
- Added owning table closure for selected columns.
- Quoted displayed column names with backticks.
- Strengthened prompt rules:
  - use exact listed table/column names only;
  - do not invent columns or tables;
  - use SQLite syntax only;
  - quote complex column names;
  - return only one SQL query.

## Command

```powershell
python src/generation/stage3_build_prompts.py --output-dir experiments/stage3_prompt_sql_generation_v2
```

## Validation

```text
placeholder_file_count = 0
```

First R-GTA top30 prompt now includes FK endpoint columns such as:

```text
Table frpm:
- `CDSCode`

Table satscores:
- `cds`

Table schools:
- `CDSCode`

Foreign keys:
- frpm.CDSCode = schools.CDSCode
- satscores.cds = schools.CDSCode
```

## Next test

Re-run R-GTA top30 with 100 examples and compare against the old result:

```text
old rgta_top30 limit100:
execution_accuracy = 0.14
pred_execution_success_rate = 0.64
api_error_rate = 0.0
```

