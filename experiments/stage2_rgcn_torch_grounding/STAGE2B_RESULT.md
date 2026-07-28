# Stage 2B Result: PyTorch R-GCN Static Schema Grounder

## Environment

```text
conda env: huaweicup
torch: 2.12.0+cpu
cuda: false
```

## Best current run

Command:

```powershell
conda run -n huaweicup python src/grounding/stage2_rgcn_torch_grounding.py --train-limit 500 --epochs 2 --use-lexical-features --output-dir experiments/stage2_rgcn_torch_grounding_hybrid_500 --top-k-examples 20
```

Output:

```text
experiments/stage2_rgcn_torch_grounding_hybrid_500/
```

## R-GTA run

Command:

```powershell
conda run -n huaweicup python src/grounding/stage2_rgcn_torch_grounding.py --encoder-type rgta --train-limit 500 --epochs 2 --use-lexical-features --output-dir experiments/stage2_rgta_torch_grounding_hybrid_500 --top-k-examples 20
```

Output:

```text
experiments/stage2_rgta_torch_grounding_hybrid_500/
```

## Metrics

| Metric | Stage 2A lexical | R-GCN + lexical | R-GTA + lexical |
|---|---:|---:|---:|
| schema_recall@10 | 0.6396 | 0.6802 | 0.6717 |
| schema_recall@20 | 0.7754 | 0.8347 | 0.8430 |
| schema_recall@30 | 0.8317 | 0.8961 | 0.9019 |
| schema_precision@10 | 0.4177 | 0.4429 | 0.4389 |
| schema_mrr | 0.8883 | 0.8755 | 0.8849 |
| table_recall@3 | 0.7173 | 0.7279 | 0.7925 |
| table_recall@5 | 0.8174 | 0.8509 | 0.9031 |
| column_recall@10 | 0.6871 | 0.7526 | 0.7451 |
| column_recall@20 | 0.7812 | 0.8649 | 0.8684 |

## Judgment

Stage 2B passes.

The pure R-GCN version underperformed lexical grounding, but Graph+Lexical encoders substantially improve recall over Stage 2A. R-GTA gives the best schema_recall@20, schema_recall@30, table_recall@5, and column_recall@20, while R-GCN gives slightly better schema_recall@10 and column_recall@10. This suggests that relation-aware graph attention is a promising schema encoder for the next stage.
