# Stage 11-B-fix1: Identity-Preserving Counterfactual Utility Gate

## Why fix1 is required

The first uncertainty-residual run reached recall@10 0.90652, below both the
independent operation-RGTA (0.91224) and legacy recurrence (0.90812). Its mean
history gate collapsed to 1.4e-9. More importantly, the old fusion applied a layer
normalization even when the gate was zero, so a closed history path did not recover
the independent prediction.

## Architecture

The controller now constructs two counterfactual graph-grounding candidates:

```text
base state --------------------> RGTA -> base logits
base state + history residual -> RGTA -> history-candidate logits
```

Final logits are a convex residual mixture:

```text
final = base + alpha * (history_candidate - base)
```

This establishes the invariant `alpha=0 => final=base` exactly. Dynamic graph
states and the controller state use the same identity-preserving mixture before
being exported to the LLM cross-attention/steering bridge.

## Counterfactual utility supervision

For every step with history, compute the supervised base and candidate losses. The
positive historical utility target is:

```text
improvement = max(L_base - L_candidate - margin, 0)
utility = 1 - exp(-improvement / temperature)
```

The gate learns this detached target. No improvement means utility zero; unlike a
plain sigmoid of a loss difference, equal candidates do not imply a half-open gate.
Both candidates receive auxiliary supervision, preventing a closed gate from
starving the history candidate of gradients.

## Server experiment

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage11_train_dynamic_grounding_controller.py \
  --train-file experiments/stage11a_dynamic_trajectories/train_trajectories.jsonl \
  --dev-file experiments/stage11a_dynamic_trajectories/dev_trajectories.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-dir experiments/stage11b_fix1_counterfactual_gate_rgta_seed42 \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 5 \
  --lr 1e-4 \
  --output-top-k 10 \
  --history-mode uncertainty_residual \
  --provisional-loss-weight 0.3 \
  --history-candidate-loss-weight 0.3 \
  --history-gate-loss-weight 0.1 \
  --history-utility-temperature 0.05 \
  --history-gate-penalty 0 \
  --device cuda \
  --seed 42
```

The primary comparison remains the seed-42 independent recall@10 of 0.91224 and
MRR of 0.79464. Gate diagnostics must also show nonzero candidate win rate,
selective rather than universal activation, and reduced gate-utility error.
