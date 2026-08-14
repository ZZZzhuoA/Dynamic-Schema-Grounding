# Stage 11-B: Uncertainty-Gated Residual History

## Motivation

Stage 11-A showed that independent operation-conditioned RGTA slightly outperformed
unconditional GRU recurrence. The failure does not reject dynamic grounding: the
independent model still re-grounds on every SQL operation using the partial SQL and
causally observed schema. It rejects the assumption that every historical latent
belief should always alter the next grounding step.

## Innovation

Stage 11-B protects the independent current-step solution and treats history as an
optional residual:

```text
current clause state -> RGTA -> provisional belief -> uncertainty
history state/belief -------------------------------> residual gate

final state = current state + uncertainty * relevance * history residual
```

The provisional belief is independently supervised. Consequently, history cannot
become the only route to a correct prediction. Normalized entropy and the top-2
margin estimate current ambiguity; the learned relevance gate decides whether the
specific historical state is useful. History is detached by default to prevent a
wrong ranking from creating an unrestricted gradient/error chain across clauses.

This is different from a prompt or a static schema selector: the gate modifies the
state that conditions relation-aware graph attention, and the resulting graph
states are exposed as neural grounding tokens for the later LLM bridge.

## Primary experiment

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage11_train_dynamic_grounding_controller.py \
  --train-file experiments/stage11a_dynamic_trajectories/train_trajectories.jsonl \
  --dev-file experiments/stage11a_dynamic_trajectories/dev_trajectories.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-dir experiments/stage11b_uncertainty_residual_rgta_seed42 \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 5 \
  --lr 1e-4 \
  --output-top-k 10 \
  --history-mode uncertainty_residual \
  --provisional-loss-weight 0.3 \
  --history-gate-penalty 0.01 \
  --device cuda \
  --seed 42
```

Compare against the existing `independent` and `legacy_recurrent` runs using the
same seed. Besides recall@10 and MRR, inspect mean gate, gate activation rate,
per-operation gate, provisional entropy, and residual norm. A useful mechanism
should improve ranking and open gates selectively; a permanently open gate is a
return to unconditional recurrence.
