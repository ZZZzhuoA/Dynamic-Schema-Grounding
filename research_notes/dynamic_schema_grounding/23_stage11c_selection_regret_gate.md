# Stage 11-C: Selection-Regret-Gated Dynamic Grounding

## Evidence motivating the stage

Stage 11-B-fix1 removed gate collapse and produced useful history candidates on
52.3% of dev steps, but it remained below independent operation-RGTA. Its gate was
trained from all-node BCE improvement while model selection uses recall@10 and MRR.
Every historical step received a gate above 0.1, showing regression toward an
average mixture rather than instance-level acceptance or rejection.

## Innovation

Stage 11-C treats historical grounding as a graph expert with a reject option. The
base and history-conditioned RGTA branches are evaluated by the actual structural
selection quality:

```text
Q(logits, gold) = Recall@K + eta * reciprocal_rank
utility = positive_part(Q_history - Q_base)
```

The detached utility supervises the gate probability. A straight-through binary
gate applies the historical graph expert only when its predicted benefit exceeds a
fixed threshold. Rejected samples exactly recover the independent logits.

The candidate and final branches also receive the Stage 10 budget-aware Top-K
coverage surrogate. This aligns representation learning, routing, checkpoint
selection, and evaluation around schema coverage rather than unrelated all-node
calibration.

The verified independent controller is loaded as a protected base expert and
frozen. Only the history residual, residual normalization, and routing gate are
trained. Therefore a rejected history candidate recovers not merely the same
formula, but the parameters of the validated independent model. This is a neural
safe-improvement constraint rather than ordinary multitask fine-tuning.

Epoch zero is evaluated and included in checkpoint selection with a reject-all gate.
The selected objective is `Recall@K + eta * MRR`, so training cannot replace the
protected baseline unless a later checkpoint improves the same ranking quality used
by the gate.

When the base is frozen, trajectories whose only supervised event is the first
history-free event have no trainable autograd path. The trainer skips these examples
and reports `train_skipped_no_trainable_path`; they cannot teach the history expert
and must not trigger a backward call on a constant frozen-base loss.

## Experiment

```bash
CUDA_VISIBLE_DEVICES=0 python src/training/stage11_train_dynamic_grounding_controller.py \
  --train-file experiments/stage11a_dynamic_trajectories/train_trajectories.jsonl \
  --dev-file experiments/stage11a_dynamic_trajectories/dev_trajectories.jsonl \
  --embedding-cache-dir experiments/stage8g_embedding_cache_corrected_qwen3_06b \
  --output-dir experiments/stage11c_selection_regret_gate_rgta_seed42 \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 5 \
  --lr 1e-4 \
  --output-top-k 10 \
  --history-mode uncertainty_residual \
  --base-checkpoint experiments/stage11a_independent_operation_rgta_seed42/dynamic_grounding_controller.pt \
  --freeze-base-controller \
  --history-gate-policy straight_through \
  --history-gate-threshold 0.5 \
  --history-utility-objective selection_regret \
  --history-mrr-weight 0.1 \
  --checkpoint-selection selection_quality \
  --history-selection-loss-weight 0.3 \
  --provisional-loss-weight 0.3 \
  --history-candidate-loss-weight 0.3 \
  --history-gate-loss-weight 0.1 \
  --history-utility-temperature 0.05 \
  --coverage-margin 0.1 \
  --coverage-temperature 0.2 \
  --device cuda \
  --seed 42
```

The acceptance rate must be between zero and one and should vary by operation. If
this stage still fails to outperform the independent controller, recurrence is
retained as a negative ablation and Stage 12 connects independent operation-RGTA
grounding tokens directly to the LLM decoder.
