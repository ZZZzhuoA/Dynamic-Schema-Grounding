# Stage 13-C-A: Frozen-GNN to Frozen-LLM Graph Adapter Alignment

## Goal

Test the smallest structural hypothesis before adding dynamic updates:

```text
frozen query-conditioned RGTA(schema graph, question)
                         |
                         v
             trainable Graph Adapter
                         |
                         v
                 frozen Code LLM
                         |
                         v
                        SQL
```

The code LLM remains the only SQL decoder. Stage 13-C-A does not predict
relational-algebra actions, pointer decisions, or partial-SQL states.

## Frozen and trainable parameters

Frozen:

- the best Stage 13-B typed RGTA checkpoint;
- the complete Qwen code LLM.

Trainable:

- graph-memory projector from RGTA space to LLM space;
- decoder-layer graph cross-attention;
- token-wise residual gate and residual scale.

The normal LLM forward pass must retain autograd. Freezing the LLM means
`requires_grad=False` on its parameters, not wrapping the forward pass in
`torch.no_grad()`. The RGTA memory is explicitly detached.

## Objectives

1. Weighted SQL teacher-forcing loss, with extra weight on schema identifiers.
2. Node-to-LLM contrastive alignment between each projected graph node and the
   frozen LLM embedding of its exact schema name.
3. A counterfactual margin requiring correct graph topology to improve gold
   schema-token log probability over a destination-corrupted graph.

The textual prompt omits foreign-key lines by default. Exact schema names and
types remain visible, while graph topology has a distinct neural path.

## Server training command

The command below uses the files produced by the verified Stage 13-A/13-B
pipeline. It assumes the Stage 13-B best checkpoint reported in the previous
experiment.

```bash
CUDA_VISIBLE_DEVICES=0,1 python src/training/stage13c_train_static_graph_adapter.py \
  --model-path /data/1_pretrained_models/Qwen2.5-Coder-32B-Instruct \
  --graph-checkpoint experiments/stage13b_typed_ra_decoder_rgta_seed42/typed_ra_decoder.pt \
  --graph-summary experiments/stage13b_typed_ra_decoder_rgta_seed42/training_summary.json \
  --train-graph-file experiments/stage13a_dsg_data_typefix1/train_examples.jsonl \
  --dev-graph-file experiments/stage13a_dsg_data_typefix1/dev_examples.jsonl \
  --embedding-cache-dir experiments/stage13b_embedding_cache_typefix1_qwen3_06b \
  --output-dir experiments/stage13c_static_graph_adapter_seed42 \
  --train-limit 1000 \
  --dev-limit 100 \
  --epochs 3 \
  --lr 2e-4 \
  --gradient-accumulation-steps 8 \
  --adapter-layer-fractions 0.25,0.5,0.75,1.0 \
  --alignment-weight 0.1 \
  --counterfactual-weight 0.5 \
  --graph-device cuda:0 \
  --device-map auto \
  --dtype bfloat16 \
  --trust-remote-code
```

Do not pass `--prompt-includes-foreign-keys` in the first controlled run.

## Acceptance criteria

The adapter is not considered successful merely because total loss falls.
Require all of the following:

- `mean_alignment_recall@1` rises above its initial/random level;
- `mean_schema_logprob_gain` is positive on dev;
- `mean_schema_improvement_rate` is greater than 0.5;
- the learned residual scale and update norm do not collapse to zero;
- the later generation test shows different outputs for correct and corrupted
  graphs and better execution accuracy for the correct graph.

If alignment succeeds, Stage 13-C-B will add autoregressive generation and
correct/zero/corrupted graph intervention. Only after that result should the
upper RGTA layer or an LLM LoRA be unfrozen.
