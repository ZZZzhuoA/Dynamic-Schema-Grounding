# Stage 12-A: Dynamic RGTA–LLM Neural Adapter

## Objective

Test whether dynamic schema grounding improves final SQL execution accuracy when it is injected into the LLM hidden state, rather than serialized as a reduced-schema prompt.

The controlled comparison is:

- `zero`: complete schema prompt and frozen base LLM, with neural grounding injection disabled;
- `none`: the same prompt and base LLM, with the trained dynamic RGTA adapter enabled.

This isolates the effect of neural grounding. Both conditions retain the complete schema, so a missed GNN candidate does not make a gold column physically unavailable to the LLM.

## Architecture

```text
complete database schema ----------------------> frozen causal LLM
                                                       |
question + generated partial SQL                       | hidden states
               |                                       v
               +--> frozen independent operation-RGTA belief
                                  |
                         weighted schema tokens
                                  |
                      cross-attention adapter
                                  |
                    uncertainty/route steering
                                  |
                         selected decoder layers
                                  |
                              next SQL token
```

The operation state is inferred from the current SQL prefix (`SELECT`, `JOIN`, `WHERE`, `GROUP BY`, or `ORDER BY`). The controller recomputes a schema belief from the actual generated prefix. It does not use gold SQL or gold schema labels at inference time.

## Training

The base LLM and the independent Stage 11 controller are frozen. Only cross-attention and steering adapters are optimized with teacher-forced SQL token loss. Gold SQL is used only to form the training trajectory and target tokens.

Adapters are inserted at configurable decoder-depth fractions. Their output scales are initialized to zero, making the initial model exactly equivalent to the frozen base LLM.

## Evaluation protocol

1. Run a 20-example smoke train.
2. Generate the same 20 examples with `none` and `zero` interventions.
3. Require non-identical SQL outputs for at least some examples; otherwise the adapter has no measurable behavioral effect.
4. Compare EX and execution success.
5. If the smoke test is functional, train on all available corrected training examples and evaluate all 1,534 BIRD dev examples.

The main claim is supported only if `none` improves full-dev EX over `zero` across seeds. Conditional grounding recall is diagnostic, not the final success metric.

## Decision rule

- Continue Stage 12 if neural injection produces stable positive EX and its random/negated interventions degrade performance.
- If EX is unchanged or worse, do not return to prompt selection. Replace token-level steering with a stronger structure, such as relation-algebra plan decoding with schema recovery or a jointly trained constrained decoder.

