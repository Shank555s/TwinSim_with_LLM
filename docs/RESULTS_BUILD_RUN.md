# TwinSim Research Report

_Generated 2026-10-04 09:55 from the TwinSim database. All numbers are measured, none are typed in._

## Research question

Can a custom language model trained from scratch improve its behavioural fidelity to predefined personas through iterative feedback-driven training?

## Methodology

**Architecture.** Decoder-only Transformer implemented from scratch (PyTorch `llm/model.py`; NumPy reference executor `llm/numpy_backend.py`). 4 pre-LayerNorm blocks, 4 heads, embedding 128, MLP 512 (tanh-GELU), learned positional embeddings, context 384 tokens, tied LM head. 973,568 parameters, randomly initialised. Trained with the `numpy` backend.

**Tokenizer.** Byte-level BPE trained on the project corpus: vocabulary 1024 (4 special + 8 structural tags + 256 bytes + merges), version tok-3f3dd7278b.

**Training.** Stage 1 (causal LM pre-training): 300 steps, 614,400 tokens, final validation loss 0.782. Stage 2 (persona specialisation, loss on response tokens only): 1200 steps, 1,442,222 tokens, final validation loss 0.297. AdamW, warm-up + cosine LR, gradient clipping 1.0.

**Data.** dataset-v1: SYNTHETIC development corpus generated from templates (8000 persona examples, 6000 pre-training documents, 400 validation examples on held-out scenarios). Not human data.

**Persona Conditioning.** Persona and scenario are serialised into tagged text (<persona> key=value lines </persona> <scenario> ... </scenario> <response>); the model continues after <response>.

**Scenarios.** 68 scenarios in 4 domains (economic, political, health, social), split into train / validation / benchmark.

**Evaluation.** Rubric evaluator `rubric-lexicon-v1` (no language model): lexicon cues -> observed Big-5 and political lean (0-10); axis fidelity = 10 - |intended - observed|; style match from surface features; overall = weighted mean x 10 (political weight 2). Degenerate output scores 0. Failure modes detected by explicit rules.

**Feedback Loop.** Each iteration: best-of-N sampling on training prompts -> evaluator ranks samples -> best samples above the threshold become weighted examples -> fine-tune on a replay mixture (new feedback / older feedback / baseline corpus) -> new checkpoint -> fixed benchmark -> promoted only if acceptance criteria pass.

**Benchmark.** 6 personas x 8 scenarios x repetitions, fixed seeds, temperature 0.7, top-k 40, top-p 0.95. Never used for training.

**Comparison.** Baseline Twin = TwinSim-v0 (frozen). Adaptive Twin = the lineage trained from feedback.

## Results

| Model | Lineage | Update | Status | Fidelity | Stability | Consistency | Failure rate | Val loss | Paired vs parent | Examples |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TwinSim-v0 | baseline | bootstrap | baseline | 77.54 | 84.24 | 68.23 | 45.8% | 0.297 | — | 0 |
| TwinSim-v1 | adaptive | feedback | rejected | 75.74 | 84.13 | 76.04 | 56.2% | 0.327 | -1.80 ± 1.11 (z -1.6) | 14 |
| TwinSim-v2 | adaptive | feedback | rejected | 76.03 | 85.27 | 70.57 | 50.0% | 0.333 | -1.51 ± 1.21 (z -1.2) | 32 |
| TwinSim-v3 | adaptive | feedback | rejected | 77.43 | 83.92 | 77.60 | 41.7% | 0.344 | -0.12 ± 1.00 (z -0.1) | 45 |
| TwinSim-v4 | adaptive | feedback | archived | 77.61 | 84.80 | 81.51 | 49.0% | 0.325 | +0.07 ± 0.74 (z 0.1) | 108 |
| TwinSim-v5 | adaptive | feedback | rejected | 77.36 | 84.56 | 77.34 | 50.0% | 0.322 | -0.24 ± 0.61 (z -0.4) | 177 |
| TwinSim-v6 | adaptive | feedback | archived | 77.62 | 85.01 | 72.66 | 47.9% | 0.323 | +0.01 ± 0.74 (z 0.0) | 236 |
| TwinSim-v7 | adaptive | feedback | rejected | 76.78 | 84.36 | 72.66 | 50.0% | 0.333 | -0.85 ± 1.43 (z -0.6) | 364 |
| TwinSim-v8 | adaptive | feedback | rejected | 77.07 | 85.94 | 72.14 | 57.3% | 0.334 | -0.55 ± 0.96 (z -0.6) | 494 |
| TwinSim-v9 | adaptive | feedback | rejected | 76.20 | 87.44 | 75.78 | 51.0% | 0.324 | -1.42 ± 1.15 (z -1.2) | 628 |
| TwinSim-v10 | adaptive | feedback | rejected | 77.20 | 84.79 | 67.97 | 43.8% | 0.326 | -0.43 ± 0.67 (z -0.6) | 757 |
| TwinSim-v11 | adaptive | control | active | 79.37 | 83.89 | 64.84 | 49.0% | 0.306 | +1.74 ± 0.65 (z 2.7) | 757 |
| TwinSim-v12 | adaptive | feedback | rejected | 78.02 | 86.09 | 78.65 | 50.0% | 0.335 | -1.35 ± 1.10 (z -1.2) | 791 |

### Learning experiment

- Initial fidelity: 77.54
- Final fidelity: 77.62
- Improvement: +0.08 points
- Initial / final stability: 84.24 / 85.01
- Initial / final failure rate: 45.8% / 47.9%
- Candidates accepted / rejected: 0 / 4

### Failure modes (active model, benchmark)

- Blue-Shift Bias: 20.8%
- Temporal Inconsistency: 20.8%
- Red-Shift Bias: 18.8%
- Generative Exaggeration: 3.1%

## Conclusion

- On the fixed benchmark, mean fidelity rose from 77.54 (TwinSim-v0, frozen baseline) to 79.37 (TwinSim-v11), a change of +1.82 points over 3 accepted update(s) (9 rejected). Paired over 48 identical benchmark items the change is +1.82 ± 0.68 (z = 2.7), so it is larger than benchmark noise.
- Twin stability 84.24 -> 83.89; failure rate 45.8% -> 49.0%.
- The active model descends from the baseline through: TwinSim-v4 (feedback), TwinSim-v6 (feedback), TwinSim-v11 (control).
- At least one accepted step is a CONTROL update (same training budget on the baseline corpus, no feedback). Gains from that step come from additional supervised training, not from feedback, and must not be attributed to feedback-driven learning.
- Feedback-trained candidates: 11; with a significant paired improvement over their parent (z >= 2): 0. Mean paired change -0.74 points.
- Control (no-feedback) candidates: TwinSim-v11 +1.74 ± 0.65 (z = 2.7).
- Caveat: improvement is measured with the same rubric evaluator that selects training examples, so part of any gain may be the model learning the rubric's cues rather than the persona. 0 human evaluation(s) have been recorded; human validation is still needed.
- All persona training data is synthetic; results show the pipeline works on a controlled development corpus and do not establish fidelity to real individuals.

## Status of claims

| Item | State |
|---|---|
| From-scratch tokenizer, Transformer, training loop, checkpoints, inference | Implemented |
| Model versions trained | Trained (13) |
| Fixed benchmark | Evaluated (14 runs) |
| Human evaluation | Implemented, no ratings yet |
| Persona training data | Placeholder (synthetic development corpus) |
| Fidelity to real people | Hypothetical (not tested) |

