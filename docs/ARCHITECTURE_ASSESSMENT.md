# TwinSim architecture assessment

## What the original project was
A single-file React app (`src/App.jsx`) that sent persona prompts to Groq's hosted Llama‑3.3‑70B and
scored the answers with a second LLM call. There was no model of its own, no training, no persistence
beyond the browser, and results depended on an external API key.

## What it is now
| Layer | Implementation |
|---|---|
| Tokenizer | Byte-level BPE trained on the project corpus (`backend/llm/tokenizer.py`), with `<pad> <bos> <eos> <unk>` and structural tags |
| Model | Decoder-only pre-LN Transformer written from scratch: PyTorch (`llm/model.py`) and a NumPy executor with hand-written backprop and a KV cache (`llm/numpy_backend.py`) |
| Training | AdamW, warm-up + cosine LR, gradient clipping, prompt-masked loss, early stopping, resumable checkpoints (`llm/training.py`, `llm/checkpoint.py`) |
| Data | Synthetic development corpus (`data/synthetic.py`, clearly labelled) plus an ingestion hook for real corpora and persona data (`data/ingestion.py`) |
| Evaluator | Rubric/lexicon scorer with 7 axes, a validity gate and 6 failure detectors; no language model (`evaluation/`) |
| Learning loop | Best-of-N harvest → replay mixture (new / replay / baseline anchor) → fine-tune → fixed benchmark → strict acceptance rule with paired statistics (`learning/`) |
| Storage | SQLite: personas, scenarios, datasets, model versions, checkpoints, training runs, trials, evaluations, feedback, training examples, human ratings, history, jobs (`database/`) |
| API | FastAPI, one background worker thread, SSE stream for live trials (`api/`, `main.py`) |
| UI | React 19, no UI/chart libraries; every view renders backend data (`src/`) |

## Known limitations
* **Synthetic data.** The persona corpus is template-generated. Results show the pipeline works on a
  controlled corpus; they say nothing about fidelity to real people.
* **Evaluator circularity.** The same rubric selects training examples and measures improvement, so a
  gain can partly reflect the model learning the rubric's cues. Human ratings and control updates exist
  to check this.
* **Small models.** The `dev` profile (~1M parameters) trains on a laptop CPU. Larger profiles exist but
  need a GPU and real data to be meaningful.
* **One worker.** Long jobs run one at a time. A job started from the CLI is not visible in the server's
  live status panel until it writes to the database.
