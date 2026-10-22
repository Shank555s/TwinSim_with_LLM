# TwinSim Research Laboratory

TwinSim builds **digital twins of personas on a language model trained from scratch** and measures how
faithfully they behave. Then it tries to improve them through a gated feedback loop and reports
honestly whether that worked.

* No external or pretrained LLM and no API keys. The tokenizer, Transformer, training loop and inference are all in this repo.
* A transparent rubric evaluator (no LLM) scores every answer on 7 axes and flags 6 failure modes.
* Every model version is benchmarked on a fixed held-out set and promoted only if it beats its parent.
* Everything is stored in SQLite and shown in a React UI: models, datasets, runs, trials, decisions and history.
* **Learning is automatic.** Run experiments; failures and successes are turned into training signals and the model is
  updated in the background. A new version goes live only if it measurably beats the active one. See
  [Automatic continual learning](#automatic-continual-learning).

---

## Quick start

```bash
# backend (Python 3.10+)
cd backend
pip install -r requirements.txt        # torch is optional; without it the NumPy backend is used
uvicorn main:app --port 8000

# frontend (Node 18+), in another terminal
npm install
npm run dev                            # http://localhost:5173 (proxies /api to :8000)
```

A production build (`npm run build`) writes `dist/`, which the FastAPI app serves directly at
http://localhost:8000.

### First run
1. On the **Dashboard**, click **Train baseline model** (one time only). This trains the tokenizer, pre-trains (stage 1),
   persona-specialises (stage 2) and benchmarks `TwinSim-v0`. The `dev` profile takes about 40 minutes on a laptop CPU.
2. **Twin Builder**: describe a persona. **Scenario Lab**: run it and watch the tokens stream, the
   fidelity gauge, trait bars, drift and the "What went wrong?" diagnosis.
3. Keep running scenarios. Nothing needs to be started: learning happens in the background and the top bar shows
   "Learning update in progress…" / "Model updated". **Model Registry → Baseline vs Adaptive** compares versions on paired benchmark items.

CLI equivalents: `python backend/scripts/train.py bootstrap|benchmark|feedback|learn|auto|retention|config|info` (`auto` runs one automatic cycle now, `retention` applies the storage limits now).

---

## UI

Flat dark interface: solid surfaces, 1px borders, small corner radii and one orange accent, with no gradients, glows or decorative animation. Inter + JetBrains Mono (self-hosted), a hand-drawn icon set, no UI or chart libraries.

| Page | What it shows (all live backend data) |
|---|---|
| Dashboard | Count-up metrics, behavioural-fidelity curve over versions, model-evolution timeline, axis scores, persona drift, recent trials |
| Twin Builder | Identity + Big Five / politics sliders, live radar and plain-language signature of the inputs |
| Scenario Lab | The main workspace: a large setup panel (scenario library or your own text, twin with its persona summary, model, all generation controls) beside the live pipeline (Persona → Transformer → Evaluator → Failure analysis → Auto-learning, driven by server-sent events), token streaming, response, gauge, trait bars, drift plot, failure diagnosis, replay / re-evaluate / compare / save |
| Experiment Lab | N×N×N studies with a fidelity matrix and blind human evaluation |
| Twin DNA | Intended vs baseline vs selected model radar, drift, axis table, communication-style match |
| Failure Analysis | Distribution, rates over model versions, detector definitions, flagged examples |
| Model Registry | Version-control cards (activate / roll back / re-benchmark / details) and the Baseline-vs-Adaptive comparison with paired statistics |
| History | Append-only event log grouped by day, searchable |
| Research Report | Methodology, results, failures, conclusion (including negative results), exports |

The top bar shows the active model, a small learning status, compute and notifications. There is no learning console: learning is not a user action. The command center polls `/api/status`
(LLM, evaluator, database, compute, training) every 2.5 s. Animation is only used to show state
(streaming text, an active pipeline stage, progress), and is turned off by `prefers-reduced-motion` or the toggle in Settings.
The sidebar links to a Privacy policy and Terms of use page; fill in your name and contact email in `src/site.js`.

## Automatic continual learning

```
experiment -> fidelity evaluation -> failure detection -> learning signal -> [queue]
   -> background cycle: corrective best-of-N on failed/weak experiments
   -> replay mixture (new + older feedback + baseline anchor) -> fine-tune a COPY of the active checkpoint
   -> candidate checkpoint -> fixed 96-item benchmark -> promotion rule -> promote | reject (active model unchanged)
```

* **Signal** (`learning/signals.py`): one compact record per experiment: fidelity, the 7 axes, which axes are weak,
  failure modes with a severity, the persona traits that matter, what to reinforce / reduce, and a training weight.
  `UNIQUE(trial_id)` means an experiment is never processed twice.
* **Failures drive training** (`learning/auto.py`): for each failed or weak experiment the active model is sampled N
  times on the *same* persona and scenario, the evaluator ranks the samples, and the best failure-free one becomes a
  weighted training example (`source = corrective`). Recurring failure modes are corrected first. High-fidelity runs
  already become positive examples (`learning/feedback.py`). Benchmark personas/scenarios are held out of learning.
* **Trigger**: `batch_trigger_trials` pending signals (10; `learning_mode = immediate` → 1), then `auto_cooldown_s`
  between cycles. One cycle at a time on the single worker thread; the HTTP API keeps serving experiments meanwhile.
* **Promotion** (`learning/promotion.py`): benchmark fidelity must rise by more than `min_improvement` **and** by at
  least `min_paired_z` standard errors on the identical paired items, validation loss and failure rate may not rise
  beyond their limits. Otherwise the candidate is kept as `rejected` and the active model does not change.
  Rollback = activate an older version in the Model Registry.
* **Crash safety**: training works on a copy; the checkpoint is complete on disk before the single `active_model`
  pointer flips; a promotion is refused if the active model changed during training; signals claimed by a dead process
  go back to the queue at startup (`learning/auto.recover`); a failing cycle is retried up to `max_signal_attempts`.

### Bounded storage (`learning/retention.py`, limits in the learning config)

| What | Limit (default) | Policy |
|---|---|---|
| Raw experiments (trial + evaluation + feedback) | `max_recent_experiments` (15) | a study counts as one. Older ones are compacted first: the signal is written and read back in the same transaction that deletes the raw rows; experiments being trained on are deferred |
| Replay buffer (`training_examples`) | `max_replay_examples` (400) | evicts from the most redundant (domain × persona profile) bucket first; never evicts untrained or human-written examples |
| Learning memory (`learning_signals`) | `max_memory_signals` (500) | consumed signals keep only compact fields; the lowest-value ones are rolled up into counters |
| Checkpoints with weights | `max_active_checkpoints` (5) | always keeps the baseline, the active model, its parent (rollback target) and any in-flight candidate; pruned versions keep their metadata and benchmark scores |
| Logs | 3000 events, 200 jobs, 100 cycles | oldest removed |

Old knowledge is carried by the checkpoint chain (each version is fine-tuned from the previous active one) plus the
replay mixture, which also contains the baseline corpus as an anchor against forgetting.

Honest limits: the model is tiny and the rubric is the same one that selects training data, so most candidates are
rejected until a gain survives the noise test. Only measured benchmark results are shown. Benchmark personas
(p1-p6) never train the model, so use your own twins in the Scenario Lab to feed learning.

## Publishing on your own domain
See `docs/DEPLOY.md` (Docker + Caddy on a VPS, DNS and HTTPS). Set `TWINSIM_PUBLIC=1` (the Dockerfile does) so
visitors can build twins, run scenarios and rate answers but cannot start training, change settings or switch
models. Run training on the server with `scripts/train.py`; the status bar shows command-line jobs as running.

---

## Results of the included run (`runtime/`, dev profile, CPU, NumPy backend)

`TwinSim-v0`: 973,568 parameters (4 layers, 4 heads, d=128, context 384, vocab 1024).
Stage 1 was 300 steps (val loss 0.782). Stage 2 was 1,200 steps (val loss 0.297).

| Model | Update | Benchmark fidelity | Paired Δ vs parent (48 items) | Status |
|---|---|---|---|---|
| TwinSim-v0 | bootstrap | **77.54** | – | baseline (active) |
| TwinSim-v1 | feedback (112 new examples) | 75.53 | −2.01 ± 1.36 (z −1.48) | rejected |
| TwinSim-v2 | feedback | 75.52 | −2.03 ± 1.22 (z −1.67) | rejected |
| TwinSim-v3 | feedback | 77.27 | −0.27 ± 1.10 (z −0.24) | rejected |
| TwinSim-v4 | control (no feedback, 400 steps) | 76.95 | −0.59 ± 1.37 (z −0.43) | rejected |

**Conclusion of this run: feedback training did not improve benchmark fidelity.** No candidate passed
the acceptance rule (> +1.0 point), and no change was beyond noise. The baseline stays active. For
reference, on the same benchmark the evaluator gives the template oracle 88.6, a bland non-committal
answer 75.5 and persona-shuffled oracle answers 67.7. So the rubric does discriminate between personas,
but v0 sits only about 2 points above a bland answer, which leaves plenty of room to improve.

## Limitations
* Persona data is **synthetic** (template generated). Nothing here measures fidelity to real people.
  `backend/data/corpus/` and `backend/data/persona_data/` accept real data; see their READMEs.
* The same rubric selects training examples and measures improvement. Gains could reflect rubric cues,
  which is why there are control updates and blind human ratings.
* Small CPU models. The `research` profiles need a GPU and real data.

See `docs/ARCHITECTURE_ASSESSMENT.md` for the architecture.

## Tests
```bash
cd backend && python -m pytest       # 54 passed (PyTorch tests are skipped when torch is not installed)
```
