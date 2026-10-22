"""Auto-generated research report (section 42). Every number is read from the database at
request time; the conclusion is produced by explicit rules over those numbers, so it can only
claim an improvement that the benchmark actually measured."""
from __future__ import annotations

import time

from database.db import Database
from database.repo import active_model, baseline_model, latest_metrics
from evaluation.fidelity import EVALUATOR_VERSION
from learning.benchmark import BENCHMARK_GENERATION, benchmark_items

from .metrics import failure_analysis, learning_curve, paired_vs_baseline

RESEARCH_QUESTION = ("Can a custom language model trained from scratch improve its behavioural fidelity to "
                     "predefined personas through iterative feedback-driven training?")


def _f(x, nd=2, pct=False):
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.{nd}f}"


def build_report(db: Database) -> dict:
    base, act = baseline_model(db), active_model(db)
    curve = learning_curve(db)
    personas, scenarios = benchmark_items(db)
    learn = db.one("SELECT * FROM experiments WHERE kind='learning' AND status='completed' ORDER BY created_at DESC LIMIT 1")
    runs = db.query("SELECT kind, steps, tokens_seen, final_train_loss, final_val_loss, output_model, backend, device "
                    "FROM training_runs WHERE status='completed' ORDER BY id")
    n_human = db.scalar("SELECT COUNT(*) FROM human_evaluations") or 0
    human_mean = db.scalar("SELECT AVG(score) FROM human_evaluations")
    ex_by_src = {r["source"]: r["n"] for r in db.query(
        "SELECT source, COUNT(*) AS n FROM training_examples WHERE example_type!='retired' GROUP BY source")}
    bench_runs = db.scalar("SELECT COUNT(*) FROM experiments WHERE kind='benchmark' AND status='completed'") or 0
    ds = db.one("SELECT * FROM datasets WHERE kind='bootstrap' ORDER BY version LIMIT 1")

    method = {}
    if base:
        mc = base["model_config"]
        method["architecture"] = (
            f"Decoder-only Transformer implemented from scratch (PyTorch `llm/model.py`; NumPy reference executor "
            f"`llm/numpy_backend.py`). {mc['num_layers']} pre-LayerNorm blocks, {mc['num_heads']} heads, embedding "
            f"{mc['embedding_dim']}, MLP {mc['embedding_dim'] * mc['mlp_ratio']} (tanh-GELU), learned positional "
            f"embeddings, context {mc['context_length']} tokens, tied LM head. {base['parameters']:,} parameters, "
            f"randomly initialised. Trained with the `{base['backend']}` backend.")
        method["tokenizer"] = (f"Byte-level BPE trained on the project corpus: vocabulary {mc['vocab_size']} "
                               f"(4 special + 8 structural tags + 256 bytes + merges), version {base['tokenizer_version']}.")
    s1 = next((r for r in runs if r["kind"] == "pretrain"), None)
    s2 = next((r for r in runs if r["kind"] == "persona"), None)
    if s1 and s2:
        method["training"] = (
            f"Stage 1 (causal LM pre-training): {s1['steps']} steps, {s1['tokens_seen']:,} tokens, final validation "
            f"loss {_f(s1['final_val_loss'], 3)}. Stage 2 (persona specialisation, loss on response tokens only): "
            f"{s2['steps']} steps, {s2['tokens_seen']:,} tokens, final validation loss {_f(s2['final_val_loss'], 3)}. "
            f"AdamW, warm-up + cosine LR, gradient clipping 1.0.")
    if ds:
        c = ds["composition"]
        method["data"] = (f"{ds['id']}: SYNTHETIC development corpus generated from templates "
                          f"({c.get('persona_train')} persona examples, {c.get('pretrain_docs')} pre-training documents, "
                          f"{c.get('persona_val')} validation examples on held-out scenarios). Not human data.")
    method["persona_conditioning"] = ("Persona and scenario are serialised into tagged text "
                                      "(<persona> key=value lines </persona> <scenario> ... </scenario> <response>); "
                                      "the model continues after <response>.")
    method["scenarios"] = (f"{db.scalar('SELECT COUNT(*) FROM scenarios WHERE deleted=0')} scenarios in 4 domains "
                           f"(economic, political, health, social), split into train / validation / benchmark.")
    method["evaluation"] = (f"Rubric evaluator `{EVALUATOR_VERSION}` (no language model): lexicon cues -> observed "
                            "Big-5 and political lean (0-10); axis fidelity = 10 - |intended - observed|; style "
                            "match from surface features; overall = weighted mean x 10 (political weight 2). "
                            "Degenerate output scores 0. Failure modes detected by explicit rules.")
    method["feedback_loop"] = ("Each iteration: best-of-N sampling on training prompts -> evaluator ranks samples -> "
                               "best samples above the threshold and free of failure modes become weighted examples -> "
                               "fine-tune on a replay mixture (new feedback / older feedback / baseline corpus) -> new "
                               "checkpoint -> fixed benchmark -> promoted only if acceptance criteria pass.")
    method["benchmark"] = (f"{len(personas)} personas x {len(scenarios)} scenarios x repetitions, fixed seeds, "
                           f"temperature {BENCHMARK_GENERATION.temperature}, top-k {BENCHMARK_GENERATION.top_k}, "
                           f"top-p {BENCHMARK_GENERATION.top_p}. Never used for training.")
    method["comparison"] = (f"Baseline Twin = {base['name'] if base else 'n/a'} (frozen). Adaptive Twin = the lineage "
                            "trained from feedback. Control updates (same budget, no feedback) separate the effect of "
                            "feedback from the effect of training longer.")

    results = {"models": curve, "learning_experiment": (learn or {}).get("summary"),
               "failures": failure_analysis(db), "human_evaluations": {"n": n_human, "mean_score": human_mean},
               "training_examples": ex_by_src, "benchmark_runs": bench_runs}

    conclusion = []
    bm = latest_metrics(db, base["name"]) if base else None
    am = latest_metrics(db, act["name"]) if act else None
    if not base or not bm:
        conclusion.append("No benchmarked model exists yet, so there are no results to report.")
    else:
        adaptive = [r for r in curve if r["lineage"] == "adaptive" and r["fidelity"] is not None]
        accepted = [r for r in adaptive if r["status"] in ("active", "archived")]
        if not adaptive:
            conclusion.append(f"Only the baseline {base['name']} has been benchmarked (fidelity {_f(bm['fidelity'])}). "
                              "No feedback-trained version exists yet, so the research question is not yet answered.")
        else:
            d = am["fidelity"] - bm["fidelity"]
            pv = paired_vs_baseline(db, act["name"])
            sig = pv is not None and abs(pv["z"]) >= 2
            if act["name"] == base["name"]:
                conclusion.append(f"{len(adaptive)} candidate(s) were produced but none passed the acceptance "
                                  f"criteria; the frozen baseline {base['name']} remains active. On this evidence, "
                                  "training did NOT improve benchmark fidelity.")
            elif d > 1.0 and sig:
                conclusion.append(f"On the fixed benchmark, mean fidelity rose from {_f(bm['fidelity'])} "
                                  f"({base['name']}, frozen baseline) to {_f(am['fidelity'])} ({act['name']}), "
                                  f"a change of {d:+.2f} points over {len(accepted)} accepted update(s) "
                                  f"({len(adaptive) - len(accepted)} rejected). Paired over {pv['n']} identical "
                                  f"benchmark items the change is {pv['mean']:+.2f} ± {pv['se']:.2f} (z = {pv['z']:.1f}), "
                                  "so it is larger than benchmark noise.")
            elif d > 1.0:
                conclusion.append(f"Mean fidelity rose from {_f(bm['fidelity'])} to {_f(am['fidelity'])} ({d:+.2f}), "
                                  + (f"but paired over identical items the change is {pv['mean']:+.2f} ± {pv['se']:.2f} "
                                     f"(z = {pv['z']:.1f}); this is not clearly larger than benchmark noise."
                                     if pv else "with no paired comparison available."))
            elif d >= -1.0:
                conclusion.append(f"Fidelity changed by only {d:+.2f} points ({_f(bm['fidelity'])} -> "
                                  f"{_f(am['fidelity'])}); this is within run-to-run noise and is not evidence of "
                                  "improvement.")
            else:
                conclusion.append(f"Fidelity decreased by {d:.2f} points; training hurt benchmark fidelity.")
            if bm.get("stability") is not None and am.get("stability") is not None:
                conclusion.append(f"Twin stability {_f(bm['stability'])} -> {_f(am['stability'])}; failure rate "
                                  f"{_f(bm['failure_rate'], pct=True)} -> {_f(am['failure_rate'], pct=True)}.")
        chain, kinds, name = [], [], act["name"] if act else None
        while name and name != base["name"]:
            m = db.one("SELECT name, parent, training_run_id FROM model_versions WHERE name=?", (name,))
            run = db.one("SELECT kind FROM training_runs WHERE id=?", (m["training_run_id"],)) if m else None
            chain.append(name)
            kinds.append(run["kind"] if run else "?")
            name = m["parent"] if m else None
        if chain:
            ups = ", ".join(f"{n} ({k})" for n, k in zip(reversed(chain), reversed(kinds)))
            conclusion.append(f"The active model descends from the baseline through: {ups}.")
            if "control" in kinds:
                conclusion.append("At least one accepted step is a CONTROL update (same training budget on the baseline "
                                  "corpus, no feedback). Gains from that step come from additional supervised training, "
                                  "not from feedback, and must not be attributed to feedback-driven learning.")
        fb_kind = {r["model"]: r["kind"] for r in curve}
        fb_only = [r for r in curve if fb_kind.get(r["model"]) in ("feedback", "micro") and r["paired_vs_parent"]]
        if fb_only:
            sig_fb = [r for r in fb_only if r["paired_vs_parent"]["z"] >= 2]
            conclusion.append(f"Feedback-trained candidates: {len(fb_only)}; with a significant paired improvement over "
                              f"their parent (z >= 2): {len(sig_fb)}. Mean paired change "
                              f"{sum(r['paired_vs_parent']['mean'] for r in fb_only) / len(fb_only):+.2f} points.")
        ctl = [r for r in curve if fb_kind.get(r["model"]) == "control" and r["paired_vs_parent"]]
        if ctl:
            conclusion.append("Control (no-feedback) candidates: " + "; ".join(
                f"{r['model']} {r['paired_vs_parent']['mean']:+.2f} ± {r['paired_vs_parent']['se']:.2f} "
                f"(z = {r['paired_vs_parent']['z']:.1f})" for r in ctl) + ".")
        conclusion.append("Caveat: improvement is measured with the same rubric evaluator that selects training "
                          "examples, so part of any gain may be the model learning the rubric's cues rather than "
                          f"the persona. {n_human} human evaluation(s) have been recorded"
                          + (f" (mean {human_mean:.1f}/10)." if n_human else "; human validation is still needed."))
        conclusion.append("All persona training data is synthetic; results show the pipeline works on a controlled "
                          "development corpus and do not establish fidelity to real individuals.")

    status = [
        {"item": "From-scratch tokenizer, Transformer, training loop, checkpoints, inference", "state": "Implemented"},
        {"item": "Model versions trained", "state": f"Trained ({len(curve)})" if curve else "Not yet"},
        {"item": "Fixed benchmark", "state": f"Evaluated ({bench_runs} runs)" if bench_runs else "Not yet run"},
        {"item": "Human evaluation", "state": f"Evaluated ({n_human} ratings)" if n_human else "Implemented, no ratings yet"},
        {"item": "Persona training data", "state": "Placeholder (synthetic development corpus)"},
        {"item": "Fidelity to real people", "state": "Hypothetical (not tested)"},
    ]
    return {"generated_at": time.time(), "title": "TwinSim Research Report", "research_question": RESEARCH_QUESTION,
            "methodology": method, "results": results, "conclusion": conclusion, "status": status}


def report_markdown(r: dict) -> str:
    L = [f"# {r['title']}", "", f"_Generated {time.strftime('%Y-%m-%d %H:%M', time.localtime(r['generated_at']))} "
         "from the TwinSim database. All numbers are measured, none are typed in._", "",
         "## Research question", "", r["research_question"], "", "## Methodology", ""]
    for k, v in r["methodology"].items():
        L += [f"**{k.replace('_', ' ').title()}.** {v}", ""]
    L += ["## Results", "", "| Model | Lineage | Update | Status | Fidelity | Stability | Consistency | Failure rate | Val loss | Paired vs parent | Examples |",
          "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in r["results"]["models"]:
        pv = m.get("paired_vs_parent")
        pvs = f"{pv['mean']:+.2f} ± {pv['se']:.2f} (z {pv['z']:.1f})" if pv else "n/a"
        L.append(f"| {m['model']} | {m['lineage']} | {m['kind']} | {m['status']} | {_f(m['fidelity'])} | {_f(m['stability'])} | "
                 f"{_f(m['consistency'])} | {_f(m['failure_rate'], pct=True)} | {_f(m['val_loss'], 3)} | {pvs} | {m['training_examples']} |")
    le = r["results"].get("learning_experiment")
    if le:
        L += ["", "### Latest learning experiment", "", f"- Initial fidelity: {_f(le.get('initial_fidelity'))}",
              f"- Final fidelity: {_f(le.get('final_fidelity'))}", f"- Improvement: {le.get('improvement', 0):+.2f} points",
              f"- Initial / final stability: {_f(le.get('initial_stability'))} / {_f(le.get('final_stability'))}",
              f"- Initial / final failure rate: {_f(le.get('initial_failure_rate'), pct=True)} / "
              f"{_f(le.get('final_failure_rate'), pct=True)}",
              f"- Candidates accepted / rejected: {le.get('accepted')} / {le.get('rejected')}"]
    fa = r["results"]["failures"]
    if fa.get("current"):
        L += ["", "### Failure modes (active model, benchmark)", ""]
        for c, v in sorted(fa["current"].items(), key=lambda kv: -kv[1]):
            L.append(f"- {fa['labels'].get(c, c)}: {v * 100:.1f}%")
    L += ["", "## Conclusion", ""] + [f"- {c}" for c in r["conclusion"]]
    L += ["", "## Status of claims", "", "| Item | State |", "|---|---|"] + \
         [f"| {s['item']} | {s['state']} |" for s in r["status"]]
    return "\n".join(L) + "\n"
