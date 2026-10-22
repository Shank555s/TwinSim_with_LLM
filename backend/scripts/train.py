"""Command-line entry point for training and experiments (same code paths as the web UI).

Examples (run from the backend/ folder):
    python scripts/train.py info
    python scripts/train.py bootstrap --profile dev
    python scripts/train.py bootstrap --profile research --persona-steps 20000     # GPU
    python scripts/train.py benchmark --model TwinSim-v0
    python scripts/train.py feedback                 # one harvest + train + benchmark iteration
    python scripts/train.py feedback --control       # ablation: same budget, no feedback
    python scripts/train.py learn --iterations 3     # full learning experiment
    python scripts/train.py auto                     # run one automatic learning cycle now (same as the background one)
    python scripts/train.py retention                # apply the storage limits now (experiments, replay, checkpoints)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import get_db  # noqa: E402
from database.repo import mark_interrupted, seed, worker_lock  # noqa: E402


class PrintCtx:
    """Minimal job context that prints progress to the terminal."""

    id = "cli"

    def __init__(self):
        self.last = 0.0

    def check(self):
        pass

    def cancelled(self):
        return False

    def stage(self, stage, progress=None, message="", data=None, force=True):
        now = time.time()
        if force or now - self.last > 5:
            p = f"{progress * 100:5.1f}%" if progress is not None else "      "
            print(f"[{time.strftime('%H:%M:%S')}] {p} {stage}: {message}", flush=True)
            self.last = now


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    b = sub.add_parser("bootstrap")
    b.add_argument("--profile", default="dev")
    b.add_argument("--pretrain-steps", type=int)
    b.add_argument("--persona-steps", type=int)
    b.add_argument("--persona-examples", type=int, default=8000)
    b.add_argument("--pretrain-docs", type=int, default=6000)
    b.add_argument("--batch-size", type=int)
    b.add_argument("--seed", type=int, default=1337)
    bm = sub.add_parser("benchmark")
    bm.add_argument("--model", required=True)
    bm.add_argument("--reps", type=int, default=2)
    f = sub.add_parser("feedback")
    f.add_argument("--no-harvest", action="store_true")
    f.add_argument("--steps", type=int)
    f.add_argument("--control", action="store_true", help="ablation: same steps on the baseline corpus only")
    le = sub.add_parser("learn")
    le.add_argument("--iterations", type=int, default=3)
    sub.add_parser("auto")
    sub.add_parser("retention")
    cfgp = sub.add_parser("config")
    cfgp.add_argument("json", help='learning-config overrides, e.g. \'{"finetune_steps": 200}\'')
    args = ap.parse_args()

    db = get_db()
    seed(db)
    mark_interrupted(db)
    ctx = PrintCtx()
    if args.cmd in ("bootstrap", "benchmark", "feedback", "learn", "auto", "retention"):
        with worker_lock():
            run(args, db, ctx)
    else:
        run(args, db, ctx)


def run(args, db, ctx):
    if args.cmd == "info":
        from llm.backend import banner, create_runner
        from llm.config import model_config
        from llm.hardware import detect_hardware, recommend_profile
        print(json.dumps(detect_hardware(), indent=2))
        print(json.dumps(recommend_profile(), indent=2))
        print(banner(create_runner(model_config("dev"))))
        for m in db.query("SELECT name, status, lineage FROM model_versions ORDER BY version"):
            print(m)
    elif args.cmd == "bootstrap":
        from learning.trainer import bootstrap
        out = bootstrap(db, ctx, profile=args.profile, pretrain_steps=args.pretrain_steps,
                        persona_steps=args.persona_steps, n_persona_examples=args.persona_examples,
                        n_pretrain_docs=args.pretrain_docs, batch_size=args.batch_size, seed=args.seed)
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "benchmark":
        from learning.benchmark import run_benchmark
        out = run_benchmark(db, ctx, args.model, repetitions=args.reps)
        print(json.dumps({k: out[k] for k in ("fidelity", "stability", "consistency", "failure_rate", "failure_rates",
                                              "axes", "by_domain")}, indent=2, default=str))
    elif args.cmd == "feedback":
        from learning.trainer import feedback_update
        if args.control:
            from experiments.runner import learning_config
            out = feedback_update(db, ctx, kind="control", harvest_first=False,
                                  steps=args.steps or learning_config(db).finetune_steps,
                                  cfg_overrides={"new_feedback_ratio": 0.0, "replay_feedback_ratio": 0.0})
        else:
            out = feedback_update(db, ctx, harvest_first=not args.no_harvest, steps=args.steps)
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "learn":
        from learning.trainer import learning_experiment
        out = learning_experiment(db, ctx, iterations=args.iterations)
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "auto":
        from learning.auto import recover, run_cycle
        recover(db)
        print(json.dumps(run_cycle(db, ctx), indent=2, default=str))
    elif args.cmd == "retention":
        from experiments.runner import learning_config
        from learning.retention import run_retention
        print(json.dumps(run_retention(db, learning_config(db)), indent=2, default=str))
    elif args.cmd == "config":
        from llm.config import LearningConfig
        cur = LearningConfig.from_dict(db.get_setting("learning_config")).to_dict()
        cur.update(json.loads(args.json))
        db.set_setting("learning_config", LearningConfig.from_dict(cur).to_dict())
        print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
