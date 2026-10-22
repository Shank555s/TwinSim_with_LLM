"""SQLite persistence (section 44). One file, created on first start; schema is idempotent.

All access goes through `Database`, which serialises writes with a lock so the API threads
and the background training worker can share one connection safely.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOME = Path(os.environ.get("TWINSIM_HOME", REPO_ROOT / "runtime"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS personas (
    id TEXT PRIMARY KEY, label TEXT, data TEXT NOT NULL, source TEXT NOT NULL,
    is_benchmark INTEGER DEFAULT 0, created_at REAL, updated_at REAL, deleted INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS scenarios (
    id TEXT PRIMARY KEY, domain TEXT NOT NULL, issue TEXT, text TEXT NOT NULL, split TEXT NOT NULL,
    source TEXT NOT NULL, created_at REAL, deleted INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY, name TEXT, kind TEXT, version INTEGER, n_examples INTEGER, sha TEXT,
    composition TEXT, path TEXT, description TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS model_versions (
    name TEXT PRIMARY KEY, version INTEGER UNIQUE, parent TEXT REFERENCES model_versions(name),
    lineage TEXT, status TEXT, created_at REAL, checkpoint_path TEXT,
    dataset_id TEXT REFERENCES datasets(id), tokenizer_version TEXT, code_version TEXT,
    training_config TEXT, model_config TEXT, parameters INTEGER, backend TEXT, notes TEXT, decision TEXT,
    training_run_id INTEGER);
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT REFERENCES model_versions(name), path TEXT,
    weights_sha TEXT, size_bytes INTEGER, created_at REAL);
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, status TEXT, base_model TEXT, output_model TEXT,
    dataset_id TEXT REFERENCES datasets(id), config TEXT, started_at REAL, finished_at REAL, steps INTEGER,
    tokens_seen INTEGER, final_train_loss REAL, final_val_loss REAL, best_val_loss REAL, log TEXT,
    error TEXT, job_id TEXT, device TEXT, backend TEXT);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY, kind TEXT, name TEXT, model_version TEXT, config TEXT, status TEXT,
    created_at REAL, finished_at REAL, summary TEXT, job_id TEXT);
CREATE TABLE IF NOT EXISTS trials (
    id TEXT PRIMARY KEY, experiment_id TEXT REFERENCES experiments(id), persona_id TEXT,
    persona_snapshot TEXT, scenario_id TEXT, scenario_snapshot TEXT, model_version TEXT, seed INTEGER,
    gen_params TEXT, response TEXT, tokens_generated INTEGER, latency_ms REAL, repetition INTEGER,
    purpose TEXT, created_at REAL, training_status TEXT, tokenizer_version TEXT, backend TEXT);
CREATE INDEX IF NOT EXISTS idx_trials_model ON trials(model_version, purpose);
CREATE INDEX IF NOT EXISTS idx_trials_exp ON trials(experiment_id);
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trial_id TEXT REFERENCES trials(id), evaluator TEXT,
    evaluator_kind TEXT, overall REAL, axes TEXT, observed TEXT, intended TEXT, drift TEXT,
    failures TEXT, explanation TEXT, style_detail TEXT, created_at REAL);
CREATE INDEX IF NOT EXISTS idx_eval_trial ON evaluations(trial_id);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trial_id TEXT REFERENCES trials(id), source TEXT NOT NULL,
    overall REAL, axes TEXT, failures TEXT, explanation TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS training_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL, feedback_id INTEGER REFERENCES feedback(id),
    trial_id TEXT REFERENCES trials(id), persona TEXT, scenario TEXT, response TEXT, weight REAL,
    example_type TEXT, created_at REAL, used_count INTEGER DEFAULT 0, last_used_run INTEGER);
CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT REFERENCES model_versions(name),
    experiment_id TEXT REFERENCES experiments(id), fidelity REAL, stability REAL, consistency REAL,
    failure_rate REAL, val_loss REAL, train_loss REAL, avg_length REAL, regression_count INTEGER,
    details TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS human_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trial_id TEXT REFERENCES trials(id), rater TEXT, score INTEGER,
    blind INTEGER, corrected_response TEXT, notes TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS history_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, type TEXT, entity_type TEXT, entity_id TEXT,
    summary TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, type TEXT, status TEXT, stage TEXT, progress REAL, message TEXT, payload TEXT,
    result TEXT, error TEXT, created_at REAL, started_at REAL, finished_at REAL, events TEXT);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
-- Automatic continual learning. One compact record per experiment (trial_id is UNIQUE, so an experiment
-- is never turned into a learning signal twice). It is both the work queue (state pending/claimed) and,
-- once consumed and compacted, the bounded long-term learning memory (persona/scenario/response dropped).
CREATE TABLE IF NOT EXISTS learning_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trial_id TEXT UNIQUE, state TEXT NOT NULL, kind TEXT,
    cycle_id INTEGER, attempts INTEGER DEFAULT 0, harvested INTEGER DEFAULT 0, compacted INTEGER DEFAULT 0,
    domain TEXT, issue TEXT, model_version TEXT, created_at REAL,
    fidelity REAL, failure_score REAL, training_weight REAL,
    traits TEXT, axes TEXT, weak_axes TEXT, failures TEXT, successes TEXT, reinforce TEXT, reduce TEXT,
    expected_behavior TEXT, persona TEXT, scenario TEXT, response TEXT);
CREATE INDEX IF NOT EXISTS idx_signals_state ON learning_signals(state);
CREATE TABLE IF NOT EXISTS learning_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT, started_at REAL, finished_at REAL, n_signals INTEGER,
    n_corrective INTEGER, n_examples INTEGER, model_before TEXT, model_after TEXT, accepted INTEGER,
    fidelity_before REAL, fidelity_after REAL, failure_rate_before REAL, failure_rate_after REAL,
    experiments_incorporated INTEGER, decision TEXT, error TEXT, job_id TEXT);
"""

# Columns added after the first release: (table, column, DDL). Applied idempotently on open.
MIGRATIONS = [
    ("model_versions", "weights_pruned", "INTEGER DEFAULT 0"),
]

JSON_COLUMNS = {
    "data", "composition", "training_config", "model_config", "decision", "config", "log", "summary",
    "persona_snapshot", "scenario_snapshot", "gen_params", "axes", "observed", "intended", "drift", "failures",
    "explanation", "style_detail", "persona", "scenario", "details", "payload", "result", "events", "value",
    "traits", "weak_axes", "successes", "reinforce", "reduce",
}


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


class Database:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else HOME / "twinsim.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(SCHEMA)
            for table, col, ddl in MIGRATIONS:
                if col not in {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
            self.conn.commit()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _enc(v):
        return json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else v

    @staticmethod
    def _row(r: sqlite3.Row | None) -> dict | None:
        if r is None:
            return None
        d = dict(r)
        for k, v in d.items():
            if k in JSON_COLUMNS and isinstance(v, str):
                try:
                    d[k] = json.loads(v)
                except json.JSONDecodeError:
                    pass
        return d

    def execute(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.conn.execute(sql, [self._enc(p) for p in params])
            self.conn.commit()
            return cur

    @contextmanager
    def transaction(self):
        """All-or-nothing group of statements (used when raw data is replaced by its compact form).
        Inside the block use the yielded connection directly: `execute()` would commit early."""
        with self.lock:
            try:
                yield self.conn
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise

    def query(self, sql: str, params: tuple | list = ()) -> list[dict]:
        with self.lock:
            return [self._row(r) for r in self.conn.execute(sql, list(params)).fetchall()]

    def one(self, sql: str, params: tuple | list = ()) -> dict | None:
        with self.lock:
            return self._row(self.conn.execute(sql, list(params)).fetchone())

    def scalar(self, sql: str, params: tuple | list = ()):
        with self.lock:
            r = self.conn.execute(sql, list(params)).fetchone()
            return r[0] if r else None

    def insert(self, table: str, row: dict) -> int | str:
        cols = list(row)
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
        cur = self.execute(sql, [row[c] for c in cols])
        return row.get("id", cur.lastrowid)

    def update(self, table: str, key: str, key_value, fields: dict) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE {table} SET {sets} WHERE {key}=?", [*fields.values(), key_value])

    # ----------------------------------------------------------------- settings
    def get_setting(self, key: str, default=None):
        r = self.one("SELECT value FROM settings WHERE key=?", (key,))
        return default if r is None else r["value"]

    def set_setting(self, key: str, value) -> None:
        self.execute("INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, json.dumps(value)))

    # ------------------------------------------------------------------ history
    def log_event(self, type_: str, summary: str, entity_type: str | None = None, entity_id=None,
                  data: dict | None = None) -> None:
        self.insert("history_events", {"ts": time.time(), "type": type_, "entity_type": entity_type,
                                       "entity_id": None if entity_id is None else str(entity_id),
                                       "summary": summary, "data": data or {}})


_db: Database | None = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db


def set_db(db: Database) -> None:
    """Used by tests to point the app at a temporary database."""
    global _db
    _db = db
