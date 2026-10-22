"""Background job queue (section 38). One worker thread runs long tasks (training, studies,
benchmarks) so the HTTP API stays responsive. Progress is persisted in the `jobs` table and the
frontend polls GET /api/jobs/{id}. Every stage update reflects real backend state."""
from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Callable

from database.db import Database, new_id

MAX_EVENTS = 400


class JobCancelled(Exception):
    pass


class JobContext:
    def __init__(self, manager: "JobManager", job_id: str):
        self.m, self.id = manager, job_id
        self._events: list[dict] = []
        self._last_flush = 0.0

    def cancelled(self) -> bool:
        return self.id in self.m.cancel_requests

    def check(self) -> None:
        if self.cancelled():
            raise JobCancelled()

    def stage(self, stage: str, progress: float | None = None, message: str = "", data: dict | None = None,
              force: bool = True) -> None:
        ev = {"t": time.time(), "stage": stage, "progress": progress, "message": message}
        if data:
            ev["data"] = data
        self._events.append(ev)
        self._events = self._events[-MAX_EVENTS:]
        now = time.time()
        if force or now - self._last_flush > 0.5:
            fields = {"stage": stage, "message": message, "events": self._events}
            if progress is not None:
                fields["progress"] = float(progress)
            self.m.db.update("jobs", "id", self.id, fields)
            self._last_flush = now


class JobManager:
    def __init__(self, db: Database):
        self.db = db
        self.q: "queue.Queue[tuple[str, Callable]]" = queue.Queue()
        self.cancel_requests: set[str] = set()
        self.current: str | None = None
        self._thread = threading.Thread(target=self._loop, name="twinsim-worker", daemon=True)
        self._thread.start()
        # jobs left 'running' by a previous process never finished
        from database.repo import mark_interrupted
        mark_interrupted(db)
        from database.repo import other_worker_alive
        stale = [] if other_worker_alive() else db.query("SELECT id FROM jobs WHERE status IN ('running','queued')")
        for j in stale:
            db.update("jobs", "id", j["id"], {"status": "failed", "error": "server restarted before the job finished",
                                              "finished_at": time.time()})

    def submit(self, type_: str, fn: Callable[[JobContext], dict], payload: dict | None = None) -> str:
        jid = new_id("job")
        self.db.insert("jobs", {"id": jid, "type": type_, "status": "queued", "stage": "queued", "progress": 0.0,
                                "message": "waiting for the worker", "payload": payload or {}, "created_at": time.time(),
                                "events": []})
        self.q.put((jid, fn))
        return jid

    def cancel(self, jid: str) -> bool:
        j = self.db.one("SELECT status FROM jobs WHERE id=?", (jid,))
        if not j or j["status"] not in ("queued", "running"):
            return False
        self.cancel_requests.add(jid)
        return True

    def busy(self) -> bool:
        return self.current is not None or not self.q.empty()

    def _loop(self) -> None:
        while True:
            jid, fn = self.q.get()
            if jid in self.cancel_requests:
                self.db.update("jobs", "id", jid, {"status": "cancelled", "finished_at": time.time()})
                continue
            self.current = jid
            ctx = JobContext(self, jid)
            self.db.update("jobs", "id", jid, {"status": "running", "started_at": time.time(), "stage": "starting"})
            try:
                from database.repo import worker_lock
                with worker_lock():
                    result = fn(ctx) or {}
                ctx.stage("done", 1.0, "completed")
                self.db.update("jobs", "id", jid, {"status": "completed", "result": result, "finished_at": time.time()})
            except JobCancelled:
                ctx.stage("cancelled", None, "cancelled by user")
                self.db.update("jobs", "id", jid, {"status": "cancelled", "finished_at": time.time()})
            except Exception as e:  # never pretend success
                tb = traceback.format_exc()
                ctx.stage("failed", None, f"{type(e).__name__}: {e}")
                self.db.update("jobs", "id", jid, {"status": "failed", "error": f"{type(e).__name__}: {e}\n{tb[-2000:]}",
                                                   "finished_at": time.time()})
                self.db.log_event("job_failed", f"{type(e).__name__}: {e}", "job", jid)
            finally:
                self.current = None
                self.cancel_requests.discard(jid)

    def wait(self, jid: str, timeout: float = 3600) -> dict:
        """Block until a job finishes (used by CLI scripts and tests)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.db.one("SELECT * FROM jobs WHERE id=?", (jid,))
            if j and j["status"] in ("completed", "failed", "cancelled"):
                return j
            time.sleep(0.2)
        raise TimeoutError(jid)
