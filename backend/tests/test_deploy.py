"""Worker locks (server + CLI running together) and public mode."""
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from database.db import get_db
from database.repo import _lock_dir, mark_interrupted, other_worker_alive, worker_lock

BACKEND = Path(__file__).resolve().parent.parent


def test_live_worker_blocks_interrupt_marking():
    d = get_db()
    d.insert("experiments", {"id": "exp-live", "kind": "study", "name": "live", "status": "running", "created_at": time.time()})
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import sys, time; sys.path.insert(0, {str(BACKEND)!r})
        from database.repo import worker_lock
        with worker_lock():
            print('locked', flush=True); time.sleep(30)
    """)], stdout=subprocess.PIPE, text=True, env=os.environ.copy())
    try:
        assert holder.stdout.readline().strip() == "locked"
        assert other_worker_alive()
        assert mark_interrupted(d) == 0
        assert d.one("SELECT status FROM experiments WHERE id='exp-live'")["status"] == "running"
    finally:
        holder.kill()
        holder.wait()
    assert not other_worker_alive()  # dead holder's lock is cleaned up
    assert mark_interrupted(d) >= 1
    assert d.one("SELECT status FROM experiments WHERE id='exp-live'")["status"] == "interrupted"


def test_own_lock_is_ignored():
    with worker_lock():
        assert (_lock_dir() / str(os.getpid())).exists()
        assert not other_worker_alive()
    assert not (_lock_dir() / str(os.getpid())).exists()


def test_public_mode_blocks_heavy_actions(tmp_path):
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(BACKEND)!r})
        from fastapi.testclient import TestClient
        from main import app
        c = TestClient(app)
        print(c.post('/api/training/feedback', json={{}}).status_code,
              c.put('/api/learning/config', json={{}}).status_code,
              c.delete('/api/personas/p1').status_code,
              c.post('/api/models/x/activate', json={{}}).status_code,
              c.post('/api/personas', json={{'label': 'visitor'}}).status_code,
              c.get('/api/status').json()['public_mode'])
    """)
    env = {**os.environ, "TWINSIM_PUBLIC": "1", "TWINSIM_HOME": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert out.stdout.split() == ["403", "403", "403", "403", "200", "True"], out.stderr[-2000:]
