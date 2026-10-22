"""Shared API state + error translation."""
from __future__ import annotations

import os

from fastapi import HTTPException

from database.db import get_db
from experiments.jobs import JobManager

_jobs: JobManager | None = None

# Public mode (TWINSIM_PUBLIC=1) is for a site on the open internet: see main.py.
PUBLIC_MODE = os.environ.get("TWINSIM_PUBLIC", "").lower() in ("1", "true", "yes")


def db():
    return get_db()


def jobs() -> JobManager:
    global _jobs
    if _jobs is None:
        _jobs = JobManager(get_db())
    return _jobs


def reset_jobs() -> None:
    global _jobs
    _jobs = None


def http_error(e: Exception) -> HTTPException:
    from llm.checkpoint import CheckpointError
    from llm.tokenizer import TokenizerError
    from twin.personas import PersonaError
    from twin.twin_engine import ModelNotTrained
    if isinstance(e, ModelNotTrained):
        return HTTPException(409, {"code": "model_not_trained", "message": str(e)})
    if isinstance(e, (CheckpointError, TokenizerError)):
        return HTTPException(500, {"code": "checkpoint_error", "message": str(e)})
    if isinstance(e, PermissionError):
        return HTTPException(403, {"code": "forbidden", "message": str(e)})
    if isinstance(e, KeyError):
        return HTTPException(404, {"code": "not_found", "message": f"not found: {e}"})
    if isinstance(e, (ValueError, PersonaError)):
        return HTTPException(422, {"code": "invalid", "message": str(e)})
    return HTTPException(500, {"code": "error", "message": f"{type(e).__name__}: {e}"})


def guarded(fn):
    """Run fn() and convert domain exceptions to HTTP errors (never report false success)."""
    try:
        return fn()
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise http_error(e)
