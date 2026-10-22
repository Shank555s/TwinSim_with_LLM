from __future__ import annotations

from fastapi import APIRouter, Body

from database import repo
from twin.scenarios import DOMAINS

from .deps import db, guarded

router = APIRouter(prefix="/api/scenarios", tags=["scenarios"])


@router.get("")
def list_scenarios(split: str | None = None):
    return {"scenarios": repo.list_scenarios(db(), split), "domains": DOMAINS}


@router.post("")
def create_scenario(data: dict = Body(...)):
    return guarded(lambda: repo.create_scenario(db(), data))


@router.delete("/{sid}")
def delete_scenario(sid: str):
    guarded(lambda: repo.delete_scenario(db(), sid))
    return {"ok": True}
