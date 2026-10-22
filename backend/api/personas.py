from __future__ import annotations

from fastapi import APIRouter, Body

from database import repo
from twin.personas import DEFAULT_PERSONA

from .deps import db, guarded

router = APIRouter(prefix="/api/personas", tags=["personas"])


@router.get("")
def list_personas():
    return {"personas": repo.list_personas(db()), "default": DEFAULT_PERSONA}


@router.get("/{pid}")
def get_persona(pid: str):
    def f():
        p = repo.get_persona(db(), pid)
        if p is None:
            raise KeyError(pid)
        return p
    return guarded(f)


@router.post("")
def create_persona(data: dict = Body(...)):
    return guarded(lambda: repo.create_persona(db(), data))


@router.put("/{pid}")
def update_persona(pid: str, data: dict = Body(...)):
    return guarded(lambda: repo.update_persona(db(), pid, data))


@router.delete("/{pid}")
def delete_persona(pid: str):
    guarded(lambda: repo.delete_persona(db(), pid))
    return {"ok": True}
