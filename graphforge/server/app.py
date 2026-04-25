"""FastAPI application — the OpenEnv server.

Endpoints (PROPOSAL.md §6.1):

  POST /reset   { task_id?: str | None, seed?: int }
                -> { episode_id, observation }
  POST /step    { episode_id, action: Action }
                -> { observation, reward, done, info }
  GET  /state?episode_id=...
                -> { ... full snapshot ... }
  POST /close   { episode_id }
                -> { closed: bool }

The handlers are thin: routing, request validation, episode lookup. The
actual per-step orchestration lives in :mod:`graphforge.server.runner`.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graphforge.actions.schema import Action
from graphforge.server.episode import GLOBAL_STORE, Episode, EpisodeStore
from graphforge.server.runner import step as runner_step
from graphforge.tasks import default_task, get_task

app = FastAPI(
    title="GraphForge OpenEnv server",
    version="0.0.1",
    description="See graphforge.server for the wire shape.",
)


# ---- request / response models --------------------------------------


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None  # reserved for variant generation, unused for tier-0


class StepRequest(BaseModel):
    episode_id: str
    # ``Action`` is itself an Annotated discriminated union; no need to
    # re-declare the discriminator on this field.
    action: Action


class CloseRequest(BaseModel):
    episode_id: str


# ---- store wiring (overridable for tests) ---------------------------


def _store() -> EpisodeStore:
    return GLOBAL_STORE


# ---- helpers --------------------------------------------------------


def _require_episode(episode_id: str) -> Episode:
    ep = _store().get(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail=f"unknown episode_id: {episode_id!r}")
    return ep


def _initial_observation(ep: Episode) -> dict[str, Any]:
    return {
        "episode_id": ep.id,
        "task": ep.task.visible_payload(),
        "turns_total": 0,
        "tokens_used_total": 0,
        "budget": ep.task.budget,
        "episode_cap": ep.task.episode_cap,
    }


# ---- endpoints ------------------------------------------------------


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    if req.task_id is None:
        task = default_task()
    else:
        t = get_task(req.task_id)
        if t is None:
            raise HTTPException(status_code=404, detail=f"unknown task_id: {req.task_id!r}")
        task = t
    ep = Episode.new(task=task)
    _store().put(ep)
    return {
        "episode_id": ep.id,
        "observation": _initial_observation(ep),
    }


@app.post("/step")
def step(req: StepRequest) -> dict:
    ep = _require_episode(req.episode_id)
    return runner_step(ep, req.action)


@app.get("/state")
def state(episode_id: str) -> dict:
    ep = _require_episode(episode_id)
    return ep.state_snapshot()


@app.post("/close")
def close(req: CloseRequest) -> dict:
    closed = _store().drop(req.episode_id)
    return {"closed": closed}


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "version": app.version}
