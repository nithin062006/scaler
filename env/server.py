"""FastAPI server for the multi-turn repo-editing environment."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from env.actions import RepoEditAction, parse_action
from env.environment import RepoEditEnvironment
from env.models import RepoEditObservation, RepoEditState

_env = RepoEditEnvironment()


def _make_app() -> FastAPI:
    app = FastAPI(title="Repo-Edit OpenEnv", version="0.3.0")

    @app.post("/reset", response_model=RepoEditObservation)
    def reset(task_id: str | None = None) -> RepoEditObservation:
        return _env.reset(task_id=task_id)

    @app.post("/step")
    def step(action_dict: dict[str, Any]) -> dict[str, Any]:
        try:
            action = parse_action(action_dict)
            obs, reward, done = _env.step(action)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"observation": obs.model_dump(), "reward": reward, "done": done}

    @app.get("/state", response_model=RepoEditState)
    def state() -> RepoEditState:
        return _env.get_state()

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok"}

    return app


app = _make_app()
__all__ = ["app"]
