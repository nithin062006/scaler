"""FastAPI server entry point for the AST code-editing environment.

Run locally::

    uvicorn env.server:app --port 8000
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from env.environment import ASTCodeEditEnvironment
from env.models import CodeEditAction, CodeEditObservation, CodeEditState

_env = ASTCodeEditEnvironment()


def _try_openenv_app() -> FastAPI | None:
    try:
        from openenv.server import create_server  # type: ignore
        return create_server(_env)
    except Exception:
        try:
            from openenv.core.env_server import create_fastapi_app as create_server  # type: ignore
            return create_server(_env)
        except Exception:
            return None


def _make_fallback_app() -> FastAPI:
    app = FastAPI(
        title="AST Code-Edit OpenEnv (fallback)",
        version="0.2.0",
        description="Single-step AST code-editing environment.",
    )

    @app.post("/reset", response_model=CodeEditObservation)
    def reset() -> CodeEditObservation:
        return _env.reset()

    @app.post("/step")
    def step(action: CodeEditAction) -> dict[str, Any]:
        try:
            obs, reward, done = _env.step(action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"observation": obs.model_dump(), "reward": reward, "done": done}

    @app.get("/state", response_model=CodeEditState)
    def state() -> CodeEditState:
        return _env.get_state()

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/")
    def index() -> dict[str, Any]:
        return {
            "name": "AST Code-Edit Environment",
            "version": "0.2.0",
            "endpoints": ["/reset", "/step", "/state", "/healthz"],
        }

    return app


_oa = _try_openenv_app()
app: FastAPI = _oa if _oa is not None else _make_fallback_app()

__all__ = ["app"]
