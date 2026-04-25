"""Env client adapters.

Two implementations of the same contract:

  * :class:`HttpEnvClient` — talks to a running FastAPI server over HTTP
    (``localhost:8000`` during training).
  * :class:`InProcessEnvClient` — drives the same FastAPI app via
    ``fastapi.testclient.TestClient``, no socket required. Used by tests
    and by single-process training jobs.

Both expose the same three operations: ``reset``, ``step``, ``close``. The
rollout code only depends on the protocol, so swapping transports doesn't
ripple through anything else.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EnvClient(Protocol):
    """Minimal client surface the rollout depends on."""

    def reset(self, task_id: str | None = None, seed: int | None = None) -> dict[str, Any]: ...

    def step(self, episode_id: str, action: dict[str, Any]) -> dict[str, Any]: ...

    def close(self, episode_id: str) -> dict[str, Any]: ...


# ---- HTTP transport --------------------------------------------------


class HttpEnvClient:
    """Thin httpx wrapper. Use during training when the env server runs out-of-process."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        # Defer the import so the dep is optional for users who only do
        # in-process drives in tests / notebooks.
        import httpx

        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def reset(self, task_id: str | None = None, seed: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if task_id is not None:
            body["task_id"] = task_id
        if seed is not None:
            body["seed"] = seed
        r = self._client.post("/reset", json=body)
        r.raise_for_status()
        return r.json()

    def step(self, episode_id: str, action: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post("/step", json={"episode_id": episode_id, "action": action})
        # 422 = malformed action payload; surface as a structured response
        # rather than raising, because the agent's job is to learn from it.
        if r.status_code == 422:
            return {
                "observation": {},
                "reward": 0.0,  # caller will overlay with MALFORMED scoring
                "done": False,
                "info": {"error": "schema_rejection", "detail": r.json()},
            }
        r.raise_for_status()
        return r.json()

    def close(self, episode_id: str) -> dict[str, Any]:
        r = self._client.post("/close", json={"episode_id": episode_id})
        r.raise_for_status()
        return r.json()

    def __enter__(self) -> "HttpEnvClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self._client.close()


# ---- in-process transport -------------------------------------------


class InProcessEnvClient:
    """Drive the FastAPI app via ``TestClient`` without a real socket."""

    def __init__(self, app: object | None = None) -> None:
        from fastapi.testclient import TestClient

        if app is None:
            from graphforge.server.app import app as default_app
            app = default_app
        self._client = TestClient(app)  # type: ignore[arg-type]

    def reset(self, task_id: str | None = None, seed: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if task_id is not None:
            body["task_id"] = task_id
        if seed is not None:
            body["seed"] = seed
        r = self._client.post("/reset", json=body)
        r.raise_for_status()
        return r.json()

    def step(self, episode_id: str, action: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post(
            "/step", json={"episode_id": episode_id, "action": action}
        )
        if r.status_code == 422:
            return {
                "observation": {},
                "reward": 0.0,
                "done": False,
                "info": {"error": "schema_rejection", "detail": r.json()},
            }
        r.raise_for_status()
        return r.json()

    def close(self, episode_id: str) -> dict[str, Any]:
        r = self._client.post("/close", json={"episode_id": episode_id})
        r.raise_for_status()
        return r.json()
