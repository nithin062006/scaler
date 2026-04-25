"""HTTP client for the repo-editing environment."""

from __future__ import annotations

from typing import Any

import httpx

from env.models import RepoEditObservation, RepoEditState


class RepoEditEnv:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def reset(self, task_id: str | None = None) -> RepoEditObservation:
        params = {"task_id": task_id} if task_id else {}
        r = self._client.post("/reset", params=params)
        r.raise_for_status()
        return RepoEditObservation.model_validate(r.json())

    def step(self, action_dict: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post("/step", json=action_dict)
        r.raise_for_status()
        return r.json()

    def state(self) -> RepoEditState:
        r = self._client.get("/state")
        r.raise_for_status()
        return RepoEditState.model_validate(r.json())

    def __enter__(self) -> "RepoEditEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()
