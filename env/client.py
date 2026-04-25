"""HTTP client for the AST code-editing environment."""

from __future__ import annotations

from typing import Any

import httpx

from env.models import CodeEditAction, CodeEditObservation, CodeEditState


class ASTCodeEditEnv:
    """OpenEnv-style HTTP client over a running env.server instance."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout)

    def reset(self) -> CodeEditObservation:
        r = self._client.post("/reset", json={})
        r.raise_for_status()
        return CodeEditObservation.model_validate(r.json())

    def step(self, action: CodeEditAction) -> dict[str, Any]:
        r = self._client.post("/step", json=action.model_dump())
        r.raise_for_status()
        return r.json()

    def state(self) -> CodeEditState:
        r = self._client.get("/state")
        r.raise_for_status()
        return CodeEditState.model_validate(r.json())

    def __enter__(self) -> "ASTCodeEditEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()
