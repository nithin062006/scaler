"""Smoke test for the InProcessEnvClient — the HTTP client is exercised
implicitly when the server is run for real, and is structurally identical
to the in-process variant."""

from __future__ import annotations

import pytest

from graphforge.server.episode import GLOBAL_STORE, EpisodeStore
from graphforge.training import InProcessEnvClient


@pytest.fixture(autouse=True)
def _isolated_store() -> None:
    fresh = EpisodeStore()
    GLOBAL_STORE._eps = fresh._eps  # type: ignore[attr-defined]


def test_full_loop():
    c = InProcessEnvClient()
    r = c.reset()
    eid = r["episode_id"]
    assert "task" in r["observation"]

    s = c.step(eid, {"kind": "add_module", "name": "m", "responsibility": "io"})
    assert s["observation"]["ok"] is True
    assert s["done"] is False

    closed = c.close(eid)
    assert closed["closed"] is True


def test_step_with_unknown_kind_returns_schema_rejection():
    c = InProcessEnvClient()
    eid = c.reset()["episode_id"]
    s = c.step(eid, {"kind": "totally_made_up"})
    assert s["info"]["error"] == "schema_rejection"
    assert s["done"] is False
