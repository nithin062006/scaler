"""End-to-end tests over the OpenEnv FastAPI server.

We drive the running app via :class:`fastapi.testclient.TestClient` (no real
HTTP socket required). Each test resets the global episode store so cases
don't leak into each other.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graphforge.server.app import app
from graphforge.server.episode import GLOBAL_STORE, EpisodeStore


@pytest.fixture(autouse=True)
def _isolated_store() -> None:
    """Replace the global store with a fresh one per test."""
    fresh = EpisodeStore()
    GLOBAL_STORE._eps = fresh._eps  # type: ignore[attr-defined]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---- /healthz --------------------------------------------------------


def test_healthz(client: TestClient):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---- /reset ----------------------------------------------------------


def test_reset_default_task(client: TestClient):
    r = client.post("/reset", json={})
    assert r.status_code == 200
    body = r.json()
    assert "episode_id" in body
    obs = body["observation"]
    assert "task" in obs
    assert obs["task"]["id"].startswith("t0.")
    # Hidden constraints must not be in the visible payload.
    assert "hidden_constraints" not in obs["task"]


def test_reset_unknown_task_404(client: TestClient):
    r = client.post("/reset", json={"task_id": "does-not-exist"})
    assert r.status_code == 404


# ---- /step (success + failure paths) --------------------------------


def _reset(client: TestClient) -> str:
    r = client.post("/reset", json={})
    return r.json()["episode_id"]


def test_step_unknown_episode_404(client: TestClient):
    r = client.post(
        "/step",
        json={
            "episode_id": "nonexistent",
            "action": {"kind": "add_module", "name": "m", "responsibility": "io"},
        },
    )
    assert r.status_code == 404


def test_step_successful_mutation(client: TestClient):
    eid = _reset(client)
    r = client.post(
        "/step",
        json={
            "episode_id": eid,
            "action": {"kind": "add_module", "name": "validators", "responsibility": "validation"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is False
    assert body["observation"]["ok"] is True
    # Per-turn cost is -0.1 plus a small token cost on a tiny payload.
    assert body["reward"] < 0
    assert body["reward"] > -0.5


def test_step_failed_mutation_costs_minus_two(client: TestClient):
    eid = _reset(client)
    # Add the module twice — second one must collide.
    payload = {
        "episode_id": eid,
        "action": {"kind": "add_module", "name": "validators", "responsibility": "validation"},
    }
    r1 = client.post("/step", json=payload)
    assert r1.status_code == 200
    assert r1.json()["observation"]["ok"] is True

    r2 = client.post("/step", json=payload)
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["observation"]["ok"] is False
    # MUTATION_FAIL (-2) + DUPLICATE_ACTION (-1) + PER_TURN_COST (-0.1) ≈ -3.1
    assert body2["reward"] < -3.0


def test_step_malformed_action_422(client: TestClient):
    """Pydantic rejects an unknown action.kind at request validation time."""
    eid = _reset(client)
    r = client.post(
        "/step",
        json={"episode_id": eid, "action": {"kind": "not_a_real_action"}},
    )
    assert r.status_code == 422


# ---- full oracle episode --------------------------------------------


def _oracle_actions() -> list[dict]:
    """Action sequence that satisfies every constraint of the tier-0 task."""
    return [
        {"kind": "add_module", "name": "validators", "responsibility": "validation"},
        {
            "kind": "add_node",
            "name": "is_email",
            "module": "validators",
            "signature": "(s: str) -> bool",
            "purity": "pure",
            "error_policy": "guard",
        },
        {
            "kind": "attach_body",
            "name": "is_email",
            "module": "validators",
            "template": "validate_with_regex",
            "args": {"pattern": "EMAIL"},
        },
        {"kind": "submit"},
    ]


def test_oracle_episode_terminates_with_positive_terminal(client: TestClient):
    eid = _reset(client)
    last_body: dict | None = None
    for action in _oracle_actions():
        r = client.post("/step", json={"episode_id": eid, "action": action})
        assert r.status_code == 200, r.text
        last_body = r.json()
    assert last_body is not None
    assert last_body["done"] is True
    info = last_body["info"]
    assert "terminal" in info
    terminal = info["terminal"]
    # All structural constraints satisfied → at least 7 (constraint count) +5 bonus.
    assert terminal["structural"] >= 7.0
    assert terminal["bonus_all_structural"] == 5.0
    assert terminal["penalty_materialize"] == 0.0
    assert terminal["total"] > 10.0
    # satisfaction breakdown is present
    assert terminal["satisfaction"]["all_satisfied"] is True


def test_partial_episode_no_all_structural_bonus(client: TestClient):
    """Submit before completing the work — no bonus, no efficiency."""
    eid = _reset(client)
    # Add only the module, then submit.
    client.post(
        "/step",
        json={
            "episode_id": eid,
            "action": {"kind": "add_module", "name": "validators", "responsibility": "validation"},
        },
    )
    r = client.post(
        "/step",
        json={"episode_id": eid, "action": {"kind": "submit"}},
    )
    body = r.json()
    assert body["done"] is True
    terminal = body["info"]["terminal"]
    assert terminal["bonus_all_structural"] == 0.0
    assert terminal["efficiency"] == 0.0
    # Some structural constraints (module_count, module_responsibility) ARE
    # satisfied, so structural reward is non-zero.
    assert terminal["structural"] > 0.0


# ---- /state and /close ----------------------------------------------


def test_state_snapshot_after_actions(client: TestClient):
    eid = _reset(client)
    client.post(
        "/step",
        json={
            "episode_id": eid,
            "action": {"kind": "add_module", "name": "m", "responsibility": "io"},
        },
    )
    r = client.get("/state", params={"episode_id": eid})
    assert r.status_code == 200
    snap = r.json()
    assert snap["turns"] == 1
    assert len(snap["graph"]["modules"]) == 1
    assert snap["graph"]["modules"][0]["name"] == "m"


def test_close_removes_episode(client: TestClient):
    eid = _reset(client)
    r = client.post("/close", json={"episode_id": eid})
    assert r.status_code == 200
    assert r.json()["closed"] is True
    # Subsequent /state must 404.
    r2 = client.get("/state", params={"episode_id": eid})
    assert r2.status_code == 404


# ---- episode cap ----------------------------------------------------


def test_episode_cap_auto_terminates(client: TestClient):
    """If we burn through the episode cap without submit, the env must
    auto-terminate with the same terminal reward shape."""
    eid = _reset(client)
    # The tier-0 cap is 20 turns. Spam a malformed-on-collision pattern.
    payload = {
        "episode_id": eid,
        "action": {"kind": "add_module", "name": "m", "responsibility": "io"},
    }
    last: dict | None = None
    for _ in range(25):
        r = client.post("/step", json=payload)
        body = r.json()
        last = body
        if body["done"]:
            break
    assert last is not None
    assert last["done"] is True
    assert "terminal" in last["info"]
    assert last["info"].get("reason") == "episode_cap_reached"
