"""End-to-end tests for the OpenEnv-compliant ``env/`` wrapper.

Tests run via ``TestClient`` against ``env.server:app`` (the fallback or the
openenv-core auto-server, whichever is available). The contract validated
here is the wire shape the hackathon will judge:

  POST /reset  -> GraphForgeObservation
  POST /step   -> {observation, reward, done}
  GET  /state  -> GraphForgeState
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from env.environment import GraphForgeEnvironment
from env.models import GraphForgeAction, GraphForgeObservation, GraphForgeState
from env.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---- direct-class tests ---------------------------------------------


def test_reset_returns_observation_with_task_payload():
    e = GraphForgeEnvironment()
    obs = e.reset()
    assert isinstance(obs, GraphForgeObservation)
    assert obs.task is not None
    assert obs.task["id"].startswith("t0.")
    # hidden constraints must be invisible to the agent
    assert "hidden_constraints" not in obs.task
    assert obs.episode_id is not None


def test_step_signature_returns_obs_reward_done():
    e = GraphForgeEnvironment()
    e.reset()
    obs, reward, done = e.step(
        GraphForgeAction(
            kind="add_module",
            payload={"name": "validators", "responsibility": "validation"},
        )
    )
    assert isinstance(obs, GraphForgeObservation)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert done is False
    assert obs.ok is True


def test_oracle_episode_terminates_with_positive_terminal():
    e = GraphForgeEnvironment()
    e.reset()
    actions = [
        GraphForgeAction(
            kind="add_module",
            payload={"name": "validators", "responsibility": "validation"},
        ),
        GraphForgeAction(
            kind="add_node",
            payload={
                "name": "is_email",
                "module": "validators",
                "signature": "(s: str) -> bool",
            },
        ),
        GraphForgeAction(
            kind="attach_body",
            payload={
                "name": "is_email",
                "module": "validators",
                "template": "validate_with_regex",
                "args": {"pattern": "EMAIL"},
            },
        ),
        GraphForgeAction(kind="submit", payload={}),
    ]
    last = None
    for a in actions:
        last = e.step(a)
    assert last is not None
    obs, reward, done = last
    assert done is True
    assert obs.terminal is not None
    assert obs.terminal["total"] > 10.0
    assert obs.terminal["satisfaction"]["all_satisfied"] is True


def test_malformed_action_scored_minus_two():
    e = GraphForgeEnvironment()
    e.reset()
    obs, reward, done = e.step(
        GraphForgeAction(kind="totally_made_up", payload={"foo": "bar"})
    )
    assert obs.outcome == "malformed"
    # MALFORMED (-2) + per_turn (-0.1) = -2.1
    assert reward == pytest.approx(-2.1)
    assert done is False


def test_get_state_after_actions():
    e = GraphForgeEnvironment()
    e.reset()
    e.step(
        GraphForgeAction(
            kind="add_module",
            payload={"name": "m", "responsibility": "io"},
        )
    )
    s = e.get_state()
    assert isinstance(s, GraphForgeState)
    assert s.turns == 1
    assert len(s.graph["modules"]) == 1
    assert s.terminated is False


# ---- HTTP-surface tests ---------------------------------------------


def test_http_reset_step_state_round_trip(client: TestClient):
    r = client.post("/reset")
    assert r.status_code == 200
    obs = r.json()
    assert obs["task"] is not None
    assert obs["episode_id"] is not None

    r2 = client.post(
        "/step",
        json={
            "kind": "add_module",
            "payload": {"name": "m", "responsibility": "io"},
        },
    )
    assert r2.status_code == 200
    body = r2.json()
    assert "observation" in body
    assert "reward" in body
    assert "done" in body
    assert body["observation"]["ok"] is True

    r3 = client.get("/state")
    assert r3.status_code == 200
    state = r3.json()
    assert state["turns"] == 1


def test_http_healthz(client: TestClient):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_http_oracle_episode_via_server(client: TestClient):
    client.post("/reset")
    actions = [
        {"kind": "add_module", "payload": {"name": "validators", "responsibility": "validation"}},
        {
            "kind": "add_node",
            "payload": {
                "name": "is_email",
                "module": "validators",
                "signature": "(s: str) -> bool",
            },
        },
        {
            "kind": "attach_body",
            "payload": {
                "name": "is_email",
                "module": "validators",
                "template": "validate_with_regex",
                "args": {"pattern": "EMAIL"},
            },
        },
        {"kind": "submit", "payload": {}},
    ]
    last = None
    for a in actions:
        r = client.post("/step", json=a)
        assert r.status_code == 200
        last = r.json()
    assert last is not None
    assert last["done"] is True
    terminal = last["observation"]["terminal"]
    assert terminal["total"] > 10.0
