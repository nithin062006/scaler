"""End-to-end rollout test.

We use the in-process env client and a scripted oracle policy to drive a
full episode of the tier-0 task. The trajectory must:

  * terminate naturally on submit
  * carry a positive terminal reward
  * have monotonically-defined returns (last-turn return == last reward)
"""

from __future__ import annotations

import math

import pytest

from graphforge.server.episode import GLOBAL_STORE, EpisodeStore
from graphforge.training import (
    InProcessEnvClient,
    ScriptedPolicy,
    Trajectory,
    render_action,
    rollout,
    trajectory_summary,
)


@pytest.fixture(autouse=True)
def _isolated_store() -> None:
    fresh = EpisodeStore()
    GLOBAL_STORE._eps = fresh._eps  # type: ignore[attr-defined]


def _oracle_completions() -> list[str]:
    actions = [
        {"kind": "add_module", "name": "validators", "responsibility": "validation"},
        {
            "kind": "add_node",
            "name": "is_email",
            "module": "validators",
            "signature": "(s: str) -> bool",
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
    return [f"Reasoning step {i}.\n{render_action(a)}" for i, a in enumerate(actions)]


def test_oracle_rollout_terminates_with_positive_terminal():
    policy = ScriptedPolicy(_oracle_completions())
    env = InProcessEnvClient()
    traj = rollout(policy=policy, env=env)
    assert isinstance(traj, Trajectory)
    assert traj.terminated_naturally is True
    assert len(traj) == 4
    assert traj.terminal_total is not None
    assert traj.terminal_total > 10.0


def test_returns_are_discounted_correctly():
    policy = ScriptedPolicy(_oracle_completions())
    env = InProcessEnvClient()
    traj = rollout(policy=policy, env=env, gamma=0.9)
    rewards = [s.reward for s in traj.samples]
    returns = [s.return_ for s in traj.samples]
    # last return == last reward (no future)
    assert math.isclose(returns[-1], rewards[-1])
    # earlier returns add discounted future
    expected = []
    running = 0.0
    for r in reversed(rewards):
        running = r + 0.9 * running
        expected.append(running)
    expected.reverse()
    for r, e in zip(returns, expected, strict=True):
        assert math.isclose(r, e)


def test_malformed_completion_costs_minus_two_no_env_step():
    """A non-tool-calling completion is scored locally as MALFORMED. The
    rollout still continues — the episode doesn't end on a parse failure."""
    completions = [
        "I'm just thinking, no action.",
        # Then recover with a real action so we can verify the loop continued.
        f"OK fine: {render_action({'kind': 'add_module', 'name': 'm', 'responsibility': 'io'})}",
        f"{render_action({'kind': 'submit'})}",
    ]
    policy = ScriptedPolicy(completions)
    env = InProcessEnvClient()
    traj = rollout(policy=policy, env=env)
    assert traj.samples[0].parse_ok is False
    assert traj.samples[0].parse_failure_code == "no_action_tag"
    # MALFORMED reward = -2 + per_turn = -2.1
    assert math.isclose(traj.samples[0].reward, -2.1)
    # The recovery action should have succeeded.
    assert traj.samples[1].parse_ok is True
    assert traj.samples[1].env_response.get("observation", {}).get("ok") is True
    # And submit should have ended the episode.
    assert traj.terminated_naturally is True


def test_summary_helper():
    policy = ScriptedPolicy(_oracle_completions())
    traj = rollout(policy=policy, env=InProcessEnvClient())
    s = trajectory_summary(traj)
    assert s["n_turns"] == 4
    assert s["terminated_naturally"] is True
    assert s["parse_failures"] == 0


def test_partial_episode_does_not_terminate_without_submit():
    """If the policy doesn't submit and we don't blow the cap, episode stays open."""
    completions = [
        f"{render_action({'kind': 'add_module', 'name': 'm', 'responsibility': 'io'})}",
        f"{render_action({'kind': 'add_node', 'name': 'f', 'module': 'm', 'signature': '() -> int'})}",
    ]
    policy = ScriptedPolicy(completions)
    env = InProcessEnvClient()
    traj = rollout(policy=policy, env=env, max_turns=2)
    assert traj.terminated_naturally is False
    assert len(traj) == 2
