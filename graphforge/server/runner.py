"""Episode runner — the per-step orchestration the server endpoints use.

Pulls together dispatcher, reward engine, constraint checker, and episode
state. Kept separate from the FastAPI app so it can be unit-tested without
spinning up an HTTP server.
"""

from __future__ import annotations

from typing import Any

from graphforge.actions import dispatch
from graphforge.actions.schema import Action, Submit
from graphforge.constraints import evaluate_all
from graphforge.materializer import materialize
from graphforge.reward.engine import (
    ActionOutcome,
    TurnReward,
    score_terminal,
    score_turn,
)
from graphforge.server.episode import (
    Episode,
    TurnRecord,
    estimate_tokens,
)
from graphforge.validator import full_check


def _classify_outcome(action: Action, ok: bool) -> ActionOutcome:
    # Schema rejection happens before this function (caught by FastAPI's
    # pydantic validation). What we see here is a successfully-parsed
    # action that either succeeded or failed at handler-time.
    return ActionOutcome.SUCCESS if ok else ActionOutcome.FAILURE


def _render_observation(ep: Episode, turn_record: TurnRecord) -> dict[str, Any]:
    return {
        "turn": turn_record.turn,
        "ok": turn_record.ok,
        "outcome": turn_record.outcome,
        "payload": turn_record.payload,
        "reward": turn_record.reward,
        "is_duplicate": turn_record.is_duplicate,
        "tokens_returned": turn_record.tokens_returned,
        "tokens_used_total": ep.tokens_used,
        "turns_total": ep.turns,
        "budget_remaining": max(0, ep.task.budget - ep.tokens_used),
        "episode_cap_remaining": max(0, ep.task.episode_cap - ep.turns),
    }


def step(ep: Episode, action: Action) -> dict[str, Any]:
    """Apply ``action`` to ``ep``. Auto-terminates on submit or cap.

    Returns a dict in the OpenEnv ``/step`` response shape:
    ``{observation, reward, done, info}``.
    """
    if ep.terminated:
        return {
            "observation": {},
            "reward": 0.0,
            "done": True,
            "info": {"error": "episode_already_terminated"},
        }

    args = action.model_dump(exclude={"kind"})
    kind = action.kind  # type: ignore[attr-defined]
    is_duplicate = ep.is_duplicate(kind, args)

    result = dispatch(ep.graph, action)
    tokens_returned = estimate_tokens(result.payload)
    outcome = _classify_outcome(action, result.ok)
    turn_reward = score_turn(
        outcome=outcome,
        is_duplicate=is_duplicate,
        tokens_returned=tokens_returned,
    )
    rec = ep.record_turn(
        kind=kind,
        args=args,
        result=result,
        outcome=outcome,
        turn_reward=turn_reward,
        is_duplicate=is_duplicate,
        tokens_returned=tokens_returned,
    )

    done = False
    info: dict[str, Any] = {}

    # Terminate on Submit.
    if isinstance(action, Submit):
        done = True
        terminal = _score_terminal(ep)
        ep.terminated = True
        ep.terminal_reward = terminal["total"]
        ep.terminal_payload = terminal
        info["terminal"] = terminal

    # Terminate on episode cap.
    if not done and ep.turns >= ep.task.episode_cap:
        done = True
        terminal = _score_terminal(ep)
        ep.terminated = True
        ep.terminal_reward = terminal["total"]
        ep.terminal_payload = terminal
        info["terminal"] = terminal
        info["reason"] = "episode_cap_reached"

    return {
        "observation": _render_observation(ep, rec),
        "reward": rec.reward + (info.get("terminal", {}).get("total", 0.0) if done else 0.0),
        "done": done,
        "info": info,
    }


def _score_terminal(ep: Episode) -> dict[str, Any]:
    """Compute terminal reward + return a serialized payload."""
    sat = evaluate_all(ep.graph, ep.task.all_constraints)
    structural, behavioral = sat.split_by_family()

    # materialization gate: try to materialize + parse-check.
    materialization_ok = False
    try:
        files = materialize(ep.graph)
        materialization_ok = full_check(files).ok
    except Exception:
        materialization_ok = False

    reward = score_terminal(
        n_structural_satisfied=len(structural.satisfied),
        n_structural_total=structural.total,
        n_behavioral_passing=len(behavioral.satisfied),
        n_behavioral_total=behavioral.total,
        materialization_ok=materialization_ok,
        type_checks_ok=None,  # mypy not wired yet
        tokens_used=ep.tokens_used,
        budget=ep.task.budget,
    )
    out = reward.to_dict()
    out["satisfaction"] = sat.to_dict()
    return out
