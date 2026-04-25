"""Trajectory generation, filtering, and SFT formatting.

Three stages:

  1. ``generate_trajectories`` — drive N rollouts with a given policy through
     the OpenEnv-compliant client/server.
  2. ``filter_by_reward`` — keep only trajectories whose terminal reward
     clears the threshold (rejection sampling).
  3. ``format_as_sft`` — turn each turn of each kept trajectory into a
     ``(prompt, completion)`` pair suitable for ``trl.SFTTrainer``.

We use the in-process env client throughout so generation doesn't require
a running HTTP server. The same data flow works against a remote server
(for HF Space evaluation) by swapping ``InProcessEnvClient`` for
``HttpEnvClient``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from graphforge.training import (
    InProcessEnvClient,
    Policy,
    ScriptedPolicy,
    Trajectory,
    rollout,
)
from graphforge.training.protocol import render_action


# ---- oracle action sequences ---------------------------------------


# The single tier-0 task is "build validators.is_email with EMAIL pattern".
# This produces the canonical positive trajectory.
ORACLE_T0_EMAIL_VALIDATOR: list[dict[str, Any]] = [
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


def oracle_completions(actions: list[dict[str, Any]]) -> list[str]:
    """Render an oracle action sequence as agent completions.

    Each completion includes a brief plausible reasoning preamble before the
    ``<action>`` block — this matches what we want the SFT'd policy to
    eventually emit.
    """
    preambles = [
        "Step 1: declare the validators module so we have somewhere to put the function.",
        "Step 2: add the is_email function with the canonical signature.",
        "Step 3: attach the validate_with_regex body with the EMAIL pattern.",
        "Step 4: the spec is satisfied — submit.",
    ]
    out: list[str] = []
    for i, action in enumerate(actions):
        preamble = preambles[i] if i < len(preambles) else f"Step {i+1}: continue."
        out.append(f"{preamble}\n{render_action(action)}")
    return out


# ---- trajectory generation -----------------------------------------


@dataclass
class TrajectoryRecord:
    """Minimal serializable wrapper around a Trajectory."""

    samples: list[dict[str, Any]] = field(default_factory=list)
    terminal_total: float | None = None
    terminated_naturally: bool = False
    parse_failures: int = 0
    source: str = "unknown"  # "oracle" | "model" | other

    @classmethod
    def from_trajectory(cls, traj: Trajectory, *, source: str) -> "TrajectoryRecord":
        return cls(
            samples=[
                {
                    "turn": s.turn,
                    "prompt_messages": s.prompt_messages,
                    "completion_text": s.completion_text,
                    "reward": s.reward,
                    "return_": s.return_,
                    "parse_ok": s.parse_ok,
                }
                for s in traj.samples
            ],
            terminal_total=traj.terminal_total,
            terminated_naturally=traj.terminated_naturally,
            parse_failures=sum(1 for s in traj.samples if not s.parse_ok),
            source=source,
        )

    @property
    def total_reward(self) -> float:
        return sum(s["reward"] for s in self.samples)


def generate_oracle_trajectories(
    n: int, *, task_id: str | None = None
) -> list[TrajectoryRecord]:
    """Run the scripted oracle ``n`` times. All produce the same trajectory
    given the deterministic env, but they're useful as SFT training data."""
    out: list[TrajectoryRecord] = []
    for _ in range(n):
        env = InProcessEnvClient()
        policy = ScriptedPolicy(oracle_completions(ORACLE_T0_EMAIL_VALIDATOR))
        traj = rollout(policy=policy, env=env, task_id=task_id)
        out.append(TrajectoryRecord.from_trajectory(traj, source="oracle"))
    return out


def generate_model_trajectories(
    policy: Policy,
    n: int,
    *,
    task_id: str | None = None,
    max_turns: int | None = None,
) -> list[TrajectoryRecord]:
    """Run a real model policy ``n`` times and capture each trajectory."""
    out: list[TrajectoryRecord] = []
    for _ in range(n):
        env = InProcessEnvClient()
        traj = rollout(
            policy=policy,
            env=env,
            task_id=task_id,
            max_turns=max_turns,
        )
        out.append(TrajectoryRecord.from_trajectory(traj, source="model"))
    return out


# ---- filtering ------------------------------------------------------


def filter_by_reward(
    records: Iterable[TrajectoryRecord], *, threshold: float
) -> list[TrajectoryRecord]:
    out: list[TrajectoryRecord] = []
    for r in records:
        if r.terminal_total is not None and r.terminal_total >= threshold:
            out.append(r)
    return out


# ---- SFT formatting -------------------------------------------------


def format_as_sft_examples(
    records: Iterable[TrajectoryRecord],
    *,
    chat_template_apply: callable | None = None,
) -> list[dict[str, str]]:
    """Convert kept trajectories into ``[{prompt, completion}, ...]`` pairs.

    Each per-turn ``TurnSample`` becomes one SFT example:

      prompt     — chat-template-rendered text of the conversation up to,
                   but not including, this turn's assistant completion.
      completion — the assistant completion text (which contains the
                   ``<action>...</action>`` tool call).

    If ``chat_template_apply`` is provided (typically
    ``tokenizer.apply_chat_template``) we use it. Otherwise we fall back to
    a plain "<role>: <content>\\n" rendering — fine for inspection but
    not what you'd actually train against.
    """
    pairs: list[dict[str, str]] = []
    for rec in records:
        for s in rec.samples:
            prompt_text = _render_messages(
                s["prompt_messages"], chat_template_apply
            )
            pairs.append(
                {"prompt": prompt_text, "completion": s["completion_text"]}
            )
    return pairs


def _render_messages(
    messages: list[dict[str, str]],
    chat_template_apply: callable | None,
) -> str:
    if chat_template_apply is not None:
        return chat_template_apply(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "\n".join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + "\n<assistant>\n"


# ---- persistence ----------------------------------------------------


def save_trajectories(
    records: Iterable[TrajectoryRecord], path: str
) -> None:
    payload = [
        {
            "terminal_total": r.terminal_total,
            "terminated_naturally": r.terminated_naturally,
            "parse_failures": r.parse_failures,
            "source": r.source,
            "samples": r.samples,
        }
        for r in records
    ]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
