"""Evaluation harness.

Drive ``n`` episodes with a given policy and return the per-episode terminal
rewards plus a small summary dict. Used for both the baseline and trained
checkpoints so the comparison is apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graphforge.training import (
    InProcessEnvClient,
    Policy,
    rollout,
)


@dataclass
class EvalResult:
    rewards: list[float]
    parse_failure_rates: list[float]   # per-episode parse-fail share
    completion_rate: float             # fraction of episodes terminating naturally
    mean_reward: float
    std_reward: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rewards": self.rewards,
            "parse_failure_rates": self.parse_failure_rates,
            "completion_rate": self.completion_rate,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
        }


def evaluate(
    policy: Policy,
    *,
    n: int,
    task_id: str | None = None,
    max_turns: int | None = None,
) -> EvalResult:
    rewards: list[float] = []
    parse_rates: list[float] = []
    completed = 0
    for _ in range(n):
        env = InProcessEnvClient()
        traj = rollout(
            policy=policy, env=env,
            task_id=task_id,
            max_turns=max_turns,
        )
        # Use the *terminal* total when available; else the sum of step rewards.
        if traj.terminal_total is not None:
            rewards.append(traj.terminal_total)
        else:
            rewards.append(traj.total_reward)
        if traj.terminated_naturally:
            completed += 1
        nf = sum(1 for s in traj.samples if not s.parse_ok)
        parse_rates.append(nf / max(1, len(traj.samples)))

    n = max(1, n)
    import statistics

    mean = statistics.fmean(rewards) if rewards else 0.0
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return EvalResult(
        rewards=rewards,
        parse_failure_rates=parse_rates,
        completion_rate=completed / n,
        mean_reward=mean,
        std_reward=std,
    )
