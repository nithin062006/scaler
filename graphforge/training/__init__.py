"""Training: multi-turn rollout for GRPO / SFT.

Public surface:

    EnvClient                 — protocol; HttpEnvClient or InProcessEnvClient
    Policy                    — protocol; ScriptedPolicy or HfPolicy
    rollout(...)              — drive one episode, return Trajectory
    Trajectory, TurnSample    — per-turn (prompt, completion, reward, return)

The rollout is environment-agnostic and policy-agnostic — see
PROPOSAL.md §7.2 for the GRPOTrainer integration story.
"""

from graphforge.training.client import (
    EnvClient,
    HttpEnvClient,
    InProcessEnvClient,
)
from graphforge.training.policy import HfPolicy, Policy, ScriptedPolicy
from graphforge.training.protocol import (
    ParseFailure,
    ParseSuccess,
    parse_completion,
    render_action,
)
from graphforge.training.rollout import (
    Trajectory,
    TurnSample,
    rollout,
    trajectory_summary,
)

__all__ = [
    "EnvClient",
    "HfPolicy",
    "HttpEnvClient",
    "InProcessEnvClient",
    "ParseFailure",
    "ParseSuccess",
    "Policy",
    "ScriptedPolicy",
    "Trajectory",
    "TurnSample",
    "parse_completion",
    "render_action",
    "rollout",
    "trajectory_summary",
]


def train_grpo(config: object) -> None:  # pragma: no cover — TODO
    raise NotImplementedError("GRPO training TODO — see PROPOSAL.md §7")


def train_sft(config: object) -> None:  # pragma: no cover — TODO
    raise NotImplementedError("SFT plan B TODO — see PROPOSAL.md §7.4")
