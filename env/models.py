"""Pydantic wire models for the multi-turn repo-editing environment."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

_cfg = ConfigDict(extra="ignore")


class RepoEditObservation(BaseModel):
    """What the env returns after reset() or step().

    Contains the current graph overview + the result of the last action.
    The agent should read action_result carefully before deciding the next step.
    """

    model_config = _cfg

    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    turn: int = 0
    max_turns: int = 15

    graph_overview: str = ""       # compact text view of the entire repo KG
    task_description: str = ""     # what the agent needs to accomplish
    action_result: str = ""        # feedback from the last action

    turn_reward: float = 0.0
    total_reward: float = 0.0
    done: bool = False

    info: dict[str, Any] = Field(default_factory=dict)


class RepoEditState(BaseModel):
    """Episode-level state snapshot."""

    model_config = _cfg

    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    turn: int = 0
    done: bool = False
    total_reward: float = 0.0
