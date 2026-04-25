"""Pydantic wire models for the AST code-editing OpenEnv environment.

Action  → CodeEditAction   {"code": "<function body>"}
Obs     → CodeEditObservation
State   → CodeEditState
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

_cfg = ConfigDict(extra="forbid")


class CodeEditAction(BaseModel):
    """Single agent action: a function body to inject into the stub.

    ``code`` is the raw Python body text (without the def line). Indentation
    is normalised by the environment.

    Example::

        CodeEditAction(code="    return s == s[::-1]")
    """

    model_config = _cfg
    code: str = Field(..., description="Function body code (indented, no def line)")


class CodeEditObservation(BaseModel):
    """What the environment returns after reset() or step().

    On reset, ``graph_text`` and ``task`` are populated.
    On step, ``reward``, ``done``, ``compile_ok``, and ``tests_passed`` reflect
    the outcome of the agent's last action.
    """

    model_config = ConfigDict(extra="ignore")

    episode_id: Optional[str] = None
    task_id: Optional[str] = None

    # graph description (present after reset)
    graph_text: str = ""
    target_function: str = ""
    target_signature: str = ""
    callers: list[str] = Field(default_factory=list)
    callees: list[str] = Field(default_factory=list)
    task_description: str = ""

    # step feedback
    ok: bool = True
    compile_ok: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    reward: float = 0.0
    done: bool = False

    info: dict[str, Any] = Field(default_factory=dict)


class CodeEditState(BaseModel):
    """Episode-level state snapshot."""

    model_config = ConfigDict(extra="ignore")

    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    done: bool = False
    reward: Optional[float] = None
    source_code: str = ""
