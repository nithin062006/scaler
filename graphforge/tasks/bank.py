"""Tier-0 task bank.

A single hand-written task that exercises every implemented subsystem
end-to-end: build a one-module ``validators`` package with an ``is_email``
function attached to ``validate_with_regex(EMAIL)``. Tier-1+ tasks land in
follow-up modules.

Variant generation (PROPOSAL.md §2.3 — ~50 concrete variants per template
× domain vocabulary) is also TODO; for now we hand-author tasks until the
env's reward-signal shape is validated end-to-end.
"""

from __future__ import annotations

from graphforge.constraints.schema import (
    AcyclicImports,
    Materializes,
    ModuleCount,
    ModuleResponsibility,
    ModuleSizeMax,
    NodeAbsent,
    NodeExists,
)
from graphforge.tasks.schema import Task


TIER_0_EMAIL_VALIDATOR = Task(
    id="t0.email_validator",
    tier=0,
    description=(
        "Build a tiny single-module package called 'validators'. It should "
        "expose a function `is_email(s: str) -> bool` that returns True for "
        "well-formed email addresses and False otherwise. Use the "
        "`validate_with_regex` body template with the EMAIL pattern. The "
        "module must materialize cleanly to runnable Python."
    ),
    visible_constraints=[
        ModuleCount(n=1),
        ModuleResponsibility(module="validators", responsibility="validation"),
        NodeExists(name="is_email", module="validators"),
        Materializes(),
    ],
    hidden_constraints=[
        # The visible constraints already pin most of this; the hidden set
        # adds shape constraints the agent must infer from the description.
        ModuleSizeMax(module="validators", n=1),
        NodeAbsent(name="main", module="validators"),
        AcyclicImports(),
    ],
    behavioral_test_names=[],  # tier-0 has no behavioral tests
    budget=4000,
    episode_cap=20,
)


_TASKS: dict[str, Task] = {
    TIER_0_EMAIL_VALIDATOR.id: TIER_0_EMAIL_VALIDATOR,
}


def list_tasks() -> list[Task]:
    return list(_TASKS.values())


def get_task(task_id: str) -> Task | None:
    return _TASKS.get(task_id)


def default_task() -> Task:
    """The task `/reset` picks when no ``task_id`` is specified."""
    return TIER_0_EMAIL_VALIDATOR
