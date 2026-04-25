"""Tests for the task schema and tier-0 bank."""

from __future__ import annotations

from graphforge.actions import dispatch
from graphforge.actions.schema import (
    AddModule,
    AddNode,
    AttachBody,
)
from graphforge.constraints import evaluate_all
from graphforge.graph import Graph
from graphforge.tasks import default_task, get_task, list_tasks


def test_default_task_is_in_bank():
    task = default_task()
    assert get_task(task.id) is task
    assert task in list_tasks()


def test_visible_payload_omits_hidden():
    task = default_task()
    payload = task.visible_payload()
    assert "hidden_constraints" not in payload
    assert len(payload["visible_constraints"]) == len(task.visible_constraints)


def test_tier_0_task_is_solvable_by_oracle_actions():
    """An oracle policy that performs the right actions reaches all-satisfied."""
    task = default_task()
    g = Graph.empty()
    # Oracle solution.
    dispatch(g, AddModule(name="validators", responsibility="validation"))
    dispatch(g, AddNode(name="is_email", module="validators", signature="(s: str) -> bool"))
    dispatch(
        g,
        AttachBody(
            name="is_email", module="validators",
            template="validate_with_regex", args={"pattern": "EMAIL"},
        ),
    )
    rep = evaluate_all(g, task.all_constraints)
    assert rep.all_satisfied is True
    assert len(rep.satisfied) == len(task.all_constraints)


def test_tier_0_task_partially_satisfied_by_partial_solution():
    task = default_task()
    g = Graph.empty()
    # Only the module — no node, no body.
    dispatch(g, AddModule(name="validators", responsibility="validation"))
    rep = evaluate_all(g, task.all_constraints)
    assert not rep.all_satisfied
    # We expect at least module_count and module_responsibility satisfied.
    satisfied_kinds = {c.kind for c in rep.satisfied}
    assert "module_count" in satisfied_kinds
    assert "module_responsibility" in satisfied_kinds
    assert "node_exists" not in satisfied_kinds
