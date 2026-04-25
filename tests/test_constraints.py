"""Tests for the constraint checker."""

from __future__ import annotations

import pytest

from graphforge.actions import dispatch
from graphforge.actions.schema import (
    AddEdge,
    AddModule,
    AddNode,
    AttachBody,
)
from graphforge.constraints import (
    AcyclicImports,
    EdgeExists,
    Materializes,
    ModuleCount,
    ModuleResponsibility,
    ModuleSizeMax,
    NodeAbsent,
    NodeExists,
    SatisfactionReport,
    check,
    evaluate_all,
)
from graphforge.graph import ArgMapping, Graph


@pytest.fixture
def g() -> Graph:
    g = Graph.empty()
    dispatch(g, AddModule(name="validators", responsibility="validation"))
    dispatch(g, AddModule(name="main", responsibility="orchestration"))
    dispatch(g, AddNode(name="is_email", module="validators", signature="(s: str) -> bool"))
    dispatch(
        g,
        AttachBody(
            name="is_email", module="validators",
            template="validate_with_regex", args={"pattern": "EMAIL"},
        ),
    )
    dispatch(g, AddNode(name="run", module="main", signature="(s: str) -> bool"))
    dispatch(
        g,
        AddEdge(
            caller="main.run", callee="validators.is_email",
            arg_mapping=[ArgMapping(caller_arg="s", callee_param="s")],
        ),
    )
    dispatch(g, AttachBody(name="run", module="main", template="passthrough_call"))
    return g


# ---- per-kind --------------------------------------------------------


def test_node_exists(g: Graph):
    assert check(g, NodeExists(name="is_email", module="validators")) is True
    assert check(g, NodeExists(name="missing", module="validators")) is False
    assert check(g, NodeExists(name="is_email", module="wrong_module")) is False


def test_node_absent(g: Graph):
    assert check(g, NodeAbsent(name="bogus", module="validators")) is True
    assert check(g, NodeAbsent(name="is_email", module="validators")) is False


def test_edge_exists(g: Graph):
    assert check(g, EdgeExists(caller="main.run", callee="validators.is_email")) is True
    assert check(g, EdgeExists(caller="main.run", callee="validators.bogus")) is False


def test_module_count(g: Graph):
    assert check(g, ModuleCount(n=2)) is True
    assert check(g, ModuleCount(n=3)) is False


def test_module_size_max(g: Graph):
    assert check(g, ModuleSizeMax(module="validators", n=1)) is True
    assert check(g, ModuleSizeMax(module="validators", n=0)) is False
    assert check(g, ModuleSizeMax(module="missing", n=10)) is True  # vacuous


def test_module_responsibility(g: Graph):
    assert check(g, ModuleResponsibility(module="validators", responsibility="validation")) is True
    assert check(g, ModuleResponsibility(module="validators", responsibility="io")) is False
    assert check(g, ModuleResponsibility(module="missing", responsibility="io")) is False


def test_acyclic_imports(g: Graph):
    assert check(g, AcyclicImports()) is True


def test_materializes(g: Graph):
    assert check(g, Materializes()) is True


def test_materializes_false_on_corrupted_graph(g: Graph):
    """Tamper to provoke a codegen failure; constraint must report False, not raise."""
    g.find_node("is_email", "validators").body_template_args = {"pattern": "DOES_NOT_EXIST"}
    assert check(g, Materializes()) is False


# ---- evaluate_all ----------------------------------------------------


def test_evaluate_all_aggregates(g: Graph):
    spec = [
        NodeExists(name="is_email", module="validators"),
        NodeExists(name="missing", module="validators"),
        ModuleCount(n=2),
        Materializes(),
    ]
    rep = evaluate_all(g, spec)
    assert rep.total == 4
    assert len(rep.satisfied) == 3
    assert len(rep.unsatisfied) == 1
    assert isinstance(rep.unsatisfied[0], NodeExists)
    assert rep.unsatisfied[0].name == "missing"
    assert rep.all_satisfied is False


def test_evaluate_all_empty_is_not_all_satisfied():
    rep = evaluate_all(Graph.empty(), [])
    assert rep.all_satisfied is False  # vacuous-success guard


def test_split_by_family(g: Graph):
    """`materializes` is in the structural family for reward scoring; the
    behavioral family is reserved for property-test results (TODO)."""
    spec = [
        NodeExists(name="is_email", module="validators"),
        Materializes(),
    ]
    rep = evaluate_all(g, spec)
    structural, behavioral = rep.split_by_family()
    assert len(structural.satisfied) == 2
    assert len(behavioral.satisfied) == 0
    sat_kinds = {c.kind for c in structural.satisfied}
    assert sat_kinds == {"node_exists", "materializes"}
