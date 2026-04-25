"""Tests for the action dispatcher.

Covers:
  * success paths for every implemented action
  * representative failure modes (collision, unknown ref, would-create-cycle,
    template arg mismatch, arg-mapping coverage)
  * atomic rollback — failed actions leave the graph byte-identical
"""

from __future__ import annotations

import pytest

from graphforge.actions import dispatch
from graphforge.actions.errors import (
    ARG_MAPPING_INVALID,
    DUPLICATE_EDGE,
    MODULE_NOT_EMPTY,
    NAME_COLLISION,
    NODE_HAS_REFERENCES,
    TEMPLATE_ARGS_INVALID,
    UNKNOWN_EDGE,
    UNKNOWN_MODULE,
    UNKNOWN_NODE,
    UNKNOWN_TEMPLATE,
    WOULD_CREATE_CYCLE,
)
from graphforge.actions.schema import (
    AddEdge,
    AddModule,
    AddNode,
    AttachBody,
    MaterializeAndValidate,
    QuerySubgraph,
    RemoveEdge,
    RemoveModule,
    RemoveNode,
    SetNodeModule,
    Submit,
)
from graphforge.graph import ArgMapping, Graph


# ---- helpers ---------------------------------------------------------


@pytest.fixture
def g() -> Graph:
    return Graph.empty()


def _seed_two_module_graph(g: Graph) -> None:
    """Seed: validators.is_email, main.register; one edge between them."""
    assert dispatch(g, AddModule(name="validators", responsibility="validation")).ok
    assert dispatch(g, AddModule(name="main", responsibility="orchestration")).ok
    assert dispatch(
        g,
        AddNode(name="is_email", module="validators", signature="(s: str) -> bool"),
    ).ok
    assert dispatch(
        g,
        AddNode(name="register", module="main", signature="(email: str) -> None"),
    ).ok
    assert dispatch(
        g,
        AddEdge(
            caller="main.register",
            callee="validators.is_email",
            arg_mapping=[ArgMapping(caller_arg="email", callee_param="s")],
        ),
    ).ok


# ---- success paths ---------------------------------------------------


def test_add_module(g: Graph):
    r = dispatch(g, AddModule(name="m", responsibility="io"))
    assert r.ok
    assert r.payload["added_module"] == "m"
    assert g.find_module("m") is not None


def test_add_node_assigns_decl_order(g: Graph):
    dispatch(g, AddModule(name="m", responsibility="io"))
    r1 = dispatch(g, AddNode(name="a", module="m", signature="() -> int"))
    r2 = dispatch(g, AddNode(name="b", module="m", signature="() -> int"))
    assert r1.ok and r2.ok
    assert r1.payload["decl_order"] == 0
    assert r2.payload["decl_order"] == 1


def test_add_edge_with_full_arg_mapping(g: Graph):
    _seed_two_module_graph(g)
    assert g.fan_out("main.register") == 1


def test_remove_edge(g: Graph):
    _seed_two_module_graph(g)
    r = dispatch(g, RemoveEdge(caller="main.register", callee="validators.is_email"))
    assert r.ok
    assert g.fan_out("main.register") == 0


def test_remove_node_after_clearing_edges(g: Graph):
    _seed_two_module_graph(g)
    dispatch(g, RemoveEdge(caller="main.register", callee="validators.is_email"))
    r = dispatch(g, RemoveNode(name="register", module="main"))
    assert r.ok


def test_remove_module_after_clearing_nodes(g: Graph):
    dispatch(g, AddModule(name="m", responsibility="io"))
    r = dispatch(g, RemoveModule(name="m"))
    assert r.ok


def test_set_node_module_rewrites_edges(g: Graph):
    _seed_two_module_graph(g)
    # Move register to a new module 'svc'.
    dispatch(g, AddModule(name="svc", responsibility="orchestration"))
    r = dispatch(
        g,
        SetNodeModule(name="register", current_module="main", new_module="svc"),
    )
    assert r.ok
    assert g.find_node("register", "svc") is not None
    assert g.find_node("register", "main") is None
    # Edge should now be svc.register -> validators.is_email
    assert g.find_edge("svc.register", "validators.is_email") is not None


def test_attach_body_with_required_args(g: Graph):
    dispatch(g, AddModule(name="v", responsibility="validation"))
    dispatch(g, AddNode(name="check", module="v", signature="(s: str) -> bool"))
    r = dispatch(
        g,
        AttachBody(
            name="check", module="v", template="validate_with_regex",
            args={"pattern": "EMAIL"},
        ),
    )
    assert r.ok
    n = g.find_node("check", "v")
    assert n is not None
    assert n.body_template == "validate_with_regex"
    assert n.body_template_args == {"pattern": "EMAIL"}


def test_query_subgraph_neighbors(g: Graph):
    _seed_two_module_graph(g)
    r = dispatch(g, QuerySubgraph(scope="neighbors:main.register"))
    assert r.ok
    assert "validators.is_email" in r.payload["callees"]


def test_submit_terminates(g: Graph):
    r = dispatch(g, Submit())
    assert r.ok
    assert r.terminal is True


# ---- failure modes ---------------------------------------------------


def test_add_module_collision(g: Graph):
    dispatch(g, AddModule(name="m", responsibility="io"))
    r = dispatch(g, AddModule(name="m", responsibility="io"))
    assert not r.ok
    assert r.payload["error"] == NAME_COLLISION


def test_add_node_unknown_module(g: Graph):
    r = dispatch(g, AddNode(name="x", module="nope", signature="() -> int"))
    assert not r.ok
    assert r.payload["error"] == UNKNOWN_MODULE


def test_add_node_collision(g: Graph):
    dispatch(g, AddModule(name="m", responsibility="io"))
    dispatch(g, AddNode(name="x", module="m", signature="() -> int"))
    r = dispatch(g, AddNode(name="x", module="m", signature="() -> int"))
    assert not r.ok
    assert r.payload["error"] == NAME_COLLISION


def test_remove_module_not_empty(g: Graph):
    dispatch(g, AddModule(name="m", responsibility="io"))
    dispatch(g, AddNode(name="x", module="m", signature="() -> int"))
    r = dispatch(g, RemoveModule(name="m"))
    assert not r.ok
    assert r.payload["error"] == MODULE_NOT_EMPTY


def test_remove_node_with_references(g: Graph):
    _seed_two_module_graph(g)
    r = dispatch(g, RemoveNode(name="register", module="main"))
    assert not r.ok
    assert r.payload["error"] == NODE_HAS_REFERENCES


def test_remove_edge_unknown(g: Graph):
    r = dispatch(g, RemoveEdge(caller="a.b", callee="c.d"))
    assert not r.ok
    assert r.payload["error"] == UNKNOWN_EDGE


def test_add_edge_unknown_endpoint(g: Graph):
    dispatch(g, AddModule(name="m", responsibility="io"))
    dispatch(g, AddNode(name="x", module="m", signature="() -> int"))
    r = dispatch(g, AddEdge(caller="m.x", callee="nope.nope"))
    assert not r.ok
    assert r.payload["error"] == UNKNOWN_NODE


def test_add_edge_duplicate(g: Graph):
    _seed_two_module_graph(g)
    r = dispatch(
        g,
        AddEdge(
            caller="main.register",
            callee="validators.is_email",
            arg_mapping=[ArgMapping(caller_arg="email", callee_param="s")],
        ),
    )
    assert not r.ok
    assert r.payload["error"] == DUPLICATE_EDGE


def test_add_edge_arg_mapping_missing_param(g: Graph):
    """Callee has param 's' but arg_mapping is empty → reject."""
    dispatch(g, AddModule(name="v", responsibility="validation"))
    dispatch(g, AddModule(name="m", responsibility="orchestration"))
    dispatch(g, AddNode(name="check", module="v", signature="(s: str) -> bool"))
    dispatch(g, AddNode(name="run", module="m", signature="(email: str) -> None"))
    r = dispatch(g, AddEdge(caller="m.run", callee="v.check", arg_mapping=[]))
    assert not r.ok
    assert r.payload["error"] == ARG_MAPPING_INVALID
    assert "s" in r.payload["missing"]


def test_add_edge_arg_mapping_unknown_callee_param(g: Graph):
    dispatch(g, AddModule(name="v", responsibility="validation"))
    dispatch(g, AddModule(name="m", responsibility="orchestration"))
    dispatch(g, AddNode(name="check", module="v", signature="(s: str) -> bool"))
    dispatch(g, AddNode(name="run", module="m", signature="(email: str) -> None"))
    r = dispatch(
        g,
        AddEdge(
            caller="m.run",
            callee="v.check",
            arg_mapping=[ArgMapping(caller_arg="email", callee_param="bogus")],
        ),
    )
    assert not r.ok
    assert r.payload["error"] == ARG_MAPPING_INVALID


def test_add_edge_would_create_cycle(g: Graph):
    """A.f -> B.g, then attempting B.g -> A.h must be rejected."""
    dispatch(g, AddModule(name="a", responsibility="io"))
    dispatch(g, AddModule(name="b", responsibility="computation"))
    dispatch(g, AddNode(name="f", module="a", signature="() -> int"))
    dispatch(g, AddNode(name="h", module="a", signature="() -> int"))
    dispatch(g, AddNode(name="g", module="b", signature="() -> int"))
    dispatch(g, AddEdge(caller="a.f", callee="b.g"))
    r = dispatch(g, AddEdge(caller="b.g", callee="a.h"))
    assert not r.ok
    assert r.payload["error"] == WOULD_CREATE_CYCLE


def test_attach_body_unknown_template(g: Graph):
    dispatch(g, AddModule(name="m", responsibility="io"))
    dispatch(g, AddNode(name="x", module="m", signature="() -> int"))
    r = dispatch(g, AttachBody(name="x", module="m", template="not_a_template"))
    assert not r.ok
    assert r.payload["error"] == UNKNOWN_TEMPLATE


def test_attach_body_missing_required_arg(g: Graph):
    dispatch(g, AddModule(name="v", responsibility="validation"))
    dispatch(g, AddNode(name="check", module="v", signature="(s: str) -> bool"))
    r = dispatch(
        g,
        AttachBody(name="check", module="v", template="validate_with_regex", args={}),
    )
    assert not r.ok
    assert r.payload["error"] == TEMPLATE_ARGS_INVALID


def test_attach_body_edge_structure_mismatch(g: Graph):
    """passthrough_call requires out_d == 1; the node has zero out-edges."""
    dispatch(g, AddModule(name="m", responsibility="orchestration"))
    dispatch(g, AddNode(name="x", module="m", signature="(a: int) -> int"))
    r = dispatch(g, AttachBody(name="x", module="m", template="passthrough_call"))
    assert not r.ok
    assert r.payload["error"] == TEMPLATE_ARGS_INVALID


def test_materialize_empty_graph_is_ok(g: Graph):
    """An empty graph projects to zero files and parses trivially."""
    r = dispatch(g, MaterializeAndValidate())
    assert r.ok
    assert r.payload["files"] == []
    assert r.payload["report"]["ok"] is True


def test_materialize_well_formed_graph(g: Graph):
    _seed_two_module_graph(g)
    # Attach bodies so codegen has something to render. validators.is_email
    # gets validate_with_regex, main.register gets passthrough_call.
    dispatch(
        g,
        AttachBody(
            name="is_email", module="validators",
            template="validate_with_regex", args={"pattern": "EMAIL"},
        ),
    )
    dispatch(
        g,
        AttachBody(name="register", module="main", template="passthrough_call"),
    )
    r = dispatch(g, MaterializeAndValidate())
    assert r.ok
    assert set(r.payload["files"]) == {"validators.py", "main.py"}
    assert r.payload["report"]["ok"] is True


def test_materialize_with_unknown_pattern_fails_cleanly(g: Graph):
    """Codegen errors from corrupt template args surface as ActionError."""
    dispatch(g, AddModule(name="v", responsibility="validation"))
    dispatch(g, AddNode(name="check", module="v", signature="(s: str) -> bool"))
    dispatch(
        g,
        AttachBody(
            name="check", module="v", template="validate_with_regex",
            args={"pattern": "EMAIL"},
        ),
    )
    # Tamper directly with the node to provoke codegen failure.
    g.find_node("check", "v").body_template_args = {"pattern": "DOES_NOT_EXIST"}
    r = dispatch(g, MaterializeAndValidate())
    assert not r.ok
    assert "unknown regex pattern" in r.payload["message"]


# ---- atomic rollback -------------------------------------------------


def test_rollback_on_failure_leaves_graph_unchanged(g: Graph):
    _seed_two_module_graph(g)
    snapshot_hash = g.structural_hash()
    # This will fail on cycle creation.
    dispatch(g, AddNode(name="b", module="validators", signature="() -> int"))
    snapshot_hash2 = g.structural_hash()
    assert snapshot_hash != snapshot_hash2  # successful add changed the hash
    # Now provoke a failure and make sure the hash doesn't change.
    pre = g.structural_hash()
    r = dispatch(g, AddModule(name="validators", responsibility="validation"))  # collision
    assert not r.ok
    assert g.structural_hash() == pre


def test_rollback_on_arg_mapping_failure(g: Graph):
    """A failed add_edge must not leave a partial edge in the graph."""
    dispatch(g, AddModule(name="v", responsibility="validation"))
    dispatch(g, AddModule(name="m", responsibility="orchestration"))
    dispatch(g, AddNode(name="check", module="v", signature="(s: str) -> bool"))
    dispatch(g, AddNode(name="run", module="m", signature="(email: str) -> None"))
    pre = g.structural_hash()
    r = dispatch(g, AddEdge(caller="m.run", callee="v.check", arg_mapping=[]))
    assert not r.ok
    assert g.structural_hash() == pre  # exactly the same — the edge did not enter the graph
