"""Tests for the canonical graph schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphforge.graph import ArgMapping, Edge, Graph, Module, Node


# ---- atomic records --------------------------------------------------


def test_module_requires_identifier_name():
    Module(name="validators", responsibility="validation")
    with pytest.raises(ValidationError):
        Module(name="not a module", responsibility="validation")
    with pytest.raises(ValidationError):
        Module(name="_private", responsibility="validation")


def test_module_responsibility_is_constrained():
    with pytest.raises(ValidationError):
        Module(name="foo", responsibility="totally-made-up")  # type: ignore[arg-type]


def test_node_signature_must_have_arrow_and_parens():
    Node(name="f", module="m", signature="(x: int) -> bool")
    with pytest.raises(ValidationError):
        Node(name="f", module="m", signature="x: int -> bool")
    with pytest.raises(ValidationError):
        Node(name="f", module="m", signature="(x: int)")


def test_node_qualified_name():
    n = Node(name="register", module="main", signature="() -> None")
    assert n.qualified_name == "main.register"


def test_edge_endpoints_must_be_qualified():
    Edge(caller="a.b", callee="c.d")
    with pytest.raises(ValidationError):
        Edge(caller="ab", callee="c.d")
    with pytest.raises(ValidationError):
        Edge(caller="a.b.c", callee="c.d")


def test_arg_mapping_requires_both_endpoints():
    ArgMapping(caller_arg="x", callee_param="y")
    with pytest.raises(ValidationError):
        ArgMapping(caller_arg="", callee_param="y")


# ---- graph helpers ---------------------------------------------------


@pytest.fixture
def linear_graph() -> Graph:
    """validators.is_email -> storage.put -> main.register (one path)."""
    g = Graph.empty()
    g.modules.extend(
        [
            Module(name="validators", responsibility="validation"),
            Module(name="storage", responsibility="storage"),
            Module(name="main", responsibility="orchestration"),
        ]
    )
    g.nodes.extend(
        [
            Node(name="is_email", module="validators", signature="(s: str) -> bool"),
            Node(name="put", module="storage", signature="(k: str, v: str) -> None"),
            Node(
                name="register",
                module="main",
                signature="(email: str, payload: str) -> None",
            ),
        ]
    )
    g.edges.extend(
        [
            Edge(
                caller="main.register",
                callee="validators.is_email",
                arg_mapping=[ArgMapping(caller_arg="email", callee_param="s")],
            ),
            Edge(
                caller="main.register",
                callee="storage.put",
                arg_mapping=[
                    ArgMapping(caller_arg="email", callee_param="k"),
                    ArgMapping(caller_arg="payload", callee_param="v"),
                ],
            ),
        ]
    )
    return g


def test_lookup(linear_graph: Graph):
    assert linear_graph.find_module("storage") is not None
    assert linear_graph.find_node("put", "storage") is not None
    assert linear_graph.find_node_qualified("main.register") is not None
    assert linear_graph.find_node_qualified("nope.nope") is None
    assert linear_graph.find_edge("main.register", "validators.is_email") is not None


def test_fan_in_fan_out(linear_graph: Graph):
    assert linear_graph.fan_out("main.register") == 2
    assert linear_graph.fan_in("validators.is_email") == 1
    assert linear_graph.fan_in("main.register") == 0


def test_import_edges_and_acyclic(linear_graph: Graph):
    edges = linear_graph.import_edges()
    assert ("main", "validators") in edges
    assert ("main", "storage") in edges
    assert linear_graph.has_module_cycle() is False


def test_call_graph_depth(linear_graph: Graph):
    # main.register -> {validators.is_email, storage.put}; both leaves.
    assert linear_graph.call_graph_depth() == 1


def test_call_graph_depth_detects_cycle():
    g = Graph.empty()
    g.modules.append(Module(name="m", responsibility="computation"))
    g.nodes.extend(
        [
            Node(name="a", module="m", signature="() -> int"),
            Node(name="b", module="m", signature="() -> int"),
        ]
    )
    g.edges.extend(
        [
            Edge(caller="m.a", callee="m.b"),
            Edge(caller="m.b", callee="m.a"),
        ]
    )
    assert g.call_graph_depth() == -1


def test_snapshot_and_structural_hash(linear_graph: Graph):
    snap = linear_graph.snapshot()
    h1 = linear_graph.structural_hash()
    h2 = snap.structural_hash()
    assert h1 == h2
    # Mutate the original; snapshot must remain unchanged.
    linear_graph.nodes.pop()
    assert snap.structural_hash() == h2
    assert linear_graph.structural_hash() != h1


def test_structural_hash_is_order_insensitive_for_modules():
    g1 = Graph.empty()
    g1.modules.extend(
        [
            Module(name="a", responsibility="io"),
            Module(name="b", responsibility="io"),
        ]
    )
    g2 = Graph.empty()
    g2.modules.extend(
        [
            Module(name="b", responsibility="io"),
            Module(name="a", responsibility="io"),
        ]
    )
    assert g1.structural_hash() == g2.structural_hash()
