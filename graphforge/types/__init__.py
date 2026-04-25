"""Type engine.

Responsibilities (PROPOSAL.md §3.1, §4.1, §5.2):

  * Parse function signatures into a typed parameter list.
  * Validate that every edge's ``arg_mapping`` has type-compatible flow
    between caller's available bindings and callee's parameter types.
  * Validate that every body template's expected types match the host
    node's signature and outgoing edges.
  * Detect ``Any`` usage for the ``no_any_types`` constraint.
  * Surface a typed view of the graph for ``query_types``.

The cheap signature parser at :mod:`graphforge.actions.signature` extracts
parameter names; this module subsumes it with full annotation parsing using
``ast.parse`` over a synthetic ``def`` so that we get Python's own grammar
for free.

Public surface (TODO):

    parse_typed_signature(sig: str) -> TypedSignature
    edge_type_flow(graph, edge) -> list[TypeIssue]
    type_view(graph, scope) -> dict
    has_any(graph) -> list[str]
"""

from __future__ import annotations


def parse_typed_signature(sig: str) -> object:  # pragma: no cover — TODO
    raise NotImplementedError("type engine — parse_typed_signature TODO")


def edge_type_flow(graph: object, edge: object) -> list[object]:  # pragma: no cover
    raise NotImplementedError("type engine — edge_type_flow TODO")


def type_view(graph: object, scope: str) -> dict[str, object]:  # pragma: no cover
    raise NotImplementedError("type engine — type_view TODO")


def has_any(graph: object) -> list[str]:  # pragma: no cover
    raise NotImplementedError("type engine — has_any TODO")
