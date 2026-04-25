"""Per-template body codegen.

Each public ``render_<template>`` function takes the host node, its outgoing
edges in deterministic order, and returns a multi-line indented body suitable
for inserting after a ``def`` line. Bodies use only stdlib and never reference
unresolved names (the orchestrator ensures imports + pattern constants are
in scope).

Codegen is intentionally simple: the goal is *runnable, readable* Python that
respects template semantics, not optimal idiomatic code.
"""

from __future__ import annotations

from graphforge.graph.schema import Edge, Graph, Node
from graphforge.materializer import patterns

INDENT = "    "


# ---- helpers ---------------------------------------------------------


def _kwargs_for(edge: Edge) -> str:
    """Render an edge's arg_mapping as ``param=arg, param2=arg2``."""
    return ", ".join(f"{m.callee_param}={m.caller_arg}" for m in edge.arg_mapping)


def _callee_name(edge: Edge) -> str:
    """The local symbol used at the call site (just the function name).

    The orchestrator emits ``from <module> import <name>`` for cross-module
    callees, so the call site can always use the bare name.
    """
    return edge.callee.split(".", 1)[1]


def _indent(lines: list[str]) -> str:
    return "\n".join(INDENT + line for line in lines)


# ---- per-template renderers -----------------------------------------


def render_passthrough_call(node: Node, out_edges: list[Edge], _g: Graph) -> str:
    if len(out_edges) != 1:
        raise ValueError(
            f"passthrough_call on {node.qualified_name} requires 1 out-edge, "
            f"got {len(out_edges)}"
        )
    e = out_edges[0]
    return _indent([f"return {_callee_name(e)}({_kwargs_for(e)})"])


def render_sequential_calls(node: Node, out_edges: list[Edge], _g: Graph) -> str:
    if not out_edges:
        raise ValueError(
            f"sequential_calls on {node.qualified_name} requires >=1 out-edge"
        )
    lines: list[str] = []
    for e in out_edges[:-1]:
        lines.append(f"{_callee_name(e)}({_kwargs_for(e)})")
    last = out_edges[-1]
    lines.append(f"return {_callee_name(last)}({_kwargs_for(last)})")
    return _indent(lines)


def render_validate_with_regex(node: Node, out_edges: list[Edge], _g: Graph) -> str:
    if out_edges:
        raise ValueError(
            f"validate_with_regex on {node.qualified_name} must have 0 out-edges"
        )
    pattern_name = str(node.body_template_args.get("pattern", ""))
    if patterns.get_pattern(pattern_name) is None:
        raise ValueError(
            f"unknown regex pattern {pattern_name!r} on {node.qualified_name}; "
            f"known: {patterns.known_patterns()}"
        )
    constant = patterns.constant_name(pattern_name)
    # The host signature is expected to be (s: str) -> bool — but we just use
    # the first parameter name, whatever it is, to be tolerant.
    from graphforge.actions.signature import parse_signature
    parsed = parse_signature(node.signature)
    if not parsed.parameters:
        raise ValueError(
            f"validate_with_regex on {node.qualified_name} requires "
            f"at least one parameter"
        )
    arg = parsed.parameters[0].name
    return _indent([f"return re.match({constant}, {arg}) is not None"])


def render_early_return_guard(node: Node, out_edges: list[Edge], _g: Graph) -> str:
    if len(out_edges) != 1:
        raise ValueError(
            f"early_return_guard on {node.qualified_name} requires 1 out-edge"
        )
    condition = str(node.body_template_args.get("condition", "True"))
    e = out_edges[0]
    return _indent(
        [
            f"if not ({condition}):",
            f"{INDENT}return None",
            f"return {_callee_name(e)}({_kwargs_for(e)})",
        ]
    )


def render_try_call_with_fallback(node: Node, out_edges: list[Edge], _g: Graph) -> str:
    if len(out_edges) != 2:
        raise ValueError(
            f"try_call_with_fallback on {node.qualified_name} requires "
            f"exactly 2 out-edges (primary, fallback)"
        )
    primary, fallback = out_edges
    return _indent(
        [
            "try:",
            f"{INDENT}return {_callee_name(primary)}({_kwargs_for(primary)})",
            "except Exception:",
            f"{INDENT}return {_callee_name(fallback)}({_kwargs_for(fallback)})",
        ]
    )


def render_leaf_constant(node: Node, out_edges: list[Edge], _g: Graph) -> str:
    if out_edges:
        raise ValueError(
            f"leaf_constant on {node.qualified_name} must have 0 out-edges"
        )
    if "value" not in node.body_template_args:
        raise ValueError(
            f"leaf_constant on {node.qualified_name} requires args.value"
        )
    value = node.body_template_args["value"]
    return _indent([f"return {value!r}"])


# ---- registry --------------------------------------------------------


_RENDERERS: dict[str, object] = {
    "passthrough_call": render_passthrough_call,
    "sequential_calls": render_sequential_calls,
    "validate_with_regex": render_validate_with_regex,
    "early_return_guard": render_early_return_guard,
    "try_call_with_fallback": render_try_call_with_fallback,
    "leaf_constant": render_leaf_constant,
}


def render_body(node: Node, out_edges: list[Edge], graph: Graph) -> str:
    """Render the body for ``node`` based on its attached body template."""
    if node.body_template is None:
        # No body attached yet — emit a placeholder so the file still parses.
        return _indent(['raise NotImplementedError("body not attached")'])
    fn = _RENDERERS.get(node.body_template)
    if fn is None:
        raise ValueError(
            f"no codegen for template {node.body_template!r} on {node.qualified_name}"
        )
    return fn(node, out_edges, graph)  # type: ignore[operator]


def template_imports(template: str | None) -> set[str]:
    """Stdlib imports a template needs, beyond cross-module function imports."""
    if template == "validate_with_regex":
        return {"re"}
    return set()
