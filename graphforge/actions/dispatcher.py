"""Atomic action dispatcher.

Applies an :class:`Action` to a :class:`Graph`. Every mutation is atomic:
the dispatcher snapshots the graph before the handler runs and restores it on
any failure. Failures surface as :class:`ActionError` with a stable code, never
as silent partial state.

Information actions (query_*, materialize_*, run_*) are routed but their
implementations live in their respective subsystems and are stubbed for now.
``submit`` returns a sentinel so the episode runner can recognize termination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graphforge.actions import errors as E
from graphforge.actions.schema import (
    Action,
    AddEdge,
    AddModule,
    AddNode,
    AttachBody,
    MaterializeAndValidate,
    QuerySpec,
    QuerySubgraph,
    QueryTypes,
    RemoveEdge,
    RemoveModule,
    RemoveNode,
    RunBehavioralTests,
    SetNodeModule,
    Submit,
)
from graphforge.actions.signature import parse_signature
from graphforge.graph.schema import (
    ArgMapping,
    Edge,
    Graph,
    Module,
    Node,
)
from graphforge.templates import get_template, validate_args


# ---- result envelope -------------------------------------------------


@dataclass
class ActionResult:
    """Envelope returned by :func:`dispatch`."""

    ok: bool
    payload: dict[str, Any]
    terminal: bool = False

    @classmethod
    def success(cls, **payload: Any) -> "ActionResult":
        return cls(ok=True, payload=payload, terminal=False)

    @classmethod
    def failure(cls, err: E.ActionError) -> "ActionResult":
        return cls(ok=False, payload=err.to_dict(), terminal=False)

    @classmethod
    def terminate(cls, **payload: Any) -> "ActionResult":
        return cls(ok=True, payload=payload, terminal=True)


# ---- dispatcher ------------------------------------------------------


def dispatch(graph: Graph, action: Action) -> ActionResult:
    """Apply ``action`` to ``graph`` in place. Atomic on failure.

    On any handler exception (including :class:`ActionError`) the graph is
    rolled back to the pre-call snapshot.
    """
    snap = graph.snapshot()
    try:
        return _route(graph, action)
    except E.ActionError as err:
        _restore(graph, snap)
        return ActionResult.failure(err)
    except Exception as exc:  # pragma: no cover — unexpected handler bug
        _restore(graph, snap)
        return ActionResult.failure(
            E.ActionError(E.SCHEMA_REJECTION, f"unhandled: {exc}")
        )


def _restore(graph: Graph, snap: Graph) -> None:
    graph.modules = snap.modules
    graph.nodes = snap.nodes
    graph.edges = snap.edges


def _route(graph: Graph, action: Action) -> ActionResult:
    # Mutations
    if isinstance(action, AddModule):
        return _h_add_module(graph, action)
    if isinstance(action, RemoveModule):
        return _h_remove_module(graph, action)
    if isinstance(action, AddNode):
        return _h_add_node(graph, action)
    if isinstance(action, RemoveNode):
        return _h_remove_node(graph, action)
    if isinstance(action, SetNodeModule):
        return _h_set_node_module(graph, action)
    if isinstance(action, AttachBody):
        return _h_attach_body(graph, action)
    if isinstance(action, AddEdge):
        return _h_add_edge(graph, action)
    if isinstance(action, RemoveEdge):
        return _h_remove_edge(graph, action)
    # Information (delegated; stubs for now)
    if isinstance(action, QuerySpec):
        return _h_query_spec(graph, action)
    if isinstance(action, QuerySubgraph):
        return _h_query_subgraph(graph, action)
    if isinstance(action, QueryTypes):
        return _h_query_types(graph, action)
    if isinstance(action, MaterializeAndValidate):
        return _h_materialize(graph, action)
    if isinstance(action, RunBehavioralTests):
        return _h_run_tests(graph, action)
    if isinstance(action, Submit):
        return _h_submit(graph, action)
    raise E.ActionError(E.SCHEMA_REJECTION, f"unknown action: {type(action).__name__}")


# ---- mutation handlers ----------------------------------------------


def _h_add_module(graph: Graph, a: AddModule) -> ActionResult:
    if graph.find_module(a.name) is not None:
        raise E.ActionError(
            E.NAME_COLLISION, f"module {a.name!r} already exists", name=a.name
        )
    graph.modules.append(Module(name=a.name, responsibility=a.responsibility))
    return ActionResult.success(added_module=a.name)


def _h_remove_module(graph: Graph, a: RemoveModule) -> ActionResult:
    mod = graph.find_module(a.name)
    if mod is None:
        raise E.ActionError(E.UNKNOWN_MODULE, f"module {a.name!r} does not exist", name=a.name)
    if any(n.module == a.name for n in graph.nodes):
        raise E.ActionError(
            E.MODULE_NOT_EMPTY,
            f"module {a.name!r} still contains nodes",
            name=a.name,
            node_count=sum(1 for n in graph.nodes if n.module == a.name),
        )
    graph.modules = [m for m in graph.modules if m.name != a.name]
    return ActionResult.success(removed_module=a.name)


def _h_add_node(graph: Graph, a: AddNode) -> ActionResult:
    if graph.find_module(a.module) is None:
        raise E.ActionError(E.UNKNOWN_MODULE, f"module {a.module!r} does not exist", name=a.module)
    if graph.find_node(a.name, a.module) is not None:
        raise E.ActionError(
            E.NAME_COLLISION,
            f"node {a.module}.{a.name} already exists",
            name=a.name,
            module=a.module,
        )
    # Surface signature parse — catches errors that the pydantic regex misses.
    try:
        parse_signature(a.signature)
    except ValueError as ve:
        raise E.ActionError(E.SCHEMA_REJECTION, str(ve), signature=a.signature) from ve
    decl_order = max((n.decl_order for n in graph.nodes), default=-1) + 1
    graph.nodes.append(
        Node(
            name=a.name,
            module=a.module,
            signature=a.signature,
            purity=a.purity,
            error_policy=a.error_policy,
            decl_order=decl_order,
        )
    )
    return ActionResult.success(added_node=f"{a.module}.{a.name}", decl_order=decl_order)


def _h_remove_node(graph: Graph, a: RemoveNode) -> ActionResult:
    n = graph.find_node(a.name, a.module)
    if n is None:
        raise E.ActionError(
            E.UNKNOWN_NODE, f"node {a.module}.{a.name} does not exist", name=a.name, module=a.module
        )
    qn = n.qualified_name
    refs = [e for e in graph.edges if e.caller == qn or e.callee == qn]
    if refs:
        raise E.ActionError(
            E.NODE_HAS_REFERENCES,
            f"node {qn} is referenced by {len(refs)} edge(s)",
            name=a.name,
            module=a.module,
            referencing_edges=[(e.caller, e.callee) for e in refs],
        )
    graph.nodes = [m for m in graph.nodes if not (m.name == a.name and m.module == a.module)]
    return ActionResult.success(removed_node=qn)


def _h_set_node_module(graph: Graph, a: SetNodeModule) -> ActionResult:
    n = graph.find_node(a.name, a.current_module)
    if n is None:
        raise E.ActionError(
            E.UNKNOWN_NODE,
            f"node {a.current_module}.{a.name} does not exist",
            name=a.name,
            module=a.current_module,
        )
    new_mod = graph.find_module(a.new_module)
    if new_mod is None:
        raise E.ActionError(
            E.UNKNOWN_MODULE,
            f"target module {a.new_module!r} does not exist",
            name=a.new_module,
        )
    if graph.find_node(a.name, a.new_module) is not None:
        raise E.ActionError(
            E.NAME_COLLISION,
            f"node named {a.name!r} already exists in {a.new_module!r}",
            name=a.name,
            module=a.new_module,
        )
    old_qn = n.qualified_name
    new_qn = f"{a.new_module}.{a.name}"
    n.module = a.new_module
    # Rewrite edge endpoints that referred to the old qualified name.
    for e in graph.edges:
        if e.caller == old_qn:
            e.caller = new_qn
        if e.callee == old_qn:
            e.callee = new_qn
    # Post-condition: rewriting must not have introduced an import cycle.
    if graph.has_module_cycle():
        raise E.ActionError(
            E.WOULD_CREATE_CYCLE,
            f"moving {old_qn} -> {new_qn} would create an import cycle",
            from_qn=old_qn,
            to_qn=new_qn,
        )
    return ActionResult.success(moved_node={"from": old_qn, "to": new_qn})


def _h_attach_body(graph: Graph, a: AttachBody) -> ActionResult:
    n = graph.find_node(a.name, a.module)
    if n is None:
        raise E.ActionError(
            E.UNKNOWN_NODE,
            f"node {a.module}.{a.name} does not exist",
            name=a.name,
            module=a.module,
        )
    spec = get_template(a.template)
    if spec is None:
        raise E.ActionError(
            E.UNKNOWN_TEMPLATE, f"unknown template {a.template!r}", template=a.template
        )
    problems = validate_args(a.template, a.args)
    if problems:
        raise E.ActionError(
            E.TEMPLATE_ARGS_INVALID,
            f"args invalid for template {a.template!r}: {'; '.join(problems)}",
            template=a.template,
            problems=problems,
        )
    out_d = graph.fan_out(n.qualified_name)
    in_d = graph.fan_in(n.qualified_name)
    if not spec.edges_ok(out_d, in_d):
        raise E.ActionError(
            E.TEMPLATE_ARGS_INVALID,
            f"template {a.template!r} requires different edge structure "
            f"(out_d={out_d}, in_d={in_d})",
            template=a.template,
            out_degree=out_d,
            in_degree=in_d,
        )
    n.body_template = a.template
    n.body_template_args = dict(a.args)
    return ActionResult.success(
        attached={"node": n.qualified_name, "template": a.template}
    )


def _h_add_edge(graph: Graph, a: AddEdge) -> ActionResult:
    caller = graph.find_node_qualified(a.caller)
    callee = graph.find_node_qualified(a.callee)
    if caller is None:
        raise E.ActionError(E.UNKNOWN_NODE, f"caller {a.caller!r} does not exist", node=a.caller)
    if callee is None:
        raise E.ActionError(E.UNKNOWN_NODE, f"callee {a.callee!r} does not exist", node=a.callee)
    if graph.find_edge(a.caller, a.callee) is not None:
        raise E.ActionError(
            E.DUPLICATE_EDGE,
            f"edge {a.caller} -> {a.callee} already exists",
            caller=a.caller,
            callee=a.callee,
        )
    # Validate arg_mapping covers all required parameters of callee.
    callee_sig = parse_signature(callee.signature)
    caller_sig = parse_signature(caller.signature)
    mapped_callee = {m.callee_param for m in a.arg_mapping}
    mapped_caller = {m.caller_arg for m in a.arg_mapping}
    missing = set(callee_sig.required_params) - mapped_callee
    if missing:
        raise E.ActionError(
            E.ARG_MAPPING_INVALID,
            f"arg_mapping is missing required callee params: {sorted(missing)}",
            missing=sorted(missing),
        )
    bogus_callee = mapped_callee - set(callee_sig.all_params)
    if bogus_callee:
        raise E.ActionError(
            E.ARG_MAPPING_INVALID,
            f"arg_mapping references unknown callee params: {sorted(bogus_callee)}",
            unknown=sorted(bogus_callee),
        )
    bogus_caller = mapped_caller - set(caller_sig.all_params)
    if bogus_caller:
        raise E.ActionError(
            E.ARG_MAPPING_INVALID,
            f"arg_mapping references unknown caller args: {sorted(bogus_caller)}",
            unknown=sorted(bogus_caller),
        )
    # Add tentatively; check post-condition.
    graph.edges.append(
        Edge(
            caller=a.caller,
            callee=a.callee,
            arg_mapping=[ArgMapping(**m.model_dump()) for m in a.arg_mapping],
        )
    )
    if graph.has_module_cycle():
        raise E.ActionError(
            E.WOULD_CREATE_CYCLE,
            f"adding edge {a.caller} -> {a.callee} would create an import cycle",
            caller=a.caller,
            callee=a.callee,
        )
    return ActionResult.success(added_edge={"caller": a.caller, "callee": a.callee})


def _h_remove_edge(graph: Graph, a: RemoveEdge) -> ActionResult:
    e = graph.find_edge(a.caller, a.callee)
    if e is None:
        raise E.ActionError(
            E.UNKNOWN_EDGE,
            f"edge {a.caller} -> {a.callee} does not exist",
            caller=a.caller,
            callee=a.callee,
        )
    graph.edges = [
        x for x in graph.edges if not (x.caller == a.caller and x.callee == a.callee)
    ]
    return ActionResult.success(removed_edge={"caller": a.caller, "callee": a.callee})


# ---- info / terminal handlers (stubs) -------------------------------


def _h_query_spec(graph: Graph, a: QuerySpec) -> ActionResult:
    # TODO: route to graphforge.constraints once tasks/specs are wired in.
    return ActionResult.success(
        not_implemented="query_spec routed via dispatcher; constraint engine TODO",
        constraint_kind=a.constraint_kind,
    )


def _h_query_subgraph(graph: Graph, a: QuerySubgraph) -> ActionResult:
    scope = a.scope
    if scope.startswith("module:"):
        mod = scope[len("module:") :]
        nodes = [n.model_dump() for n in graph.nodes_in_module(mod)]
        edges = [
            e.model_dump()
            for e in graph.edges
            if e.caller.split(".")[0] == mod and e.callee.split(".")[0] == mod
        ]
        return ActionResult.success(scope=scope, nodes=nodes, edges=edges)
    if scope.startswith("neighbors:"):
        qn = scope[len("neighbors:") :]
        return ActionResult.success(
            scope=scope,
            callers=graph.callers_of(qn),
            callees=graph.callees_of(qn),
        )
    if scope.startswith("path:"):
        # TODO: shortest-path search over call graph.
        return ActionResult.success(
            scope=scope, not_implemented="path search TODO"
        )
    raise E.ActionError(E.SCHEMA_REJECTION, f"unrecognized subgraph scope {scope!r}")


def _h_query_types(graph: Graph, a: QueryTypes) -> ActionResult:
    # TODO: delegate to graphforge.types.
    return ActionResult.success(
        scope=a.scope, not_implemented="type engine TODO"
    )


def _h_materialize(graph: Graph, a: MaterializeAndValidate) -> ActionResult:
    """Project the graph to source and run the parse-only validator gate.

    Heavier validation gates (mypy --strict, import-resolution, behavioral
    tests) are added to this action's report as their subsystems land.
    """
    from graphforge.materializer import materialize as _materialize
    from graphforge.validator import full_check

    try:
        files = _materialize(graph)
    except ValueError as ve:
        # Codegen rejected the graph (e.g. unknown pattern, template/edge
        # structure mismatch missed by the dispatcher's preconditions).
        raise E.ActionError(
            E.SCHEMA_REJECTION, f"materialization failed: {ve}"
        ) from ve
    report = full_check(files)
    return ActionResult.success(
        files=list(files.keys()),
        bytes_total=sum(len(s) for s in files.values()),
        report=report.to_dict(),
    )


def _h_run_tests(graph: Graph, a: RunBehavioralTests) -> ActionResult:
    # TODO: delegate to graphforge.behavioral.
    raise E.ActionError(
        E.SCHEMA_REJECTION, "run_behavioral_tests is not yet implemented"
    )


def _h_submit(graph: Graph, a: Submit) -> ActionResult:
    return ActionResult.terminate(submitted=True)
