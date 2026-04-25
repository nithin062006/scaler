"""Canonical graph schema.

The graph is the single source of truth for an in-progress program. Every
materialization is a deterministic function of (graph, template library).

Wire format mirrors the JSON shape documented in PROPOSAL.md §3.1, exactly:

    {
      "modules": [{"name": ..., "responsibility": ...}, ...],
      "nodes":   [{"name": ..., "module": ..., "signature": ...,
                   "body_template": ..., "body_template_args": {...},
                   "purity": ..., "error_policy": ..., "decl_order": ...}, ...],
      "edges":   [{"caller": "<module>.<name>",
                   "callee": "<module>.<name>",
                   "arg_mapping": [{"caller_arg": ..., "callee_param": ...}, ...]}, ...]
    }

This module enforces shape and well-formedness only. Higher-order invariants
(unique names, edge endpoints exist, no cycles, type-flow compatibility) are
enforced by the action dispatcher and the type engine, not the schema, so
that callers can build partial / invalid graphs and inspect why they fail.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ----------------------------------------------------------------------
# Enumerated tags
# ----------------------------------------------------------------------

# Responsibility tags constrain which kinds of nodes a module is allowed to
# host. The canonical set; new tags are added intentionally because tasks
# encode constraints against this vocabulary.
ResponsibilityTag = Literal[
    "io",
    "validation",
    "transform",
    "orchestration",
    "storage",
    "formatting",
    "lookup",
    "policy",
    "logging",
    "computation",
]

Purity = Literal["pure", "impure"]

# How a function handles errors in its body. "guard" means it includes a
# guard / try-except. "propagate" means it deliberately lets errors flow up.
# "none" is the default — no claim either way.
ErrorPolicy = Literal["guard", "propagate", "none"]


# ----------------------------------------------------------------------
# Atomic records
# ----------------------------------------------------------------------


class Module(BaseModel):
    """A declared module — one Python file at materialization time."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    name: str = Field(..., min_length=1)
    responsibility: ResponsibilityTag

    @field_validator("name")
    @classmethod
    def _name_is_identifier(cls, v: str) -> str:
        if not v.isidentifier():
            raise ValueError(f"module name {v!r} is not a Python identifier")
        if v.startswith("_"):
            raise ValueError(f"module name {v!r} must not start with an underscore")
        return v


class Node(BaseModel):
    """A declared function. ``body_template`` may be unset until attach_body."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    name: str = Field(..., min_length=1)
    module: str = Field(..., min_length=1)
    signature: str = Field(..., min_length=2)  # e.g., "(x: int) -> bool"
    body_template: Optional[str] = None
    body_template_args: dict[str, object] = Field(default_factory=dict)
    purity: Purity = "impure"
    error_policy: ErrorPolicy = "none"
    decl_order: int = 0

    @field_validator("name")
    @classmethod
    def _name_is_identifier(cls, v: str) -> str:
        if not v.isidentifier():
            raise ValueError(f"node name {v!r} is not a Python identifier")
        return v

    @field_validator("signature")
    @classmethod
    def _signature_shape(cls, v: str) -> str:
        # Cheap surface check; the type engine does the real parse.
        if not v.lstrip().startswith("("):
            raise ValueError(f"signature must start with '(': got {v!r}")
        if "->" not in v:
            raise ValueError(f"signature must include '->' return arrow: got {v!r}")
        return v

    # Convenience -----------------------------------------------------

    @property
    def qualified_name(self) -> str:
        """``<module>.<name>`` — the canonical address used on edges."""
        return f"{self.module}.{self.name}"


class ArgMapping(BaseModel):
    """How an edge wires a caller's argument to a callee's parameter."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    caller_arg: str = Field(..., min_length=1)
    callee_param: str = Field(..., min_length=1)


class Edge(BaseModel):
    """A CALLS edge. Endpoints are qualified node names ``<module>.<name>``."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    caller: str = Field(..., min_length=3)
    callee: str = Field(..., min_length=3)
    arg_mapping: list[ArgMapping] = Field(default_factory=list)

    @field_validator("caller", "callee")
    @classmethod
    def _qualified(cls, v: str) -> str:
        if v.count(".") != 1:
            raise ValueError(
                f"edge endpoint {v!r} is not qualified (expected '<module>.<name>')"
            )
        mod, name = v.split(".")
        if not mod.isidentifier() or not name.isidentifier():
            raise ValueError(f"edge endpoint {v!r} has non-identifier parts")
        return v


# ----------------------------------------------------------------------
# Graph
# ----------------------------------------------------------------------


class Graph(BaseModel):
    """Canonical graph state. Mutable; cloned via ``snapshot``/``restore``."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    modules: list[Module] = Field(default_factory=list)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    # ----- lookup ----------------------------------------------------

    def find_module(self, name: str) -> Optional[Module]:
        for m in self.modules:
            if m.name == name:
                return m
        return None

    def find_node(self, name: str, module: str) -> Optional[Node]:
        for n in self.nodes:
            if n.name == name and n.module == module:
                return n
        return None

    def find_node_qualified(self, qualified: str) -> Optional[Node]:
        if qualified.count(".") != 1:
            return None
        mod, nm = qualified.split(".")
        return self.find_node(nm, mod)

    def find_edge(self, caller: str, callee: str) -> Optional[Edge]:
        for e in self.edges:
            if e.caller == caller and e.callee == callee:
                return e
        return None

    def nodes_in_module(self, module: str) -> list[Node]:
        return [n for n in self.nodes if n.module == module]

    def callers_of(self, qualified: str) -> list[str]:
        return [e.caller for e in self.edges if e.callee == qualified]

    def callees_of(self, qualified: str) -> list[str]:
        return [e.callee for e in self.edges if e.caller == qualified]

    def fan_in(self, qualified: str) -> int:
        return len(self.callers_of(qualified))

    def fan_out(self, qualified: str) -> int:
        return len(self.callees_of(qualified))

    # ----- structural derivations ------------------------------------

    def import_edges(self) -> set[tuple[str, str]]:
        """Set of (caller_module, callee_module) pairs from cross-module edges."""
        out: set[tuple[str, str]] = set()
        for e in self.edges:
            cm = e.caller.split(".")[0]
            tm = e.callee.split(".")[0]
            if cm != tm:
                out.add((cm, tm))
        return out

    def has_module_cycle(self) -> bool:
        """True iff the cross-module import graph contains a directed cycle."""
        adj: dict[str, set[str]] = {m.name: set() for m in self.modules}
        for src, dst in self.import_edges():
            adj.setdefault(src, set()).add(dst)
            adj.setdefault(dst, set())
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {k: WHITE for k in adj}

        def visit(u: str) -> bool:
            color[u] = GRAY
            for v in adj.get(u, ()):
                if color[v] == GRAY:
                    return True
                if color[v] == WHITE and visit(v):
                    return True
            color[u] = BLACK
            return False

        return any(color[u] == WHITE and visit(u) for u in adj)

    def call_graph_depth(self) -> int:
        """Longest path length (in edges) in the function call DAG.

        If the call graph is cyclic, returns the special value -1 (callers
        should treat this as an invariant violation).
        """
        adj: dict[str, list[str]] = {n.qualified_name: [] for n in self.nodes}
        for e in self.edges:
            adj.setdefault(e.caller, []).append(e.callee)
            adj.setdefault(e.callee, [])
        memo: dict[str, int] = {}
        ON_STACK = -2

        def dfs(u: str) -> int:
            if u in memo:
                if memo[u] == ON_STACK:
                    return -1
                return memo[u]
            memo[u] = ON_STACK
            best = 0
            for v in adj.get(u, ()):
                d = dfs(v)
                if d == -1:
                    return -1
                best = max(best, d + 1)
            memo[u] = best
            return best

        results = [dfs(u) for u in adj]
        if any(r == -1 for r in results):
            return -1
        return max(results, default=0)

    # ----- copying / hashing -----------------------------------------

    def snapshot(self) -> "Graph":
        """Deep copy. Used by the dispatcher for atomic action rollback."""
        return self.model_copy(deep=True)

    def structural_hash(self) -> str:
        """Stable SHA-256 over a canonical JSON projection.

        Insensitive to list ordering on the dimensions where order is not
        semantically meaningful (modules, nodes), but sensitive to
        ``decl_order`` because that affects materialized output.
        """
        canon: dict[str, object] = {
            "modules": sorted(
                [m.model_dump() for m in self.modules],
                key=lambda d: d["name"],
            ),
            "nodes": sorted(
                [n.model_dump() for n in self.nodes],
                key=lambda d: (d["module"], d["name"]),
            ),
            "edges": sorted(
                [e.model_dump() for e in self.edges],
                key=lambda d: (d["caller"], d["callee"]),
            ),
        }
        blob = json.dumps(canon, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # ----- factories -------------------------------------------------

    @classmethod
    def empty(cls) -> "Graph":
        return cls()
