"""Constraint checker dispatch.

Each constraint kind has a small ``_check_*`` function. ``check`` routes by
isinstance and ``evaluate_all`` reports which constraints from a list are
satisfied or not.

Behavioral / materialization constraints (currently just ``materializes``)
delegate to the materializer and validator subsystems.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from graphforge.constraints.schema import (
    AcyclicImports,
    Constraint,
    EdgeExists,
    Materializes,
    ModuleCount,
    ModuleResponsibility,
    ModuleSizeMax,
    NodeAbsent,
    NodeExists,
    STRUCTURAL_KINDS,
)
from graphforge.graph.schema import Graph


@dataclass
class SatisfactionReport:
    satisfied: list[Constraint] = field(default_factory=list)
    unsatisfied: list[Constraint] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.satisfied) + len(self.unsatisfied)

    @property
    def all_satisfied(self) -> bool:
        return self.total > 0 and not self.unsatisfied

    def split_by_family(self) -> tuple["SatisfactionReport", "SatisfactionReport"]:
        """Split into (structural, behavioral) sub-reports.

        Useful for the reward engine, which scores the two families with
        different magnitudes per PROPOSAL.md §5.2.
        """
        sr = SatisfactionReport()
        br = SatisfactionReport()
        for c in self.satisfied:
            (sr if c.kind in STRUCTURAL_KINDS else br).satisfied.append(c)
        for c in self.unsatisfied:
            (sr if c.kind in STRUCTURAL_KINDS else br).unsatisfied.append(c)
        return sr, br

    def to_dict(self) -> dict[str, object]:
        return {
            "satisfied": [c.model_dump() for c in self.satisfied],
            "unsatisfied": [c.model_dump() for c in self.unsatisfied],
            "total": self.total,
            "all_satisfied": self.all_satisfied,
        }


# ---- per-kind checkers ----------------------------------------------


def _check_node_exists(g: Graph, c: NodeExists) -> bool:
    return g.find_node(c.name, c.module) is not None


def _check_node_absent(g: Graph, c: NodeAbsent) -> bool:
    return g.find_node(c.name, c.module) is None


def _check_edge_exists(g: Graph, c: EdgeExists) -> bool:
    return g.find_edge(c.caller, c.callee) is not None


def _check_module_count(g: Graph, c: ModuleCount) -> bool:
    return len(g.modules) == c.n


def _check_module_size_max(g: Graph, c: ModuleSizeMax) -> bool:
    return len(g.nodes_in_module(c.module)) <= c.n


def _check_module_responsibility(g: Graph, c: ModuleResponsibility) -> bool:
    m = g.find_module(c.module)
    return m is not None and m.responsibility == c.responsibility


def _check_acyclic_imports(g: Graph, _c: AcyclicImports) -> bool:
    return not g.has_module_cycle()


def _check_materializes(g: Graph, _c: Materializes) -> bool:
    # Imported lazily so that callers who don't use this checker don't pay
    # the cost of pulling the materializer/validator graph.
    from graphforge.materializer import materialize
    from graphforge.validator import full_check

    try:
        files = materialize(g)
    except Exception:
        return False
    return full_check(files).ok


# ---- dispatch --------------------------------------------------------


def check(graph: Graph, constraint: Constraint) -> bool:
    if isinstance(constraint, NodeExists):
        return _check_node_exists(graph, constraint)
    if isinstance(constraint, NodeAbsent):
        return _check_node_absent(graph, constraint)
    if isinstance(constraint, EdgeExists):
        return _check_edge_exists(graph, constraint)
    if isinstance(constraint, ModuleCount):
        return _check_module_count(graph, constraint)
    if isinstance(constraint, ModuleSizeMax):
        return _check_module_size_max(graph, constraint)
    if isinstance(constraint, ModuleResponsibility):
        return _check_module_responsibility(graph, constraint)
    if isinstance(constraint, AcyclicImports):
        return _check_acyclic_imports(graph, constraint)
    if isinstance(constraint, Materializes):
        return _check_materializes(graph, constraint)
    raise ValueError(f"unknown constraint kind: {constraint!r}")


def evaluate_all(graph: Graph, constraints: list[Constraint]) -> SatisfactionReport:
    rep = SatisfactionReport()
    for c in constraints:
        if check(graph, c):
            rep.satisfied.append(c)
        else:
            rep.unsatisfied.append(c)
    return rep
