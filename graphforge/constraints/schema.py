"""Constraint schemas (tier-0 subset).

Constraints are pydantic discriminated-union members keyed on ``kind``.
Tier-0 carves out the smallest set sufficient to express a real task and
exercise the reward engine end-to-end. The remaining vocabulary in
PROPOSAL.md §2.2 (fan_in_max, dag_depth_max, type_consistency,
behavioral_test_passes, …) lands on top of this same shape as new
discriminated members + checker functions.

Each constraint member is a pure data record. Behavior lives in
:mod:`graphforge.constraints.checker`.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from graphforge.graph.schema import ResponsibilityTag

_cfg = ConfigDict(extra="forbid")


# ---- structural ------------------------------------------------------


class NodeExists(BaseModel):
    model_config = _cfg
    kind: Literal["node_exists"] = "node_exists"
    name: str
    module: str


class NodeAbsent(BaseModel):
    model_config = _cfg
    kind: Literal["node_absent"] = "node_absent"
    name: str
    module: str


class EdgeExists(BaseModel):
    model_config = _cfg
    kind: Literal["edge_exists"] = "edge_exists"
    caller: str  # qualified
    callee: str  # qualified


class ModuleCount(BaseModel):
    model_config = _cfg
    kind: Literal["module_count"] = "module_count"
    n: int = Field(..., ge=0)


class ModuleSizeMax(BaseModel):
    model_config = _cfg
    kind: Literal["module_size_max"] = "module_size_max"
    module: str
    n: int = Field(..., ge=0)


class ModuleResponsibility(BaseModel):
    model_config = _cfg
    kind: Literal["module_responsibility"] = "module_responsibility"
    module: str
    responsibility: ResponsibilityTag


class AcyclicImports(BaseModel):
    model_config = _cfg
    kind: Literal["acyclic_imports"] = "acyclic_imports"


# ---- behavioral / materialization -----------------------------------


class Materializes(BaseModel):
    model_config = _cfg
    kind: Literal["materializes"] = "materializes"


# ---- discriminated union --------------------------------------------

Constraint = Annotated[
    Union[
        NodeExists,
        NodeAbsent,
        EdgeExists,
        ModuleCount,
        ModuleSizeMax,
        ModuleResponsibility,
        AcyclicImports,
        Materializes,
    ],
    Field(discriminator="kind"),
]


# Set of kinds considered "structural" for the reward engine's per-constraint
# +1 magnitude. The "behavioral" family is reserved for property-test results
# (BehavioralTestPasses, TODO) which earn the higher +3 magnitude. The
# ``materializes`` constraint is structural for scoring purposes; the more
# severe "Materialization fails: -8" penalty in PROPOSAL.md §5.2 is an
# independent gate driven by the materializer raising or returning parse
# errors, not by this constraint kind.
STRUCTURAL_KINDS = {
    "node_exists",
    "node_absent",
    "edge_exists",
    "module_count",
    "module_size_max",
    "module_responsibility",
    "acyclic_imports",
    "materializes",
}


__all__ = [
    "AcyclicImports",
    "Constraint",
    "EdgeExists",
    "Materializes",
    "ModuleCount",
    "ModuleResponsibility",
    "ModuleSizeMax",
    "NodeAbsent",
    "NodeExists",
    "STRUCTURAL_KINDS",
]
