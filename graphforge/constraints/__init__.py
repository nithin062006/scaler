"""Constraint vocabulary and dispatch.

Three families (PROPOSAL.md §2.2):

  * Structural — node_exists, edge_exists, module_count, acyclic_imports,
    fan_in_max, fan_out_max, dag_depth_max, internal_only, …
  * Type / signature — signature_matches, return_type, arg_type,
    type_consistency, no_any_types, pure_function (TODO)
  * Behavioral / materialization — materializes, imports_resolve,
    type_checks, behavioral_test_passes, error_handling_present|absent

Currently shipped: tier-0 subset of structural + ``materializes``. Additional
kinds land as new discriminated members in :mod:`schema` and matching
``_check_*`` functions in :mod:`checker`.
"""

from graphforge.constraints.checker import (
    SatisfactionReport,
    check,
    evaluate_all,
)
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
    "SatisfactionReport",
    "check",
    "evaluate_all",
]
