"""Structured action errors.

Every failure mode in the action dispatcher surfaces as an :class:`ActionError`
with a stable ``code`` so the agent can be trained against deterministic error
strings (see PROPOSAL.md §4.4 — "failures return structured errors describing
the cause"). Codes are kept short and stable across versions.
"""

from __future__ import annotations

from typing import Any


class ActionError(Exception):
    """Raised by action handlers; caught and reported by the dispatcher."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, **self.details}


# ---- canonical codes -------------------------------------------------
# Schema layer
SCHEMA_REJECTION = "schema_rejection"
# Pre-condition layer
UNKNOWN_MODULE = "unknown_module"
UNKNOWN_NODE = "unknown_node"
UNKNOWN_EDGE = "unknown_edge"
NAME_COLLISION = "name_collision"
MODULE_NOT_EMPTY = "module_not_empty"
NODE_HAS_REFERENCES = "node_has_references"
DUPLICATE_EDGE = "duplicate_edge"
UNKNOWN_TEMPLATE = "unknown_template"
TEMPLATE_ARGS_INVALID = "template_args_invalid"
RESPONSIBILITY_MISMATCH = "responsibility_mismatch"
ARG_MAPPING_INVALID = "arg_mapping_invalid"
# Post-condition layer
WOULD_CREATE_CYCLE = "would_create_cycle"
TYPE_MISMATCH = "type_mismatch"
