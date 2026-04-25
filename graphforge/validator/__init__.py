"""Materialization validator: parse + import + mypy --strict.

Responsibilities (PROPOSAL.md §6.2 — "subprocess validation is bounded"):

  * ``parse_check`` — call ``compile(source, ...)`` per file; report errors.
  * ``import_check`` — write to a temp directory, attempt
    ``importlib.import_module`` in a subprocess; report errors. (TODO)
  * ``mypy_check`` — invoke ``mypy --strict`` in a subprocess against the
    materialized tree, capturing exit code and parsed errors. (TODO)
  * Hard timeouts: 8s per type-check, 12s for behavioral runs.
  * Cache results keyed on ``Graph.structural_hash`` so we don't re-run
    mypy after non-structural changes.

Currently only parse-checking is implemented; ``full_check`` will be extended
in place as the deeper gates land.
"""

from graphforge.validator.parse import (
    ParseError,
    ValidationReport,
    full_check,
    parse_check,
)

__all__ = ["ParseError", "ValidationReport", "full_check", "parse_check"]
