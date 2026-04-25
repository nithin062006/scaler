"""Round-trip parser: Python source -> Graph.

Responsibilities (PROPOSAL.md §3.4):

  * Walk an AST per module file.
  * Recover function declarations as :class:`Node` objects.
  * Recover ``from x import y`` lines as cross-module edges (best-effort).
  * Recognize body templates by structural pattern matching against the
    template library, and recover ``body_template`` + ``body_template_args``.
  * Produce a :class:`Graph` identical (per ``structural_hash``) to the one
    that produced the source via :mod:`graphforge.materializer`.

The round-trip parser is unit-tested against every body template + every
constraint pattern. If it fails to round-trip, the materializer emits a
warning and the graph is treated as canonical.

Public surface (TODO):

    parse_program(files: dict[str, str]) -> Graph
    parse_directory(path: Path) -> Graph
"""

from __future__ import annotations


def parse_program(files: dict[str, str]) -> object:  # pragma: no cover — TODO
    raise NotImplementedError("round-trip parser TODO — see PROPOSAL.md §3.4")
