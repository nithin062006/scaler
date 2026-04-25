"""Graph -> Python source projection.

Responsibilities (PROPOSAL.md §3.3):

  * Emit one ``<module>.py`` per declared module.
  * Emit functions in :attr:`Node.decl_order` order.
  * Compute ``from <module> import <name>`` lines from cross-module edges,
    deduplicated and sorted.
  * Expand body templates with the node's ``body_template_args`` to produce
    a runnable function body.

The materializer is total over well-formed graphs: every dispatcher-accepted
graph must produce parseable source. Round-trip correctness (the produced
source re-parses to the same graph) is enforced by tests in
:mod:`graphforge.parser` (TODO).
"""

from graphforge.materializer.materialize import materialize

__all__ = ["materialize"]
