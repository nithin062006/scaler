"""Body template registry.

The full library is roughly 25 templates spanning common patterns
(passthrough_call, sequential_calls, validate_with_regex, dispatch_on_type,
try_call_with_fallback, accumulate, compose, ...). See PROPOSAL.md §3.2.

This file holds a *seed* registry sufficient for the action dispatcher and
its tests to exercise the attach_body code path. Each entry declares only
the metadata the dispatcher needs:

  * ``args_schema``  — required arg names and their JSON-shape hint
  * ``required_edges`` — predicate that the node has the right edges to
    support this template (e.g., passthrough_call needs exactly one out-edge)

Codegen (template -> Python source) and full type signatures live in
:mod:`graphforge.templates.library` and are TODO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    args_schema: dict[str, type]
    description: str
    # Predicate that takes (out_degree, in_degree) of the host node and
    # returns True iff the template is attachable. Real validation
    # (template <-> signature compatibility, type flow) is the type engine.
    edges_ok: Callable[[int, int], bool] = lambda out_d, in_d: True  # noqa: E731


_REGISTRY: dict[str, TemplateSpec] = {
    "passthrough_call": TemplateSpec(
        name="passthrough_call",
        args_schema={},
        description="Call exactly one downstream function and return its result.",
        edges_ok=lambda out_d, in_d: out_d == 1,
    ),
    "sequential_calls": TemplateSpec(
        name="sequential_calls",
        args_schema={},
        description="Call each downstream function in declaration order; return the last.",
        edges_ok=lambda out_d, in_d: out_d >= 1,
    ),
    "validate_with_regex": TemplateSpec(
        name="validate_with_regex",
        args_schema={"pattern": str},
        description="Apply a named regex pattern to the input; return bool.",
        edges_ok=lambda out_d, in_d: out_d == 0,
    ),
    "early_return_guard": TemplateSpec(
        name="early_return_guard",
        args_schema={"condition": str},
        description="Guard with an early-return; otherwise delegate to one downstream call.",
        edges_ok=lambda out_d, in_d: out_d == 1,
    ),
    "try_call_with_fallback": TemplateSpec(
        name="try_call_with_fallback",
        args_schema={},
        description="Try the first out-edge; on exception, delegate to the second.",
        edges_ok=lambda out_d, in_d: out_d == 2,
    ),
    "leaf_constant": TemplateSpec(
        name="leaf_constant",
        args_schema={"value": object},
        description="Return a literal constant. Leaf node.",
        edges_ok=lambda out_d, in_d: out_d == 0,
    ),
}


def known_templates() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_template(name: str) -> TemplateSpec | None:
    return _REGISTRY.get(name)


def validate_args(name: str, args: dict[str, object]) -> list[str]:
    """Return a list of human-readable problems with ``args``.

    Empty list means the args satisfy the schema.
    """
    spec = _REGISTRY.get(name)
    if spec is None:
        return [f"unknown template: {name!r}"]
    problems: list[str] = []
    extra = set(args) - set(spec.args_schema)
    missing = set(spec.args_schema) - set(args)
    for k in sorted(missing):
        problems.append(f"missing arg {k!r}")
    for k in sorted(extra):
        problems.append(f"unexpected arg {k!r}")
    for k, T in spec.args_schema.items():
        if k in args and T is not object and not isinstance(args[k], T):
            problems.append(
                f"arg {k!r} should be {T.__name__}, got {type(args[k]).__name__}"
            )
    return problems
