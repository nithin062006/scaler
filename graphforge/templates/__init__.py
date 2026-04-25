"""Body template library.

Templates are the constrained building blocks for function bodies. See
PROPOSAL.md §3.2. The seed set is in :mod:`graphforge.templates.registry`;
the full ~25-entry library and codegen live in :mod:`library` (TODO).
"""

from graphforge.templates.registry import (
    TemplateSpec,
    get_template,
    known_templates,
    validate_args,
)

__all__ = ["TemplateSpec", "get_template", "known_templates", "validate_args"]
