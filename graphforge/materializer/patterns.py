"""Named regex patterns for ``validate_with_regex`` template.

Patterns are referenced by name in the graph (e.g. ``args={"pattern": "EMAIL"}``)
and resolved here at materialization time. The registry keeps task definitions
domain-agnostic — a task constraint can name a pattern without leaking the
regex itself into the graph schema.

Add new patterns sparingly; every name here becomes part of the constraint
vocabulary that tasks can use.
"""

from __future__ import annotations

# name -> (regex string, brief description)
_PATTERNS: dict[str, str] = {
    "EMAIL": r"[^@\s]+@[^@\s]+\.[^@\s]+",
    "HEXCOLOR": r"#[0-9a-fA-F]{6}",
    "PHONE": r"\+?\d{10,15}",
    "ALPHANUM": r"[A-Za-z0-9]+",
    "URL": r"https?://[^\s]+",
}


def known_patterns() -> list[str]:
    return sorted(_PATTERNS.keys())


def get_pattern(name: str) -> str | None:
    return _PATTERNS.get(name)


def constant_name(name: str) -> str:
    """Module-level constant name we emit for a given pattern name."""
    return f"_PATTERN_{name}"
