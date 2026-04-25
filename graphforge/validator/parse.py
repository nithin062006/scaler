"""Parse-only validator.

Calls Python's own ``compile()`` on each materialized file and reports any
syntax / lexer errors. This is the cheapest gate; the agent receives this
feedback as part of ``materialize_and_validate``.

Heavier checks (import-resolution, ``mypy --strict``, behavioral tests) live
elsewhere in :mod:`graphforge.validator` and :mod:`graphforge.behavioral` —
intentionally separated so the agent can pay for verification incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParseError:
    filename: str
    line: int | None
    column: int | None
    message: str


@dataclass
class ValidationReport:
    parse_errors: list[ParseError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.parse_errors

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "parse_errors": [
                {
                    "filename": e.filename,
                    "line": e.line,
                    "column": e.column,
                    "message": e.message,
                }
                for e in self.parse_errors
            ],
        }


def parse_check(files: dict[str, str]) -> list[ParseError]:
    """Compile each ``files[name]`` source. Return collected errors.

    An empty list means every file parsed cleanly.
    """
    errors: list[ParseError] = []
    for filename, source in files.items():
        try:
            compile(source, filename, "exec")
        except SyntaxError as e:
            errors.append(
                ParseError(
                    filename=filename,
                    line=e.lineno,
                    column=e.offset,
                    message=e.msg,
                )
            )
    return errors


def full_check(files: dict[str, str]) -> ValidationReport:
    """Run every validator gate that's currently implemented.

    Today: parse-only. ``mypy --strict`` and import-resolution are added in
    follow-up commits but the report shape stays the same.
    """
    return ValidationReport(parse_errors=parse_check(files))
