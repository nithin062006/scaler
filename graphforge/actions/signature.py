"""Cheap signature parser.

Used by the dispatcher to validate ``add_edge`` arg-mappings against the
callee's parameter list. Real type flow validation (caller_arg type vs
callee_param type) is the type engine; this module only extracts parameter
*names* from a signature string of the form::

    (a: int, b: str = "x", *, c: bool) -> bool

Annotations are tolerated as opaque text. Defaults are tolerated and treated
as making the parameter optional.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    name: str
    annotation: str | None
    has_default: bool


@dataclass(frozen=True)
class ParsedSignature:
    parameters: list[Parameter]
    return_annotation: str

    @property
    def required_params(self) -> list[str]:
        return [p.name for p in self.parameters if not p.has_default]

    @property
    def all_params(self) -> list[str]:
        return [p.name for p in self.parameters]


_SIG_RE = re.compile(r"^\s*\((?P<params>.*)\)\s*->\s*(?P<ret>.+?)\s*$", re.DOTALL)


def parse_signature(sig: str) -> ParsedSignature:
    """Parse a function signature string. Lenient — caller validates more deeply.

    Raises ``ValueError`` on signatures that fail surface checks. The schema
    layer (Node validator) already requires ``(`` and ``->``; this is the
    secondary parse used at dispatch time.
    """
    m = _SIG_RE.match(sig)
    if not m:
        raise ValueError(f"could not parse signature: {sig!r}")
    raw_params = m.group("params").strip()
    ret = m.group("ret").strip()

    params: list[Parameter] = []
    if raw_params:
        for piece in _split_top_level(raw_params, ","):
            piece = piece.strip()
            if not piece or piece in {"*", "/"}:
                continue
            if piece.startswith("**"):
                piece = piece[2:].lstrip()
            elif piece.startswith("*"):
                piece = piece[1:].lstrip()
            has_default = False
            if "=" in piece:
                # split off default at top-level '=' (ignore ones inside [..]).
                head, default = _split_default(piece)
                piece = head.strip()
                has_default = default is not None
            name = piece
            annotation: str | None = None
            if ":" in piece:
                name, annotation = piece.split(":", 1)
                name = name.strip()
                annotation = annotation.strip()
            if not name.isidentifier():
                raise ValueError(f"unparseable parameter {piece!r} in {sig!r}")
            params.append(Parameter(name=name, annotation=annotation, has_default=has_default))

    return ParsedSignature(parameters=params, return_annotation=ret)


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split ``s`` on ``sep`` at bracket-depth 0."""
    out: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def _split_default(piece: str) -> tuple[str, str | None]:
    """Split off ``= default`` at bracket-depth 0. Returns (head, default | None)."""
    depth = 0
    for i, ch in enumerate(piece):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "=" and depth == 0:
            return piece[:i], piece[i + 1 :]
    return piece, None
