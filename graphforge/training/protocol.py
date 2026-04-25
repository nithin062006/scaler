"""Tool-call wire format.

The agent emits a single tool call per turn as a JSON object wrapped in
``<action>...</action>`` tags::

    Some optional reasoning text the model writes before the call.
    <action>
    {"kind": "add_module", "name": "validators", "responsibility": "validation"}
    </action>

Why this format and not OpenAI / Qwen native tool-calling:

* It's tokenizer-agnostic. We don't depend on any chat-template's tool-call
  hooks, so we can swap models freely.
* It's easy for a 0.5B model to emit reliably with a few in-context examples.
* It's easy to fail cleanly: malformed output produces a structured
  ``ParseFailure`` that maps to MALFORMED in the reward engine.

If the model emits multiple ``<action>`` blocks we take the *last* one; this
matches "the agent reasoned, then committed to one action" and avoids
rewarding an early stutter.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

ACTION_OPEN = "<action>"
ACTION_CLOSE = "</action>"

_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)


@dataclass(frozen=True)
class ParseSuccess:
    action: dict[str, object]
    raw: str  # the JSON text we extracted, for debugging


@dataclass(frozen=True)
class ParseFailure:
    code: str
    message: str
    raw: str


ParseResult = ParseSuccess | ParseFailure


def parse_completion(text: str) -> ParseResult:
    """Extract a tool call from a model completion.

    On success, returns ``ParseSuccess`` whose ``action`` is a JSON dict
    suitable to forward to ``/step``. On any failure path returns a
    ``ParseFailure`` with a stable code:

      * ``no_action_tag``       — neither tag found
      * ``unclosed_tag``        — open tag without close
      * ``invalid_json``        — tags found but body wasn't JSON
      * ``not_an_object``       — JSON parsed but isn't a dict
      * ``missing_kind``        — dict is missing the ``kind`` field
    """
    if ACTION_OPEN not in text:
        return ParseFailure("no_action_tag", "no <action> tag found", raw=text)
    if ACTION_CLOSE not in text:
        return ParseFailure("unclosed_tag", "<action> tag never closed", raw=text)

    matches = _ACTION_RE.findall(text)
    if not matches:
        return ParseFailure(
            "no_action_tag",
            "<action> tags present but body could not be extracted",
            raw=text,
        )
    body = matches[-1].strip()  # take the last action emitted
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        return ParseFailure("invalid_json", f"json error: {e.msg}", raw=body)

    if not isinstance(obj, dict):
        return ParseFailure(
            "not_an_object",
            f"action body must be a JSON object, got {type(obj).__name__}",
            raw=body,
        )
    if "kind" not in obj:
        return ParseFailure("missing_kind", "action object lacks 'kind' field", raw=body)

    return ParseSuccess(action=obj, raw=body)


def render_action(action: dict[str, object]) -> str:
    """Render an action dict in the on-the-wire format. Used by tests and
    by scripted policies."""
    return f"{ACTION_OPEN}\n{json.dumps(action)}\n{ACTION_CLOSE}"
