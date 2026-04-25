"""Prompt / conversation builder.

Produces the message list the policy sees, in HF chat-template-compatible
shape: ``[{"role": "system", "content": ...}, {"role": "user", ...}, ...]``.

The system prompt is short and stable across episodes; the per-task user
turn is the natural-language description plus the visible constraints
(rendered compactly so we don't burn context on JSON).

After each step, the env's observation is appended as a ``user`` turn —
this is the role that's typically used for tool-result injection in the
absence of a dedicated ``tool`` role in the chat template.
"""

from __future__ import annotations

import json
from typing import Any

from graphforge.training.protocol import ACTION_CLOSE, ACTION_OPEN

Message = dict[str, str]


SYSTEM_PROMPT = f"""You are an agent that builds Python programs by mutating a typed function-call graph.

You don't write source code directly. Instead, each turn you emit exactly one tool call.
The environment applies the call to a graph, replies with an observation, and the cycle repeats.
At the end, the graph is materialized into Python and scored against a hidden specification.

# Tool call format

Your reply each turn should end with one tool call like this:

    {ACTION_OPEN}
    {{"kind": "add_module", "name": "validators", "responsibility": "validation"}}
    {ACTION_CLOSE}

Reasoning before the call is fine; the parser takes the last <action> block.
Malformed output (no tag, bad JSON, missing 'kind') costs reward.

# Available tools

Graph mutations:
  add_module(name, responsibility)
  remove_module(name)
  add_node(name, module, signature, purity?, error_policy?)
  remove_node(name, module)
  set_node_module(name, current_module, new_module)
  attach_body(name, module, template, args?)
  add_edge(caller, callee, arg_mapping?)            # caller/callee are "<module>.<name>"
  remove_edge(caller, callee)

Information (cheap):
  query_subgraph(scope)        # "module:<name>" | "neighbors:<qualified>" | "path:<from>:<to>"
  query_spec(constraint_kind?) # how many constraints satisfied
  query_types(scope)           # type view (TODO)

Information (expensive — token cost):
  materialize_and_validate()   # project graph to Python, parse-check
  run_behavioral_tests()       # property tests (TODO)

Terminal:
  submit()                     # ends episode and triggers final scoring

# Reward shape

Per turn:
  successful mutation         0
  failed mutation            -2
  malformed output           -2
  duplicate of prior action  -1
  per-turn cost              -0.1
  token cost on response     -0.0008 * tokens

Terminal:
  +1 per structural constraint satisfied
  +5 if all structural constraints satisfied
  +5 * (budget_remaining / budget) if all satisfied  (token-efficiency bonus)
  -8 if materialization fails

Plan before you act. Failed actions and reading expensive responses cost reward."""


def initial_messages(task_visible: dict[str, Any]) -> list[Message]:
    """Build the conversation seed for a fresh episode.

    ``task_visible`` is the dict returned by ``Task.visible_payload()``.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _format_task_user_turn(task_visible)},
    ]


def append_observation(
    messages: list[Message], observation: dict[str, Any]
) -> list[Message]:
    """Append an env observation as a user turn. Returns a new list."""
    return list(messages) + [
        {"role": "user", "content": _format_observation(observation)},
    ]


def append_completion(messages: list[Message], completion: str) -> list[Message]:
    return list(messages) + [{"role": "assistant", "content": completion}]


# ---- formatting -----------------------------------------------------


def _format_task_user_turn(task_visible: dict[str, Any]) -> str:
    desc = task_visible.get("description", "(no description)")
    cs = task_visible.get("visible_constraints", [])
    rendered = "\n".join(f"  - {_format_constraint(c)}" for c in cs) or "  (none)"
    tier = task_visible.get("tier")
    cap = task_visible.get("episode_cap")
    budget = task_visible.get("budget")
    return (
        f"# Task (tier {tier})\n"
        f"{desc}\n\n"
        f"# Visible constraints (the spec also has hidden constraints; you must "
        f"interpret the description, not just satisfy this checklist)\n"
        f"{rendered}\n\n"
        f"# Limits\n"
        f"  episode_cap: {cap} turns\n"
        f"  budget: {budget} tokens\n"
    )


def _format_constraint(c: dict[str, Any]) -> str:
    kind = c.get("kind", "?")
    rest = {k: v for k, v in c.items() if k != "kind"}
    if not rest:
        return kind
    inside = ", ".join(f"{k}={v!r}" for k, v in rest.items())
    return f"{kind}({inside})"


def _format_observation(obs: dict[str, Any]) -> str:
    """Render a /step observation tersely — the agent doesn't need every field.

    Returns a multi-line string with the action outcome, the payload, and
    running counters. Kept concise to control token cost.
    """
    payload_text = json.dumps(obs.get("payload", {}), indent=2, default=str)
    if len(payload_text) > 800:
        payload_text = payload_text[:800] + "\n  …(truncated)"
    return (
        f"# Observation\n"
        f"  ok: {obs.get('ok')}\n"
        f"  outcome: {obs.get('outcome')}\n"
        f"  duplicate: {obs.get('is_duplicate')}\n"
        f"  reward: {obs.get('reward')}\n"
        f"  turns_total: {obs.get('turns_total')}\n"
        f"  tokens_used_total: {obs.get('tokens_used_total')}\n"
        f"  budget_remaining: {obs.get('budget_remaining')}\n"
        f"  episode_cap_remaining: {obs.get('episode_cap_remaining')}\n"
        f"  payload: {payload_text}\n"
    )
