"""Prompt templates for the multi-turn repo-editing agent."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a code-editing agent that works on real Python repositories.

You receive a Knowledge Graph (KG) of the repository — parsed from its AST — \
with nodes representing modules, classes, functions, and methods, \
and edges representing contains / calls / imports / inherits relationships.

Your goal: make a specific code change to the repository as described in the task.

You work in a multi-turn conversation. Each turn you output EXACTLY ONE action as
a JSON object inside a ```json code block, like this:

```json
{"kind": "query", "keywords": "validate", "node_type": "function"}
```

Available actions:
──────────────────────────────────────────────
query         Search the graph by keyword.
              ```json
              {"kind": "query", "keywords": "validate priority", "node_type": "function"}
              ```
              node_type: "all" | "function" | "class" | "method" | "module"

inspect       View the full source of a node.
              ```json
              {"kind": "inspect", "node_id": "function:validators.py:validate_title"}
              ```

add_node      Add a new function or class to a module.
              ```json
              {"kind": "add_node", "parent_id": "module:validators.py",
               "name": "validate_priority", "node_type": "function",
               "code": "def validate_priority(priority: str) -> bool:\\n    ..."}
              ```

update_node   Replace an existing node's source.
              ```json
              {"kind": "update_node", "node_id": "function:api.py:create_task",
               "new_code": "def create_task(...):\\n    ..."}
              ```

submit        Apply all changes and run the test suite. Ends the episode.
              ```json
              {"kind": "submit"}
              ```
──────────────────────────────────────────────

Strategy:
1. Use query + inspect to find the right location.
2. Use add_node or update_node to implement the change.
3. Use submit when confident.

Output ONLY the ```json block — no other text needed."""


def format_observation(obs_dict: dict) -> str:
    """Format a RepoEditObservation dict as the user turn content."""
    lines = [
        f"== Turn {obs_dict.get('turn', '?')} / {obs_dict.get('max_turns', '?')} ==",
        "",
        "## Last Action Result",
        obs_dict.get("action_result", "(start of episode)"),
        "",
        "## Repository Knowledge Graph",
        obs_dict.get("graph_overview", ""),
        "",
        "## Your Task",
        obs_dict.get("task_description", ""),
        "",
        "Emit your next action:",
    ]
    return "\n".join(lines)


_ACTION_KINDS = ("query", "inspect", "add_node", "update_node", "remove_node", "submit")


def extract_action_json(completion: str) -> dict | None:
    """Parse action JSON from several formats the model might produce.

    Priority order:
      1. ```json ... ``` code block  (preferred — matches new prompt format)
      2. <action>...</action> tags   (legacy format, still accepted)
      3. action_kind {...}           (fallback for models that skip wrappers)
      4. Any bare {"kind": ...} JSON
    """
    import json, re

    # 1. ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", completion, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. <action>...</action>
    m = re.search(r"<action>(.*?)</action>", completion, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. action_kind {"arg": ...}  — model knows the kind but skips wrappers
    for kind in _ACTION_KINDS:
        m = re.search(rf"\b{re.escape(kind)}\s+(\{{[^{{}}]*\}})", completion, re.DOTALL)
        if m:
            try:
                args = json.loads(m.group(1))
                args.setdefault("kind", kind)
                return args
            except json.JSONDecodeError:
                pass
        if kind == "submit" and re.search(r"\bsubmit\b", completion):
            return {"kind": "submit"}

    # 4. Any bare JSON object that has a "kind" field
    for m in re.finditer(r"\{[^{}]+\}", completion):
        try:
            obj = json.loads(m.group())
            if "kind" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    return None
