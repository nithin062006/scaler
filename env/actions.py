"""Action schema for the multi-turn repo-editing environment.

All actions are expressed as JSON dicts with a "kind" discriminator.
The agent emits one action per turn inside <action>...</action> XML tags.

Actions
-------
query        Search the knowledge graph for relevant nodes.
inspect      View the full source of a specific node.
add_node     Insert a new function or class into a module/class.
update_node  Replace the source of an existing node.
remove_node  Delete a node from the graph.
submit       Apply all pending changes, run tests, end the episode.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


_cfg = ConfigDict(extra="forbid")


class QueryAction(BaseModel):
    model_config = _cfg
    kind: Literal["query"] = "query"
    keywords: str
    node_type: str = "all"   # "all" | "function" | "class" | "module" | "method"


class InspectAction(BaseModel):
    model_config = _cfg
    kind: Literal["inspect"] = "inspect"
    node_id: str


class AddNodeAction(BaseModel):
    model_config = _cfg
    kind: Literal["add_node"] = "add_node"
    parent_id: str       # node_id of the parent (module or class)
    name: str            # name of the new function/class
    node_type: str       # "function" | "class"
    code: str            # full source of the new node (incl. def/class line)


class UpdateNodeAction(BaseModel):
    model_config = _cfg
    kind: Literal["update_node"] = "update_node"
    node_id: str         # which node to replace
    new_code: str        # full replacement source (incl. def/class line)


class RemoveNodeAction(BaseModel):
    model_config = _cfg
    kind: Literal["remove_node"] = "remove_node"
    node_id: str


class SubmitAction(BaseModel):
    model_config = _cfg
    kind: Literal["submit"] = "submit"


RepoEditAction = (
    QueryAction
    | InspectAction
    | AddNodeAction
    | UpdateNodeAction
    | RemoveNodeAction
    | SubmitAction
)


def parse_action(raw: dict) -> RepoEditAction:
    """Dispatch raw dict to the correct action model."""
    kind = raw.get("kind", "")
    mapping = {
        "query": QueryAction,
        "inspect": InspectAction,
        "add_node": AddNodeAction,
        "update_node": UpdateNodeAction,
        "remove_node": RemoveNodeAction,
        "submit": SubmitAction,
    }
    cls = mapping.get(kind)
    if cls is None:
        raise ValueError(f"Unknown action kind: {kind!r}. Valid: {list(mapping)}")
    return cls.model_validate(raw)
