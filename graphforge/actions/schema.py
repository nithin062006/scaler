"""Action message schemas.

These are the wire shapes accepted by the dispatcher. Every action is a
discriminated-union member keyed on ``kind``.

The action vocabulary mirrors PROPOSAL.md §4. Total surface:

  Graph mutations
    add_module, remove_module
    add_node, remove_node, set_node_module, attach_body
    add_edge, remove_edge
  Information
    query_spec, query_subgraph, query_types,
    materialize_and_validate, run_behavioral_tests
  Terminal
    submit

Note: the proposal abstract states "eleven actions"; the section-4 listing
contains fourteen. We implement the section-4 set; the abstract count will
be corrected in the next revision of PROPOSAL.md.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from graphforge.graph.schema import ArgMapping, ErrorPolicy, Purity, ResponsibilityTag


# Common config: forbid unknown fields, fail loudly on schema drift.
_cfg = ConfigDict(extra="forbid")


# ---- mutations -------------------------------------------------------


class AddModule(BaseModel):
    model_config = _cfg
    kind: Literal["add_module"] = "add_module"
    name: str
    responsibility: ResponsibilityTag


class RemoveModule(BaseModel):
    model_config = _cfg
    kind: Literal["remove_module"] = "remove_module"
    name: str


class AddNode(BaseModel):
    model_config = _cfg
    kind: Literal["add_node"] = "add_node"
    name: str
    module: str
    signature: str
    purity: Purity = "impure"
    error_policy: ErrorPolicy = "none"


class RemoveNode(BaseModel):
    model_config = _cfg
    kind: Literal["remove_node"] = "remove_node"
    name: str
    module: str


class SetNodeModule(BaseModel):
    model_config = _cfg
    kind: Literal["set_node_module"] = "set_node_module"
    name: str
    current_module: str
    new_module: str


class AttachBody(BaseModel):
    model_config = _cfg
    kind: Literal["attach_body"] = "attach_body"
    name: str
    module: str
    template: str
    args: dict[str, object] = Field(default_factory=dict)


class AddEdge(BaseModel):
    model_config = _cfg
    kind: Literal["add_edge"] = "add_edge"
    caller: str
    callee: str
    arg_mapping: list[ArgMapping] = Field(default_factory=list)


class RemoveEdge(BaseModel):
    model_config = _cfg
    kind: Literal["remove_edge"] = "remove_edge"
    caller: str
    callee: str


# ---- information actions --------------------------------------------


class QuerySpec(BaseModel):
    model_config = _cfg
    kind: Literal["query_spec"] = "query_spec"
    constraint_kind: Optional[str] = None


class QuerySubgraph(BaseModel):
    model_config = _cfg
    kind: Literal["query_subgraph"] = "query_subgraph"
    scope: str  # "module:<name>" | "neighbors:<qualified>" | "path:<from>:<to>"


class QueryTypes(BaseModel):
    model_config = _cfg
    kind: Literal["query_types"] = "query_types"
    scope: str  # "all" | "module:<name>" | "node:<qualified>"


class MaterializeAndValidate(BaseModel):
    model_config = _cfg
    kind: Literal["materialize_and_validate"] = "materialize_and_validate"


class RunBehavioralTests(BaseModel):
    model_config = _cfg
    kind: Literal["run_behavioral_tests"] = "run_behavioral_tests"
    materialized: bool = True


# ---- terminal --------------------------------------------------------


class Submit(BaseModel):
    model_config = _cfg
    kind: Literal["submit"] = "submit"


# ---- discriminated union --------------------------------------------

Action = Annotated[
    Union[
        AddModule,
        RemoveModule,
        AddNode,
        RemoveNode,
        SetNodeModule,
        AttachBody,
        AddEdge,
        RemoveEdge,
        QuerySpec,
        QuerySubgraph,
        QueryTypes,
        MaterializeAndValidate,
        RunBehavioralTests,
        Submit,
    ],
    Field(discriminator="kind"),
]


__all__ = [
    "Action",
    "AddModule",
    "RemoveModule",
    "AddNode",
    "RemoveNode",
    "SetNodeModule",
    "AttachBody",
    "AddEdge",
    "RemoveEdge",
    "QuerySpec",
    "QuerySubgraph",
    "QueryTypes",
    "MaterializeAndValidate",
    "RunBehavioralTests",
    "Submit",
]
