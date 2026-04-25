"""Tests for the materializer + per-template codegen.

Strategy: for each template, build a minimal graph that exercises it,
materialize, parse-check (must succeed), and where possible *execute* the
resulting code in a fresh namespace and assert behavior. Multi-module graphs
are written to a tmp directory and imported via importlib so cross-module
imports are exercised end-to-end.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

from graphforge.actions import dispatch
from graphforge.actions.schema import (
    AddEdge,
    AddModule,
    AddNode,
    AttachBody,
)
from graphforge.graph import ArgMapping, Graph
from graphforge.materializer import materialize
from graphforge.validator import parse_check


# ---- helpers ---------------------------------------------------------


def _exec_single_module(source: str) -> dict:
    """Exec a single-module source string into a fresh namespace."""
    ns: dict = {"__name__": "synth", "__builtins__": __builtins__}
    code = compile(source, "synth.py", "exec")
    exec(code, ns)
    return ns


def _write_and_import(files: dict[str, str], tmp_path: Path, top: str):
    """Write files to ``tmp_path``, import the module named ``top``."""
    for name, src in files.items():
        (tmp_path / name).write_text(src)
    sys.path.insert(0, str(tmp_path))
    # Drop any cached versions from prior tests in the same session.
    for k in list(sys.modules):
        if k == top or k.startswith(top + "."):
            sys.modules.pop(k, None)
    try:
        return importlib.import_module(top)
    finally:
        sys.path.remove(str(tmp_path))


# ---- empty / no-body graphs -----------------------------------------


def test_empty_graph_materializes():
    g = Graph.empty()
    files = materialize(g)
    assert files == {}


def test_empty_module_emits_minimal_file():
    g = Graph.empty()
    dispatch(g, AddModule(name="m", responsibility="io"))
    files = materialize(g)
    assert "m.py" in files
    assert "from __future__ import annotations" in files["m.py"]
    assert parse_check(files) == []


def test_node_without_body_emits_placeholder():
    g = Graph.empty()
    dispatch(g, AddModule(name="m", responsibility="io"))
    dispatch(g, AddNode(name="f", module="m", signature="(x: int) -> int"))
    files = materialize(g)
    assert "def f(x: int) -> int:" in files["m.py"]
    assert "NotImplementedError" in files["m.py"]
    assert parse_check(files) == []


# ---- leaf_constant ---------------------------------------------------


def test_leaf_constant_executes():
    g = Graph.empty()
    dispatch(g, AddModule(name="m", responsibility="computation"))
    dispatch(g, AddNode(name="ans", module="m", signature="() -> int"))
    dispatch(g, AttachBody(name="ans", module="m", template="leaf_constant", args={"value": 42}))
    files = materialize(g)
    assert parse_check(files) == []
    ns = _exec_single_module(files["m.py"])
    assert ns["ans"]() == 42


def test_leaf_constant_string_repr_is_safe():
    g = Graph.empty()
    dispatch(g, AddModule(name="m", responsibility="computation"))
    dispatch(g, AddNode(name="hello", module="m", signature="() -> str"))
    dispatch(
        g,
        AttachBody(
            name="hello", module="m", template="leaf_constant",
            args={"value": "she said \"hi\"\nworld"},
        ),
    )
    files = materialize(g)
    assert parse_check(files) == []
    ns = _exec_single_module(files["m.py"])
    assert ns["hello"]() == "she said \"hi\"\nworld"


# ---- validate_with_regex --------------------------------------------


def test_validate_with_regex_true_and_false():
    g = Graph.empty()
    dispatch(g, AddModule(name="v", responsibility="validation"))
    dispatch(g, AddNode(name="is_email", module="v", signature="(s: str) -> bool"))
    dispatch(
        g,
        AttachBody(
            name="is_email", module="v", template="validate_with_regex",
            args={"pattern": "EMAIL"},
        ),
    )
    files = materialize(g)
    assert "import re" in files["v.py"]
    assert "_PATTERN_EMAIL" in files["v.py"]
    assert parse_check(files) == []
    ns = _exec_single_module(files["v.py"])
    # Behavioral pin — these match/no-match assertions are the regression
    # guard. If the emitted pattern's escaping is wrong (e.g. \s rendered
    # as the literal characters '\' + 's'), the True case below fails.
    assert ns["is_email"]("alice@example.com") is True
    assert ns["is_email"]("not an email") is False


def test_validate_with_regex_unknown_pattern_fails_codegen():
    g = Graph.empty()
    dispatch(g, AddModule(name="v", responsibility="validation"))
    dispatch(g, AddNode(name="is_x", module="v", signature="(s: str) -> bool"))
    # Bypass the dispatcher's template-arg gate by attaching with a known
    # template + valid args, then mutating the args directly to test the
    # codegen-side defensive check.
    dispatch(
        g,
        AttachBody(
            name="is_x", module="v", template="validate_with_regex",
            args={"pattern": "EMAIL"},
        ),
    )
    g.find_node("is_x", "v").body_template_args = {"pattern": "DOES_NOT_EXIST"}
    with pytest.raises(ValueError, match="unknown regex pattern"):
        materialize(g)


# ---- passthrough_call -----------------------------------------------


def test_passthrough_call_single_module():
    g = Graph.empty()
    dispatch(g, AddModule(name="m", responsibility="orchestration"))
    dispatch(g, AddNode(name="inner", module="m", signature="(x: int) -> int"))
    dispatch(
        g,
        AttachBody(name="inner", module="m", template="leaf_constant", args={"value": 7}),
    )
    dispatch(g, AddNode(name="outer", module="m", signature="(x: int) -> int"))
    dispatch(
        g,
        AddEdge(
            caller="m.outer",
            callee="m.inner",
            arg_mapping=[ArgMapping(caller_arg="x", callee_param="x")],
        ),
    )
    dispatch(g, AttachBody(name="outer", module="m", template="passthrough_call"))
    files = materialize(g)
    assert parse_check(files) == []
    ns = _exec_single_module(files["m.py"])
    assert ns["outer"](1) == 7


# ---- sequential_calls -----------------------------------------------


def test_sequential_calls_returns_last(tmp_path: Path):
    """Two callees; sequential_calls returns the result of the last one."""
    g = Graph.empty()
    dispatch(g, AddModule(name="m", responsibility="orchestration"))
    dispatch(g, AddNode(name="a", module="m", signature="() -> int"))
    dispatch(g, AddNode(name="b", module="m", signature="() -> int"))
    dispatch(g, AddNode(name="run", module="m", signature="() -> int"))
    dispatch(g, AttachBody(name="a", module="m", template="leaf_constant", args={"value": 1}))
    dispatch(g, AttachBody(name="b", module="m", template="leaf_constant", args={"value": 99}))
    dispatch(g, AddEdge(caller="m.run", callee="m.a"))
    dispatch(g, AddEdge(caller="m.run", callee="m.b"))
    dispatch(g, AttachBody(name="run", module="m", template="sequential_calls"))
    files = materialize(g)
    assert parse_check(files) == []
    ns = _exec_single_module(files["m.py"])
    assert ns["run"]() == 99


# ---- try_call_with_fallback -----------------------------------------


def test_try_call_with_fallback():
    """Primary raises; fallback returns. Result must equal fallback."""
    g = Graph.empty()
    dispatch(g, AddModule(name="m", responsibility="orchestration"))
    # primary will be a function that raises — easiest way is leaf_constant
    # of an unrepresentable value? No. Inject by post-edit: attach a body
    # we control directly.
    dispatch(g, AddNode(name="primary", module="m", signature="() -> int"))
    dispatch(g, AddNode(name="fallback", module="m", signature="() -> int"))
    dispatch(g, AddNode(name="run", module="m", signature="() -> int"))
    dispatch(
        g,
        AttachBody(name="fallback", module="m", template="leaf_constant", args={"value": 7}),
    )
    # For 'primary' we need a body that actually raises. We'll attach a
    # placeholder body (no template) and the materializer will emit
    # NotImplementedError, which IS raised at call-time.
    dispatch(g, AddEdge(caller="m.run", callee="m.primary"))
    dispatch(g, AddEdge(caller="m.run", callee="m.fallback"))
    dispatch(g, AttachBody(name="run", module="m", template="try_call_with_fallback"))
    files = materialize(g)
    assert parse_check(files) == []
    ns = _exec_single_module(files["m.py"])
    assert ns["run"]() == 7  # primary raises NotImplementedError -> fallback wins


# ---- early_return_guard ---------------------------------------------


def test_early_return_guard_passes_through():
    g = Graph.empty()
    dispatch(g, AddModule(name="m", responsibility="orchestration"))
    dispatch(g, AddNode(name="inner", module="m", signature="(x: int) -> int"))
    dispatch(g, AddNode(name="outer", module="m", signature="(x: int) -> int"))
    dispatch(
        g, AttachBody(name="inner", module="m", template="leaf_constant", args={"value": 5}),
    )
    dispatch(
        g,
        AddEdge(
            caller="m.outer",
            callee="m.inner",
            arg_mapping=[ArgMapping(caller_arg="x", callee_param="x")],
        ),
    )
    dispatch(
        g,
        AttachBody(
            name="outer", module="m", template="early_return_guard",
            args={"condition": "x > 0"},
        ),
    )
    files = materialize(g)
    assert parse_check(files) == []
    ns = _exec_single_module(files["m.py"])
    assert ns["outer"](5) == 5  # condition true -> delegates to inner -> 5
    assert ns["outer"](-1) is None  # condition false -> early return None


# ---- multi-module integration ---------------------------------------


def test_multi_module_with_cross_module_import(tmp_path: Path):
    """Two modules; main.run -> validators.is_email. Imports must resolve."""
    g = Graph.empty()
    dispatch(g, AddModule(name="validators", responsibility="validation"))
    dispatch(g, AddModule(name="main", responsibility="orchestration"))
    dispatch(g, AddNode(name="is_email", module="validators", signature="(s: str) -> bool"))
    dispatch(
        g,
        AttachBody(
            name="is_email", module="validators",
            template="validate_with_regex", args={"pattern": "EMAIL"},
        ),
    )
    dispatch(g, AddNode(name="run", module="main", signature="(s: str) -> bool"))
    dispatch(
        g,
        AddEdge(
            caller="main.run",
            callee="validators.is_email",
            arg_mapping=[ArgMapping(caller_arg="s", callee_param="s")],
        ),
    )
    dispatch(g, AttachBody(name="run", module="main", template="passthrough_call"))
    files = materialize(g)
    assert parse_check(files) == []
    assert "from validators import is_email" in files["main.py"]

    main_mod = _write_and_import(files, tmp_path, "main")
    assert main_mod.run("alice@example.com") is True
    assert main_mod.run("nope") is False


# ---- determinism -----------------------------------------------------


def test_materialization_is_deterministic():
    g = Graph.empty()
    dispatch(g, AddModule(name="m", responsibility="io"))
    dispatch(g, AddNode(name="a", module="m", signature="() -> int"))
    dispatch(g, AddNode(name="b", module="m", signature="() -> int"))
    dispatch(g, AttachBody(name="a", module="m", template="leaf_constant", args={"value": 1}))
    dispatch(g, AttachBody(name="b", module="m", template="leaf_constant", args={"value": 2}))
    files1 = materialize(g)
    files2 = materialize(g)
    assert files1 == files2
