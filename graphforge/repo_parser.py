"""Parse a Python repository (directory tree) into a KnowledgeGraph.

Usage
-----
    from graphforge.repo_parser import parse_repo
    kg = parse_repo("/path/to/my_package")

What it extracts
----------------
  Nodes  : repo, package, module, class, function, method
  Edges  : contains, calls (same-file), imports, inherits

Cross-file call resolution is best-effort: if function A in file X calls
function B and B appears anywhere in the graph, an edge is added.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

from graphforge.knowledge_graph import KGEdge, KGNode, KnowledgeGraph


# ── helpers ───────────────────────────────────────────────────────────────────

def _node_id(node_type: str, file_path: str, *names: str) -> str:
    parts = [node_type, file_path] + list(names)
    return ":".join(p for p in parts if p)


def _sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    for arg in node.args.args:
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        args.append(f"{arg.arg}{ann}")
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"({', '.join(args)}){ret}"


def _source_slice(source_lines: list[str], start: int, end: int) -> str:
    """1-indexed, inclusive."""
    return "\n".join(source_lines[start - 1 : end])


def _direct_calls(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Collect names of directly called functions (Name-style calls only)."""
    calls: set[str] = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.add(node.func.id)
    return calls


# ── single-file parser ────────────────────────────────────────────────────────

def _parse_file(
    file_path: str,       # relative to repo root
    abs_path: str,
    kg: KnowledgeGraph,
    parent_id: str,
) -> None:
    try:
        source = Path(abs_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return

    try:
        tree = ast.parse(source, filename=abs_path)
    except SyntaxError:
        return

    lines = source.splitlines()
    mod_id = _node_id("module", file_path)

    # Module node
    mod_doc = ast.get_docstring(tree) or ""
    kg.add_node(KGNode(
        node_id=mod_id,
        node_type="module",
        name=Path(file_path).stem,
        file_path=file_path,
        line_start=1,
        line_end=len(lines),
        source=source,
        docstring=mod_doc,
    ))
    kg.add_edge(KGEdge("contains", parent_id, mod_id))

    # Import edges (resolve module names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imp_id = _node_id("module", alias.name.replace(".", "/") + ".py")
                kg.add_edge(KGEdge("imports", mod_id, imp_id))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imp_id = _node_id("module", node.module.replace(".", "/") + ".py")
            kg.add_edge(KGEdge("imports", mod_id, imp_id))

    # Top-level classes and functions
    func_name_to_id: dict[str, str] = {}   # for call resolution within file

    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            _parse_class(stmt, file_path, lines, kg, mod_id, func_name_to_id)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _parse_function(stmt, file_path, lines, kg, mod_id, func_name_to_id)

    # Same-file call edges
    _resolve_calls(func_name_to_id, kg)


def _parse_class(
    cls_node: ast.ClassDef,
    file_path: str,
    lines: list[str],
    kg: KnowledgeGraph,
    parent_id: str,
    func_name_to_id: dict[str, str],
) -> None:
    cls_id = _node_id("class", file_path, cls_node.name)
    doc = ast.get_docstring(cls_node) or ""
    kg.add_node(KGNode(
        node_id=cls_id,
        node_type="class",
        name=cls_node.name,
        file_path=file_path,
        line_start=cls_node.lineno,
        line_end=cls_node.end_lineno,
        source=_source_slice(lines, cls_node.lineno, cls_node.end_lineno),
        docstring=doc,
    ))
    kg.add_edge(KGEdge("contains", parent_id, cls_id))

    # Inheritance edges
    for base in cls_node.bases:
        if isinstance(base, ast.Name):
            base_id = _node_id("class", file_path, base.id)
            kg.add_edge(KGEdge("inherits", cls_id, base_id))

    # Methods
    for item in cls_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _parse_method(item, file_path, lines, kg, cls_id, cls_node.name, func_name_to_id)


def _parse_function(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: str,
    lines: list[str],
    kg: KnowledgeGraph,
    parent_id: str,
    func_name_to_id: dict[str, str],
) -> None:
    fn_id = _node_id("function", file_path, fn.name)
    doc = ast.get_docstring(fn) or ""
    kg.add_node(KGNode(
        node_id=fn_id,
        node_type="function",
        name=fn.name,
        file_path=file_path,
        line_start=fn.lineno,
        line_end=fn.end_lineno,
        source=_source_slice(lines, fn.lineno, fn.end_lineno),
        docstring=doc,
        metadata={"signature": _sig(fn), "calls": list(_direct_calls(fn))},
    ))
    kg.add_edge(KGEdge("contains", parent_id, fn_id))
    func_name_to_id[fn.name] = fn_id


def _parse_method(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: str,
    lines: list[str],
    kg: KnowledgeGraph,
    parent_id: str,
    class_name: str,
    func_name_to_id: dict[str, str],
) -> None:
    method_id = _node_id("method", file_path, class_name, fn.name)
    doc = ast.get_docstring(fn) or ""
    kg.add_node(KGNode(
        node_id=method_id,
        node_type="method",
        name=fn.name,
        file_path=file_path,
        line_start=fn.lineno,
        line_end=fn.end_lineno,
        source=_source_slice(lines, fn.lineno, fn.end_lineno),
        docstring=doc,
        metadata={"signature": _sig(fn), "calls": list(_direct_calls(fn))},
    ))
    kg.add_edge(KGEdge("contains", parent_id, method_id))
    # register under unqualified name too for call resolution
    func_name_to_id[fn.name] = method_id


def _resolve_calls(func_name_to_id: dict[str, str], kg: KnowledgeGraph) -> None:
    """Add calls edges based on direct-call names collected during parse."""
    for fn_id, node in [(nid, n) for nid, n in kg._nodes.items()
                        if n.node_type in ("function", "method")]:
        calls: list[str] = node.metadata.get("calls", [])
        for callee_name in calls:
            if callee_name in func_name_to_id:
                callee_id = func_name_to_id[callee_name]
                if callee_id != fn_id:
                    kg.add_edge(KGEdge("calls", fn_id, callee_id))


# ── repo walker ───────────────────────────────────────────────────────────────

def parse_repo(repo_path: str, exclude_dirs: set[str] | None = None) -> KnowledgeGraph:
    """Walk repo_path recursively and return a KnowledgeGraph.

    Parameters
    ----------
    repo_path : str
        Absolute or relative path to the root of the repo.
    exclude_dirs : set[str], optional
        Directory names to skip (e.g. {"__pycache__", ".git", "tests"}).
    """
    if exclude_dirs is None:
        exclude_dirs = {"__pycache__", ".git", ".venv", "venv", "env",
                        "node_modules", ".mypy_cache", ".pytest_cache", "dist", "build"}

    abs_root = str(Path(repo_path).resolve())
    kg = KnowledgeGraph(repo_path=repo_path)

    # Root repo node
    repo_name = Path(abs_root).name
    repo_id = _node_id("repo", "", repo_name)
    kg.add_node(KGNode(
        node_id=repo_id,
        node_type="repo",
        name=repo_name,
        file_path="",
    ))

    # Walk directory tree
    for dirpath, dirnames, filenames in os.walk(abs_root):
        # Prune excluded dirs in-place (modifies os.walk traversal)
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        rel_dir = os.path.relpath(dirpath, abs_root)
        if rel_dir == ".":
            rel_dir = ""

        parent_id = repo_id
        if rel_dir:
            pkg_id = _node_id("package", rel_dir)
            if pkg_id not in kg._nodes:
                kg.add_node(KGNode(
                    node_id=pkg_id,
                    node_type="package",
                    name=Path(rel_dir).name,
                    file_path=rel_dir,
                ))
                kg.add_edge(KGEdge("contains", repo_id, pkg_id))
            parent_id = pkg_id

        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            rel_file = os.path.join(rel_dir, fname) if rel_dir else fname
            abs_file = os.path.join(dirpath, fname)
            _parse_file(rel_file, abs_file, kg, parent_id)

    return kg
