"""AST-based DAG parser and code injection utilities.

parse_source(source, module_name) -> CodeDAG
    Parses a Python source string and returns a structured DAG with nodes
    (module, function, imported_module) and typed edges (contains, calls, imports).

inject_function_body(source, func_name, new_body) -> str
    Replaces the body of func_name in source with new_body, preserving the
    def line and any docstring. Used by the environment's step() method.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


# ── DAG data model ────────────────────────────────────────────────────────────

@dataclass
class DAGNode:
    name: str
    node_type: str    # "module" | "function" | "class" | "imported_module"
    signature: str = ""
    is_stub: bool = False
    body_summary: str = ""


@dataclass
class DAGEdge:
    edge_type: str    # "contains" | "calls" | "imports"
    source: str
    target: str


@dataclass
class FunctionInfo:
    name: str
    signature: str
    is_stub: bool
    start_line: int   # 1-indexed
    end_line: int     # 1-indexed, inclusive
    has_docstring: bool
    docstring_end_line: int   # 1-indexed; == start_line when no docstring


@dataclass
class CodeDAG:
    module_name: str
    nodes: list[DAGNode] = field(default_factory=list)
    edges: list[DAGEdge] = field(default_factory=list)
    function_infos: dict[str, FunctionInfo] = field(default_factory=dict)

    def callers_of(self, func_name: str) -> list[str]:
        return [e.source for e in self.edges if e.edge_type == "calls" and e.target == func_name]

    def callees_of(self, func_name: str) -> list[str]:
        return [e.target for e in self.edges if e.edge_type == "calls" and e.source == func_name]

    def stub_functions(self) -> list[str]:
        return [n.name for n in self.nodes if n.node_type == "function" and n.is_stub]


# ── helpers ───────────────────────────────────────────────────────────────────

def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    parts = []
    for arg in node.args.args:
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        parts.append(f"{arg.arg}{ann}")
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"({', '.join(parts)}){ret}"


def _is_stub(node: ast.FunctionDef | ast.AsyncFunctionDef, source: str) -> bool:
    func_src = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])
    if "# STUB" in func_src:
        return True
    # body that is just "raise NotImplementedError"
    stmts = [s for s in node.body
              if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if len(stmts) == 1 and isinstance(stmts[0], ast.Raise):
        exc = stmts[0].exc
        if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
            return True
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
            return True
    return False


def _extract_calls(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    calls: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.add(child.func.id)
    return calls


# ── main parser ───────────────────────────────────────────────────────────────

def parse_source(source: str, module_name: str = "module") -> CodeDAG:
    """Parse Python source into a CodeDAG."""
    tree = ast.parse(source)
    dag = CodeDAG(module_name=module_name)
    dag.nodes.append(DAGNode(name=module_name, node_type="module"))

    func_names: set[str] = set()

    # imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imp = alias.asname or alias.name
                dag.nodes.append(DAGNode(name=imp, node_type="imported_module"))
                dag.edges.append(DAGEdge("imports", module_name, imp))
        elif isinstance(node, ast.ImportFrom) and node.module:
            dag.nodes.append(DAGNode(name=node.module, node_type="imported_module"))
            dag.edges.append(DAGEdge("imports", module_name, node.module))

    # top-level functions and classes
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _signature(node)
            stub = _is_stub(node, source)
            has_doc = (
                bool(node.body)
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
            )
            doc_end = node.body[0].end_lineno if has_doc else node.lineno

            dag.nodes.append(DAGNode(
                name=node.name,
                node_type="function",
                signature=sig,
                is_stub=stub,
                body_summary="STUB — needs implementation" if stub else "(implemented)",
            ))
            dag.edges.append(DAGEdge("contains", module_name, node.name))
            dag.function_infos[node.name] = FunctionInfo(
                name=node.name,
                signature=sig,
                is_stub=stub,
                start_line=node.lineno,
                end_line=node.end_lineno,
                has_docstring=has_doc,
                docstring_end_line=doc_end,
            )
            func_names.add(node.name)

        elif isinstance(node, ast.ClassDef):
            dag.nodes.append(DAGNode(name=node.name, node_type="class"))
            dag.edges.append(DAGEdge("contains", module_name, node.name))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qname = f"{node.name}.{item.name}"
                    dag.nodes.append(DAGNode(
                        name=qname,
                        node_type="function",
                        signature=_signature(item),
                        is_stub=_is_stub(item, source),
                    ))
                    dag.edges.append(DAGEdge("contains", node.name, qname))
                    func_names.add(qname)

    # call edges (same-module only)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for callee in _extract_calls(node):
                if callee in func_names and callee != node.name:
                    dag.edges.append(DAGEdge("calls", node.name, callee))

    return dag


# ── code injection ────────────────────────────────────────────────────────────

def inject_function_body(source: str, func_name: str, new_body: str) -> str:
    """Replace the body of func_name in source with new_body.

    Preserves the def line and any docstring. new_body should be the raw body
    text (with or without indentation — we normalise it).
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != func_name:
            continue

        # Determine where to keep up to (def line + optional docstring)
        has_doc = (
            bool(node.body)
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        )
        keep_until = node.body[0].end_lineno if has_doc else node.lineno
        # keep_until is 1-indexed; lines[:keep_until] gives 0..keep_until-1

        before = lines[:keep_until]
        after = lines[node.end_lineno:]   # everything after the function

        # Normalise body indent: strip common leading whitespace, then re-add 4 spaces.
        raw_lines = new_body.splitlines()
        # find minimum indent of non-empty lines
        min_indent = min(
            (len(l) - len(l.lstrip()) for l in raw_lines if l.strip()),
            default=0,
        )
        body_lines: list[str] = []
        for raw_line in raw_lines:
            if raw_line.strip():
                body_lines.append("    " + raw_line[min_indent:] + "\n")
            else:
                body_lines.append("\n")

        if not body_lines:
            body_lines = ["    pass\n"]

        return "".join(before + body_lines + after)

    raise ValueError(f"Function {func_name!r} not found in source")


# ── DAG → text description (for prompts) ─────────────────────────────────────

def dag_to_text(dag: CodeDAG) -> str:
    """Render the DAG as a concise human-readable block for the agent prompt."""
    lines: list[str] = [f"## Module: {dag.module_name}", "", "### Nodes"]

    for n in dag.nodes:
        if n.node_type == "module":
            lines.append(f"- [MODULE]   {n.name}")
        elif n.node_type == "function":
            status = "[ STUB ]" if n.is_stub else "[ready ]"
            lines.append(f"- [FUNC]  {status}  {n.name}{n.signature}")
        elif n.node_type == "class":
            lines.append(f"- [CLASS]   {n.name}")
        elif n.node_type == "imported_module":
            lines.append(f"- [IMPORT]  {n.name}")

    lines += ["", "### Edges"]
    for e in dag.edges:
        lines.append(f"- {e.source}  --{e.edge_type}-->  {e.target}")

    return "\n".join(lines)
