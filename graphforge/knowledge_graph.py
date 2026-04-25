"""In-memory Knowledge Graph for a Python repository.

Mirrors the structure of a Neo4j property graph but lives in RAM:

Nodes
-----
  repo        — the repository root
  package     — a directory containing __init__.py
  module      — a .py file
  class       — a class definition
  function    — a top-level or nested function / async function
  method      — a method inside a class

Edges (directed)
-----------------
  contains    — parent → child (repo→package, package→module, module→class, …)
  calls       — function/method → function/method (same-file same-package)
  imports     — module → module  (from x import y  /  import x)
  inherits    — class → class

Each node stores the actual source lines so the agent can read/edit them.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Iterable


# ── node & edge ───────────────────────────────────────────────────────────────

@dataclass
class KGNode:
    node_id: str          # unique key, e.g. "function:validators.py:validate_title"
    node_type: str        # module | class | function | method | package | repo
    name: str             # short identifier
    file_path: str        # relative path from repo root (empty for repo/package)
    line_start: int = 0
    line_end: int = 0
    source: str = ""      # full source text of this node (incl. def line)
    docstring: str = ""
    metadata: dict = field(default_factory=dict)

    def brief(self) -> str:
        """One-line summary for graph overviews."""
        loc = f"  [{self.file_path}:{self.line_start}]" if self.file_path else ""
        return f"[{self.node_type.upper():<8}] {self.node_id}{loc}"


@dataclass
class KGEdge:
    edge_type: str   # contains | calls | imports | inherits
    source_id: str
    target_id: str


# ── knowledge graph ───────────────────────────────────────────────────────────

class KnowledgeGraph:
    """Property graph for a repository.

    Supports rich queries used by the agent and reward checker.
    """

    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path
        self._nodes: dict[str, KGNode] = {}
        self._edges: list[KGEdge] = []

    # ── mutation ──────────────────────────────────────────────────────────────

    def add_node(self, node: KGNode) -> None:
        self._nodes[node.node_id] = node

    def add_edge(self, edge: KGEdge) -> None:
        self._edges.append(edge)

    def update_node_source(self, node_id: str, new_source: str) -> None:
        """Replace a node's source and recount lines."""
        node = self._nodes[node_id]
        node.source = new_source
        lines = new_source.splitlines()
        node.line_end = node.line_start + len(lines) - 1

    def insert_node(
        self,
        parent_id: str,
        new_node: KGNode,
    ) -> None:
        """Add new_node to the graph and wire a contains edge from parent."""
        self._nodes[new_node.node_id] = new_node
        self._edges.append(KGEdge("contains", parent_id, new_node.node_id))

    def remove_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        self._edges = [e for e in self._edges
                       if e.source_id != node_id and e.target_id != node_id]

    # ── queries ───────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> KGNode | None:
        return self._nodes.get(node_id)

    def all_nodes(self, node_type: str | None = None) -> list[KGNode]:
        nodes = list(self._nodes.values())
        if node_type:
            nodes = [n for n in nodes if n.node_type == node_type]
        return nodes

    def children_of(self, node_id: str) -> list[KGNode]:
        child_ids = {e.target_id for e in self._edges
                     if e.source_id == node_id and e.edge_type == "contains"}
        return [self._nodes[cid] for cid in child_ids if cid in self._nodes]

    def parent_of(self, node_id: str) -> KGNode | None:
        for e in self._edges:
            if e.target_id == node_id and e.edge_type == "contains":
                return self._nodes.get(e.source_id)
        return None

    def callers_of(self, node_id: str) -> list[KGNode]:
        caller_ids = {e.source_id for e in self._edges
                      if e.target_id == node_id and e.edge_type == "calls"}
        return [self._nodes[cid] for cid in caller_ids if cid in self._nodes]

    def callees_of(self, node_id: str) -> list[KGNode]:
        callee_ids = {e.target_id for e in self._edges
                      if e.source_id == node_id and e.edge_type == "calls"}
        return [self._nodes[cid] for cid in callee_ids if cid in self._nodes]

    def imports_of(self, module_id: str) -> list[KGNode]:
        imp_ids = {e.target_id for e in self._edges
                   if e.source_id == module_id and e.edge_type == "imports"}
        return [self._nodes[i] for i in imp_ids if i in self._nodes]

    def search(self, keywords: str, node_type: str | None = None) -> list[KGNode]:
        """Fuzzy keyword search over node names, docstrings, and source."""
        kws = keywords.lower().split()
        results: list[KGNode] = []
        for node in self._nodes.values():
            if node_type and node.node_type != node_type:
                continue
            haystack = f"{node.name} {node.docstring} {node.source}".lower()
            if all(kw in haystack for kw in kws):
                results.append(node)
        return results

    def subgraph(self, root_id: str, depth: int = 2) -> list[KGNode]:
        """BFS from root_id up to depth hops; returns all encountered nodes."""
        visited: set[str] = set()
        frontier = {root_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                for e in self._edges:
                    if e.source_id == nid and e.target_id not in visited:
                        next_frontier.add(e.target_id)
            frontier = next_frontier
        visited.update(frontier)
        return [self._nodes[nid] for nid in visited if nid in self._nodes]

    # ── text representations ──────────────────────────────────────────────────

    def overview(self, max_chars: int = 3000) -> str:
        """Compact multi-line overview of the repo graph, capped to avoid LLM context overflow."""
        lines: list[str] = [f"## Repository: {self.repo_path}", ""]
        modules = self.all_nodes("module")
        all_fns  = self.all_nodes("function")
        all_cls  = self.all_nodes("class")
        lines.append(f"  {len(modules)} modules · {len(all_fns)} functions · {len(all_cls)} classes")
        lines.append("")

        for mod in sorted(modules, key=lambda n: n.file_path):
            children = self.children_of(mod.node_id)
            funcs   = [c for c in children if c.node_type in ("function", "method")]
            classes = [c for c in children if c.node_type == "class"]
            summary = []
            if classes:
                summary.append(f"{len(classes)} class{'es' if len(classes)>1 else ''}")
            if funcs:
                summary.append(f"{len(funcs)} fn{'s' if len(funcs)>1 else ''}")
            lines.append(f"  [{mod.file_path}]  ({', '.join(summary) or 'empty'})")
            for cls in sorted(classes, key=lambda n: n.name):
                methods = [c for c in self.children_of(cls.node_id) if c.node_type == "method"]
                mnames  = ", ".join(m.name for m in sorted(methods, key=lambda n: n.line_start))
                lines.append(f"    class {cls.name}  →  {mnames or '(no methods)'}")
                lines.append(f"      node_id: {cls.node_id}")
            for fn in sorted(funcs, key=lambda n: n.line_start):
                lines.append(f"    def {fn.name}{fn.metadata.get('signature', '')}")
                lines.append(f"      node_id: {fn.node_id}")

            # Stop expanding if we are already near the character cap
            current = "\n".join(lines)
            if len(current) > max_chars:
                remaining = len(modules) - (modules.index(mod) + 1)
                if remaining:
                    lines.append(f"\n  ... [{remaining} more modules not shown — use query() to explore]")
                break

        return "\n".join(lines)

    def node_detail(self, node_id: str) -> str:
        """Full inspection view of a single node."""
        node = self._nodes.get(node_id)
        if node is None:
            return f"[ERROR] node_id {node_id!r} not found in graph."
        lines = [
            f"## Node: {node.node_id}",
            f"type     : {node.node_type}",
            f"file     : {node.file_path}  (lines {node.line_start}–{node.line_end})",
        ]
        if node.docstring:
            lines.append(f"docstring: {node.docstring[:120]}")
        callers = self.callers_of(node_id)
        callees = self.callees_of(node_id)
        if callers:
            lines.append("called by: " + ", ".join(n.name for n in callers))
        if callees:
            lines.append("calls    : " + ", ".join(n.name for n in callees))
        children = self.children_of(node_id)
        if children:
            lines.append("contains : " + ", ".join(c.name for c in children))
        lines += ["", "### Source", "```python", node.source or "(no source)", "```"]
        return "\n".join(lines)

    def snapshot(self) -> "KnowledgeGraph":
        """Deep copy — used to preserve state before mutations."""
        import copy
        return copy.deepcopy(self)
