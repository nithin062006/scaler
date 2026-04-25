"""Auto-generate training tasks from any Python repository.

Pipeline
--------
1. Parse the repo with AST → KnowledgeGraph
2. Find public functions that have doctest examples (>>> in docstring)
3. Extract those examples as runnable assertions
4. Replace the function body with `raise NotImplementedError` — the agent
   must re-implement it from the docstring alone
5. Return RepoTask objects ready for GRPO training — no hand-writing needed

Usage
-----
    from graphforge.task_generator import generate_tasks
    tasks = generate_tasks("/tmp/humanize/src/humanize", n_tasks=6)
    for t in tasks:
        print(t.task_id, "→", t.description[:60])
"""

from __future__ import annotations

import ast
import doctest
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphforge.knowledge_graph import KGNode, KnowledgeGraph
from graphforge.repo_parser import parse_repo


# ── Task dataclass (mirrors env.tasks.RepoTask but lives here to avoid circular import) ──

@dataclass
class AutoTask:
    task_id: str
    repo_name: str
    repo_path: str          # absolute path to the repo source directory
    description: str
    test_code: str          # uses short import: from <repo_name>.<module> import <func>
    stubbed_node_id: str    # the node whose body was replaced
    original_source: str    # saved so env can restore on reset
    max_turns: int = 12
    difficulty: int = 0
    hints: list[str] = field(default_factory=list)


# ── Doctest extraction ────────────────────────────────────────────────────────

def _extract_all_examples(docstring: str) -> list[tuple[str, str]]:
    """Return ALL doctest lines as (source, want) — want is '' for setup lines."""
    if not docstring:
        return []
    parser = doctest.DocTestParser()
    try:
        examples = parser.get_examples(docstring, name="<doc>")
        return [(ex.source.strip(), ex.want.strip()) for ex in examples]
    except Exception:
        return []


def _to_assertion(expr: str, expected: str) -> str | None:
    """Convert one doctest example to a Python assertion.

    - True/False expected → assert (expr) is True/False
    - Traceback expected  → skip
    - Non-literal         → skip
    """
    if not expected or expected.startswith("Traceback"):
        return None
    if expected in ("True", "False"):
        return f"assert ({expr}) is {expected}, f'got {{repr({expr})}}'"
    try:
        ast.literal_eval(expected)
    except (ValueError, SyntaxError):
        return None
    return f"assert {expr} == {expected}, f'got {{repr({expr})}}'"


def _build_test_code(func_name: str, module_stem: str, repo_name: str,
                     all_examples: list[tuple[str, str]]) -> str | None:
    """Build complete test code including setup lines then assertions."""
    import_line = f"from {repo_name}.{module_stem} import {func_name}"
    setup_lines: list[str] = []
    assertion_lines: list[str] = []

    for expr, expected in all_examples:
        if not expected:
            setup_lines.append(expr)
        else:
            a = _to_assertion(expr, expected)
            if a and func_name in a:   # only keep assertions that call our function
                assertion_lines.append(a)

    if len(assertion_lines) < 2:
        return None
    parts = [import_line] + setup_lines + assertion_lines
    return "\n".join(parts)


# ── Function stubbing ─────────────────────────────────────────────────────────

def _stub_function(source: str) -> str:
    """Replace a function body with `raise NotImplementedError`, keeping signature + docstring."""
    dedented = textwrap.dedent(source)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return source

    lines = dedented.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        body = node.body
        indent = "    " * (node.col_offset // 4 + 1)

        # Keep signature lines (everything up to and including the colon)
        sig_end = body[0].lineno - 1  # 0-indexed line where body starts

        # Keep docstring if present
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            keep_until = body[0].end_lineno  # inclusive, 1-indexed
        else:
            keep_until = sig_end

        kept = "\n".join(lines[:keep_until])
        stub = kept.rstrip() + f"\n{indent}raise NotImplementedError\n"
        return stub

    return source


# ── Candidate selection ───────────────────────────────────────────────────────

def _score_candidate(node: KGNode, examples: list) -> int:
    """Higher = better training signal. Prefer more examples and longer docstrings."""
    return len(examples) * 3 + min(len(node.docstring or ""), 200) // 20


def _find_candidates(kg: KnowledgeGraph, repo_name: str) -> list[tuple[KGNode, str, int]]:
    """Return (node, test_code, score) for all viable candidates."""
    candidates = []
    for node in kg.all_nodes("function"):
        if node.name.startswith("_"):
            continue
        if not node.docstring or not node.source:
            continue
        module_stem = Path(node.file_path).stem if node.file_path else None
        if not module_stem:
            continue

        examples = _extract_all_examples(node.docstring)
        if not examples:
            continue

        test_code = _build_test_code(node.name, module_stem, repo_name, examples)
        if not test_code:
            continue

        score = _score_candidate(node, examples)
        candidates.append((node, test_code, score))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_tasks(
    repo_source_dir: str,
    n_tasks: int = 4,
    max_turns: int = 12,
) -> tuple[KnowledgeGraph, list[AutoTask]]:
    """Parse a Python repo directory and auto-generate training tasks.

    Args:
        repo_source_dir: Path to the Python package source directory.
                         e.g. '/tmp/humanize/src/humanize'
        n_tasks: How many tasks to generate (picks highest-scoring candidates).
        max_turns: Max turns per episode.

    Returns:
        (kg, tasks) — the Knowledge Graph and the list of AutoTask objects.
    """
    repo_source_dir = str(Path(repo_source_dir).resolve())
    repo_name = Path(repo_source_dir).name
    kg = parse_repo(repo_source_dir)

    candidates = _find_candidates(kg, repo_name)
    if not candidates:
        raise ValueError(
            f"No suitable candidates found in {repo_source_dir}. "
            "Make sure functions have doctest examples (>>> in docstring)."
        )

    selected = candidates[:n_tasks]
    tasks: list[AutoTask] = []

    for node, test_code, score in selected:
        stubbed = _stub_function(node.source)
        desc = textwrap.dedent(f"""\
            Implement the function `{node.name}` in `{node.file_path}`.

            {node.docstring.strip() if node.docstring else 'No docstring available.'}
        """).strip()

        task = AutoTask(
            task_id=f"auto.{repo_name}.{node.name}",
            repo_name=repo_name,
            repo_path=repo_source_dir,
            description=desc,
            test_code=test_code,
            stubbed_node_id=node.node_id,
            original_source=node.source,
            max_turns=max_turns,
            difficulty=min(2, max(0, score // 8)),
            hints=[
                f"Look at {node.file_path} to understand the module style.",
                f"The function signature is: {node.name}{node.metadata.get('signature', '(...)')}",
            ],
        )
        tasks.append(task)

    return kg, tasks
