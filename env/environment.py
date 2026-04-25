"""Multi-turn repo-editing OpenEnv environment.

Episode flow
------------
reset()   Parse the target repo into a KnowledgeGraph. Return an observation
          containing the full graph overview and the task description.

step()    The agent emits one RepoEditAction per turn:
          - query       → search results (information, no graph mutation)
          - inspect     → full node source (information)
          - add_node    → insert new function/class into the live graph
          - update_node → replace a node's source in the live graph
          - remove_node → delete a node
          - submit      → materialise all changes back to disk (temp), run tests,
                          compute reward, end episode

Reward structure (sparse — designed for long-horizon RL)
---------------------------------------------------------
  Per-turn cost    : -0.05   (forces efficiency)
  Malformed action : -0.2
  On submit
    all tests pass : +1.0
    partial pass   : +0.5 * (n_pass / n_total)
    compile error  : 0.0
  Episode cap hit  : 0.0

This sparse reward deliberately requires the agent to plan, navigate, and
execute across many turns — it cannot succeed by guessing on the first turn.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import tempfile
import textwrap
import traceback
import uuid
from pathlib import Path
from typing import Any

from env.actions import (
    AddNodeAction,
    InspectAction,
    QueryAction,
    RemoveNodeAction,
    RepoEditAction,
    SubmitAction,
    UpdateNodeAction,
    parse_action,
)
from env.models import RepoEditObservation, RepoEditState
from env.tasks import SAMPLE_REPOS_DIR, TASK_BANK, RepoTask, all_task_ids, get_task
from graphforge.knowledge_graph import KGEdge, KGNode, KnowledgeGraph
from graphforge.repo_parser import parse_repo, _node_id

try:
    from openenv.core import Environment  # type: ignore
    _HAS_OPENENV = True
except Exception:
    _HAS_OPENENV = False
    from typing import Generic, TypeVar
    A = TypeVar("A")
    O = TypeVar("O")
    S = TypeVar("S")

    class Environment(Generic[A, O, S]):  # type: ignore[no-redef]
        def reset(self) -> O: ...
        def step(self, action: A) -> tuple[O, float, bool]: ...
        def get_state(self) -> S: ...


# ── constants ─────────────────────────────────────────────────────────────────

PER_TURN_COST = -0.05
MALFORMED_PENALTY = -0.2


# ── materialiser (graph → disk) ───────────────────────────────────────────────

def _materialise_changes(
    kg: KnowledgeGraph,
    repo_src_path: Path,
    tmp_dir: str,
) -> dict[str, str]:
    """Write mutated module sources to tmp_dir. Returns {rel_path: source}."""
    files: dict[str, str] = {}
    for node in kg.all_nodes("module"):
        if not node.file_path:
            continue
        # Re-assemble module source from its children's current sources
        # For simplicity: use the node.source field (which we keep in sync)
        files[node.file_path] = node.source
        dest = Path(tmp_dir) / node.file_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(node.source, encoding="utf-8")
    # Copy non-py files (like __init__.py markers) from original
    for root, _, fnames in os.walk(str(repo_src_path)):
        for fname in fnames:
            if fname.endswith(".py"):
                continue
            src = Path(root) / fname
            rel = src.relative_to(repo_src_path)
            dst = Path(tmp_dir) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    return files


# ── code injection into module source ─────────────────────────────────────────

def _apply_add_node(
    module_source: str,
    code: str,
    class_name: str | None = None,
) -> str:
    """Insert code into module_source.

    If class_name is given, the code is indented and appended inside the class
    body. Otherwise it is appended at module level.
    """
    new_code = textwrap.dedent(code).strip()
    if class_name is None:
        return module_source.rstrip() + "\n\n\n" + new_code + "\n"

    # Insert indented method just before the end of the class block
    indented = "\n".join("    " + line for line in new_code.splitlines())
    # Find the class definition via AST and splice
    try:
        tree = ast.parse(module_source)
        lines = module_source.splitlines(keepends=True)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                insert_at = node.end_lineno  # 1-indexed, inclusive last line of class
                before = "".join(lines[:insert_at])
                after = "".join(lines[insert_at:])
                return before.rstrip() + "\n\n" + indented + "\n" + after
    except Exception:
        pass
    # Fallback: append at module level
    return module_source.rstrip() + "\n\n\n" + indented + "\n"


def _apply_update_node(
    module_source: str,
    old_source: str,
    new_code: str,
) -> str:
    """Replace old_source verbatim in module_source with new_code."""
    new_code_clean = textwrap.dedent(new_code).strip()
    if old_source in module_source:
        return module_source.replace(old_source, new_code_clean, 1)
    # Fallback: try stripping indentation differences
    return module_source + "\n\n# PATCHED\n" + new_code_clean + "\n"


def _apply_remove_node(module_source: str, old_source: str) -> str:
    if old_source in module_source:
        return module_source.replace(old_source, "", 1)
    return module_source


def _validate_python(source: str) -> tuple[bool, str]:
    try:
        ast.parse(source)
        return True, ""
    except SyntaxError as exc:
        return False, str(exc)


# ── environment ───────────────────────────────────────────────────────────────

class RepoEditEnvironment(
    Environment[RepoEditAction, RepoEditObservation, RepoEditState]
):
    """Multi-turn OpenEnv environment for repository-level code editing.

    The agent receives a Knowledge Graph of a real Python repo and must
    navigate it to find the right location, then apply the correct edit.
    Reward is sparse: only granted on a passing submit().
    """

    def __init__(self, task_id: str | None = None) -> None:
        self._configured_task_id = task_id
        self._task: RepoTask | None = None
        self._kg: KnowledgeGraph | None = None
        self._episode_id: str | None = None
        self._turn: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0
        self._history: list[dict[str, Any]] = []

    # ----- OpenEnv contract ---------------------------------------------------

    def reset(self, task_id: str | None = None) -> RepoEditObservation:
        tid = task_id or self._configured_task_id or _pick_random_task()
        task = TASK_BANK.get(tid)
        if task is None:
            raise ValueError(f"Unknown task_id: {tid!r}. Available: {all_task_ids()}")

        repo_path = str(SAMPLE_REPOS_DIR / task.repo_name)
        self._task = task
        self._kg = parse_repo(repo_path)
        self._episode_id = str(uuid.uuid4())[:8]
        self._turn = 0
        self._done = False
        self._total_reward = 0.0
        self._history = []

        return RepoEditObservation(
            episode_id=self._episode_id,
            task_id=tid,
            turn=0,
            max_turns=task.max_turns,
            graph_overview=self._kg.overview(),
            task_description=task.description,
            action_result="Episode started. Use query/inspect to navigate, then add_node/update_node to edit, then submit.",
            done=False,
        )

    def step(self, action: RepoEditAction) -> tuple[RepoEditObservation, float, bool]:
        if self._task is None or self._kg is None:
            raise RuntimeError("step() called before reset()")
        if self._done:
            return self._terminal_obs("Episode already done."), 0.0, True

        self._turn += 1
        turn_reward = PER_TURN_COST

        # Dispatch
        try:
            result_text, extra_reward, done = self._dispatch(action)
            turn_reward += extra_reward
        except Exception as exc:
            result_text = f"[ERROR] {exc}"
            turn_reward += MALFORMED_PENALTY
            done = False

        self._total_reward += turn_reward

        # Episode cap
        if not done and self._turn >= self._task.max_turns:
            done = True
            result_text += f"\n[Episode cap reached: {self._task.max_turns} turns]"

        self._done = done
        self._history.append({
            "turn": self._turn,
            "action_kind": getattr(action, "kind", "unknown"),
            "reward": turn_reward,
        })

        obs = RepoEditObservation(
            episode_id=self._episode_id,
            task_id=self._task.task_id,
            turn=self._turn,
            max_turns=self._task.max_turns,
            graph_overview=self._kg.overview(),
            task_description=self._task.description,
            action_result=result_text,
            turn_reward=turn_reward,
            total_reward=self._total_reward,
            done=done,
        )
        return obs, turn_reward, done

    def get_state(self) -> RepoEditState:
        return RepoEditState(
            episode_id=self._episode_id,
            task_id=self._task.task_id if self._task else None,
            turn=self._turn,
            done=self._done,
            total_reward=self._total_reward,
        )

    @property
    def state(self) -> RepoEditState:
        return self.get_state()

    # ----- action dispatch ----------------------------------------------------

    def _dispatch(
        self, action: RepoEditAction
    ) -> tuple[str, float, bool]:
        """Returns (result_text, extra_reward, done)."""
        kg = self._kg
        assert kg is not None

        if isinstance(action, QueryAction):
            nt = None if action.node_type == "all" else action.node_type
            results = kg.search(action.keywords, node_type=nt)
            if not results:
                return f"No nodes found for query: {action.keywords!r}", 0.0, False
            lines = [f"Found {len(results)} node(s) matching {action.keywords!r}:"]
            for n in results[:10]:
                lines.append(f"  {n.node_id}  ({n.file_path}:{n.line_start})")
            return "\n".join(lines), 0.0, False

        if isinstance(action, InspectAction):
            detail = kg.node_detail(action.node_id)
            return detail, 0.0, False

        if isinstance(action, AddNodeAction):
            parent = kg.get_node(action.parent_id)
            if parent is None:
                return f"[ERROR] parent_id {action.parent_id!r} not found.", MALFORMED_PENALTY, False
            ok, err = _validate_python(action.code)
            if not ok:
                return f"[SYNTAX ERROR in your code] {err}", MALFORMED_PENALTY, False

            # Append to parent module's source
            module_node = _find_module_for(kg, action.parent_id)
            if module_node is None:
                return f"[ERROR] could not find module for parent {action.parent_id!r}", MALFORMED_PENALTY, False

            parent_node = kg.get_node(action.parent_id)
            class_name = parent_node.name if parent_node and parent_node.node_type == "class" else None
            module_node.source = _apply_add_node(module_node.source, action.code, class_name=class_name)

            # Register the new node in the KG
            ntype = action.node_type if action.node_type in ("function", "class", "method") else "function"
            new_id = _node_id(ntype, module_node.file_path, action.name)
            new_node = KGNode(
                node_id=new_id,
                node_type=ntype,
                name=action.name,
                file_path=module_node.file_path,
                line_start=module_node.line_end,
                line_end=module_node.line_end + action.code.count("\n") + 1,
                source=textwrap.dedent(action.code).strip(),
            )
            kg.insert_node(action.parent_id, new_node)
            return f"Added {ntype} `{action.name}` to `{module_node.file_path}`.\nNew node_id: {new_id}", 0.0, False

        if isinstance(action, UpdateNodeAction):
            target = kg.get_node(action.node_id)
            if target is None:
                return f"[ERROR] node_id {action.node_id!r} not found.", MALFORMED_PENALTY, False
            ok, err = _validate_python(action.new_code)
            if not ok:
                return f"[SYNTAX ERROR in your code] {err}", MALFORMED_PENALTY, False

            module_node = _find_module_for(kg, action.node_id)
            if module_node is None:
                return f"[ERROR] could not find module for {action.node_id!r}", MALFORMED_PENALTY, False

            old_source = target.source
            module_node.source = _apply_update_node(module_node.source, old_source, action.new_code)
            target.source = textwrap.dedent(action.new_code).strip()
            return f"Updated `{action.node_id}`.", 0.0, False

        if isinstance(action, RemoveNodeAction):
            target = kg.get_node(action.node_id)
            if target is None:
                return f"[ERROR] node_id {action.node_id!r} not found.", MALFORMED_PENALTY, False
            module_node = _find_module_for(kg, action.node_id)
            if module_node:
                module_node.source = _apply_remove_node(module_node.source, target.source)
            kg.remove_node(action.node_id)
            return f"Removed `{action.node_id}`.", 0.0, False

        if isinstance(action, SubmitAction):
            return self._run_submit()

        return f"[ERROR] unrecognised action type: {type(action)}", MALFORMED_PENALTY, False

    def _run_submit(self) -> tuple[str, float, bool]:
        """Materialise changes into sys.modules, run tests, restore."""
        kg = self._kg
        task = self._task
        assert kg is not None and task is not None

        pkg_prefix = f"graphforge.sample_repos.{task.repo_name}"
        _patch_modules(kg, pkg_prefix)
        try:
            reward, msg = _run_tests(task.test_code)
        finally:
            _restore_modules(pkg_prefix)

        return f"[SUBMIT RESULT]\n{msg}", reward, True

    def _terminal_obs(self, msg: str) -> RepoEditObservation:
        return RepoEditObservation(
            episode_id=self._episode_id,
            task_id=self._task.task_id if self._task else None,
            turn=self._turn,
            max_turns=self._task.max_turns if self._task else 0,
            graph_overview="",
            task_description="",
            action_result=msg,
            done=True,
            total_reward=self._total_reward,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_module_for(kg: KnowledgeGraph, node_id: str) -> KGNode | None:
    """Walk up the parent chain until we hit a module node."""
    current_id = node_id
    seen: set[str] = set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        node = kg.get_node(current_id)
        if node and node.node_type == "module":
            return node
        parent = kg.parent_of(current_id)
        if parent is None:
            break
        current_id = parent.node_id
    return None


def _patch_modules(kg: KnowledgeGraph, pkg_prefix: str) -> None:
    """Compile & exec each modified module source, inject into sys.modules."""
    import importlib
    import types

    # Backup original modules
    _MODULE_BACKUP.clear()
    for key, mod in list(sys.modules.items()):
        if key.startswith(pkg_prefix):
            _MODULE_BACKUP[key] = mod

    # Remove stale cached copies
    for key in list(sys.modules):
        if key.startswith(pkg_prefix):
            del sys.modules[key]

    # For each module node in the KG, create a fresh module from modified source
    for node in kg.all_nodes("module"):
        if not node.file_path or node.file_path == "__init__.py":
            continue
        # e.g. file_path = "validators.py" → full_name = "graphforge.sample_repos.task_manager.validators"
        stem = Path(node.file_path).stem
        full_name = f"{pkg_prefix}.{stem}"
        new_mod = types.ModuleType(full_name)
        new_mod.__file__ = node.file_path
        new_mod.__package__ = pkg_prefix
        try:
            exec(compile(node.source, node.file_path, "exec"), new_mod.__dict__)  # noqa: S102
        except Exception:
            pass   # leave module partially populated; tests will catch it
        sys.modules[full_name] = new_mod


_MODULE_BACKUP: dict[str, Any] = {}


def _restore_modules(pkg_prefix: str) -> None:
    """Remove patched modules and restore originals from backup."""
    for key in list(sys.modules):
        if key.startswith(pkg_prefix):
            del sys.modules[key]
    sys.modules.update(_MODULE_BACKUP)
    _MODULE_BACKUP.clear()


def _run_tests(test_code: str) -> tuple[float, str]:
    """Execute test_code in a clean namespace. Returns (reward, message)."""
    try:
        exec(compile(test_code, "<tests>", "exec"), {})  # noqa: S102
        return 1.0, "✓ All tests passed!"
    except AssertionError as exc:
        return 0.0, f"✗ Test failed: {exc}"
    except Exception:
        return 0.0, f"✗ Exception during tests:\n{traceback.format_exc(limit=5)}"


def _pick_random_task() -> str:
    import random
    return random.choice(all_task_ids())
