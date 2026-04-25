"""Multi-turn repo-editing tasks.

Each Task specifies:
  - A target repo to work on (points to a sample_repos/ subdir)
  - A natural-language description of the change to make
  - A set of test functions (Python code strings) that verify the change
  - The maximum number of turns allowed

Training tasks are deliberately structured to require multi-step navigation:
  1. The agent must QUERY the graph to find relevant nodes
  2. INSPECT nodes to understand the existing code
  3. ADD or UPDATE nodes to implement the change
  4. SUBMIT to trigger compilation + test execution

This sparse reward structure forces the agent to develop structured planning
and state tracking across long trajectories — the core theme of this project.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from pathlib import Path


SAMPLE_REPOS_DIR = Path(__file__).resolve().parent.parent / "graphforge" / "sample_repos"


@dataclass
class RepoTask:
    task_id: str
    repo_name: str                    # subdirectory name under sample_repos/
    description: str                  # natural-language task for the agent
    test_code: str                    # Python code that tests the change
    max_turns: int = 15
    difficulty: int = 0               # 0=easy, 1=medium, 2=hard
    hints: list[str] = field(default_factory=list)


TASK_BANK: dict[str, RepoTask] = {}


def _reg(task: RepoTask) -> RepoTask:
    TASK_BANK[task.task_id] = task
    return task


# ── Task 0: add validate_due_date ────────────────────────────────────────────

_reg(RepoTask(
    task_id="t0.validate_due_date",
    repo_name="task_manager",
    description=textwrap.dedent("""\
        Add a function `validate_due_date(due_date) -> bool` to `validators.py`.

        The function should return True if:
          - due_date is None (no deadline), OR
          - due_date is a datetime.date instance

        It should return False for any other type (strings, integers, etc.).
    """).strip(),
    test_code=textwrap.dedent("""\
        from datetime import date
        from graphforge.sample_repos.task_manager.validators import validate_due_date
        assert validate_due_date(None)            is True,  "None is valid (no deadline)"
        assert validate_due_date(date(2025, 1, 1)) is True,  "date object is valid"
        assert validate_due_date("2025-01-01")    is False, "string is not valid"
        assert validate_due_date(20250101)        is False, "int is not valid"
        assert validate_due_date([])              is False, "list is not valid"
    """).strip(),
    max_turns=12,
    hints=[
        "Look in validators.py to see the style of existing validators.",
        "The function signature should be: def validate_due_date(due_date) -> bool",
        "Import datetime.date inside the function or at the top of validators.py.",
    ],
))

# ── Task 1: add Task.is_overdue ───────────────────────────────────────────────

_reg(RepoTask(
    task_id="t1.is_overdue",
    repo_name="task_manager",
    description=textwrap.dedent("""\
        Add a method `is_overdue(self, today: date) -> bool` to the `Task`
        class in `models.py`.

        The method should return True if:
          - the task has a due_date AND
          - today is strictly after the due_date AND
          - the task is not yet done

        It should return False if there is no due_date, or if the task is done,
        or if today <= due_date.
    """).strip(),
    test_code=textwrap.dedent("""\
        from datetime import date
        from graphforge.sample_repos.task_manager.models import Task

        t_past   = Task("x", "low", [], due_date=date(2020, 1, 1))
        t_future = Task("y", "low", [], due_date=date(2099, 1, 1))
        t_none   = Task("z", "low", [], due_date=None)
        t_done   = Task("d", "low", [], due_date=date(2020, 1, 1))
        t_done.complete()

        today = date.today()
        assert t_past.is_overdue(today)   is True,  "past due date → overdue"
        assert t_future.is_overdue(today) is False, "future due date → not overdue"
        assert t_none.is_overdue(today)   is False, "no due date → not overdue"
        assert t_done.is_overdue(today)   is False, "done task → not overdue"
    """).strip(),
    max_turns=15,
    difficulty=1,
    hints=[
        "The Task class is in models.py.",
        "The method should check self.due_date, today, and self.done.",
    ],
))

# ── Task 2: add TaskStore.find_by_tag ─────────────────────────────────────────

_reg(RepoTask(
    task_id="t2.find_by_tag",
    repo_name="task_manager",
    description=textwrap.dedent("""\
        Add a method `find_by_tag(self, tag: str) -> list[Task]` to the
        `TaskStore` class in `storage.py`.

        The method should return a list of all tasks that have `tag` in
        their `tags` list. Return an empty list if no tasks match.
    """).strip(),
    test_code=textwrap.dedent("""\
        from graphforge.sample_repos.task_manager.models import Task
        from graphforge.sample_repos.task_manager.storage import TaskStore

        store = TaskStore()
        store.add(Task("t1", "high",   ["python", "backend"], None))
        store.add(Task("t2", "low",    ["frontend"],          None))
        store.add(Task("t3", "medium", ["python"],            None))

        result = store.find_by_tag("python")
        assert len(result) == 2, f"Expected 2, got {len(result)}"
        titles = {t.title for t in result}
        assert titles == {"t1", "t3"}, f"Wrong titles: {titles}"

        empty = store.find_by_tag("devops")
        assert empty == [], f"Expected [], got {empty}"
    """).strip(),
    max_turns=15,
    difficulty=1,
))

# ── Task 3 (hard): enforce priority validation in api.create_task ─────────────

_reg(RepoTask(
    task_id="t3.enforce_priority",
    repo_name="task_manager",
    description=textwrap.dedent("""\
        Update the `create_task` function in `api.py` so that it validates
        the `priority` argument using `validate_priority` from `validators.py`.

        If the priority is invalid, raise `ValueError` with a clear message.
        The existing validations for title and tags must still work.

        Note: `validate_priority` already exists in validators.py.
        You must import and call it inside `create_task`.
    """).strip(),
    test_code=textwrap.dedent("""\
        from graphforge.sample_repos.task_manager import api as _api
        _api.reset_store()  # clean state between runs

        # valid priority passes through
        t = _api.create_task("Buy milk", priority="high")
        assert t.priority == "high"

        # invalid priority raises ValueError
        raised = False
        try:
            _api.create_task("Bad task", priority="urgent")
        except ValueError:
            raised = True
        assert raised, "create_task should raise ValueError for invalid priority"

        # title validation still works
        raised2 = False
        try:
            _api.create_task("", priority="low")
        except ValueError:
            raised2 = True
        assert raised2, "create_task should still reject empty title"
    """).strip(),
    max_turns=18,
    difficulty=2,
    hints=[
        "api.py already imports validate_title and validate_tags from validators.",
        "You need to also import validate_priority and call it in create_task.",
    ],
))


# ── test runner ───────────────────────────────────────────────────────────────

def run_tests(task: RepoTask) -> tuple[bool, str]:
    """Execute task.test_code and return (passed, message)."""
    # Reload all task_manager modules to pick up any source-level changes
    _reload_task_manager()
    try:
        exec(compile(task.test_code, "<test>", "exec"), {})  # noqa: S102
        return True, "All assertions passed."
    except AssertionError as exc:
        return False, f"AssertionError: {exc}"
    except Exception:
        return False, traceback.format_exc(limit=5)


def _reload_task_manager() -> None:
    """Force-reload all task_manager submodules so edits take effect."""
    prefix = "graphforge.sample_repos.task_manager"
    to_reload = [k for k in sys.modules if k.startswith(prefix)]
    for mod_name in to_reload:
        del sys.modules[mod_name]


def all_task_ids() -> list[str]:
    return list(TASK_BANK.keys())


def get_task(task_id: str) -> RepoTask | None:
    return TASK_BANK.get(task_id)
