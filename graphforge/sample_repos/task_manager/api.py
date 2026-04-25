"""High-level API layer that wires models, storage, and validators together."""

from __future__ import annotations

from graphforge.sample_repos.task_manager.models import Task
from graphforge.sample_repos.task_manager.storage import TaskStore
from graphforge.sample_repos.task_manager.validators import validate_tags, validate_title

_store = TaskStore()


def create_task(
    title: str,
    priority: str = "medium",
    tags: list[str] | None = None,
) -> Task:
    """Create and persist a new task.

    Raises ValueError if title or tags are invalid.
    """
    if not validate_title(title):
        raise ValueError(f"Invalid title: {title!r}")
    resolved_tags = tags or []
    if not validate_tags(resolved_tags):
        raise ValueError(f"Invalid tags: {resolved_tags!r}")
    task = Task(title=title, priority=priority, tags=resolved_tags)
    _store.add(task)
    return task


def get_all_tasks() -> list[Task]:
    """Return every task in the store."""
    return _store.all()


def complete_task(title: str) -> bool:
    """Mark a task done by title. Returns True if found, False otherwise."""
    task = _store.find_by_title(title)
    if task:
        task.complete()
        return True
    return False


def reset_store() -> None:
    """Clear the store — used by tests between runs."""
    global _store
    _store = TaskStore()
