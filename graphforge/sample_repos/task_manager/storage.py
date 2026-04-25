"""In-memory task storage."""

from __future__ import annotations

from typing import Optional

from graphforge.sample_repos.task_manager.models import Task


class TaskStore:
    """Simple in-memory list-backed store for Task objects."""

    def __init__(self) -> None:
        self._tasks: list[Task] = []

    def add(self, task: Task) -> None:
        """Append task to the store."""
        self._tasks.append(task)

    def all(self) -> list[Task]:
        """Return all tasks."""
        return list(self._tasks)

    def find_by_title(self, title: str) -> Optional[Task]:
        """Return the first task whose title matches, or None."""
        for t in self._tasks:
            if t.title == title:
                return t
        return None

    def find_done(self) -> list[Task]:
        """Return all completed tasks."""
        return [t for t in self._tasks if t.done]

    def find_pending(self) -> list[Task]:
        """Return all incomplete tasks."""
        return [t for t in self._tasks if not t.done]
