"""Domain models for the task manager."""

from __future__ import annotations

from datetime import date
from typing import Optional


class Task:
    """A single task in the task manager."""

    def __init__(
        self,
        title: str,
        priority: str,
        tags: list[str],
        due_date: Optional[date] = None,
    ) -> None:
        self.title = title
        self.priority = priority   # expected: "low" | "medium" | "high"
        self.tags = tags
        self.due_date = due_date
        self.done = False

    def complete(self) -> None:
        """Mark this task as done."""
        self.done = True

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "priority": self.priority,
            "tags": self.tags,
            "done": self.done,
            "due_date": str(self.due_date) if self.due_date else None,
        }


class User:
    """A user who owns tasks."""

    def __init__(self, username: str, email: str) -> None:
        self.username = username
        self.email = email

    def display(self) -> str:
        return f"{self.username} <{self.email}>"
