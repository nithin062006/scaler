"""Input validation functions for the task manager."""

from __future__ import annotations

VALID_PRIORITIES = {"low", "medium", "high"}


def validate_title(title: str) -> bool:
    """Return True if title is a non-empty string <= 200 chars."""
    return isinstance(title, str) and 0 < len(title) <= 200


def validate_tags(tags: object) -> bool:
    """Return True if tags is a list of strings."""
    return isinstance(tags, list) and all(isinstance(t, str) for t in tags)


def validate_email(email: str) -> bool:
    """Return True if email looks like a valid address (contains @ and .)."""
    return isinstance(email, str) and "@" in email and "." in email.split("@")[-1]


def validate_priority(priority: str) -> bool:
    """Return True if priority is one of 'low', 'medium', or 'high'."""
    return priority in VALID_PRIORITIES
