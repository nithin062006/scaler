"""Behavioral test runner.

Responsibilities (PROPOSAL.md §2.1, §6.2):

  * Run a property-based test suite (hypothesis) against materialized code,
    in a sandboxed subprocess with timeout + memory limit.
  * Tests are part of the task definition; their bodies are *hidden* from
    the agent. The agent sees only test names and pass/fail at submission.
  * Distinguish failures (assertion) from errors (timeout, crash) — both
    count as test failures, but they're surfaced separately for diagnostics.

Public surface (TODO):

    run_tests(files, tests, timeout=12.0) -> dict[str, TestResult]
"""

from __future__ import annotations


def run_tests(  # pragma: no cover — TODO
    files: dict[str, str],
    tests: list[object],
    timeout: float = 12.0,
) -> dict[str, object]:
    raise NotImplementedError("behavioral runner TODO — see PROPOSAL.md §6.2")
