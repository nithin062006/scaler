"""Tests for the parse-only validator."""

from __future__ import annotations

from graphforge.validator import full_check, parse_check


def test_parse_check_clean():
    files = {"a.py": "def f(x: int) -> int:\n    return x + 1\n"}
    assert parse_check(files) == []


def test_parse_check_catches_syntax_error():
    files = {"a.py": "def f(x: int) -> int\n    return x + 1\n"}  # missing colon
    errs = parse_check(files)
    assert len(errs) == 1
    assert errs[0].filename == "a.py"
    assert errs[0].line is not None


def test_parse_check_multiple_files():
    files = {
        "a.py": "def f(): return 1\n",
        "b.py": "def g): return 2\n",  # syntax error
        "c.py": "def h(): return 3\n",
    }
    errs = parse_check(files)
    assert {e.filename for e in errs} == {"b.py"}


def test_full_check_report():
    good = {"a.py": "def f(): return 1\n"}
    bad = {"a.py": "def f(:\n"}
    assert full_check(good).ok is True
    assert full_check(bad).ok is False


def test_validation_report_to_dict():
    bad = {"a.py": "def f(:\n"}
    report = full_check(bad)
    d = report.to_dict()
    assert d["ok"] is False
    assert len(d["parse_errors"]) == 1
    assert d["parse_errors"][0]["filename"] == "a.py"
