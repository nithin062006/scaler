"""Tests for the cheap signature parser used by the dispatcher."""

from __future__ import annotations

import pytest

from graphforge.actions.signature import parse_signature


def test_simple():
    s = parse_signature("(x: int) -> bool")
    assert s.all_params == ["x"]
    assert s.required_params == ["x"]
    assert s.return_annotation == "bool"


def test_multiple_with_default():
    s = parse_signature("(x: int, y: str = 'a', z: bool = False) -> None")
    assert s.all_params == ["x", "y", "z"]
    assert s.required_params == ["x"]


def test_no_params():
    s = parse_signature("() -> None")
    assert s.all_params == []
    assert s.required_params == []


def test_keyword_only_marker():
    s = parse_signature("(x: int, *, y: str) -> bool")
    assert s.all_params == ["x", "y"]
    assert s.required_params == ["x", "y"]


def test_var_args():
    s = parse_signature("(*args: int, **kwargs: str) -> None")
    assert s.all_params == ["args", "kwargs"]


def test_complex_annotations():
    s = parse_signature("(x: dict[str, int], y: list[tuple[int, str]] = []) -> dict[str, int]")
    assert s.all_params == ["x", "y"]
    assert s.required_params == ["x"]
    assert "dict" in s.return_annotation


def test_unparseable():
    with pytest.raises(ValueError):
        parse_signature("not a signature at all")
