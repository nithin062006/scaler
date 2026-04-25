"""Tests for the tool-call wire format."""

from __future__ import annotations

from graphforge.training.protocol import (
    ParseFailure,
    ParseSuccess,
    parse_completion,
    render_action,
)


def test_happy_path():
    text = """I'll add the validators module first.

<action>
{"kind": "add_module", "name": "validators", "responsibility": "validation"}
</action>"""
    r = parse_completion(text)
    assert isinstance(r, ParseSuccess)
    assert r.action["kind"] == "add_module"
    assert r.action["name"] == "validators"


def test_no_action_tag():
    r = parse_completion("just thinking out loud, no tool call here")
    assert isinstance(r, ParseFailure)
    assert r.code == "no_action_tag"


def test_unclosed_tag():
    r = parse_completion("<action>\n{\"kind\": \"add_module\"}")
    assert isinstance(r, ParseFailure)
    assert r.code == "unclosed_tag"


def test_invalid_json():
    r = parse_completion("<action>\nnot json at all\n</action>")
    assert isinstance(r, ParseFailure)
    assert r.code == "invalid_json"


def test_not_an_object():
    r = parse_completion("<action>\n[1,2,3]\n</action>")
    assert isinstance(r, ParseFailure)
    assert r.code == "not_an_object"


def test_missing_kind():
    r = parse_completion('<action>\n{"name": "validators"}\n</action>')
    assert isinstance(r, ParseFailure)
    assert r.code == "missing_kind"


def test_takes_last_action_when_multiple_emitted():
    text = """First try:
<action>
{"kind": "add_module", "name": "first", "responsibility": "io"}
</action>
On reflection, I want this instead:
<action>
{"kind": "add_module", "name": "second", "responsibility": "io"}
</action>"""
    r = parse_completion(text)
    assert isinstance(r, ParseSuccess)
    assert r.action["name"] == "second"


def test_render_then_parse_round_trips():
    action = {"kind": "add_node", "name": "f", "module": "m", "signature": "() -> int"}
    rendered = render_action(action)
    r = parse_completion("preamble " + rendered)
    assert isinstance(r, ParseSuccess)
    assert r.action == action
