"""Tests for the reward engine.

Pins every branch of PROPOSAL.md §5 so the magnitudes don't drift silently.
"""

from __future__ import annotations

import math

import pytest

from graphforge.reward import (
    ALL_BEHAVIORAL_BONUS,
    ALL_STRUCTURAL_BONUS,
    ActionOutcome,
    BEHAVIORAL_PER_PASS,
    DUPLICATE_ACTION,
    MATERIALIZE_FAIL_PENALTY,
    MUTATION_FAIL,
    PER_TURN_COST,
    SCHEMA_REJECTION,
    STRUCTURAL_PER_SAT,
    TOKEN_EFFICIENCY_MAX,
    TYPE_CHECK_BONUS,
    score_terminal,
    score_turn,
)


# ---- per-turn --------------------------------------------------------


def test_turn_success_baseline():
    r = score_turn(outcome=ActionOutcome.SUCCESS, is_duplicate=False, tokens_returned=0)
    assert r.base == 0.0
    assert r.duplicate == 0.0
    assert r.per_turn == PER_TURN_COST
    assert r.token_cost == 0.0
    assert r.total == PER_TURN_COST


def test_turn_failure_penalty():
    r = score_turn(outcome=ActionOutcome.FAILURE, is_duplicate=False, tokens_returned=0)
    assert r.base == MUTATION_FAIL
    assert r.total == MUTATION_FAIL + PER_TURN_COST


def test_turn_malformed_penalty():
    r = score_turn(outcome=ActionOutcome.MALFORMED, is_duplicate=False, tokens_returned=0)
    assert r.base == SCHEMA_REJECTION


def test_turn_duplicate_stacks_with_success():
    r = score_turn(outcome=ActionOutcome.SUCCESS, is_duplicate=True, tokens_returned=0)
    assert r.duplicate == DUPLICATE_ACTION
    assert r.total == PER_TURN_COST + DUPLICATE_ACTION


def test_turn_token_cost_scales():
    r = score_turn(
        outcome=ActionOutcome.SUCCESS, is_duplicate=False,
        tokens_returned=1000, alpha=0.001,
    )
    assert math.isclose(r.token_cost, -1.0)
    assert math.isclose(r.total, -1.0 + PER_TURN_COST)


def test_turn_negative_tokens_clamped():
    r = score_turn(outcome=ActionOutcome.SUCCESS, is_duplicate=False, tokens_returned=-50)
    assert r.token_cost == 0.0


# ---- terminal --------------------------------------------------------


def test_terminal_all_pass_tier0_no_behavioral():
    """Tier-0: 4 structural, 0 behavioral. All structural pass; mat ok."""
    r = score_terminal(
        n_structural_satisfied=4,
        n_structural_total=4,
        n_behavioral_passing=0,
        n_behavioral_total=0,
        materialization_ok=True,
        type_checks_ok=None,  # mypy not wired yet
        tokens_used=200,
        budget=4000,
    )
    assert r.structural == 4 * STRUCTURAL_PER_SAT
    assert r.behavioral == 0.0
    assert r.bonus_all_structural == ALL_STRUCTURAL_BONUS
    # Zero behavioral tests means no all-behavioral bonus...
    assert r.bonus_all_behavioral == 0.0
    # ...but the efficiency gate treats it as vacuously passing.
    assert r.efficiency > 0.0
    assert r.bonus_type_checks == 0.0  # type_checks_ok is None
    assert r.penalty_materialize == 0.0


def test_terminal_partial_structural_no_efficiency():
    r = score_terminal(
        n_structural_satisfied=2,
        n_structural_total=4,
        n_behavioral_passing=0,
        n_behavioral_total=0,
        materialization_ok=True,
        type_checks_ok=None,
        tokens_used=10,
        budget=4000,
    )
    assert r.structural == 2 * STRUCTURAL_PER_SAT
    assert r.bonus_all_structural == 0.0
    assert r.efficiency == 0.0  # gated off by partial structural


def test_terminal_materialization_fails():
    r = score_terminal(
        n_structural_satisfied=2,
        n_structural_total=4,
        n_behavioral_passing=0,
        n_behavioral_total=0,
        materialization_ok=False,
        type_checks_ok=None,
        tokens_used=100,
        budget=4000,
    )
    assert r.penalty_materialize == MATERIALIZE_FAIL_PENALTY
    assert r.total == 2 + MATERIALIZE_FAIL_PENALTY  # rest are zero


def test_terminal_with_behavioral_all_passing():
    r = score_terminal(
        n_structural_satisfied=10,
        n_structural_total=10,
        n_behavioral_passing=3,
        n_behavioral_total=3,
        materialization_ok=True,
        type_checks_ok=True,
        tokens_used=500,
        budget=4000,
    )
    assert r.behavioral == 3 * BEHAVIORAL_PER_PASS
    assert r.bonus_all_structural == ALL_STRUCTURAL_BONUS
    assert r.bonus_all_behavioral == ALL_BEHAVIORAL_BONUS
    assert r.bonus_type_checks == TYPE_CHECK_BONUS
    assert r.efficiency > 0.0


def test_terminal_efficiency_exact_value():
    r = score_terminal(
        n_structural_satisfied=2,
        n_structural_total=2,
        n_behavioral_passing=0,
        n_behavioral_total=0,
        materialization_ok=True,
        type_checks_ok=None,
        tokens_used=1000,
        budget=4000,
    )
    # ratio = (4000-1000)/4000 = 0.75 → efficiency = 5 * 0.75 = 3.75
    assert math.isclose(r.efficiency, TOKEN_EFFICIENCY_MAX * 0.75)


def test_terminal_efficiency_zero_when_overrun():
    r = score_terminal(
        n_structural_satisfied=2,
        n_structural_total=2,
        n_behavioral_passing=0,
        n_behavioral_total=0,
        materialization_ok=True,
        type_checks_ok=None,
        tokens_used=10000,  # over budget
        budget=4000,
    )
    assert r.efficiency == 0.0


def test_terminal_rejects_negative_counts():
    with pytest.raises(ValueError):
        score_terminal(
            n_structural_satisfied=-1, n_structural_total=2,
            n_behavioral_passing=0, n_behavioral_total=0,
            materialization_ok=True, type_checks_ok=None,
            tokens_used=0, budget=100,
        )


def test_terminal_zero_constraints_no_all_structural_bonus():
    """Vacuous-success guard: no constraints → no all-structural bonus."""
    r = score_terminal(
        n_structural_satisfied=0, n_structural_total=0,
        n_behavioral_passing=0, n_behavioral_total=0,
        materialization_ok=True, type_checks_ok=None,
        tokens_used=0, budget=100,
    )
    assert r.bonus_all_structural == 0.0
