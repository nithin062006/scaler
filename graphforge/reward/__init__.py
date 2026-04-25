"""Reward engine — see :mod:`graphforge.reward.engine`.

Per-turn (dense, small) and terminal (sparse, large) reward computation
following PROPOSAL.md §5.
"""

from graphforge.reward.engine import (
    ActionOutcome,
    ALL_BEHAVIORAL_BONUS,
    ALL_STRUCTURAL_BONUS,
    ALPHA_TOKEN_COST,
    BEHAVIORAL_PER_PASS,
    DUPLICATE_ACTION,
    MATERIALIZE_FAIL_PENALTY,
    MUTATION_FAIL,
    PER_TURN_COST,
    SCHEMA_REJECTION,
    STRUCTURAL_PER_SAT,
    TYPE_CHECK_BONUS,
    TOKEN_EFFICIENCY_MAX,
    TerminalReward,
    TurnReward,
    score_terminal,
    score_turn,
)

__all__ = [
    "ALPHA_TOKEN_COST",
    "ALL_BEHAVIORAL_BONUS",
    "ALL_STRUCTURAL_BONUS",
    "ActionOutcome",
    "BEHAVIORAL_PER_PASS",
    "DUPLICATE_ACTION",
    "MATERIALIZE_FAIL_PENALTY",
    "MUTATION_FAIL",
    "PER_TURN_COST",
    "SCHEMA_REJECTION",
    "STRUCTURAL_PER_SAT",
    "TOKEN_EFFICIENCY_MAX",
    "TYPE_CHECK_BONUS",
    "TerminalReward",
    "TurnReward",
    "score_terminal",
    "score_turn",
]
