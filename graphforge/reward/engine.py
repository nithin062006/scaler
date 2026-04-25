"""Reward engine — per-turn (dense, small) and terminal (sparse, large).

Implementation follows PROPOSAL.md §5 verbatim. The two halves are pure
functions over lightweight envelopes so the server can call them without
threading state through the reward module.

Decisions worth flagging:

* ``All-behavioral-passing`` bonus is awarded only when there is at least
  one behavioral test. The gate for the token-efficiency bonus, however,
  treats zero behavioral tests as vacuously satisfied (so a tier-0 task
  with no behavioral tests can still earn token-efficiency reward).
* ``type_checks_ok`` is tri-state: ``True`` / ``False`` / ``None``. ``None``
  means the type-check gate didn't run (e.g. mypy isn't wired yet); the
  +3 bonus is suppressed in that case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Coefficients (PROPOSAL.md §5.1). Override at call time if you want.
ALPHA_TOKEN_COST: float = 0.0008
PER_TURN_COST: float = -0.1
MUTATION_FAIL: float = -2.0
SCHEMA_REJECTION: float = -2.0
DUPLICATE_ACTION: float = -1.0

# Terminal magnitudes (§5.2)
STRUCTURAL_PER_SAT: float = 1.0
BEHAVIORAL_PER_PASS: float = 3.0
ALL_STRUCTURAL_BONUS: float = 5.0
ALL_BEHAVIORAL_BONUS: float = 5.0
TYPE_CHECK_BONUS: float = 3.0
MATERIALIZE_FAIL_PENALTY: float = -8.0
TOKEN_EFFICIENCY_MAX: float = 5.0


# ---- per-turn -------------------------------------------------------


class ActionOutcome(str, Enum):
    """Coarse classification used by ``score_turn``.

    ``SUCCESS``    — mutation or info action returned ``ok=True``.
    ``FAILURE``    — handler raised :class:`ActionError` (rollback path).
    ``MALFORMED``  — pydantic schema rejected the action at parse time.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class TurnReward:
    base: float          # outcome-dependent component
    duplicate: float     # 0 or DUPLICATE_ACTION
    per_turn: float      # PER_TURN_COST
    token_cost: float    # alpha * tokens_returned, negated

    @property
    def total(self) -> float:
        return self.base + self.duplicate + self.per_turn + self.token_cost

    def to_dict(self) -> dict[str, float]:
        return {
            "base": self.base,
            "duplicate": self.duplicate,
            "per_turn": self.per_turn,
            "token_cost": self.token_cost,
            "total": self.total,
        }


def score_turn(
    *,
    outcome: ActionOutcome,
    is_duplicate: bool,
    tokens_returned: int,
    alpha: float = ALPHA_TOKEN_COST,
    per_turn_cost: float = PER_TURN_COST,
) -> TurnReward:
    if outcome is ActionOutcome.SUCCESS:
        base = 0.0
    elif outcome is ActionOutcome.FAILURE:
        base = MUTATION_FAIL
    else:  # MALFORMED
        base = SCHEMA_REJECTION
    return TurnReward(
        base=base,
        duplicate=DUPLICATE_ACTION if is_duplicate else 0.0,
        per_turn=per_turn_cost,
        token_cost=-alpha * max(0, tokens_returned),
    )


# ---- terminal -------------------------------------------------------


@dataclass(frozen=True)
class TerminalReward:
    structural: float           # +1 per structural constraint satisfied
    behavioral: float           # +3 per behavioral test passing
    bonus_all_structural: float
    bonus_all_behavioral: float
    bonus_type_checks: float
    penalty_materialize: float  # 0 or MATERIALIZE_FAIL_PENALTY
    efficiency: float           # gated by all-structural AND all-behavioral

    components: dict[str, object] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (
            self.structural
            + self.behavioral
            + self.bonus_all_structural
            + self.bonus_all_behavioral
            + self.bonus_type_checks
            + self.penalty_materialize
            + self.efficiency
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "structural": self.structural,
            "behavioral": self.behavioral,
            "bonus_all_structural": self.bonus_all_structural,
            "bonus_all_behavioral": self.bonus_all_behavioral,
            "bonus_type_checks": self.bonus_type_checks,
            "penalty_materialize": self.penalty_materialize,
            "efficiency": self.efficiency,
            "total": self.total,
            "components": self.components,
        }


def score_terminal(
    *,
    n_structural_satisfied: int,
    n_structural_total: int,
    n_behavioral_passing: int,
    n_behavioral_total: int,
    materialization_ok: bool,
    type_checks_ok: bool | None,
    tokens_used: int,
    budget: int,
) -> TerminalReward:
    if n_structural_satisfied < 0 or n_structural_total < 0:
        raise ValueError("structural counts must be non-negative")
    if n_behavioral_passing < 0 or n_behavioral_total < 0:
        raise ValueError("behavioral counts must be non-negative")
    if budget <= 0:
        raise ValueError("budget must be positive")

    structural = STRUCTURAL_PER_SAT * n_structural_satisfied
    behavioral = BEHAVIORAL_PER_PASS * n_behavioral_passing

    all_structural = (
        n_structural_total > 0 and n_structural_satisfied == n_structural_total
    )
    all_behavioral_present_and_passing = (
        n_behavioral_total > 0 and n_behavioral_passing == n_behavioral_total
    )
    bonus_all_structural = ALL_STRUCTURAL_BONUS if all_structural else 0.0
    bonus_all_behavioral = (
        ALL_BEHAVIORAL_BONUS if all_behavioral_present_and_passing else 0.0
    )

    if type_checks_ok is True:
        bonus_type_checks = TYPE_CHECK_BONUS
    else:
        bonus_type_checks = 0.0

    penalty_materialize = (
        0.0 if materialization_ok else MATERIALIZE_FAIL_PENALTY
    )

    # Efficiency bonus is gated on all-structural AND all-behavioral satisfied.
    # When n_behavioral_total == 0 the behavioral half is vacuously satisfied
    # for the gate's purposes (otherwise tier-0 tasks could never earn it).
    behavioral_gate_ok = (
        n_behavioral_total == 0
        or n_behavioral_passing == n_behavioral_total
    )
    efficiency = 0.0
    if all_structural and behavioral_gate_ok:
        ratio = max(0.0, (budget - tokens_used) / budget)
        efficiency = TOKEN_EFFICIENCY_MAX * ratio

    return TerminalReward(
        structural=structural,
        behavioral=behavioral,
        bonus_all_structural=bonus_all_structural,
        bonus_all_behavioral=bonus_all_behavioral,
        bonus_type_checks=bonus_type_checks,
        penalty_materialize=penalty_materialize,
        efficiency=efficiency,
        components={
            "n_structural_satisfied": n_structural_satisfied,
            "n_structural_total": n_structural_total,
            "n_behavioral_passing": n_behavioral_passing,
            "n_behavioral_total": n_behavioral_total,
            "materialization_ok": materialization_ok,
            "type_checks_ok": type_checks_ok,
            "tokens_used": tokens_used,
            "budget": budget,
        },
    )
