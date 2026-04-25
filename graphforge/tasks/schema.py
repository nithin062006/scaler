"""Task data model.

A *task* is the agent-facing unit of work. The visible portion is what the
agent sees at reset — natural-language description plus the visible subset
of constraints. The hidden portion drives reward but is invisible to the
policy, forcing the agent to interpret the description rather than mechanically
satisfying a fully-revealed checklist (PROPOSAL.md §2.1).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from graphforge.constraints.schema import Constraint


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(..., min_length=1)
    tier: int = Field(..., ge=0, le=3)
    description: str = Field(..., min_length=1)
    visible_constraints: list[Constraint] = Field(default_factory=list)
    hidden_constraints: list[Constraint] = Field(default_factory=list)
    # Behavioral test names are visible to the agent at reset; bodies live in
    # the test runner (TODO) and are hidden. Empty for tier-0.
    behavioral_test_names: list[str] = Field(default_factory=list)
    budget: int = Field(..., gt=0)
    episode_cap: int = Field(..., gt=0)

    @property
    def all_constraints(self) -> list[Constraint]:
        return list(self.visible_constraints) + list(self.hidden_constraints)

    def visible_payload(self) -> dict[str, object]:
        """Subset of the task that's exposed to the agent at reset."""
        return {
            "id": self.id,
            "tier": self.tier,
            "description": self.description,
            "visible_constraints": [c.model_dump() for c in self.visible_constraints],
            "behavioral_test_names": list(self.behavioral_test_names),
            "budget": self.budget,
            "episode_cap": self.episode_cap,
        }
