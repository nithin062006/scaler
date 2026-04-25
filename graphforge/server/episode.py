"""Episode state — one per active OpenEnv session.

The server holds episodes in an in-memory dict keyed by ``episode_id``.
Episodes are entirely self-contained: they own a :class:`Graph`, a
:class:`Task`, and the running history. There is no leakage between
episodes (PROPOSAL.md §6.2 — "episode isolation").

Token accounting is a server-side concern. We use a coarse character-based
estimate (``len(json) // 4``) until a real tokenizer is wired in. The
estimate is consistent across baseline and trained runs because both go
through the same envelope.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from graphforge.actions.dispatcher import ActionResult
from graphforge.graph.schema import Graph
from graphforge.reward.engine import ActionOutcome, TurnReward
from graphforge.tasks.schema import Task


# ---- token estimation -----------------------------------------------


def estimate_tokens(payload: Any) -> int:
    """Coarse token estimate over a JSON-serializable payload.

    ~4 chars / token is the GPT-style rule of thumb. The exact tokenizer
    matters for training-time reward magnitudes; this estimate is a
    placeholder that's monotone in the size of the payload, which is
    enough to drive the 'prefer cheap queries over expensive ones' shaping
    while we wait on the real Qwen tokenizer.
    """
    try:
        s = json.dumps(payload, default=str)
    except Exception:
        s = str(payload)
    return max(0, len(s) // 4)


# ---- history records ------------------------------------------------


@dataclass
class TurnRecord:
    turn: int
    action_kind: str
    action_args: dict[str, Any]
    outcome: str       # ActionOutcome value
    ok: bool
    reward: float
    payload: dict[str, Any] = field(default_factory=dict)
    is_duplicate: bool = False
    tokens_returned: int = 0


# ---- episode --------------------------------------------------------


@dataclass
class Episode:
    id: str
    task: Task
    graph: Graph = field(default_factory=Graph.empty)
    history: list[TurnRecord] = field(default_factory=list)
    tokens_used: int = 0
    turns: int = 0
    terminated: bool = False
    terminal_reward: float | None = None
    terminal_payload: dict[str, Any] | None = None

    @classmethod
    def new(cls, task: Task) -> "Episode":
        return cls(id=str(uuid.uuid4()), task=task)

    # ----- duplicate detection ---------------------------------------

    def is_duplicate(self, kind: str, args: dict[str, Any]) -> bool:
        """True iff an identical (kind, args) action was tried this episode."""
        for r in self.history:
            if r.action_kind == kind and r.action_args == args:
                return True
        return False

    # ----- bookkeeping -----------------------------------------------

    def record_turn(
        self,
        kind: str,
        args: dict[str, Any],
        result: ActionResult,
        outcome: ActionOutcome,
        turn_reward: TurnReward,
        is_duplicate: bool,
        tokens_returned: int,
    ) -> TurnRecord:
        rec = TurnRecord(
            turn=self.turns,
            action_kind=kind,
            action_args=args,
            outcome=outcome.value,
            ok=result.ok,
            reward=turn_reward.total,
            payload=result.payload,
            is_duplicate=is_duplicate,
            tokens_returned=tokens_returned,
        )
        self.history.append(rec)
        self.turns += 1
        self.tokens_used += tokens_returned
        return rec

    # ----- snapshot --------------------------------------------------

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "episode_id": self.id,
            "task": self.task.visible_payload(),
            "turns": self.turns,
            "tokens_used": self.tokens_used,
            "budget": self.task.budget,
            "episode_cap": self.task.episode_cap,
            "terminated": self.terminated,
            "graph": {
                "modules": [m.model_dump() for m in self.graph.modules],
                "nodes": [n.model_dump() for n in self.graph.nodes],
                "edges": [e.model_dump() for e in self.graph.edges],
            },
            "history": [
                {
                    "turn": r.turn,
                    "action_kind": r.action_kind,
                    "ok": r.ok,
                    "reward": r.reward,
                }
                for r in self.history
            ],
            "terminal_reward": self.terminal_reward,
        }


# ---- in-memory store ------------------------------------------------


class EpisodeStore:
    """Thin wrapper around a dict so we can swap in a TTL cache later."""

    def __init__(self) -> None:
        self._eps: dict[str, Episode] = {}

    def put(self, ep: Episode) -> None:
        self._eps[ep.id] = ep

    def get(self, episode_id: str) -> Episode | None:
        return self._eps.get(episode_id)

    def drop(self, episode_id: str) -> bool:
        return self._eps.pop(episode_id, None) is not None

    def __len__(self) -> int:
        return len(self._eps)


# Singleton store. The server module holds onto this for the lifetime of
# the process. Tests can construct their own EpisodeStore for isolation.
GLOBAL_STORE = EpisodeStore()
