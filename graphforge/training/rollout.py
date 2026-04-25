"""Multi-turn rollout — the bridge between the env and a policy.

For each turn:

  1. The policy is sampled, given the conversation so far. It returns a
     single text completion.
  2. The completion is parsed to extract the tool call. If parsing fails,
     a synthetic ``schema_rejection`` step is recorded with the reward
     engine's MALFORMED magnitude and the loop continues.
  3. The tool call is forwarded to the env via ``EnvClient.step``. The env
     returns ``{observation, reward, done, info}``.
  4. The observation is appended to the conversation as a user turn.
  5. We stop on ``done`` or when ``episode_cap`` is reached.

After the loop we compute discounted returns from each turn and produce a
list of ``TurnSample(prompt_messages, completion_text, reward, return_)``
tuples — exactly the shape ``trl.GRPOTrainer`` consumes when wrapped with
a custom reward function.

The rollout is environment-agnostic via :class:`EnvClient` and
policy-agnostic via :class:`Policy`. Both come from sibling modules; the
rollout function never imports torch or httpx directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graphforge.reward.engine import (
    DUPLICATE_ACTION,
    PER_TURN_COST,
    SCHEMA_REJECTION,
)
from graphforge.training.client import EnvClient
from graphforge.training.policy import Policy
from graphforge.training.prompt import (
    Message,
    append_completion,
    append_observation,
    initial_messages,
)
from graphforge.training.protocol import (
    ParseFailure,
    ParseSuccess,
    parse_completion,
)


# ---- per-turn record -------------------------------------------------


@dataclass
class TurnSample:
    """Single (prompt, completion, reward, return) tuple for the trainer.

    ``prompt_messages`` is the conversation up to (but not including) the
    assistant's completion at this turn.
    """

    turn: int
    prompt_messages: list[Message]
    completion_text: str
    reward: float
    return_: float = 0.0

    # Diagnostics; not consumed by the trainer.
    parse_ok: bool = True
    parse_failure_code: str | None = None
    env_response: dict[str, Any] = field(default_factory=dict)
    done: bool = False


@dataclass
class Trajectory:
    episode_id: str
    task_id: str
    samples: list[TurnSample] = field(default_factory=list)
    terminated_naturally: bool = False
    terminal_total: float | None = None

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.samples)

    def __len__(self) -> int:
        return len(self.samples)


# ---- rollout ---------------------------------------------------------


def rollout(
    *,
    policy: Policy,
    env: EnvClient,
    task_id: str | None = None,
    seed: int | None = None,
    gamma: float = 0.97,
    max_turns: int | None = None,
    auto_close: bool = True,
) -> Trajectory:
    """Run one episode end-to-end. Returns a :class:`Trajectory`.

    ``max_turns`` overrides the task's ``episode_cap`` if specified
    (useful for unit tests). Otherwise the env's own cap fires first.
    ``auto_close`` calls ``env.close`` when the episode ends.
    """
    reset_resp = env.reset(task_id=task_id, seed=seed)
    episode_id = reset_resp["episode_id"]
    task_visible = reset_resp["observation"]["task"]
    cap = max_turns or task_visible["episode_cap"]

    messages = initial_messages(task_visible)
    samples: list[TurnSample] = []
    done = False
    terminal_total: float | None = None

    for turn_idx in range(cap):
        # 1. Sample the policy.
        completion = policy.sample(messages)
        prompt_at_turn = list(messages)  # snapshot before appending the assistant turn

        # 2. Parse the tool call.
        parsed = parse_completion(completion)

        if isinstance(parsed, ParseFailure):
            # Synthetic step — env never sees the action. Reward mirrors
            # the MALFORMED branch of score_turn (no token cost because
            # nothing came back from the env).
            reward = SCHEMA_REJECTION + PER_TURN_COST
            sample = TurnSample(
                turn=turn_idx,
                prompt_messages=prompt_at_turn,
                completion_text=completion,
                reward=reward,
                parse_ok=False,
                parse_failure_code=parsed.code,
            )
            samples.append(sample)
            messages = append_completion(messages, completion)
            messages = append_observation(
                messages,
                {
                    "ok": False,
                    "outcome": "malformed",
                    "is_duplicate": False,
                    "reward": reward,
                    "payload": {"error": parsed.code, "message": parsed.message},
                    "turns_total": turn_idx + 1,
                    "tokens_used_total": 0,
                    "budget_remaining": task_visible["budget"],
                    "episode_cap_remaining": cap - (turn_idx + 1),
                },
            )
            continue

        # 3. Forward to env.
        assert isinstance(parsed, ParseSuccess)
        env_resp = env.step(episode_id, parsed.action)

        info = env_resp.get("info", {})
        # The env client returns a synthetic response on FastAPI 422 — that's
        # a schema_rejection (e.g. unknown kind, missing required field).
        # Score it the same as a parse-side malformed completion.
        is_schema_rejection = info.get("error") == "schema_rejection"
        if is_schema_rejection:
            reward = SCHEMA_REJECTION + PER_TURN_COST
        else:
            reward = float(env_resp.get("reward", 0.0))
        done = bool(env_resp.get("done", False))

        # The embedded observation carries duplicate flags etc.
        obs = env_resp.get("observation", {})

        sample = TurnSample(
            turn=turn_idx,
            prompt_messages=prompt_at_turn,
            completion_text=completion,
            reward=reward,
            env_response=env_resp,
            done=done,
            parse_ok=not is_schema_rejection,
            parse_failure_code="env_schema_rejection" if is_schema_rejection else None,
        )
        samples.append(sample)

        messages = append_completion(messages, completion)
        messages = append_observation(messages, obs)

        if done:
            terminal_total = info.get("terminal", {}).get("total")
            break

    if auto_close:
        try:
            env.close(episode_id)
        except Exception:
            pass

    _fill_returns(samples, gamma=gamma)

    return Trajectory(
        episode_id=episode_id,
        task_id=task_visible.get("id", ""),
        samples=samples,
        terminated_naturally=done,
        terminal_total=terminal_total,
    )


# ---- discounted returns ---------------------------------------------


def _fill_returns(samples: list[TurnSample], *, gamma: float) -> None:
    """In-place fill of ``return_`` on each sample.

    return_t = r_t + gamma * return_{t+1}, with return_{T+1} = 0.
    """
    running = 0.0
    for s in reversed(samples):
        running = s.reward + gamma * running
        s.return_ = running


# ---- helper for stub-policy demo ------------------------------------


def trajectory_summary(traj: Trajectory) -> dict[str, Any]:
    return {
        "episode_id": traj.episode_id,
        "task_id": traj.task_id,
        "n_turns": len(traj),
        "total_reward": traj.total_reward,
        "terminated_naturally": traj.terminated_naturally,
        "terminal_total": traj.terminal_total,
        "parse_failures": sum(1 for s in traj.samples if not s.parse_ok),
    }
