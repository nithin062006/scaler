"""AST-based code-editing OpenEnv environment.

Pipeline per episode
--------------------
reset()  Load a task, parse its source with the AST parser, return an
         observation containing the DAG description + task prompt.

step()   The agent emits a CodeEditAction{"code": "..."}. The env injects
         the snippet into the stub function, compiles the result, runs the
         task's test cases, and returns (obs, reward, done=True). This is a
         single-step environment — every episode ends after one action.

Reward
------
  0.00   Syntax / compile error
  0.20   Compiles but crashes at module-level import
  0.30   Compiles + runs, function not found
  0.50   Compiles + runs, 0 tests pass
  0.50 + 0.50 * (passed / total)   up to 1.00 for all tests passing
"""

from __future__ import annotations

import signal
import uuid
from contextlib import contextmanager
from typing import Any, Generator

from env.ast_parser import CodeDAG, dag_to_text, inject_function_body, parse_source
from env.models import CodeEditAction, CodeEditObservation, CodeEditState
from env.tasks import TASK_BANK, Task, TestCase, all_task_ids

try:
    from openenv.core import Environment  # type: ignore
    _HAS_OPENENV = True
except Exception:
    _HAS_OPENENV = False
    from typing import Generic, TypeVar
    A = TypeVar("A")
    O = TypeVar("O")
    S = TypeVar("S")

    class Environment(Generic[A, O, S]):  # type: ignore[no-redef]
        def reset(self) -> O: ...
        def step(self, action: A) -> tuple[O, float, bool]: ...
        def get_state(self) -> S: ...


# ── timeout helper ────────────────────────────────────────────────────────────

@contextmanager
def _time_limit(seconds: int) -> Generator[None, None, None]:
    def _handler(signum: int, frame: object) -> None:
        raise TimeoutError("execution timed out")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ── reward computation ────────────────────────────────────────────────────────

def _run_test(func: Any, tc: TestCase, timeout: int = 2) -> bool:
    try:
        with _time_limit(timeout):
            result = func(*tc.args, **tc.kwargs)
        return result == tc.expected
    except Exception:
        return False


def compute_reward(new_source: str, func_name: str, test_cases: list[TestCase]) -> tuple[float, bool, int]:
    """Return (reward, compile_ok, tests_passed)."""
    try:
        compiled = compile(new_source, "<env>", "exec")
    except SyntaxError:
        return 0.0, False, 0

    ns: dict[str, Any] = {}
    try:
        with _time_limit(5):
            exec(compiled, ns)  # noqa: S102
    except Exception:
        return 0.2, True, 0

    func = ns.get(func_name)
    if func is None:
        return 0.3, True, 0

    passed = sum(1 for tc in test_cases if _run_test(func, tc))
    n = len(test_cases)
    reward = 0.5 + 0.5 * (passed / n) if n > 0 else 0.5
    return reward, True, passed


# ── environment ───────────────────────────────────────────────────────────────

class ASTCodeEditEnvironment(
    Environment[CodeEditAction, CodeEditObservation, CodeEditState]
):
    """OpenEnv-compliant single-step code-editing environment.

    The agent sees a Python module as a DAG and must fill in one stub function.
    """

    def __init__(self, task_id: str | None = None) -> None:
        self._configured_task_id = task_id
        self._task: Task | None = None
        self._dag: CodeDAG | None = None
        self._episode_id: str | None = None
        self._done: bool = False
        self._last_reward: float | None = None
        self._current_source: str = ""

    # ----- OpenEnv contract -----------------------------------------------

    def reset(self, task_id: str | None = None) -> CodeEditObservation:
        tid = task_id or self._configured_task_id or _pick_random_task()
        task = TASK_BANK.get(tid)
        if task is None:
            raise ValueError(f"Unknown task_id: {tid!r}. Available: {all_task_ids()}")

        self._task = task
        self._dag = parse_source(task.source_code, task.module_name)
        self._episode_id = str(uuid.uuid4())[:8]
        self._done = False
        self._last_reward = None
        self._current_source = task.source_code

        fi = self._dag.function_infos.get(task.target_function)

        return CodeEditObservation(
            episode_id=self._episode_id,
            task_id=tid,
            graph_text=dag_to_text(self._dag),
            target_function=task.target_function,
            target_signature=fi.signature if fi else "",
            callers=self._dag.callers_of(task.target_function),
            callees=self._dag.callees_of(task.target_function),
            task_description=task.description,
            ok=True,
            done=False,
        )

    def step(self, action: CodeEditAction) -> tuple[CodeEditObservation, float, bool]:
        if self._task is None:
            raise RuntimeError("step() called before reset()")
        if self._done:
            obs = CodeEditObservation(
                episode_id=self._episode_id,
                task_id=self._task.task_id,
                ok=False,
                done=True,
                reward=self._last_reward or 0.0,
                info={"error": "episode_already_done"},
            )
            return obs, 0.0, True

        task = self._task
        try:
            new_source = inject_function_body(
                self._current_source, task.target_function, action.code
            )
        except Exception as exc:
            reward, compile_ok, passed = 0.0, False, 0
            new_source = self._current_source
            info: dict[str, Any] = {"inject_error": str(exc)}
        else:
            reward, compile_ok, passed = compute_reward(
                new_source, task.target_function, task.test_cases
            )
            info = {}

        self._current_source = new_source
        self._done = True
        self._last_reward = reward

        obs = CodeEditObservation(
            episode_id=self._episode_id,
            task_id=task.task_id,
            ok=True,
            compile_ok=compile_ok,
            tests_passed=passed,
            tests_total=len(task.test_cases),
            reward=reward,
            done=True,
            info=info,
        )
        return obs, reward, True

    def get_state(self) -> CodeEditState:
        return CodeEditState(
            episode_id=self._episode_id,
            task_id=self._task.task_id if self._task else None,
            done=self._done,
            reward=self._last_reward,
            source_code=self._current_source,
        )

    @property
    def state(self) -> CodeEditState:
        return self.get_state()


# ── helpers ───────────────────────────────────────────────────────────────────

def _pick_random_task() -> str:
    import random
    return random.choice(all_task_ids())
