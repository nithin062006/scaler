"""Task bank for the AST code-editing environment.

Each Task has a Python source file with one STUB function and a set of test
cases. The environment presents the module as a DAG; the agent fills in the
stub; the env compiles and runs the tests to compute the reward.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field


@dataclass
class TestCase:
    args: tuple
    kwargs: dict
    expected: object
    description: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.args, list):
            self.args = tuple(self.args)


@dataclass
class Task:
    task_id: str
    description: str
    module_name: str
    source_code: str       # full Python source; stub marked with # STUB
    target_function: str   # name of function to implement
    test_cases: list[TestCase] = field(default_factory=list)
    difficulty: int = 0    # 0=easy, 1=medium, 2=hard


TASK_BANK: dict[str, Task] = {}


def _reg(task: Task) -> Task:
    TASK_BANK[task.task_id] = task
    return task


# ── Task 0: palindrome ────────────────────────────────────────────────────────

_reg(Task(
    task_id="t0.palindrome",
    description="Implement is_palindrome(s) — returns True if s reads the same forwards and backwards.",
    module_name="string_utils",
    source_code=textwrap.dedent("""\
        def reverse_string(s: str) -> str:
            return s[::-1]


        def is_palindrome(s: str) -> bool:
            # STUB
            raise NotImplementedError


        def check_word(word: str) -> str:
            label = "a palindrome" if is_palindrome(word) else "not a palindrome"
            return f"{word} is {label}"
    """),
    target_function="is_palindrome",
    test_cases=[
        TestCase(("racecar",), {}, True),
        TestCase(("hello",), {}, False),
        TestCase(("",), {}, True),
        TestCase(("abcba",), {}, True),
        TestCase(("python",), {}, False),
    ],
))

# ── Task 1: fibonacci ─────────────────────────────────────────────────────────

_reg(Task(
    task_id="t1.fibonacci",
    description="Implement fibonacci(n) — returns the nth Fibonacci number (0-indexed, fib(0)=0, fib(1)=1).",
    module_name="math_utils",
    source_code=textwrap.dedent("""\
        def fibonacci(n: int) -> int:
            # STUB
            raise NotImplementedError


        def fibonacci_sequence(length: int) -> list:
            return [fibonacci(i) for i in range(length)]


        def is_fibonacci_number(n: int) -> bool:
            return n in fibonacci_sequence(20)
    """),
    target_function="fibonacci",
    test_cases=[
        TestCase((0,), {}, 0),
        TestCase((1,), {}, 1),
        TestCase((2,), {}, 1),
        TestCase((5,), {}, 5),
        TestCase((10,), {}, 55),
    ],
    difficulty=1,
))

# ── Task 2: count_vowels ──────────────────────────────────────────────────────

_reg(Task(
    task_id="t2.count_vowels",
    description="Implement count_vowels(text) — returns the number of vowels (a e i o u, case-insensitive) in text.",
    module_name="text_utils",
    source_code=textwrap.dedent("""\
        VOWELS = set("aeiouAEIOU")


        def is_vowel(char: str) -> bool:
            return char in VOWELS


        def count_vowels(text: str) -> int:
            # STUB
            raise NotImplementedError


        def vowel_ratio(text: str) -> float:
            if not text:
                return 0.0
            return count_vowels(text) / len(text)
    """),
    target_function="count_vowels",
    test_cases=[
        TestCase(("hello",), {}, 2),
        TestCase(("",), {}, 0),
        TestCase(("xyz",), {}, 0),
        TestCase(("aeiou",), {}, 5),
        TestCase(("Hello World",), {}, 3),
    ],
))

# ── Task 3: find_max ──────────────────────────────────────────────────────────

_reg(Task(
    task_id="t3.find_max",
    description="Implement find_max(numbers) — returns the maximum value in a non-empty list without using built-in max().",
    module_name="list_utils",
    source_code=textwrap.dedent("""\
        def find_max(numbers: list) -> float:
            # STUB
            raise NotImplementedError


        def find_min(numbers: list) -> float:
            return -find_max([-x for x in numbers])


        def normalize(numbers: list) -> list:
            m = find_max(numbers)
            return [x / m for x in numbers] if m != 0 else list(numbers)
    """),
    target_function="find_max",
    test_cases=[
        TestCase(([1, 3, 2],), {}, 3),
        TestCase(([-1, -3, -2],), {}, -1),
        TestCase(([5],), {}, 5),
        TestCase(([0, 0, 0],), {}, 0),
        TestCase(([1, 2, 3, 4, 5],), {}, 5),
    ],
))

# ── Task 4: flatten_list ──────────────────────────────────────────────────────

_reg(Task(
    task_id="t4.flatten",
    description="Implement flatten_list(nested) — flattens one level of nesting from a list of lists.",
    module_name="list_utils",
    source_code=textwrap.dedent("""\
        def is_nested(item: object) -> bool:
            return isinstance(item, list)


        def flatten_list(nested: list) -> list:
            # STUB
            raise NotImplementedError


        def total_elements(nested: list) -> int:
            return len(flatten_list(nested))
    """),
    target_function="flatten_list",
    test_cases=[
        TestCase(([[1, 2], [3, 4]],), {}, [1, 2, 3, 4]),
        TestCase(([[1], [2], [3]],), {}, [1, 2, 3]),
        TestCase(([[], [1, 2]],), {}, [1, 2]),
        TestCase(([[],],), {}, []),
        TestCase(([[1, 2, 3]],), {}, [1, 2, 3]),
    ],
))


def get_task(task_id: str) -> Task | None:
    return TASK_BANK.get(task_id)


def all_task_ids() -> list[str]:
    return list(TASK_BANK.keys())
