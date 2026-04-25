"""Registry of training repos with their clone URLs and source paths.

Add a new repo by appending to REGISTRY. The pipeline will clone it,
parse it, and auto-generate tasks from its doctests.

Each entry:
    name        short identifier used in task_ids
    url         git clone URL (depth-1 clone)
    src_hint    subdirectory containing the Python package
                (tried as: <clone>/<hint>, <clone>/src/<hint>, <clone>)
    n_tasks     max tasks to pull from this repo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RepoSpec:
    name: str
    url: str
    src_hint: str
    n_tasks: int = 6


REGISTRY: list[RepoSpec] = [
    # ── string / text ────────────────────────────────────────────────────────
    RepoSpec(
        name="humanize",
        url="https://github.com/jmoiron/humanize.git",
        src_hint="src/humanize",
        n_tasks=6,
    ),
    RepoSpec(
        name="wcwidth",
        url="https://github.com/jquast/wcwidth.git",
        src_hint="wcwidth",
        n_tasks=6,
    ),
    RepoSpec(
        name="inflect",
        url="https://github.com/jaraco/inflect.git",
        src_hint="inflect",
        n_tasks=4,
    ),

    # ── iteration / functional ───────────────────────────────────────────────
    RepoSpec(
        name="boltons",
        url="https://github.com/mahmoud/boltons.git",
        src_hint="boltons",
        n_tasks=10,
    ),
    RepoSpec(
        name="more-itertools",
        url="https://github.com/more-itertools/more-itertools.git",
        src_hint="more_itertools",
        n_tasks=8,
    ),
    RepoSpec(
        name="toolz",
        url="https://github.com/pytoolz/toolz.git",
        src_hint="toolz",
        n_tasks=6,
    ),

    # ── data transformation / ETL ────────────────────────────────────────────
    RepoSpec(
        name="petl",
        url="https://github.com/petl-developers/petl.git",
        src_hint="src/petl",
        n_tasks=8,
    ),
    RepoSpec(
        name="pydash",
        url="https://github.com/dgilland/pydash.git",
        src_hint="src/pydash",
        n_tasks=8,
    ),

]

# Repos that were evaluated and produced 0 tasks (no literal-eval-able doctests):
#   num2words, parse, dateutil — omitted from REGISTRY


def _find_src(clone_dir: str, hint: str) -> str:
    for candidate in [
        f"{clone_dir}/{hint}",
        f"{clone_dir}/src/{hint}",
        clone_dir,
    ]:
        if Path(candidate).is_dir():
            return candidate
    return clone_dir


def load_all_tasks(
    clone_root: str = "/tmp/train_repos",
    registry: list[RepoSpec] | None = None,
    verbose: bool = True,
) -> list:
    """Clone every repo in the registry and return all AutoTask objects.

    Args:
        clone_root: Directory under which repos are cloned.
        registry:   Use a custom registry; defaults to REGISTRY.
        verbose:    Print progress.

    Returns:
        Flat list of AutoTask objects from all repos.
    """
    import subprocess
    from pathlib import Path
    from graphforge.task_generator import generate_tasks

    specs = registry or REGISTRY
    all_tasks = []
    Path(clone_root).mkdir(parents=True, exist_ok=True)

    for spec in specs:
        clone_dir = str(Path(clone_root) / spec.name)
        if not Path(clone_dir).exists():
            if verbose:
                print(f"Cloning {spec.name} ...")
            subprocess.check_call(
                ["git", "clone", "--depth", "1", "-q", spec.url, clone_dir]
            )

        src = _find_src(clone_dir, spec.src_hint)
        try:
            kg, tasks = generate_tasks(src, n_tasks=spec.n_tasks)
            all_tasks.extend(tasks)
            if verbose:
                print(f"  {spec.name}: {len(tasks)} tasks  "
                      f"(DAG {len(kg._nodes)} nodes)")
        except Exception as exc:
            if verbose:
                print(f"  {spec.name}: SKIPPED — {exc}")

    if verbose:
        print(f"\nTotal auto-tasks: {len(all_tasks)}")
    return all_tasks
