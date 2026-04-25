"""One-command Mac/local runner.

Skips the slow ``n_explore`` live-model rollouts entirely (oracle trajectories
carry the SFT dataset). Real model load, real SFT, real before/after eval —
just sized for a 5-minute round-trip on M-series MPS.

Run from repo root:

    python -m training.run_local
"""

from __future__ import annotations

import time
from pathlib import Path

from training.config import TrainConfig
from training.train import run


def _patch_eval_progress() -> None:
    """Add per-episode progress prints inside evaluate().

    The ``graphforge.training`` package __init__ does
    ``from graphforge.training.rollout import rollout`` which shadows the
    submodule attribute with the function. ``import x.y.z as alias`` reads
    that shadowed attribute, so it returns the function — not the module.
    Workaround: import the submodule (which populates ``sys.modules``) and
    grab the module object back out of ``sys.modules`` by full name.
    """
    import sys
    import graphforge.training.rollout  # noqa: F401 — registers in sys.modules
    _r_mod = sys.modules["graphforge.training.rollout"]

    if getattr(_r_mod, "_progress_patched", False):
        return
    _orig = _r_mod.rollout

    def _verbose(*a, **kw):
        t0 = time.time()
        traj = _orig(*a, **kw)
        pf = sum(1 for s in traj.samples if not s.parse_ok)
        term = traj.terminal_total
        term_s = f"{term:+.2f}" if term is not None else "-"
        print(
            f"  ep: {len(traj):2d} turns, {time.time() - t0:5.1f}s, "
            f"parse_fail={pf}, terminal={term_s}",
            flush=True,
        )
        return traj

    _r_mod.rollout = _verbose
    _r_mod._progress_patched = True

    # training/eval.py (the top-level training package, not graphforge.training)
    # did ``from graphforge.training import rollout`` — that bound a reference
    # to the function into training.eval's module namespace. Rebind it there
    # too so ``evaluate()`` calls the verbose wrapper.
    import training.eval  # noqa: F401
    sys.modules["training.eval"].rollout = _verbose


def main() -> None:
    _patch_eval_progress()

    cfg = TrainConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        task_id="t0.email_validator",
        # Inference budget: tight, so 3 baseline + 3 trained eval episodes
        # fit in ~1-2 min on M2/M3 MPS.
        max_new_tokens=48,
        episode_cap=6,
        # Skip exploratory rollouts entirely — oracle trajectories alone
        # are enough to drive the SFT signal.
        n_oracle=20,
        n_explore=0,
        reward_threshold=5.0,
        # Short SFT — produces a real loss curve in <2 min.
        epochs=1,
        learning_rate=2e-4,
        batch_size=1,
        gradient_accumulation_steps=2,
        use_lora=True,
        n_eval_episodes=3,
        out_dir=Path("outputs"),
        plots_dir=Path("plots"),
    )
    summary = run(cfg)
    print("=" * 60)
    print(f"baseline mean reward = {summary['baseline_eval']['mean_reward']:+.2f}")
    print(f"trained mean reward  = {summary['trained_eval']['mean_reward']:+.2f}")
    print("=" * 60)
    print("Plots written to: plots/")
    print("Summary JSON:     outputs/summary.json")


if __name__ == "__main__":
    main()
