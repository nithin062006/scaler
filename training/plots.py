"""Plotting helpers — matplotlib only, hackathon-judge-friendly.

Each plotting function:
  * accepts plain Python data (no matplotlib types in the API),
  * writes a PNG to ``plots_dir``,
  * uses clearly-labeled axes with units.

The four required plots per the rules:
  1. reward curve (vs trajectory index during eval)
  2. loss curve (vs training step during SFT)
  3. baseline-vs-trained comparison (same chart)
  4. terminal-reward histograms

Returns the filepath of each saved plot for embedding in the README.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless — Colab + CI safe
import matplotlib.pyplot as plt
import numpy as np


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_eval_reward_curve(
    rewards: Sequence[float],
    *,
    label: str,
    out_path: Path,
    title: str | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = list(range(1, len(rewards) + 1))
    ax.plot(xs, rewards, marker="o", linewidth=1.2, label=label)
    if len(rewards) > 1:
        running = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        ax.plot(xs, running, linestyle="--", linewidth=1.0, label="running mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Terminal reward")
    ax.set_title(title or f"Terminal reward per episode ({label})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_loss_curve(
    steps: Sequence[int],
    losses: Sequence[float],
    *,
    out_path: Path,
    title: str = "SFT training loss",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, losses, color="tab:red", linewidth=1.2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (cross-entropy)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_baseline_vs_trained(
    *,
    baseline_rewards: Sequence[float],
    trained_rewards: Sequence[float],
    out_path: Path,
) -> Path:
    """Same-graph comparison — bars (mean ± std) and overlaid scatter."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: mean ± std bars.
    ax = axes[0]
    means = [float(np.mean(baseline_rewards)) if baseline_rewards else 0.0,
             float(np.mean(trained_rewards)) if trained_rewards else 0.0]
    stds = [float(np.std(baseline_rewards)) if baseline_rewards else 0.0,
            float(np.std(trained_rewards)) if trained_rewards else 0.0]
    bars = ax.bar(["Baseline", "Trained"], means, yerr=stds, capsize=6,
                  color=["#888", "tab:blue"])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean terminal reward (± std)")
    ax.set_title("Baseline vs trained — terminal reward")
    ax.grid(True, axis="y", alpha=0.3)

    # Right: histograms overlaid.
    ax = axes[1]
    bins = _shared_bins(baseline_rewards, trained_rewards, n=15)
    ax.hist(baseline_rewards, bins=bins, alpha=0.55, label="Baseline", color="#888")
    ax.hist(trained_rewards, bins=bins, alpha=0.65, label="Trained", color="tab:blue")
    ax.set_xlabel("Terminal reward")
    ax.set_ylabel("Episode count")
    ax.set_title("Distribution of terminal reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_reward_histogram(
    rewards: Sequence[float],
    *,
    label: str,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if rewards:
        ax.hist(rewards, bins=15, color="tab:blue", alpha=0.8)
        ax.axvline(float(np.mean(rewards)), color="red", linestyle="--",
                   linewidth=1.2, label=f"mean={np.mean(rewards):.2f}")
        ax.legend()
    ax.set_xlabel("Terminal reward")
    ax.set_ylabel("Episode count")
    ax.set_title(f"Reward distribution — {label}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _shared_bins(a: Sequence[float], b: Sequence[float], n: int = 15) -> np.ndarray:
    pool = list(a) + list(b)
    if not pool:
        return np.linspace(-10, 20, n + 1)
    lo = float(min(pool))
    hi = float(max(pool))
    if hi == lo:
        hi = lo + 1.0
    return np.linspace(lo - 0.5, hi + 0.5, n + 1)
