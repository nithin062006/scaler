"""Plotting helpers — matplotlib only, judge-friendly.

Functions called by train.py
-----------------------------
  plot_reward_curve(history, out_path)        reward vs training step
  plot_loss_curve(history, out_path)          loss vs training step
  plot_baseline_vs_trained(base, trained, p)  before/after comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract(history: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    steps, vals = [], []
    for entry in history:
        if key in entry and "step" in entry:
            steps.append(int(entry["step"]))
            vals.append(float(entry[key]))
    return steps, vals


# ── public API ────────────────────────────────────────────────────────────────

def plot_reward_curve(history: list[dict[str, Any]], out_path: Path) -> Path:
    """Reward vs training step (from GRPO log history)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    steps, rewards = _extract(history, "reward")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if steps:
        ax.plot(steps, rewards, marker="o", markersize=3, linewidth=1.2,
                color="tab:blue", label="GRPO reward")
        if len(rewards) >= 5:
            window = max(3, len(rewards) // 8)
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1:], smoothed, linewidth=2.0,
                    color="tab:orange", label=f"smoothed (w={window})")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward  [0 – 1]")
    ax.set_title("GRPO reward during training")
    ax.set_ylim(bottom=0, top=1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_loss_curve(history: list[dict[str, Any]], out_path: Path) -> Path:
    """Loss vs training step."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    steps, losses = _extract(history, "loss")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if steps:
        ax.plot(steps, losses, color="tab:red", linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss (GRPO)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_baseline_vs_trained(
    baseline: dict[str, Any],
    trained: dict[str, Any],
    out_path: Path,
) -> Path:
    """Side-by-side comparison: per-task bars + overall mean."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    task_ids = list(baseline.get("per_task", {}).keys())
    base_means = [float(np.mean(baseline["per_task"].get(t, [0.0]))) for t in task_ids]
    train_means = [float(np.mean(trained["per_task"].get(t, [0.0]))) for t in task_ids]

    x = np.arange(len(task_ids))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: per-task bars
    ax = axes[0]
    b1 = ax.bar(x - width / 2, base_means, width, label="Before GRPO", color="#888")
    b2 = ax.bar(x + width / 2, train_means, width, label="After GRPO", color="tab:blue")
    for bar, val in zip(list(b1) + list(b2), base_means + train_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(task_ids, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean reward  [0 – 1]")
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-task reward: before vs after GRPO")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # right: overall scatter / distribution
    ax = axes[1]
    base_all = [r for rs in baseline.get("per_task", {}).values() for r in rs]
    train_all = [r for rs in trained.get("per_task", {}).values() for r in rs]
    bins = np.linspace(0, 1, 12)
    ax.hist(base_all, bins=bins, alpha=0.6, label=f"Before (μ={np.mean(base_all):.2f})",
            color="#888")
    ax.hist(train_all, bins=bins, alpha=0.7, label=f"After  (μ={np.mean(train_all):.2f})",
            color="tab:blue")
    ax.set_xlabel("Reward  [0 – 1]")
    ax.set_ylabel("Count")
    ax.set_title("Overall reward distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("GRPO Training Results", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
