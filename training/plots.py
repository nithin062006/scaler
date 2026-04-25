"""Reviewer-quality matplotlib helpers for the training pipeline.

Public functions called by train.py
-------------------------------------
  plot_reward_curve(history, out_path)        reward vs training step + smoothed
  plot_loss_curve(history, out_path)          loss vs training step + smoothed
  plot_baseline_vs_trained(base, trained, p)  histogram + summary comparison
  plot_summary_panel(base, trained, hist, p)  4-panel figure for README / paper
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── shared style ──────────────────────────────────────────────────────────────

_BLUE   = "#1f77b4"
_ORANGE = "#ff7f0e"
_GRAY   = "#7f7f7f"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── data extraction helpers ───────────────────────────────────────────────────

def _extract_series(history: Any, key: str) -> tuple[list[int], list[float]]:
    """Handle two storage formats.

    Format A (GRPO/TRL log_history): list[dict] with 'step' and metric keys.
    Format B (legacy SFT dict):       {'steps': [...], '<key>_history': [...]}
    """
    if isinstance(history, list):
        steps, vals = [], []
        for entry in history:
            if key in entry and "step" in entry:
                steps.append(int(entry["step"]))
                vals.append(float(entry[key]))
        return steps, vals

    if isinstance(history, dict):
        # Legacy format: {'steps': [...], 'loss_history': [...], ...}
        step_list = history.get("steps", [])
        val_list  = history.get(f"{key}_history", history.get(key, []))
        if step_list and val_list:
            n = min(len(step_list), len(val_list))
            return list(step_list[:n]), [float(v) for v in val_list[:n]]
        # Flat reward list (e.g. dry-run synthetic)
        if isinstance(history.get(key), list):
            vals = [float(v) for v in history[key]]
            return list(range(1, len(vals) + 1)), vals

    return [], []


def _smooth(vals: list[float], window: int | None = None) -> list[float]:
    if len(vals) < 3:
        return vals
    w = window or max(3, len(vals) // 8)
    kernel = np.ones(w) / w
    return list(np.convolve(vals, kernel, mode="same"))


# ── public API ────────────────────────────────────────────────────────────────

def plot_reward_curve(history: Any, out_path: Path) -> Path:
    """GRPO reward per training step (raw + smoothed rolling average)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    steps, rewards = _extract_series(history, "reward")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if steps:
        ax.plot(steps, rewards, color=_BLUE, alpha=0.35, linewidth=1.0,
                label="per-step reward")
        smoothed = _smooth(rewards)
        ax.plot(steps, smoothed, color=_BLUE, linewidth=2.2,
                label=f"rolling avg (w={max(3, len(rewards)//8)})")
        ax.axhline(np.mean(rewards), color=_GRAY, linestyle="--", linewidth=1.0,
                   label=f"mean = {np.mean(rewards):.3f}")
    else:
        ax.text(0.5, 0.5, "No reward data", transform=ax.transAxes,
                ha="center", va="center", color=_GRAY)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward  [0.0 – 1.0]")
    ax.set_title("GRPO reward during training")
    ax.set_ylim(-0.15, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_loss_curve(history: Any, out_path: Path) -> Path:
    """Training loss per step (raw + smoothed)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    steps, losses = _extract_series(history, "loss")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if steps:
        ax.plot(steps, losses, color=_ORANGE, alpha=0.35, linewidth=1.0,
                label="per-step loss")
        smoothed = _smooth(losses)
        ax.plot(steps, smoothed, color=_ORANGE, linewidth=2.2,
                label=f"rolling avg (w={max(3, len(losses)//8)})")
    else:
        ax.text(0.5, 0.5, "No loss data", transform=ax.transAxes,
                ha="center", va="center", color=_GRAY)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training loss (GRPO / LoRA fine-tuning)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_baseline_vs_trained(
    baseline: dict[str, Any],
    trained: dict[str, Any],
    out_path: Path,
) -> Path:
    """Two-panel comparison: reward histogram + per-domain summary."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_all  = [r for rs in baseline.get("per_task", {}).values() for r in rs]
    train_all = [r for rs in trained.get("per_task", {}).values() for r in rs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── left: reward histogram ────────────────────────────────────────────────
    ax = axes[0]
    bins = np.linspace(-0.15, 1.05, 14)
    base_mu  = float(np.mean(base_all))  if base_all  else 0.0
    train_mu = float(np.mean(train_all)) if train_all else 0.0

    ax.hist(base_all,  bins=bins, alpha=0.65, color=_GRAY,
            label=f"Before GRPO  (μ = {base_mu:.3f})", edgecolor="white")
    ax.hist(train_all, bins=bins, alpha=0.70, color=_BLUE,
            label=f"After GRPO   (μ = {train_mu:.3f})", edgecolor="white")
    ax.axvline(base_mu,  color=_GRAY,   linestyle="--", linewidth=1.5)
    ax.axvline(train_mu, color=_BLUE,   linestyle="--", linewidth=1.5)

    ax.set_xlabel("Episode reward  [0.0 – 1.0]")
    ax.set_ylabel("Number of episodes")
    ax.set_title("Reward distribution before vs. after GRPO")
    ax.legend()
    ax.grid(True, alpha=0.25)

    # ── right: summary bar chart ──────────────────────────────────────────────
    ax = axes[1]

    # Group tasks by domain prefix (e.g. "auto.humanize.xxx" → "humanize")
    def _domain(tid: str) -> str:
        parts = tid.split(".")
        return parts[1] if len(parts) >= 3 and parts[0] == "auto" else "built-in"

    all_tids = sorted(set(baseline.get("per_task", {})) | set(trained.get("per_task", {})))
    domains: dict[str, list[str]] = {}
    for tid in all_tids:
        domains.setdefault(_domain(tid), []).append(tid)

    domain_labels = sorted(domains)
    base_domain_means  = []
    train_domain_means = []
    for d in domain_labels:
        tids = domains[d]
        bvals = [r for t in tids for r in baseline.get("per_task", {}).get(t, [])]
        tvals = [r for t in tids for r in trained.get("per_task", {}).get(t, [])]
        base_domain_means.append(float(np.mean(bvals))  if bvals  else 0.0)
        train_domain_means.append(float(np.mean(tvals)) if tvals else 0.0)

    x = np.arange(len(domain_labels))
    w = 0.35
    bars_b = ax.bar(x - w/2, base_domain_means,  w, color=_GRAY,  label="Before GRPO", edgecolor="white")
    bars_t = ax.bar(x + w/2, train_domain_means, w, color=_BLUE, label="After GRPO",  edgecolor="white")

    for bar, val in zip(list(bars_b) + list(bars_t),
                        base_domain_means + train_domain_means):
        if val > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(domain_labels, rotation=30, ha="right")
    ax.set_ylabel("Mean episode reward  [0.0 – 1.0]")
    ax.set_ylim(0, 1.2)
    ax.set_title("Per-domain mean reward: before vs. after GRPO")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    delta = train_mu - base_mu
    fig.suptitle(
        f"GRPO Training Results  ·  Δ mean reward = {delta:+.3f}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_summary_panel(
    baseline: dict[str, Any],
    trained: dict[str, Any],
    history: Any,
    out_path: Path,
) -> Path:
    """4-panel summary figure: loss · reward curve · histogram · domain bars.

    Designed for the README / paper — one image that tells the full story.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    steps_r, rewards = _extract_series(history, "reward")
    steps_l, losses  = _extract_series(history, "loss")
    base_all  = [r for rs in baseline.get("per_task", {}).values() for r in rs]
    train_all = [r for rs in trained.get("per_task", {}).values() for r in rs]
    base_mu   = float(np.mean(base_all))  if base_all  else 0.0
    train_mu  = float(np.mean(train_all)) if train_all else 0.0

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ── panel A: loss ─────────────────────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[0, 0])
    if steps_l:
        ax_loss.plot(steps_l, losses, color=_ORANGE, alpha=0.3, linewidth=1.0)
        ax_loss.plot(steps_l, _smooth(losses), color=_ORANGE, linewidth=2.2,
                     label="smoothed loss")
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("A  Training Loss")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.2)

    # ── panel B: reward curve ─────────────────────────────────────────────────
    ax_rew = fig.add_subplot(gs[0, 1])
    if steps_r:
        ax_rew.plot(steps_r, rewards, color=_BLUE, alpha=0.3, linewidth=1.0)
        ax_rew.plot(steps_r, _smooth(rewards), color=_BLUE, linewidth=2.2,
                    label="smoothed reward")
        ax_rew.axhline(np.mean(rewards), color=_GRAY, linestyle="--",
                       linewidth=1.0, label=f"mean={np.mean(rewards):.3f}")
    ax_rew.set_xlabel("Training step")
    ax_rew.set_ylabel("Reward  [0.0 – 1.0]")
    ax_rew.set_ylim(-0.15, 1.05)
    ax_rew.set_title("B  GRPO Reward During Training")
    ax_rew.legend(fontsize=9)
    ax_rew.grid(True, alpha=0.2)

    # ── panel C: histogram ────────────────────────────────────────────────────
    ax_hist = fig.add_subplot(gs[1, 0])
    bins = np.linspace(-0.15, 1.05, 13)
    ax_hist.hist(base_all, bins=bins, alpha=0.65, color=_GRAY,
                 label=f"Before  μ={base_mu:.3f}", edgecolor="white")
    ax_hist.hist(train_all, bins=bins, alpha=0.70, color=_BLUE,
                 label=f"After   μ={train_mu:.3f}", edgecolor="white")
    ax_hist.axvline(base_mu,  color=_GRAY, linestyle="--", linewidth=1.5)
    ax_hist.axvline(train_mu, color=_BLUE, linestyle="--", linewidth=1.5)
    ax_hist.set_xlabel("Episode reward  [0.0 – 1.0]")
    ax_hist.set_ylabel("Number of episodes")
    ax_hist.set_title("C  Reward Distribution: Before vs. After GRPO")
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.2)

    # ── panel D: domain bar chart ─────────────────────────────────────────────
    ax_dom = fig.add_subplot(gs[1, 1])

    def _domain(tid: str) -> str:
        parts = tid.split(".")
        return parts[1] if len(parts) >= 3 and parts[0] == "auto" else "built-in"

    all_tids = sorted(set(baseline.get("per_task", {})) | set(trained.get("per_task", {})))
    domains: dict[str, list[str]] = {}
    for tid in all_tids:
        domains.setdefault(_domain(tid), []).append(tid)

    domain_labels = sorted(domains)
    base_dm, train_dm = [], []
    for d in domain_labels:
        tids = domains[d]
        bv = [r for t in tids for r in baseline.get("per_task", {}).get(t, [])]
        tv = [r for t in tids for r in trained.get("per_task", {}).get(t, [])]
        base_dm.append(float(np.mean(bv)) if bv else 0.0)
        train_dm.append(float(np.mean(tv)) if tv else 0.0)

    x = np.arange(len(domain_labels))
    w = 0.35
    ax_dom.bar(x - w/2, base_dm,  w, color=_GRAY,  label="Before GRPO", edgecolor="white")
    ax_dom.bar(x + w/2, train_dm, w, color=_BLUE, label="After GRPO",  edgecolor="white")
    ax_dom.set_xticks(x)
    ax_dom.set_xticklabels(domain_labels, rotation=30, ha="right")
    ax_dom.set_ylabel("Mean episode reward  [0.0 – 1.0]")
    ax_dom.set_ylim(0, 1.2)
    ax_dom.set_title("D  Per-domain Mean Reward")
    ax_dom.legend(fontsize=9)
    ax_dom.grid(True, axis="y", alpha=0.2)

    delta = train_mu - base_mu
    fig.suptitle(
        f"GraphForge GRPO Training Summary  ·  Qwen2.5-0.5B  ·  Δ μ-reward = {delta:+.3f}",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
