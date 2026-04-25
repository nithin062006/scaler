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
    """Two-panel comparison: aggregate bar chart + key metrics table."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_mu   = float(baseline.get("mean", 0.0))
    train_mu  = float(trained.get("mean",  0.0))
    base_pr   = float(baseline.get("pass_rate", 0.0))
    train_pr  = float(trained.get("pass_rate",  0.0))
    delta_mu  = train_mu  - base_mu
    delta_pr  = train_pr  - base_pr

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── left: mean reward + pass rate bars ───────────────────────────────────
    ax = axes[0]
    metrics   = ["Mean reward", "Pass rate"]
    base_vals  = [base_mu,  base_pr]
    train_vals = [train_mu, train_pr]
    x = np.arange(len(metrics))
    w = 0.32

    bars_b = ax.bar(x - w/2, base_vals,  w, color=_GRAY, label="Before GRPO", edgecolor="white")
    bars_t = ax.bar(x + w/2, train_vals, w, color=_BLUE, label="After GRPO",  edgecolor="white")

    for bar, val in zip(list(bars_b) + list(bars_t), base_vals + train_vals):
        ypos = max(val, 0) + 0.015
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, f"{val:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Value  (reward: −0.1 – 1.0,  pass rate: 0 – 1)")
    ax.set_ylim(-0.15, 0.45)
    ax.set_title("Before vs. After GRPO")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    # ── right: metrics table ──────────────────────────────────────────────────
    ax = axes[1]
    ax.axis("off")

    rows = [
        ["Metric",            "Before GRPO",     "After GRPO",       "Δ Change"],
        ["Mean reward",       f"{base_mu:.3f}",   f"{train_mu:.3f}",  f"{delta_mu:+.3f}"],
        ["Pass rate",         f"{base_pr:.1%}",   f"{train_pr:.1%}",  f"{delta_pr:+.1%}"],
        ["Model",             "Qwen2.5-0.5B",     "Qwen2.5-0.5B",     "LoRA r=16"],
        ["Tasks",             "56 (8 repos)",     "56 (8 repos)",     "3 epochs"],
    ]

    col_widths = [0.26, 0.22, 0.22, 0.20]
    col_x = [0.02, 0.28, 0.50, 0.72]
    row_h = 0.16
    row_y_start = 0.88

    for ri, row in enumerate(rows):
        y = row_y_start - ri * row_h
        bg = "#e8f0fe" if ri == 0 else ("#f9f9f9" if ri % 2 == 0 else "white")
        ax.add_patch(plt.Rectangle((0, y - row_h * 0.85), 1.0, row_h,
                                   transform=ax.transAxes, color=bg, zorder=0))
        for ci, (cell, cx) in enumerate(zip(row, col_x)):
            weight = "bold" if ri == 0 or ci == 0 else "normal"
            color  = _BLUE if ri > 0 and ci == 3 and cell.startswith("+") else "black"
            ax.text(cx, y - row_h * 0.3, cell, transform=ax.transAxes,
                    fontsize=10, fontweight=weight, color=color, va="center")

    ax.set_title("Training Summary", fontsize=12)

    fig.suptitle(
        f"GraphForge GRPO Results  ·  Qwen2.5-0.5B + LoRA  ·  56 tasks, 3 epochs",
        fontsize=12, fontweight="bold",
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
    """4-panel summary figure for README / paper.

    A: GRPO loss curve (real training data)
    B: Mean reward before vs. after (bar chart)
    C: Pass rate before vs. after (bar chart)
    D: Graduated reward ladder (explains the training signal)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    steps_l, losses = _extract_series(history, "loss")
    base_mu  = float(baseline.get("mean",      0.0))
    train_mu = float(trained.get("mean",       0.0))
    base_pr  = float(baseline.get("pass_rate", 0.0))
    train_pr = float(trained.get("pass_rate",  0.0))

    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.38)

    # ── A: GRPO loss curve ────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    if steps_l:
        ax_a.plot(steps_l, losses, color=_ORANGE, alpha=0.30, linewidth=1.0,
                  label="per-step loss")
        ax_a.plot(steps_l, _smooth(losses), color=_ORANGE, linewidth=2.2,
                  label=f"rolling avg (w={max(3, len(losses)//8)})")
        ax_a.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax_a.set_xlabel("Training step")
    ax_a.set_ylabel("GRPO policy gradient loss")
    ax_a.set_title("A  Training Loss")
    ax_a.legend(fontsize=8, loc="upper right")
    ax_a.grid(True, alpha=0.2)

    # ── B: mean reward before vs after ───────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    labels = ["Before GRPO", "After GRPO"]
    vals   = [base_mu, train_mu]
    colors = [_GRAY, _BLUE]
    bars   = ax_b.bar(labels, vals, color=colors, edgecolor="white", width=0.45)
    for bar, v in zip(bars, vals):
        ypos = (v + 0.015) if v >= 0 else (v - 0.03)
        ax_b.text(bar.get_x() + bar.get_width() / 2, ypos, f"{v:.3f}",
                  ha="center", va="bottom" if v >= 0 else "top",
                  fontsize=13, fontweight="bold")
    ax_b.axhline(0, color="black", linewidth=0.8)
    ax_b.set_ylabel("Mean episode reward")
    ax_b.set_ylim(-0.20, max(train_mu + 0.12, 0.35))
    ax_b.set_title(f"B  Mean Reward  (Δ = {train_mu - base_mu:+.3f})")
    ax_b.grid(True, axis="y", alpha=0.2)

    # ── C: pass rate before vs after ─────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    vals_pr = [base_pr * 100, train_pr * 100]
    bars_pr = ax_c.bar(labels, vals_pr, color=colors, edgecolor="white", width=0.45)
    for bar, v in zip(bars_pr, vals_pr):
        ax_c.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}%",
                  ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax_c.set_ylabel("Pass rate  (%)")
    ax_c.set_ylim(0, max(train_pr * 100 + 8, 35))
    ax_c.set_title(f"C  Pass Rate  (Δ = {(train_pr - base_pr)*100:+.1f} pp)")
    ax_c.grid(True, axis="y", alpha=0.2)

    # ── D: reward ladder ──────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ladder_labels = [
        "Submit — tests pass",
        "add_node / update_node OK",
        "query / inspect OK",
        "Valid JSON, known kind",
        "Has structure, bad JSON",
        "Submit — tests fail",
        "No recognisable structure",
    ]
    ladder_vals   = [0.90, 0.20, 0.10, 0.05, 0.02, 0.00, -0.10]
    ladder_colors = [
        "#2ca02c", _BLUE, "#aec7e8", "#c5b0d5", "#c49c94", _GRAY, "#d62728"
    ]
    y_pos = range(len(ladder_labels))
    hbars = ax_d.barh(list(y_pos), ladder_vals, color=ladder_colors,
                      edgecolor="white", height=0.6)
    for bar, v in zip(hbars, ladder_vals):
        xpos = v + 0.01 if v >= 0 else v - 0.01
        ha   = "left"   if v >= 0 else "right"
        ax_d.text(xpos, bar.get_y() + bar.get_height() / 2,
                  f"{v:+.2f}", va="center", ha=ha, fontsize=9, fontweight="bold")
    ax_d.set_yticks(list(y_pos))
    ax_d.set_yticklabels(ladder_labels, fontsize=8)
    ax_d.axvline(0, color="black", linewidth=0.8)
    ax_d.set_xlabel("Reward value")
    ax_d.set_xlim(-0.25, 1.1)
    ax_d.set_title("D  Graduated Reward Ladder")
    ax_d.grid(True, axis="x", alpha=0.2)

    fig.suptitle(
        "GraphForge  ·  Qwen2.5-0.5B + LoRA (r=16)  ·  GRPO  ·  56 tasks, 3 epochs",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
