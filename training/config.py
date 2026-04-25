"""TrainConfig — hyperparameters for the GRPO training run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    # ── model ─────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_completion_length: int = 300   # max tokens the model generates
    temperature: float = 0.9

    # ── LoRA ──────────────────────────────────────────────────────────────
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # ── GRPO ──────────────────────────────────────────────────────────────
    num_generations: int = 4           # completions per prompt (G)
    epochs: int = 3
    learning_rate: float = 5e-6
    batch_size: int = 1                # per device
    gradient_accumulation_steps: int = 4
    samples_per_task: int = 10         # dataset rows per task

    # ── eval ──────────────────────────────────────────────────────────────
    n_eval_per_task: int = 4           # completions to sample at eval time

    # ── output ────────────────────────────────────────────────────────────
    out_dir: Path = field(default_factory=lambda: Path("outputs"))
    plots_dir: Path = field(default_factory=lambda: Path("plots"))
    seed: int = 42
    dry_run: bool = False              # skip model load; use fixed completions

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        d["plots_dir"] = str(self.plots_dir)
        return d
