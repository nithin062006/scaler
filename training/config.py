"""TrainConfig — single source of truth for run-time hyperparameters.

Keep this small and serializable. The Colab notebook overrides fields by
constructing a ``TrainConfig`` directly; CLI users override via argparse
in ``train.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    # ---- model + tokenizer ------------------------------------------
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 256       # per agent turn
    temperature: float = 0.7
    top_p: float = 0.95

    # ---- task / env -------------------------------------------------
    task_id: str = "t0.email_validator"
    episode_cap: int = 20

    # ---- trajectory generation --------------------------------------
    n_explore: int = 30             # rollouts with the live model
    n_oracle: int = 20              # scripted oracle rollouts (seed)
    reward_threshold: float = 5.0   # keep terminal_reward >= threshold

    # ---- SFT --------------------------------------------------------
    epochs: int = 2
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # ---- eval -------------------------------------------------------
    n_eval_episodes: int = 20

    # ---- output -----------------------------------------------------
    out_dir: Path = field(default_factory=lambda: Path("outputs"))
    plots_dir: Path = field(default_factory=lambda: Path("plots"))
    seed: int = 42

    # ---- runtime niceties -------------------------------------------
    # If True the script never tries to load an LM and falls back to a
    # ScriptedPolicy — useful for plot/data sanity checks on machines
    # without a GPU.
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        d["plots_dir"] = str(self.plots_dir)
        return d
