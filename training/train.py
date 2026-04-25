"""GRPO training script for the AST code-editing environment.

Pipeline
--------
1. Build a dataset of DAG-context prompts (one per task × samples_per_task).
2. Baseline eval  — sample the untrained model on all tasks; record rewards.
3. GRPO training  — TRL GRPOTrainer with per-completion reward from the env.
4. Trained eval   — sample the fine-tuned model; record rewards.
5. Plots          — reward curve, loss curve, before/after comparison.

Reward function (called inside GRPO)
-------------------------------------
Extract the code block from the completion, inject into the task's stub,
compile, run tests. Returns a float in [0, 1].

Usage
-----
    python -m training.train                        # defaults
    python -m training.train --model Qwen/Qwen2.5-1.5B-Instruct
    python -m training.train --dry-run              # no GPU needed
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

# ── optional Unsloth (faster) ─────────────────────────────────────────────────
try:
    from unsloth import FastLanguageModel  # type: ignore
    _USE_UNSLOTH = True
except ImportError:
    _USE_UNSLOTH = False

import torch

from training.config import TrainConfig

# ── env imports ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.ast_parser import dag_to_text, inject_function_body, parse_source
from env.environment import compute_reward
from env.tasks import TASK_BANK, all_task_ids

# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a code-editing agent. You are given a Python module represented as a \
directed graph (DAG) with nodes (module, function, class) and edges \
(contains, calls, imports).

Your task: implement the body of the STUB function shown in the graph.

Rules:
- Emit ONLY the function body inside <code>...</code> tags.
- Do NOT include the def line.
- Use standard 4-space indentation.
- Your code will be compiled and tested automatically.

Example response:
<code>
    return s == s[::-1]
</code>"""


def build_user_prompt(task_id: str) -> str:
    task = TASK_BANK[task_id]
    dag = parse_source(task.source_code, task.module_name)
    fi = dag.function_infos.get(task.target_function)
    callers = dag.callers_of(task.target_function)
    callees = dag.callees_of(task.target_function)

    return f"""{dag_to_text(dag)}

## Target function to implement
Name      : {task.target_function}
Signature : {fi.signature if fi else '(unknown)'}
Called by : {', '.join(callers) or '(none)'}
Calls     : {', '.join(callees) or '(none)'}

## Task description
{task.description}

Implement `{task.target_function}` now:"""


# ── code extraction ───────────────────────────────────────────────────────────

def extract_code(completion: str) -> str:
    """Pull text from <code>...</code>; fall back to markdown fences."""
    m = re.search(r"<code>(.*?)</code>", completion, re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```(?:python)?\n(.*?)```", completion, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return completion.strip()


# ── reward function for GRPO ──────────────────────────────────────────────────

def reward_fn(
    prompts: list[Any],
    completions: list[str],
    task_ids: list[str] | None = None,
    **_: Any,
) -> list[float]:
    """Score each completion by compiling + running tests in the live env."""
    rewards: list[float] = []
    for i, completion in enumerate(completions):
        tid = task_ids[i] if task_ids else all_task_ids()[0]
        task = TASK_BANK.get(tid)
        if task is None:
            rewards.append(0.0)
            continue
        code = extract_code(completion)
        try:
            new_source = inject_function_body(task.source_code, task.target_function, code)
            r, _, _ = compute_reward(new_source, task.target_function, task.test_cases)
        except Exception:
            r = 0.0
        rewards.append(r)
    return rewards


# ── dataset ───────────────────────────────────────────────────────────────────

def build_dataset(tokenizer: Any, cfg: TrainConfig) -> Any:
    from datasets import Dataset  # type: ignore

    rows: list[dict[str, Any]] = []
    for tid in all_task_ids():
        user_msg = build_user_prompt(tid)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for _ in range(cfg.samples_per_task):
            rows.append({"prompt": prompt, "task_ids": tid})

    random.shuffle(rows)
    return Dataset.from_list(rows)


# ── eval (no gradient) ────────────────────────────────────────────────────────

def evaluate(model: Any, tokenizer: Any, cfg: TrainConfig) -> dict[str, Any]:
    """Sample cfg.n_eval_per_task completions per task; return mean rewards."""
    from transformers import pipeline  # type: ignore

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg.max_completion_length,
        do_sample=True,
        temperature=cfg.temperature,
        pad_token_id=tokenizer.eos_token_id,
    )

    per_task: dict[str, list[float]] = {}
    for tid in all_task_ids():
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(tid)},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        rewards_here: list[float] = []
        for _ in range(cfg.n_eval_per_task):
            out = pipe(prompt, return_full_text=False)
            completion: str = out[0]["generated_text"]  # type: ignore
            rs = reward_fn([prompt], [completion], task_ids=[tid])
            rewards_here.append(rs[0])
        per_task[tid] = rewards_here

    all_r = [r for rs in per_task.values() for r in rs]
    return {
        "per_task": per_task,
        "mean": sum(all_r) / len(all_r) if all_r else 0.0,
    }


# ── dry-run oracle completions ────────────────────────────────────────────────

_ORACLE: dict[str, str] = {
    "t0.palindrome": "return s == s[::-1]",
    "t1.fibonacci": "if n <= 1:\n    return n\nreturn fibonacci(n - 1) + fibonacci(n - 2)",
    "t2.count_vowels": "return sum(1 for c in text if c in VOWELS)",
    "t3.find_max": (
        "result = numbers[0]\n"
        "for x in numbers[1:]:\n"
        "    if x > result:\n"
        "        result = x\n"
        "return result"
    ),
    "t4.flatten": "result = []\nfor sub in nested:\n    result.extend(sub)\nreturn result",
}


def dry_run_eval(untrained: bool = False) -> dict[str, Any]:
    per_task: dict[str, list[float]] = {}
    for tid in all_task_ids():
        task = TASK_BANK[tid]
        if untrained:
            per_task[tid] = [0.0] * 4
        else:
            code = _ORACLE.get(tid, "pass")
            new_src = inject_function_body(task.source_code, task.target_function, code)
            r, _, _ = compute_reward(new_src, task.target_function, task.test_cases)
            per_task[tid] = [r] * 4
    all_r = [r for rs in per_task.values() for r in rs]
    return {"per_task": per_task, "mean": sum(all_r) / len(all_r) if all_r else 0.0}


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: TrainConfig) -> tuple[Any, Any]:
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device == "cuda" else torch.float32

    if _USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=2048,
            dtype=dtype,
            load_in_4bit=(device == "cuda"),
        )
        if cfg.use_lora:
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                use_gradient_checkpointing="unsloth",
            )
        print("Loaded with Unsloth + 4-bit quant")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device not in ("cuda",):
            model = model.to(device)
        if cfg.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
            lora_cfg = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
        print(f"Loaded with Transformers on {device}")

    return model, tokenizer


# ── main training loop ────────────────────────────────────────────────────────

def run(cfg: TrainConfig) -> dict[str, Any]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("AST Code-Edit  —  GRPO Training")
    print(f"  model  : {cfg.model_name}")
    print(f"  tasks  : {all_task_ids()}")
    print(f"  lora   : {cfg.use_lora} (r={cfg.lora_r})")
    print(f"  dry_run: {cfg.dry_run}")
    print(f"{'='*60}\n")

    # ── baseline eval ──────────────────────────────────────────────────────
    print("── Baseline eval (before training) ──")
    if cfg.dry_run:
        baseline = dry_run_eval(untrained=True)
        model, tokenizer = None, None
    else:
        model, tokenizer = load_model_and_tokenizer(cfg)
        baseline = evaluate(model, tokenizer, cfg)

    print(f"Baseline mean reward: {baseline['mean']:.3f}")
    (cfg.out_dir / "baseline_eval.json").write_text(json.dumps(baseline, indent=2))

    # ── GRPO training ──────────────────────────────────────────────────────
    training_history: list[dict[str, Any]] = []
    if cfg.dry_run:
        training_history = [
            {"step": i + 1, "reward": 0.05 + 0.018 * i, "loss": 1.9 - 0.042 * i}
            for i in range(45)
        ]
        print("\n── (dry-run: skipping GRPO) ──")
    else:
        from trl import GRPOConfig, GRPOTrainer  # type: ignore

        dataset = build_dataset(tokenizer, cfg)

        grpo_cfg = GRPOConfig(
            output_dir=str(cfg.out_dir / "grpo_checkpoint"),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            num_generations=cfg.num_generations,
            max_completion_length=cfg.max_completion_length,
            temperature=cfg.temperature,
            logging_steps=1,
            save_steps=50,
            report_to="none",
            seed=cfg.seed,
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=grpo_cfg,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        print("\n── GRPO training ──")
        trainer.train()

        for entry in trainer.state.log_history:
            if "reward" in entry or "loss" in entry:
                training_history.append(entry)

        (cfg.out_dir / "training_history.json").write_text(
            json.dumps(training_history, indent=2)
        )

    # ── trained eval ───────────────────────────────────────────────────────
    print("\n── Trained eval (after training) ──")
    if cfg.dry_run:
        trained = dry_run_eval(untrained=False)
    else:
        trained = evaluate(model, tokenizer, cfg)

    print(f"Trained mean reward : {trained['mean']:.3f}")
    (cfg.out_dir / "trained_eval.json").write_text(json.dumps(trained, indent=2))

    # ── plots ──────────────────────────────────────────────────────────────
    from training.plots import (
        plot_baseline_vs_trained,
        plot_loss_curve,
        plot_reward_curve,
    )

    plot_reward_curve(training_history, cfg.plots_dir / "reward_curve.png")
    plot_loss_curve(training_history, cfg.plots_dir / "loss_curve.png")
    plot_baseline_vs_trained(baseline, trained, cfg.plots_dir / "comparison.png")

    delta = trained["mean"] - baseline["mean"]
    print(f"\nPlots saved to {cfg.plots_dir}/")
    print(f"  Baseline → {baseline['mean']:.3f}")
    print(f"  Trained  → {trained['mean']:.3f}")
    print(f"  Δ        → {delta:+.3f}")

    return {
        "baseline": baseline,
        "trained": trained,
        "training_history": training_history,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        model_name=args.model,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        samples_per_task=args.samples_per_task,
        out_dir=Path(args.out_dir),
        plots_dir=Path(args.plots_dir),
        dry_run=args.dry_run,
        use_lora=not args.no_lora,
    )
    run(cfg)
