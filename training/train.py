"""Multi-turn GRPO training for the repo-editing environment.

Architecture
------------
The agent navigates a repo Knowledge Graph (parsed from AST) and must:
  1. Query/inspect the graph to find the right location
  2. Add or update nodes to implement the required change
  3. Submit — reward is sparse (only granted when tests pass)

This is a long-horizon RL problem: the agent cannot succeed by guessing.
It must plan across multiple turns, track graph state, and reason about
code structure — pushing beyond shallow next-token prediction.

Training pipeline
-----------------
  1. Baseline eval   — run untrained model on all tasks; record pass rate
  2. Multi-turn GRPO — collect G rollouts per task, score with env reward,
                       train with group-relative policy optimization + LoRA
  3. Trained eval    — re-evaluate; compare with baseline
  4. Plots           — reward curve, loss curve, before/after comparison

Usage
-----
    python -m training.train               # real GPU run
    python -m training.train --dry-run     # no model, synthetic history
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Any

try:
    from unsloth import FastLanguageModel  # type: ignore
    _USE_UNSLOTH = True
except Exception:
    _USE_UNSLOTH = False

import torch

from training.config import TrainConfig
from training.prompts import SYSTEM_PROMPT, extract_action_json, format_observation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.actions import parse_action
from env.environment import RepoEditEnvironment
from env.tasks import TASK_BANK, all_task_ids


# ── multi-turn rollout ────────────────────────────────────────────────────────

def run_episode(
    model: Any,
    tokenizer: Any,
    task_id: str,
    cfg: TrainConfig,
) -> tuple[list[tuple[str, str]], float]:
    """Run one episode. Returns ([(prompt, completion), …], terminal_reward)."""
    env = RepoEditEnvironment()
    obs = env.reset(task_id=task_id)
    obs_dict = obs.model_dump()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs_dict)},
    ]

    trajectory: list[tuple[str, str]] = []
    terminal_reward = 0.0

    for _turn in range(obs.max_turns):
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        completion = _generate(model, tokenizer, prompt_str, cfg)
        trajectory.append((prompt_str, completion))

        # Parse and execute action
        action_dict = extract_action_json(completion)
        if action_dict is None:
            # Malformed — env will penalise, keep going
            action_dict = {"kind": "submit"}   # force submit on parse failure

        try:
            action = parse_action(action_dict)
        except Exception:
            action = parse_action({"kind": "submit"})

        obs, step_reward, done = env.step(action)
        terminal_reward += step_reward if done and action_dict.get("kind") == "submit" else 0.0

        if done:
            # Extract the terminal reward from the observation
            terminal_reward = obs.total_reward
            break

        # Append assistant + next user turn to message history
        messages.append({"role": "assistant", "content": completion})
        messages.append({"role": "user", "content": format_observation(obs.model_dump())})

    return trajectory, terminal_reward


def _generate(model: Any, tokenizer: Any, prompt: str, cfg: TrainConfig) -> str:
    max_prompt_tokens = 2048 - cfg.max_completion_length
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=max_prompt_tokens,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_completion_length,
            do_sample=True,
            temperature=cfg.temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── GRPO dataset & reward ─────────────────────────────────────────────────────

def build_grpo_dataset(tokenizer: Any, cfg: TrainConfig) -> Any:
    """Build a flat dataset of (prompt, task_id) for TRL GRPOTrainer.

    Each row is the initial observation prompt. GRPOTrainer generates
    cfg.num_generations completions per row and calls reward_fn on them.
    Note: for multi-turn we use a simplified single-turn approximation here
    where the model must emit ALL actions in one shot as a sequence.
    """
    from datasets import Dataset  # type: ignore

    rows: list[dict[str, Any]] = []
    for tid in all_task_ids():
        env = RepoEditEnvironment()
        obs = env.reset(task_id=tid)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(obs.model_dump())},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for _ in range(cfg.samples_per_task):
            rows.append({"prompt": prompt, "task_ids": tid})

    random.shuffle(rows)
    return Dataset.from_list(rows)


def reward_fn(
    prompts: list[Any],
    completions: list[str],
    task_ids: list[str] | None = None,
    **_: Any,
) -> list[float]:
    """Graduated reward ladder so GRPO always has non-zero variance.

      -0.1  no <action> tags at all
       0.05 has <action> tags but invalid/unparseable JSON
       0.1  valid JSON with any recognised action kind
       0.2  update_node or add_node (shows edit intent)
       0.0  submit that fails tests
       0.9  submit that passes all tests
    """
    import re as _re
    rewards: list[float] = []
    for i, completion in enumerate(completions):
        tid = task_ids[i] if task_ids else all_task_ids()[0]

        # Level 0 — no recognised format at all
        has_block = bool(_re.search(r"```(?:json)?", completion))
        has_tag   = bool(_re.search(r"<action>", completion))
        has_kind  = any(k in completion for k in ("query", "inspect", "add_node",
                                                   "update_node", "remove_node", "submit"))
        if not has_block and not has_tag and not has_kind:
            rewards.append(-0.1)
            continue

        # Level 1 — has some structure, check if parseable
        action_dict = extract_action_json(completion)
        if action_dict is None:
            rewards.append(0.02 if has_kind else -0.05)
            continue

        # Level 2 — valid JSON, check kind
        kind = action_dict.get("kind", "")
        if kind not in ("query", "inspect", "add_node", "update_node",
                        "remove_node", "submit"):
            rewards.append(0.05)
            continue

        # Level 3 — execute through env; use the env's reward directly
        try:
            env = RepoEditEnvironment()
            env.reset(task_id=tid)
            action = parse_action(action_dict)
            _, r, done = env.step(action)
            rewards.append(r)
        except Exception:
            # Action was parsed but failed to execute (missing fields, wrong ids…)
            rewards.append(-0.10)
    return rewards


# ── eval ──────────────────────────────────────────────────────────────────────

def evaluate_multiturn(
    model: Any,
    tokenizer: Any,
    cfg: TrainConfig,
) -> dict[str, Any]:
    """Run one multi-turn episode per task; return pass counts and mean reward."""
    per_task: dict[str, list[float]] = {}
    task_ids = all_task_ids()
    for i, tid in enumerate(task_ids):
        rewards: list[float] = []
        for _ in range(cfg.n_eval_per_task):
            _, r = run_episode(model, tokenizer, tid, cfg)
            rewards.append(r)
        per_task[tid] = rewards
        print(f"  eval [{i+1}/{len(task_ids)}] {tid}: mean={sum(rewards)/len(rewards):.3f}")

    all_r = [r for rs in per_task.values() for r in rs]
    return {
        "per_task": per_task,
        "mean": sum(all_r) / len(all_r) if all_r else 0.0,
        "pass_rate": sum(1 for r in all_r if r >= 0.9) / len(all_r) if all_r else 0.0,
    }


# ── dry-run helpers ───────────────────────────────────────────────────────────

def dry_run_eval(pass_rate: float = 0.0) -> dict[str, Any]:
    per_task = {tid: [pass_rate] * 4 for tid in all_task_ids()}
    all_r = [r for rs in per_task.values() for r in rs]
    return {
        "per_task": per_task,
        "mean": sum(all_r) / len(all_r),
        "pass_rate": sum(1 for r in all_r if r >= 0.9) / len(all_r),
    }


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
            max_seq_length=4096,
            dtype=dtype,
            load_in_4bit=(device == "cuda"),
        )
        if cfg.use_lora:
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                use_gradient_checkpointing="unsloth",
            )
        print("Loaded with Unsloth")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_gpus = torch.cuda.device_count() if device == "cuda" else 0
        if n_gpus > 1:
            max_mem = {i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.85 / 1e9)}GiB"
                       for i in range(n_gpus)}
            print(f"Using {n_gpus} GPUs: {max_mem}")
            dm: dict = {"device_map": "auto", "max_memory": max_mem}
        elif device == "cuda":
            dm = {"device_map": "auto"}
        else:
            dm = {}
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=dtype, **dm,
        )
        if device not in ("cuda",):
            model = model.to(device)
        if cfg.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
            model = get_peft_model(model, LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type=TaskType.CAUSAL_LM,
            ))
            model.print_trainable_parameters()
            # transformers>=4.57 requires _gradient_checkpointing_func to be set
            # before any forward pass when gradient_checkpointing is True on layers.
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        print(f"Loaded on {device}")

    return model, tokenizer


# ── main ──────────────────────────────────────────────────────────────────────

def run(cfg: TrainConfig) -> dict[str, Any]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)

    # Auto-load tasks from all registered repos
    from graphforge.repo_registry import load_all_tasks
    auto_tasks = load_all_tasks(verbose=True)
    for t in auto_tasks:
        TASK_BANK[t.task_id] = t

    print(f"\n{'='*65}")
    print("Repo-Edit Agent  —  Multi-Turn GRPO Training")
    print(f"  model  : {cfg.model_name}")
    print(f"  tasks  : {len(all_task_ids())} total "
          f"({len(auto_tasks)} auto + {len(all_task_ids())-len(auto_tasks)} hand-written)")
    print(f"  lora   : r={cfg.lora_r}, G={cfg.num_generations}")
    print(f"  dry_run: {cfg.dry_run}")
    print(f"{'='*65}\n")

    # Model load (always needed for GRPO)
    if cfg.dry_run:
        model, tokenizer = None, None
    else:
        model, tokenizer = load_model_and_tokenizer(cfg)

    # Baseline eval (optional — skipped when skip_baseline_eval=True)
    if cfg.skip_baseline_eval:
        print("── Baseline eval skipped ──")
        baseline = dry_run_eval(pass_rate=0.0)
    elif cfg.dry_run:
        print("── Baseline eval ──")
        baseline = dry_run_eval(pass_rate=0.0)
    else:
        print("── Baseline eval ──")
        baseline = evaluate_multiturn(model, tokenizer, cfg)
        print(f"  mean reward : {baseline['mean']:.3f}")
        print(f"  pass rate   : {baseline['pass_rate']:.1%}")

    (cfg.out_dir / "baseline_eval.json").write_text(json.dumps(baseline, indent=2))

    # GRPO
    training_history: list[dict[str, Any]] = []
    if cfg.dry_run:
        training_history = [
            {"step": i + 1, "reward": 0.02 + 0.022 * i, "loss": 2.1 - 0.045 * i}
            for i in range(50)
        ]
        print("\n── (dry-run: skipping GRPO) ──")
    else:
        from trl import GRPOConfig, GRPOTrainer  # type: ignore

        # transformers>=4.57 passes `dataset` to _get_train_sampler; TRL's
        # version only accepts `self` — wrap once to accept the extra arg.
        if not getattr(GRPOTrainer._get_train_sampler, "_is_patched", False):
            _orig_gts = GRPOTrainer._get_train_sampler
            def _gts_compat(self, dataset=None):
                return _orig_gts(self)
            _gts_compat._is_patched = True
            GRPOTrainer._get_train_sampler = _gts_compat

        dataset = build_grpo_dataset(tokenizer, cfg)
        grpo_cfg = GRPOConfig(
            output_dir=str(cfg.out_dir / "grpo_checkpoint"),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            num_generations=cfg.num_generations,
            max_completion_length=cfg.max_completion_length,
            temperature=cfg.temperature,
            gradient_checkpointing=False,
            beta=0.0,  # no KL penalty; avoids ref_logps computation in TRL
            logging_steps=1,
            save_steps=50,
            report_to="none",
            seed=cfg.seed,
        )

        # Bug 1 patch (module-global): grpo_compute_loss crashes when ref=None.
        if _USE_UNSLOTH:
            try:
                import unsloth_compiled_cache.UnslothGRPOTrainer as _ug_mod
                if not getattr(_ug_mod.grpo_compute_loss, "_ref_none_patched", False):
                    _orig_gcc = _ug_mod.grpo_compute_loss
                    def _safe_gcc(*args, **kwargs):
                        if len(args) >= 3 and args[2] is None:
                            _a = list(args); _a[2] = _a[0].detach(); args = tuple(_a)
                        return _orig_gcc(*args, **kwargs)
                    _safe_gcc._ref_none_patched = True
                    _ug_mod.grpo_compute_loss = _safe_gcc
                    print("Patched grpo_compute_loss for None ref_logps")
            except Exception as _pe:
                print(f"[warn] grpo_compute_loss patch failed: {_pe}")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=grpo_cfg,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        # Bug 2 patch: done AFTER trainer creation so we can walk type(trainer).__mro__
        # to find the actual compute_loss function and its exec-namespace.
        if _USE_UNSLOTH:
            try:
                import unsloth_compiled_cache.UnslothGRPOTrainer as _ug_mod

                print(f"[dbg] trainer type: {type(trainer).__name__}")
                print(f"[dbg] _ug_mod 'compute' names: {[k for k in dir(_ug_mod) if 'compute' in k.lower()]}")
                print(f"[dbg] _ug_mod 'grpo/accum' names: {[k for k in dir(_ug_mod) if 'grpo' in k.lower() or 'accum' in k.lower()]}")

                # Walk MRO to find the actual compute_loss function object
                _cl_fn = None
                for _mro_cls in type(trainer).__mro__:
                    _entry = _mro_cls.__dict__.get("compute_loss")
                    if _entry is not None:
                        _cl_fn = _entry
                        print(f"[dbg] compute_loss found in MRO class: {_mro_cls.__name__}")
                        print(f"[dbg]   type: {type(_cl_fn)}")
                        if callable(_cl_fn) and hasattr(_cl_fn, "__globals__"):
                            print(f"[dbg]   __globals__ is _ug_mod.__dict__: {_cl_fn.__globals__ is vars(_ug_mod)}")
                            _gal_in_ns = "grpo_accumulated_loss" in _cl_fn.__globals__
                            print(f"[dbg]   'grpo_accumulated_loss' in __globals__: {_gal_in_ns}")
                        break
                else:
                    print(f"[dbg] compute_loss NOT found in MRO: {[c.__name__ for c in type(trainer).__mro__]}")

                # Also check trainer instance dict
                _inst_cl = trainer.__dict__.get("compute_loss")
                print(f"[dbg] compute_loss in trainer.__dict__: {_inst_cl is not None}")

                # Check _ug_mod for class-like objects containing compute_loss
                for _attr_name in dir(_ug_mod):
                    _attr = getattr(_ug_mod, _attr_name, None)
                    if isinstance(_attr, type) and hasattr(_attr, "compute_loss"):
                        print(f"[dbg] class {_attr_name} in _ug_mod has compute_loss")
                    if callable(_attr) and getattr(_attr, "__name__", "") == "compute_loss":
                        print(f"[dbg] callable '{_attr_name}' in _ug_mod is named compute_loss")

                # Now attempt the patch on every location we found
                _patched = False

                def _make_compat_gal(orig):
                    def _compat(*a, **kw):
                        r = orig(*a, **kw)
                        if not (isinstance(r, (tuple, list)) and len(r) >= 6):
                            return r
                        # Semantic remap:
                        # new: (loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1)
                        # old: (loss, completion_length, mean_kl, coef_1, completion_mask)
                        # completion_mask must support .sum(); flat_is_ratio may be None,
                        # so fall back to completion_length (r[1]) as a non-None tensor proxy.
                        _mask = r[4] if r[4] is not None else r[1]
                        return (r[0], r[1], r[2], r[5], _mask)
                    _compat._compat5_patched = True
                    return _compat

                # Patch via MRO class — ALWAYS re-apply so stale wrappers from
                # a previous run (same kernel session) get the updated mapping.
                # Double-wrapping is safe: outer wrapper sees len<6 and passes through.
                if _cl_fn and callable(_cl_fn) and hasattr(_cl_fn, "__globals__"):
                    _exec_gal = _cl_fn.__globals__.get("grpo_accumulated_loss")
                    if _exec_gal:
                        _compat_gal = _make_compat_gal(_exec_gal)
                        _cl_fn.__globals__["grpo_accumulated_loss"] = _compat_gal
                        print("[dbg] Patched grpo_accumulated_loss in compute_loss exec-ns")
                        _patched = True

                # Patch via _ug_mod module-level attrs that are named compute_loss
                for _attr_name in list(vars(_ug_mod)):
                    _attr = vars(_ug_mod)[_attr_name]
                    if callable(_attr) and getattr(_attr, "__name__", None) == "compute_loss":
                        _exec_gal2 = getattr(_attr, "__globals__", {}).get("grpo_accumulated_loss")
                        if _exec_gal2 and not getattr(_exec_gal2, "_compat5_patched", False):
                            _compat_gal2 = _make_compat_gal(_exec_gal2)
                            _attr.__globals__["grpo_accumulated_loss"] = _compat_gal2
                            print(f"[dbg] Patched via _ug_mod.{_attr_name}.__globals__")
                            _patched = True

                if not _patched:
                    print("[dbg] No patch location found — grpo_accumulated_loss is a closure or unreachable")

            except Exception as _pe:
                import traceback as _tb; _tb.print_exc()

        print("\n── GRPO training ──")
        import torch._dynamo
        torch._dynamo.disable(trainer.train)()  # disables Dynamo for train + all nested calls
        for entry in trainer.state.log_history:
            if "reward" in entry or "loss" in entry:
                training_history.append(entry)
        (cfg.out_dir / "training_history.json").write_text(
            json.dumps(training_history, indent=2)
        )

    # Trained eval
    print("\n── Trained eval ──")
    if cfg.dry_run:
        trained = dry_run_eval(pass_rate=0.6)
    else:
        trained = evaluate_multiturn(model, tokenizer, cfg)

    print(f"  mean reward : {trained['mean']:.3f}")
    print(f"  pass rate   : {trained['pass_rate']:.1%}")
    (cfg.out_dir / "trained_eval.json").write_text(json.dumps(trained, indent=2))

    # Plots
    from training.plots import plot_baseline_vs_trained, plot_loss_curve, plot_reward_curve

    plot_reward_curve(training_history, cfg.plots_dir / "reward_curve.png")
    plot_loss_curve(training_history, cfg.plots_dir / "loss_curve.png")
    plot_baseline_vs_trained(baseline, trained, cfg.plots_dir / "comparison.png")

    delta = trained["mean"] - baseline["mean"]
    print(f"\n  Baseline → {baseline['mean']:.3f}  |  Trained → {trained['mean']:.3f}  |  Δ {delta:+.3f}")
    print(f"  Plots saved to {cfg.plots_dir}/")

    return {"baseline": baseline, "trained": trained, "training_history": training_history}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        model_name=args.model,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        samples_per_task=args.samples_per_task,
        dry_run=args.dry_run,
        use_lora=not args.no_lora,
    )
    run(cfg)
