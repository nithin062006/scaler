"""GraphForge agent demo — runs trained (or base) model on the environment."""
from __future__ import annotations

import sys
sys.path.insert(0, "/app")

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

from env.environment import RepoEditEnvironment
from env.tasks import TASK_BANK, all_task_ids
from graphforge.repo_registry import load_all_tasks
from training.prompts import SYSTEM_PROMPT, extract_action_json, format_observation
from env.actions import parse_action

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_REPO  = "nithin04/graphforge-lora"

# ── load tasks ────────────────────────────────────────────────────────────────
print("Loading tasks...")
for t in load_all_tasks(verbose=False):
    TASK_BANK[t.task_id] = t
TASK_IDS = all_task_ids()
print(f"  {len(TASK_IDS)} tasks ready")

# ── load model ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
print(f"Loading model on {device}...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)
if device == "cpu":
    model = model.to(device)

lora_loaded = False
try:
    from peft import PeftModel  # type: ignore
    model = PeftModel.from_pretrained(model, LORA_REPO)
    lora_loaded = True
    print(f"LoRA loaded from {LORA_REPO}")
except Exception:
    print(f"No LoRA found at {LORA_REPO} — using base model")

model.eval()
print("Model ready.")


# ── episode runner ────────────────────────────────────────────────────────────
def run_episode(task_id: str) -> str:
    env = RepoEditEnvironment()
    obs = env.reset(task_id=task_id)

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": format_observation(obs.model_dump())},
    ]

    lines: list[str] = [
        f"Task : {task_id}",
        f"Desc : {obs.task_description}",
        "=" * 60,
    ]

    for turn in range(obs.max_turns):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=1792,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )

        action_dict = extract_action_json(completion) or {"kind": "submit"}
        kind = action_dict.get("kind", "?")
        lines.append(f"\nTurn {turn + 1}  action={kind}")

        try:
            action = parse_action(action_dict)
            obs, reward, done = env.step(action)
            lines.append(f"  reward={reward:.3f}  done={done}")
            if kind != "submit":
                lines.append(f"  {obs.action_result[:120]}")
            if done:
                lines.append(f"\n{'='*60}")
                lines.append(f"Episode done  total_reward={obs.total_reward:.3f}")
                break
        except Exception as e:
            lines.append(f"  error: {e}")
            break

        messages.append({"role": "assistant", "content": completion})
        messages.append({"role": "user",      "content": format_observation(obs.model_dump())})

    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
model_label = f"`{BASE_MODEL}` + LoRA ✓" if lora_loaded else f"`{BASE_MODEL}` (base only — LoRA not yet uploaded)"

with gr.Blocks(title="GraphForge Agent Demo") as demo:
    gr.Markdown(f"""
# GraphForge — Agent Demo
**Model:** {model_label}

Run the trained agent on a repo-editing task. The agent queries the knowledge graph,
makes edits, then submits — reward is 0.90 if all tests pass.
""")
    with gr.Row():
        task_dd = gr.Dropdown(TASK_IDS, value=TASK_IDS[0], label="Task", scale=4)
        run_btn = gr.Button("▶ Run Episode", variant="primary", scale=1)
    output = gr.Textbox(label="Episode log", lines=35, max_lines=60)
    run_btn.click(run_episode, inputs=task_dd, outputs=output)

demo.launch(server_name="0.0.0.0", server_port=7860)
