"""Entry point for HF Space training run.

Runs GRPO training, saves LoRA weights to HF Hub, then serves a
simple status page so the Space doesn't exit.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, "/app")

# Shared state updated by training thread; read by HTTP handler
_status: dict = {"phase": "starting", "baseline": None, "trained": None, "pass_rate": None}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        phase = _status["phase"]
        baseline = _status["baseline"]
        trained = _status["trained"]
        pass_rate = _status["pass_rate"]

        if phase == "done" and trained is not None:
            body = f"""<!DOCTYPE html><html><body style="font-family:monospace;padding:2rem">
<h2>GraphForge GRPO Training — Complete</h2>
<table border=1 cellpadding=8>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Baseline mean reward</td><td>{baseline:.3f}</td></tr>
<tr><td>Trained mean reward</td><td>{trained:.3f}</td></tr>
<tr><td>Delta</td><td>{trained - baseline:+.3f}</td></tr>
<tr><td>Pass rate</td><td>{pass_rate:.1%}</td></tr>
</table>
<p>LoRA weights: <a href="https://huggingface.co/nithin04/graphforge-lora">nithin04/graphforge-lora</a></p>
</body></html>"""
        else:
            body = f"""<!DOCTYPE html><html><head>
<meta http-equiv="refresh" content="30">
</head><body style="font-family:monospace;padding:2rem">
<h2>GraphForge GRPO Training — In Progress</h2>
<p>Phase: <strong>{phase}</strong></p>
<p>This page refreshes every 30 seconds.</p>
</body></html>"""

        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_):
        pass


def _start_server() -> None:
    server = HTTPServer(("0.0.0.0", 7860), _Handler)
    print("Status server listening on :7860")
    server.serve_forever()


def main() -> None:
    # Start HTTP server immediately so HF Space health check passes
    t = threading.Thread(target=_start_server, daemon=True)
    t.start()

    from training.config import TrainConfig
    from training.train import run

    out_dir = Path("outputs_hf")
    plots_dir = Path("plots_hf")

    cfg = TrainConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        dry_run=False,
        use_lora=True, lora_r=16, lora_alpha=32,
        num_generations=8,
        epochs=3,
        learning_rate=5e-6,
        batch_size=1,
        gradient_accumulation_steps=4,
        samples_per_task=6,
        n_eval_per_task=4,
        max_completion_length=256,
        temperature=0.9,
        out_dir=out_dir,
        plots_dir=plots_dir,
        skip_baseline_eval=True,
    )

    print("=" * 60)
    print("GraphForge GRPO Training — HF Space")
    print("=" * 60)

    _status["phase"] = "GRPO training"
    summary = run(cfg)

    baseline_mean = summary["baseline"]["mean"]
    trained_mean  = summary["trained"]["mean"]
    pass_rate     = summary["trained"]["pass_rate"]

    _status.update({"phase": "done", "baseline": baseline_mean,
                    "trained": trained_mean, "pass_rate": pass_rate})

    print(f"\nBaseline → {baseline_mean:.3f}")
    print(f"Trained  → {trained_mean:.3f}")
    print(f"Delta    : {trained_mean - baseline_mean:+.3f}")
    print(f"Pass rate: {pass_rate:.1%}")

    # Push LoRA weights to HF Hub
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        import glob
        from huggingface_hub import HfApi

        ckpts = sorted(
            glob.glob(str(out_dir / "grpo_checkpoint/checkpoint-*")),
            key=lambda p: int(p.split("-")[-1]),
        )
        if ckpts:
            final_ckpt = ckpts[-1]
            repo_id = "nithin04/graphforge-lora"
            api = HfApi(token=hf_token)
            try:
                api.create_repo(repo_id, exist_ok=True, private=False)
            except Exception:
                pass
            api.upload_folder(
                folder_path=final_ckpt,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"GRPO LoRA — trained={trained_mean:.3f} pass={pass_rate:.1%}",
            )
            print(f"\nModel pushed to hf.co/{repo_id}")

        (out_dir / "summary.json").write_text(json.dumps({
            "baseline": baseline_mean,
            "trained": trained_mean,
            "delta": trained_mean - baseline_mean,
            "pass_rate": pass_rate,
        }, indent=2))
    else:
        print("\nHF_TOKEN not set — model not pushed to Hub")

    # Keep main thread alive (daemon server thread will die if main exits)
    t.join()


if __name__ == "__main__":
    main()
