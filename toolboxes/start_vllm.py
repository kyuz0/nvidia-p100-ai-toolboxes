#!/usr/bin/env python3
"""
vLLM Launcher for NVIDIA Tesla P100 (sm_60 / Pascal)
=====================================================
Interactive TUI to select and serve models via the sasha0552 Pascal-compatible
vLLM fork. Tailored for 4x P100 16GB HBM2 GPUs.

Key P100 constraints baked into this launcher:
  - No FlashAttention (Pascal lacks the hardware; xformers fallback is used)
  - No FP8 / Marlin quantization kernels (require Ampere+)
  - enforce-eager recommended to avoid CUDA graph issues on sm_60
  - VLLM_USE_V1=0 to stay on the stable legacy engine
  - Conservative gpu-memory-utilization (0.85) given 16GB VRAM budget
"""
import sys
import os
import shutil
import tempfile
import subprocess
from pathlib import Path

# ─── Hardware Constants ──────────────────────────────────────────────────────
VRAM_PER_GPU_GB = 16  # P100 = 16 GB HBM2

HOST = os.getenv("HOST", "0.0.0.0")
PORT = os.getenv("PORT", "8000")

# ─── Model Catalogue ────────────────────────────────────────────────────────
# Each entry is sized for 16 GB VRAM per P100.
# Only FP16 / GPTQ-4bit / AWQ-4bit quants are usable (no Marlin, no FP8).
# Context lengths are set conservatively to leave headroom for KV cache.
MODEL_TABLE = {
    # --- 1-GPU models (fit in single 16 GB P100) ---

    # Qwen 2.5 7B Instruct – native FP16, ~14 GB
    "Qwen/Qwen2.5-7B-Instruct": {
        "trust_remote": False,
        "valid_tp": [1],
        "ctx": 4096,
        "gpu_util": 0.85,
        "enforce_eager": True,
    },

    # Llama 3.1 8B Instruct – native FP16, ~15 GB
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "trust_remote": False,
        "valid_tp": [1, 2],
        "ctx": 4096,
        "gpu_util": 0.85,
        "enforce_eager": True,
    },

    # Qwen 2.5 14B Instruct GPTQ-Int4 – ~8 GB weights
    "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4": {
        "trust_remote": True,
        "valid_tp": [1, 2],
        "ctx": 4096,
        "gpu_util": 0.85,
        "enforce_eager": True,
    },

    # --- 2-GPU models (split across 2x P100 = 32 GB) ---

    # Qwen 2.5 32B Instruct AWQ – ~17 GB weights
    "Qwen/Qwen2.5-32B-Instruct-AWQ": {
        "trust_remote": True,
        "valid_tp": [2],
        "ctx": 4096,
        "gpu_util": 0.85,
        "enforce_eager": True,
    },

    # Llama 3.1 70B Instruct AWQ – ~36 GB weights
    "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4": {
        "trust_remote": False,
        "valid_tp": [4],
        "ctx": 4096,
        "gpu_util": 0.85,
        "enforce_eager": True,
    },
}

MODELS_TO_RUN = list(MODEL_TABLE.keys())


# ─── GPU Detection ──────────────────────────────────────────────────────────
def detect_gpus():
    """Detect NVIDIA GPUs via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if res.returncode == 0:
            lines = [l.strip() for l in res.stdout.strip().split("\n") if l.strip()]
            return len(lines)
    except FileNotFoundError:
        pass
    # Fallback
    return 1


def get_discovered_models(gpu_count):
    """Filter MODEL_TABLE to models compatible with the available GPU count."""
    compatible = []
    for m in MODELS_TO_RUN:
        if m in MODEL_TABLE:
            valid_tps = MODEL_TABLE[m].get("valid_tp", [1])
            if min(valid_tps) <= gpu_count:
                compatible.append(m)
    return compatible


# ─── TUI Helpers ─────────────────────────────────────────────────────────────
def check_dependencies():
    if not shutil.which("dialog"):
        print("Error: 'dialog' is required.  Install it:  apt-get install dialog")
        sys.exit(1)


def run_dialog(args):
    """Run dialog(1) and return the user's selection (from stderr)."""
    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        try:
            subprocess.run(["dialog"] + args, stderr=tf, check=True)
            tf.seek(0)
            return tf.read().strip()
        except subprocess.CalledProcessError:
            return None  # cancelled


def nuke_vllm_cache():
    """Clear vLLM / Triton caches to avoid stale CUDA graphs."""
    for cache_dir, label in [
        (Path.home() / ".cache" / "vllm", "vLLM"),
        (Path.home() / ".triton" / "cache", "Triton"),
    ]:
        if cache_dir.exists():
            try:
                print(f"  Clearing {label} cache at {cache_dir}...", end="", flush=True)
                subprocess.run(["rm", "-rf", str(cache_dir)], check=True)
                print(" Done.")
            except Exception as e:
                print(f" Failed: {e}")


# ─── Configuration & Launch ──────────────────────────────────────────────────
def configure_and_launch(model_idx, models, gpu_count):
    model_id = models[model_idx]
    config = MODEL_TABLE[model_id]

    valid_tps = config.get("valid_tp", [1])
    max_tp = max(valid_tps)

    current_tp = min(gpu_count, max_tp)
    current_ctx = config.get("ctx", 4096)
    current_util = config.get("gpu_util", 0.85)
    current_seqs = 1
    use_eager = config.get("enforce_eager", True)  # Default ON for P100
    clear_cache = True

    name = model_id.split("/")[-1]

    while True:
        cache_status = "YES" if clear_cache else "NO"
        eager_status = "YES" if use_eager else "NO"

        menu_args = [
            "--clear", "--backtitle", f"NVIDIA P100 vLLM Launcher (GPUs: {gpu_count})",
            "--title", f"Configuration: {name}",
            "--menu", "Customize Launch Parameters:", "20", "65", "8",
            "1", f"Tensor Parallelism:   {current_tp}",
            "2", f"Concurrent Requests:  {current_seqs}",
            "3", f"Context Length:       {current_ctx}",
            "4", f"GPU Utilization:      {current_util}",
            "5", f"Erase vLLM Cache:     {cache_status}",
            "6", f"Force Eager Mode:     {eager_status}",
            "7", "LAUNCH SERVER",
        ]

        choice = run_dialog(menu_args)
        if not choice:
            return False  # Back / Cancel

        if choice == "1":
            new_tp = run_dialog([
                "--title", "Tensor Parallelism",
                "--rangebox", f"Set TP Size (1-{max_tp})", "10", "40",
                "1", str(max_tp), str(current_tp),
            ])
            if new_tp:
                current_tp = int(new_tp)

        elif choice == "2":
            new_seqs = run_dialog([
                "--title", "Concurrent Requests",
                "--menu", "Select Max Concurrent Requests:", "12", "40", "4",
                "1", "1 (Latency Focus)",
                "4", "4 (Balanced)",
                "8", "8 (Throughput)",
                "16", "16 (Max Load)",
            ])
            if new_seqs:
                current_seqs = int(new_seqs)

        elif choice == "3":
            new_ctx = run_dialog([
                "--title", "Context Length",
                "--inputbox",
                f"Context length (current: {current_ctx}).\n"
                "P100 has no FlashAttention — keep this low.",
                "10", "50", str(current_ctx),
            ])
            if new_ctx:
                current_ctx = int(new_ctx)

        elif choice == "4":
            new_util = run_dialog([
                "--title", "GPU Memory Utilization",
                "--inputbox",
                f"GPU utilization 0.0–1.0 (current: {current_util}).\n"
                "P100 has 16 GB HBM2. Stay ≤ 0.90 to avoid OOM.",
                "10", "50", str(current_util),
            ])
            if new_util:
                current_util = float(new_util)

        elif choice == "5":
            clear_cache = not clear_cache

        elif choice == "6":
            use_eager = not use_eager

        elif choice == "7":
            break

    # ── Build Command ────────────────────────────────────────────────────
    subprocess.run(["clear"])

    if clear_cache:
        nuke_vllm_cache()

    cmd = [
        "vllm", "serve", model_id,
        "--host", HOST,
        "--port", PORT,
        "--tensor-parallel-size", str(current_tp),
        "--max-num-seqs", str(current_seqs),
        "--max-model-len", str(current_ctx),
        "--gpu-memory-utilization", str(current_util),
        "--dtype", "half",
    ]

    if config.get("trust_remote"):
        cmd.append("--trust-remote-code")
    if use_eager:
        cmd.append("--enforce-eager")

    # ── Environment ──────────────────────────────────────────────────────
    env = os.environ.copy()

    # Force legacy V0 engine — the V1 engine has known issues on sm_60
    env["VLLM_USE_V1"] = "0"

    # Disable torch.compile cache (avoids stale artefacts across versions)
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Launching: {name}")
    print(f"  Config:    TP={current_tp} | Seqs={current_seqs} | "
          f"Ctx={current_ctx} | Util={current_util}")
    print(f"  Eager:     {eager_status}")
    if current_tp > gpu_count:
        print(f"  ⚠ WARNING: Model requires TP={current_tp} but only "
              f"{gpu_count} GPUs detected.")
    if clear_cache:
        print("  Action:    Cleared vLLM/Triton caches")
    print(f"\n  Command:   {' '.join(cmd)}")
    print()
    print("  Environment overrides:")
    print("    VLLM_USE_V1=0")
    print("    VLLM_DISABLE_COMPILE_CACHE=1")
    print("=" * 60)
    print()

    os.execvpe("vllm", cmd, env)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    check_dependencies()
    gpu_count = detect_gpus()
    models = get_discovered_models(gpu_count)

    if not models:
        print("No compatible models found for the detected GPU configuration.")
        sys.exit(1)

    while True:
        menu_items = []
        for i, m_id in enumerate(models):
            name = m_id.split("/")[-1]
            cfg = MODEL_TABLE[m_id]
            tp_info = f"TP{min(cfg['valid_tp'])}"
            menu_items.extend([str(i), f"{name}  [{tp_info}]"])

        choice = run_dialog([
            "--clear",
            "--backtitle", f"NVIDIA P100 vLLM Launcher (GPUs: {gpu_count})",
            "--title", "Select Model",
            "--menu", "Choose a model to serve:", "20", "70", "10",
        ] + menu_items)

        if not choice:
            subprocess.run(["clear"])
            print("Selection cancelled.")
            sys.exit(0)

        configure_and_launch(int(choice), models, gpu_count)


if __name__ == "__main__":
    main()
