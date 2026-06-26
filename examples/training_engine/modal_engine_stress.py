"""Oversubscription stress test of the continuous MultiLoRA engine on Modal GPUs.

Same image/overlay/volume strategy as ``modal_engine.py``, but launches the
stress driver, which submits *far* more LoRA jobs (default 50,000) than there are
GPU slots and lets the central coordinator page them through the resident slots.
Only ``--multi-lora-n-adapters`` adapters are ever in VRAM, so the engine handles
a huge job queue without OOM.

Usage
-----
Provision the base model (shared with the multi_lora example; no-op if present)::

    modal run examples/training_engine/modal_engine_stress.py::provision

Run the stress test (single H200, detached)::

    modal run -d examples/training_engine/modal_engine_stress.py::train

    # Tune it:
    modal run -d examples/training_engine/modal_engine_stress.py::train \
        --num-jobs 50000 --max-steps 400 --n-adapters 16 --examples-per-job 1
"""

import os
import subprocess
from pathlib import Path

import modal

# ── Paths ──────────────────────────────────────────────────────────────────

MILES_ROOT = "/root/miles"
ASSETS_PATH = "/assets"
MEGATRON_BRIDGE_REMOTE = "/root/Megatron-Bridge"

if modal.is_local():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    MEGATRON_BRIDGE_LOCAL = Path("~/Megatron-Bridge").expanduser()
else:
    REPO_ROOT = Path(MILES_ROOT)
    MEGATRON_BRIDGE_LOCAL = Path(MEGATRON_BRIDGE_REMOTE)

ROOT_LINKS = {
    "Qwen3-4B": f"{ASSETS_PATH}/Qwen3-4B",
    "gsm8k": f"{ASSETS_PATH}/gsm8k",
}

DOCKER_IMAGE = "radixark/miles:dev-202606200114"

# ── Image (identical overlay strategy to modal_engine.py) ────────────────────

image = (
    modal.Image.from_registry(DOCKER_IMAGE)
    .entrypoint([])
    .run_commands(
        "rm -rf /root/.cache/huggingface 2>/dev/null || true",
        "rm -rf /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/ 2>/dev/null || true",
    )
    .env({"LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"})
    .add_local_dir(
        str(MEGATRON_BRIDGE_LOCAL),
        remote_path=MEGATRON_BRIDGE_REMOTE,
        copy=True,
        ignore=[
            "**/.git",
            "**/__pycache__",
            "**/*.pyc",
            "3rdparty/**",
            "docs/**",
            "tutorials/**",
            "tests/**",
            "examples/**",
            "*.png",
            "uv.lock",
        ],
    )
    .run_commands(
        "pip uninstall -y megatron-bridge megatron_bridge || true",
        "python3 -c \"import importlib.util as u, os, shutil; "
        "s = u.find_spec('megatron.bridge'); "
        "p = os.path.dirname(s.origin) if s and getattr(s, 'origin', None) else None; "
        "shutil.rmtree(p) if p else None; print('removed stale megatron.bridge:', p)\" || true",
        f"pip install --no-build-isolation --no-deps -e {MEGATRON_BRIDGE_REMOTE}",
        "python3 -c \"import importlib.util as u; "
        "print('megatron.bridge ->', u.find_spec('megatron.bridge').origin)\"",
        f"rm -rf {MILES_ROOT}",
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=MILES_ROOT,
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
    )
)

# ── Volumes (reuse the multi_lora assets cache) ──────────────────────────────

assets_volume = modal.Volume.from_name("miles-multilora-assets", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

volumes = {
    ASSETS_PATH: assets_volume,
    "/root/.cache/huggingface": hf_cache_volume,
}

app = modal.App("miles-training-engine-stress")


def _link_assets_into_root() -> None:
    for name, target in ROOT_LINKS.items():
        link = f"/root/{name}"
        if os.path.islink(link) or os.path.exists(link):
            subprocess.run(["rm", "-rf", link], check=True)
        os.symlink(target, link)


@app.function(
    image=image,
    volumes=volumes,
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def provision():
    """Populate the assets volume with Qwen3-4B + gsm8k (shared with multi_lora)."""
    from huggingface_hub import snapshot_download

    assets_volume.reload()
    snapshot_download("Qwen/Qwen3-4B", local_dir=ROOT_LINKS["Qwen3-4B"])
    snapshot_download("zhuzilin/gsm8k", repo_type="dataset", local_dir=ROOT_LINKS["gsm8k"])
    assets_volume.commit()
    print("Provisioned Qwen3-4B + gsm8k into the assets volume.")


@app.function(
    image=image,
    gpu="H200:1",
    volumes=volumes,
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(
    num_jobs: int = 50000,
    max_steps: int = 300,
    n_adapters: int = 16,
    examples_per_job: int = 1,
    log_every: int = 20,
):
    """Launch the oversubscription stress run on one H200.

    The v0 manifests for 50k jobs are written to fast local disk (not a volume).
    """
    assets_volume.reload()
    hf_cache_volume.reload()

    script_path = Path(MILES_ROOT) / "examples" / "training_engine" / "run_engine_stress.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"bash script not found: {script_path}")

    _link_assets_into_root()

    env = {
        **os.environ,
        "ENGINE_OUTPUT_ROOT": "/root/stress_artifacts",
        "ENGINE_NUM_JOBS": str(num_jobs),
        "ENGINE_MAX_STEPS": str(max_steps),
        "ENGINE_N_ADAPTERS": str(n_adapters),
        "ENGINE_EXAMPLES_PER_JOB": str(examples_per_job),
        "ENGINE_LOG_EVERY": str(log_every),
    }
    print(
        f"Stress run: {num_jobs} jobs -> {n_adapters} slots "
        f"({num_jobs // max(1, n_adapters)}x oversubscription), max_steps={max_steps}",
        flush=True,
    )
    subprocess.run(["bash", str(script_path)], cwd=MILES_ROOT, env=env, check=True)
    print("Stress run complete.")
