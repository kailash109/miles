"""Serve the continuous MultiLoRA training engine on a Modal GPU and demo clients.

Launches two subprocesses inside one H200 container:
  1. the engine server  (``run_engine_serve.sh`` -> ``serve_engine.py``), and
  2. a client runner     (``engine_client_demo.py``) that waits a few minutes for
     the engine to warm up, then opens a few HTTP clients that each submit a job.

Usage
-----
Provision the base model + dataset (shared cache; no-op if present)::

    modal run examples/training_engine/modal_serve_engine.py::provision

Serve + run the demo clients (single H200, detached)::

    modal run -d examples/training_engine/modal_serve_engine.py::demo

    # Tune it:
    modal run -d examples/training_engine/modal_serve_engine.py::demo \
        --wait-seconds 300 --num-clients 3 --n-adapters 16

Real-math multi-job RL demo (3 GRPO jobs on gsm8k / dapo-math / deepscaler).
Provision the datasets once, then run for ~50 steps per job (2x H200, detached)::

    modal run examples/training_engine/modal_serve_engine.py::provision_math
    modal run -d examples/training_engine/modal_serve_engine.py::demo_math_rl

    # Tune it:
    modal run -d examples/training_engine/modal_serve_engine.py::demo_math_rl \
        --target-steps 50 --n-samples 8 --prompts-per-iter 8 --max-new-tokens 512
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

# ── Image ────────────────────────────────────────────────────────────────────

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

ADAPTER_STORE_PATH = "/adapter_store"

assets_volume = modal.Volume.from_name("miles-multilora-assets", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
# Persistent store for HF-PEFT adapters so paged-out jobs are still servable.
adapter_store_volume = modal.Volume.from_name("miles-engine-adapter-store", create_if_missing=True)

volumes = {
    ASSETS_PATH: assets_volume,
    "/root/.cache/huggingface": hf_cache_volume,
    ADAPTER_STORE_PATH: adapter_store_volume,
}

app = modal.App("miles-training-engine-serve")


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


# Real math datasets for the multi-job RL demo (one per client). gsm8k is read
# locally by the engine server (--prompt-data); all three are loaded by the
# client via the HF datasets cache (the huggingface-cache volume).
MATH_RL_DATASETS = [
    "zhuzilin/gsm8k",
    "zhuzilin/dapo-math-17k",
    "agentica-org/DeepScaleR-Preview-Dataset",
]


@app.function(
    image=image,
    volumes=volumes,
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def provision_math():
    """Provision Qwen3-4B + gsm8k (engine) and warm the HF cache for the three
    math RL datasets (gsm8k, dapo-math-17k, DeepScaleR) used by demo_math_rl."""
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    assets_volume.reload()
    hf_cache_volume.reload()

    # Base model + gsm8k parquet for the engine server's --prompt-data.
    snapshot_download("Qwen/Qwen3-4B", local_dir=ROOT_LINKS["Qwen3-4B"])
    snapshot_download("zhuzilin/gsm8k", repo_type="dataset", local_dir=ROOT_LINKS["gsm8k"])
    assets_volume.commit()

    # Warm the datasets the client loads via load_dataset() into the HF cache.
    for repo in MATH_RL_DATASETS:
        print(f"Warming HF cache for {repo} ...", flush=True)
        load_dataset(repo, split="train[:2000]")
    hf_cache_volume.commit()
    print("Provisioned base model + warmed math RL datasets.")


def _run_demo(
    mode: str,
    max_wait: int,
    num_clients: int,
    n_adapters: int,
    api_port: int,
    client_script_name: str = "engine_client_demo.py",
    extra_client_env = None,
):
    """Serve the engine and run a few demo clients against it."""
    assets_volume.reload()
    hf_cache_volume.reload()
    _link_assets_into_root()

    serve_script = Path(MILES_ROOT) / "examples" / "training_engine" / "run_engine_serve.sh"
    client_script = Path(MILES_ROOT) / "examples" / "training_engine" / client_script_name
    for p in (serve_script, client_script):
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")

    # 1) Serve the coordinator/engine (background subprocess).
    enable_generation = "1" if mode == "rl" else "0"
    server_env = {
        **os.environ,
        "ENGINE_API_HOST": "0.0.0.0",
        "ENGINE_API_PORT": str(api_port),
        "ENGINE_N_ADAPTERS": str(n_adapters),
        "ENGINE_ENABLE_GENERATION": enable_generation,
    }
    if mode == "rl":
        server_env["ENGINE_ADAPTER_STORE"] = ADAPTER_STORE_PATH
    print(f"Starting engine server (mode={mode}, port={api_port}, slots={n_adapters}) ...", flush=True)
    server = subprocess.Popen(["bash", str(serve_script)], cwd=MILES_ROOT, env=server_env)

    # 2) Client runner: waits for warm-up, then opens a few clients that submit jobs.
    client_env = {
        **os.environ,
        "ENGINE_BASE_URL": f"http://localhost:{api_port}",
        "ENGINE_CLIENT_MAX_WAIT": str(max_wait),
        "ENGINE_NUM_CLIENTS": str(num_clients),
        "ENGINE_BASE_MODEL": "/root/Qwen3-4B/",
        "ENGINE_DEMO_MODE": mode,
        **(extra_client_env or {}),
    }
    try:
        print(f"Client runner: polling until ready (max {max_wait}s), then {num_clients} clients.", flush=True)
        subprocess.run(["python3", str(client_script)], cwd=MILES_ROOT, env=client_env, check=True)
    finally:
        print("Demo clients finished; stopping engine server.", flush=True)
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


@app.function(
    image=image,
    gpu="H200:1",
    volumes=volumes,
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def demo(max_wait: int = 1800, num_clients: int = 3, n_adapters: int = 16, api_port: int = 8000):
    """SFT demo on a single H200 (train-only, no sglang)."""
    _run_demo("sft", max_wait, num_clients, n_adapters, api_port)


@app.function(
    image=image,
    gpu="H200:2",
    volumes=volumes,
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def demo_rl(max_wait: int = 1800, num_clients: int = 3, n_adapters: int = 16, api_port: int = 8000):
    """Online-RL demo on 2x H200 (disaggregated: 1 trainer GPU + 1 sglang GPU)."""
    _run_demo("rl", max_wait, num_clients, n_adapters, api_port)


@app.function(
    image=image,
    gpu="H200:2",
    volumes=volumes,
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def demo_math_rl(
    max_wait: int = 1800,
    n_adapters: int = 16,
    api_port: int = 8000,
    target_steps: int = 50,
    n_samples: int = 8,
    prompts_per_iter: int = 8,
    max_new_tokens: int = 512,
):
    """Online-RL on real math datasets: 3 concurrent GRPO jobs (gsm8k, dapo-math,
    deepscaler), each trained for ``target_steps`` steps on 2x H200.

    Run ``provision_math`` once first to populate the base model + datasets.
    """
    _run_demo(
        "rl",
        max_wait,
        3,
        n_adapters,
        api_port,
        client_script_name="math_rl_clients.py",
        extra_client_env={
            "ENGINE_TARGET_STEPS": str(target_steps),
            "ENGINE_N_SAMPLES": str(n_samples),
            "ENGINE_PROMPTS_PER_ITER": str(prompts_per_iter),
            "ENGINE_MAX_NEW_TOKENS": str(max_new_tokens),
        },
    )
