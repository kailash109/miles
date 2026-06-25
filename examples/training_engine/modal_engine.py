"""Run the continuous MultiLoRA training engine on Modal GPUs (real Megatron).

This mirrors ``examples/multi_lora/modal_train.py``: it uses the radixark/miles
image (so torch/Megatron-Bridge/SGLang/Ray are all present), overlays Osmosis'
Megatron-Bridge fork and this checkout, and launches a bash script that runs the
engine via ``ray job submit``.

The engine itself (``examples/training_engine/engine_driver.py``) builds the
real Megatron MultiLoRA model through Miles' train actor, then spins up the
continuous engine, submits two synthetic SFT jobs, trains their LoRA adapters,
and publishes versioned adapter artifacts.

Usage
-----
Provision the base model into the shared assets volume (reused from the
multi_lora example, so this is a no-op if you already ran that)::

    modal run examples/training_engine/modal_engine.py::provision

Run the engine (single H200; detached so it survives a disconnect)::

    modal run -d examples/training_engine/modal_engine.py::train

Artifacts are written to the ``miles-training-engine-artifacts`` volume under
``/artifacts/engine``.
"""

import os
import subprocess
from pathlib import Path

import modal

# ── Paths ──────────────────────────────────────────────────────────────────

MILES_ROOT = "/root/miles"
ASSETS_PATH = "/assets"
ARTIFACTS_PATH = "/artifacts"
MEGATRON_BRIDGE_REMOTE = "/root/Megatron-Bridge"

if modal.is_local():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    MEGATRON_BRIDGE_LOCAL = Path("~/Megatron-Bridge").expanduser()
else:
    REPO_ROOT = Path(MILES_ROOT)
    MEGATRON_BRIDGE_LOCAL = Path(MEGATRON_BRIDGE_REMOTE)

# Symlink cache-volume assets onto the /root/<name> paths the script expects.
ROOT_LINKS = {
    "Qwen3-4B": f"{ASSETS_PATH}/Qwen3-4B",
    "gsm8k": f"{ASSETS_PATH}/gsm8k",
}

DOCKER_IMAGE = "radixark/miles:dev-202606200114"

# ── Image (same overlay strategy as examples/multi_lora/modal_train.py) ──────

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

# ── Volumes ──────────────────────────────────────────────────────────────────

assets_volume = modal.Volume.from_name("miles-multilora-assets", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
artifacts_volume = modal.Volume.from_name("miles-training-engine-artifacts", create_if_missing=True)

volumes = {
    ASSETS_PATH: assets_volume,
    "/root/.cache/huggingface": hf_cache_volume,
    ARTIFACTS_PATH: artifacts_volume,
}

app = modal.App("miles-training-engine")


def _link_assets_into_root() -> None:
    for name, target in ROOT_LINKS.items():
        link = f"/root/{name}"
        if os.path.islink(link) or os.path.exists(link):
            subprocess.run(["rm", "-rf", link], check=True)
        os.symlink(target, link)


# ── Provision base model + (unused but referenced) gsm8k ──────────────────────


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


# ── Train: run the continuous engine on one H200 ──────────────────────────────


@app.function(
    image=image,
    gpu="H200:1",
    volumes=volumes,
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(script: str = "run_engine.sh"):
    """Launch the engine bash script (default ``run_engine.sh``) on one H200."""
    assets_volume.reload()
    hf_cache_volume.reload()

    script_path = Path(MILES_ROOT) / "examples" / "training_engine" / script
    if not script_path.exists():
        raise FileNotFoundError(f"bash script not found: {script_path}")

    _link_assets_into_root()

    print(f"Running {script_path} from {MILES_ROOT} on H200:1")
    subprocess.run(
        ["bash", str(script_path)],
        cwd=MILES_ROOT,
        env={**os.environ, "ENGINE_OUTPUT_ROOT": f"{ARTIFACTS_PATH}/engine"},
        check=True,
    )
    artifacts_volume.commit()
    print(f"Engine run complete. Adapter artifacts under {ARTIFACTS_PATH}/engine on the volume.")
