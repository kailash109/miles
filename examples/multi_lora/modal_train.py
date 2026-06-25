"""Run the multi-LoRA example on a single 8xH200 Modal node.

This wraps the bash scripts in ``examples/multi_lora`` (``run.sh``,
``run_dev.sh``, ...) so they run unchanged on Modal. It provisions a cache
volume with the base model + datasets, overlays this local miles checkout into
the image, and launches the chosen bash script on one ``H200:8`` container.

It also bakes Osmosis' Megatron-Bridge fork (``~/Megatron-Bridge``) into the
image and installs it over the image's stock copy, since miles' multi-LoRA path
depends on that fork's ``megatron.bridge`` (MultiLoRA, etc.).

Usage
-----
First populate the cache volume (downloads Qwen3-4B + gsm8k + dapo-math-17k):

    modal run examples/multi_lora/modal_train.py::provision

Then run training (detached so it survives a disconnect):

    modal run -d examples/multi_lora/modal_train.py::train

Pick a different bash script (default ``run.sh``):

    modal run -d examples/multi_lora/modal_train.py::train --script run_dev.sh

Secrets (create once with ``modal secret create``):
  - ``huggingface-secret``  → HF_TOKEN          (gated/private model downloads)
  - ``wandb-secret``        → WANDB_API_KEY      (optional; enables --use-wandb)
"""

import os
import subprocess
from pathlib import Path

import modal

# ── Paths ──────────────────────────────────────────────────────────────────

MILES_ROOT = "/root/miles"
ASSETS_PATH = "/assets"  # cache volume: base model + datasets live here.
MEGATRON_BRIDGE_REMOTE = "/root/Megatron-Bridge"

# Local source dirs are only needed when building the image (i.e. locally). In
# the container the module is imported from /root/modal_train.py, where these
# `__file__`/`~` paths don't resolve, so guard them.
if modal.is_local():
    # examples/multi_lora/modal_train.py → repo root is two levels up.
    REPO_ROOT = Path(__file__).resolve().parents[2]
    # Osmosis' fork of Megatron-Bridge. miles imports `megatron.bridge`
    # (MultiLoRA, AutoBridge, ...) and must use this fork, not the stock copy.
    MEGATRON_BRIDGE_LOCAL = Path("~/Megatron-Bridge").expanduser()
else:
    REPO_ROOT = Path(MILES_ROOT)
    MEGATRON_BRIDGE_LOCAL = Path(MEGATRON_BRIDGE_REMOTE)

# The bash scripts hardcode /root/<name>; we symlink these onto the cache volume
# so the scripts run unmodified.
ROOT_LINKS = {
    "Qwen3-4B": f"{ASSETS_PATH}/Qwen3-4B",
    "gsm8k": f"{ASSETS_PATH}/gsm8k",
    "dapo-math-17k": f"{ASSETS_PATH}/dapo-math-17k",
}

# Pinned dated dev tag (radixark/miles prunes old dated tags; `latest`/`dev`
# drift). Bump to a current tag from https://hub.docker.com/r/radixark/miles/tags
# if this one is removed.
DOCKER_IMAGE = "radixark/miles:dev-202606200114"

# ── Image ────────────────────────────────────────────────────────────────────

image = (
    modal.Image.from_registry(DOCKER_IMAGE)
    .entrypoint([])
    .run_commands(
        # The HF cache dir is replaced by a volume mount below.
        "rm -rf /root/.cache/huggingface 2>/dev/null || true",
        # TE loads system cuDNN via absolute paths; the pip build has H200
        # symbol mismatches, so drop it and let the system libs win.
        "rm -rf /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/ 2>/dev/null || true",
    )
    .env({"LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"})
    # Overlay Osmosis' Megatron-Bridge fork and install it editable so that
    # `import megatron.bridge` resolves here. `megatron` is a namespace package
    # (no megatron/__init__.py), so megatron.core from /root/Megatron-LM is
    # unaffected. This must win regardless of the Ray runtime PYTHONPATH, hence
    # a real install rather than a PYTHONPATH tweak.
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
        # Remove the image's stock megatron-bridge (pip-managed or copied-in)
        # so the fork is the only megatron.bridge on the import path.
        "pip uninstall -y megatron-bridge megatron_bridge || true",
        "python3 -c \"import importlib.util as u, os, shutil; "
        "s = u.find_spec('megatron.bridge'); "
        "p = os.path.dirname(s.origin) if s and getattr(s, 'origin', None) else None; "
        "shutil.rmtree(p) if p else None; print('removed stale megatron.bridge:', p)\" || true",
        f"pip install --no-build-isolation --no-deps -e {MEGATRON_BRIDGE_REMOTE}",
        # Fail the build early if the fork is not the resolved megatron.bridge.
        "python3 -c \"import importlib.util as u; "
        "print('megatron.bridge ->', u.find_spec('megatron.bridge').origin)\"",
        # Drop the image's bundled miles checkout before overlaying ours.
        # add_local_dir merges rather than replaces, so a stale layout in the
        # image (e.g. a miles/ray/rollout/ package) would otherwise shadow this
        # checkout's miles/ray/rollout.py and break imports.
        f"rm -rf {MILES_ROOT}",
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=MILES_ROOT,
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
    )
)

# ── Volumes ───────────────────────────────────────────────────────────────────

assets_volume = modal.Volume.from_name("miles-multilora-assets", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

volumes = {
    ASSETS_PATH: assets_volume,
    "/root/.cache/huggingface": hf_cache_volume,
}

app = modal.App("miles-multi-lora")


def _link_assets_into_root() -> None:
    """Point the /root/<name> paths the bash scripts expect at the cache volume."""
    for name, target in ROOT_LINKS.items():
        link = f"/root/{name}"
        if os.path.islink(link) or os.path.exists(link):
            subprocess.run(["rm", "-rf", link], check=True)
        os.symlink(target, link)


# ── Provision: download base model + datasets into the cache volume ────────────


@app.function(
    image=image,
    volumes=volumes,
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def provision():
    """Populate the assets volume with Qwen3-4B + gsm8k + dapo-math-17k.

    Mirrors ``examples/multi_lora/provision.sh`` (minus the torch_dist
    conversion, which ``run.sh`` does not need since it uses bridge mode).
    """
    from huggingface_hub import snapshot_download

    assets_volume.reload()

    snapshot_download("Qwen/Qwen3-4B", local_dir=ROOT_LINKS["Qwen3-4B"])
    snapshot_download(
        "zhuzilin/gsm8k", repo_type="dataset", local_dir=ROOT_LINKS["gsm8k"]
    )
    snapshot_download(
        "zhuzilin/dapo-math-17k",
        repo_type="dataset",
        local_dir=ROOT_LINKS["dapo-math-17k"],
    )

    assets_volume.commit()
    print("Provisioned base model and datasets into the assets volume.")


# ── Train: run the multi-LoRA bash script on one 8xH200 node ───────────────────


@app.function(
    image=image,
    gpu="H200:8",
    volumes=volumes,
    timeout=24 * 60 * 60,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train(script: str = "run.sh"):
    """Launch a multi-LoRA bash script (default ``run.sh``) on 8xH200."""
    assets_volume.reload()
    hf_cache_volume.reload()

    script_path = Path(MILES_ROOT) / "examples" / "multi_lora" / script
    if not script_path.exists():
        raise FileNotFoundError(f"bash script not found: {script_path}")

    _link_assets_into_root()

    # wandb reads ~/.netrc; logging in here covers the Ray workers too since
    # they share this node's home dir. No-op when the secret is absent.
    if wandb_key := os.environ.get("WANDB_API_KEY"):
        subprocess.run(["wandb", "login", "--relogin", wandb_key], check=False)

    print(f"Running {script_path} from {MILES_ROOT} on 8xH200")
    subprocess.run(
        ["bash", str(script_path)],
        cwd=MILES_ROOT,
        env={**os.environ},
        check=True,
    )
