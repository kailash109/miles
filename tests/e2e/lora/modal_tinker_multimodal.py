"""Run Qwen3-VL Tinker validation on four temporary Modal H100s.

From the miles checkout, with Modal credentials configured:
    modal run tests/e2e/lora/modal_tinker_multimodal.py

The checkpoint is cached in a Modal Volume. GPUs stop when the test finishes.
"""

import os
import subprocess
from pathlib import Path

import modal

app = modal.App("miles-tinker-multimodal-validation")
model_cache = modal.Volume.from_name("miles-tinker-multimodal-models", create_if_missing=True)
image = (
    modal.Image.from_registry("radixark/miles@sha256:360616b7678698a8c6b358b57e80c18b8bd3758da3d4753adb20f95432e8a511")
    .entrypoint([])
    .run_commands(
        "git clone --filter=blob:none https://github.com/radixark/Megatron-LM.git /opt/miles-megatron"
        " && git -C /opt/miles-megatron checkout a84b105473eca9c9c6e49dbf1c8aa460c5cf796e"
        " && pip install --no-deps --no-build-isolation -e /opt/miles-megatron",
        "pip install --no-deps --no-build-isolation git+https://github.com/radixark/Megatron-Bridge.git@8cd3466d14d2337c8492827b3712482c2b3e4866",
    )
    .pip_install("tinker==0.26.2", "fastapi", "uvicorn", "transformers==5.12.1")
    .env({"PYTHONPATH": "/workspace/miles:/opt/miles-megatron", "CUDA_DEVICE_MAX_CONNECTIONS": "1"})
    .add_local_dir(
        Path(__file__).resolve().parents[3], "/workspace/miles", ignore=[".git", "__pycache__", ".pytest_cache"]
    )
)


@app.function(image=image, volumes={"/models": model_cache}, timeout=3600, gpu="H100:4", cpu=32, memory=196608)
def validate():
    from huggingface_hub import snapshot_download

    os.chdir("/workspace/miles")
    model_path = "/models/Qwen3-VL-30B-A3B-Instruct"
    snapshot_download("Qwen/Qwen3-VL-30B-A3B-Instruct", local_dir=model_path)
    model_cache.commit()
    env = dict(os.environ, MILES_MULTIMODAL_CHECKPOINT=model_path, MILES_MEGATRON_PATH="/opt/miles-megatron")
    subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "tests/e2e/lora/test_tinker_multimodal_gateway.py",
            "-s",
            "-x",
            "--confcutdir=tests/e2e/lora",
            "-o",
            "addopts=",
        ],
        env=env,
        check=True,
    )


@app.local_entrypoint()
def main():
    validate.remote()
