"""Shared Qwen3-VL launch args and official SDK image prompt construction."""

import io
import os
import shlex

import tinker

BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"


def image_prompt(processor, images, text):
    rendered = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [*({"type": "image", "image": image} for image in images), {"type": "text", "text": text}],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    parts = rendered.split(processor.image_token)
    assert len(parts) == len(images) + 1
    chunks = []
    for prefix, image in zip(parts[:-1], images, strict=True):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_ids = processor(text=processor.image_token, images=[image], add_special_tokens=False)["input_ids"][0]
        chunks.extend(
            [
                tinker.types.EncodedTextChunk(tokens=processor.tokenizer.encode(prefix, add_special_tokens=False)),
                tinker.types.ImageChunk(data=buffer.getvalue(), format="png", expected_tokens=len(image_ids)),
            ]
        )
    chunks.append(
        tinker.types.EncodedTextChunk(tokens=processor.tokenizer.encode(parts[-1], add_special_tokens=False))
    )
    return tinker.ModelInput(chunks=chunks)


def qwen3_vl_serve_args(checkpoint):
    megatron_path = os.environ.get("MILES_MEGATRON_PATH", "/root/Megatron-LM")
    return (
        f"--hf-checkpoint {shlex.quote(checkpoint)} --model-type qwen3-vl-30B-A3B "
        f"--megatron-path {shlex.quote(megatron_path)} "
        "--num-gpus-per-node 4 --actor-num-gpus 2 --rollout-num-gpus 2 "
        "--tp 2 --ep 2 --n-adapters 1 --target-modules attn "
        f"--extra-args '--tinker-base-model {BASE_MODEL} "
        "--sglang-context-length 4096 --sglang-cuda-graph-backend-decode disabled'"
    )
