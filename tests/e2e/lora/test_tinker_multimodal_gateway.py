"""Qwen3-VL images through the miles gateway: sampling, loss, backward, and Adam.

Requires four GPUs and MILES_MULTIMODAL_CHECKPOINT pointing to the downloaded
Qwen/Qwen3-VL-30B-A3B-Instruct checkpoint.
"""

import io
import math
import os
import shlex

from huggingface_hub import snapshot_download
from PIL import Image
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.tinker_gateway import running_gateway
from transformers import AutoProcessor

import tinker

register_cuda_ci(
    est_time=1200,
    suite="stage-c-4-gpu-h200",
    labels=["multi-lora"],
    hardware=["hopper"],
)

BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def _image_prompt(processor, color):
    image = Image.new("RGB", (128, 128), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    text = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Name the solid color in this image. Answer with one word."},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    before, after = text.split(processor.image_token)
    image_tokens = processor(text=processor.image_token, images=[image], add_special_tokens=False)["input_ids"][0]
    return tinker.ModelInput(
        chunks=[
            tinker.types.EncodedTextChunk(tokens=processor.tokenizer.encode(before, add_special_tokens=False)),
            tinker.types.ImageChunk(data=buffer.getvalue(), format="png", expected_tokens=len(image_tokens)),
            tinker.types.EncodedTextChunk(tokens=processor.tokenizer.encode(after, add_special_tokens=False)),
        ]
    )


def _datum(prompt, completion):
    return tinker.Datum(
        model_input=prompt.append(tinker.types.EncodedTextChunk(tokens=completion[:-1])),
        loss_fn_inputs={
            "target_tokens": [0] * (prompt.length - 1) + completion,
            "weights": [0.0] * (prompt.length - 1) + [1.0] * len(completion),
        },
    )


def test_qwen3_vl_tinker_images():
    checkpoint = os.environ["MILES_MULTIMODAL_CHECKPOINT"]
    processor = AutoProcessor.from_pretrained(checkpoint)
    megatron_path = os.environ.get("MILES_MEGATRON_PATH", "/root/Megatron-LM")
    serve_args = (
        f"--hf-checkpoint {shlex.quote(checkpoint)} --model-type qwen3-vl-30B-A3B "
        f"--megatron-path {shlex.quote(megatron_path)} "
        "--num-gpus-per-node 4 --actor-num-gpus 2 --rollout-num-gpus 2 "
        "--tp 2 --ep 2 --n-adapters 1 --target-modules attn "
        f"--extra-args '--tinker-base-model {BASE_MODEL} "
        "--sglang-context-length 4096 --sglang-cuda-graph-backend-decode disabled'"
    )
    with running_gateway(serve_args=serve_args) as base_url:
        client = tinker.ServiceClient(base_url=base_url, api_key="tml-miles-multimodal-validation")
        sampler = client.create_sampling_client(base_model=BASE_MODEL)
        for color in ("red", "blue"):
            prompt = _image_prompt(processor, color)
            sampled = sampler.sample(
                prompt,
                num_samples=1,
                sampling_params=tinker.SamplingParams(max_tokens=16, temperature=0.0),
            ).result()
            answer = processor.tokenizer.decode(sampled.sequences[0].tokens, skip_special_tokens=True)
            print(f"{color} image: {answer}", flush=True)
            assert color in answer.lower()

        trainer = client.create_lora_training_client(
            base_model=BASE_MODEL,
            rank=8,
            train_mlp=False,
            train_unembed=False,
        )
        image_datum = _datum(
            _image_prompt(processor, "red"), processor.tokenizer.encode("green", add_special_tokens=False)
        )
        text_datum = _datum(tinker.ModelInput.from_ints([100, 200]), [300])
        blue_datum = _datum(
            _image_prompt(processor, "blue"), processor.tokenizer.encode("green", add_special_tokens=False)
        )
        image_scores = trainer.forward([image_datum, blue_datum], loss_fn="cross_entropy").result()
        red_logprob = image_scores.loss_fn_outputs[0]["logprobs"].data[-1]
        blue_logprob = image_scores.loss_fn_outputs[1]["logprobs"].data[-1]
        assert abs(red_logprob - blue_logprob) > 1e-3, "training logits must depend on the image pixels"
        data = [text_datum, image_datum]
        before = trainer.forward(data, loss_fn="cross_entropy").result()
        backward = trainer.forward_backward(data, loss_fn="cross_entropy").result()
        for result in (before, backward):
            assert math.isfinite(result.metrics["loss:sum"])
            assert len(result.loss_fn_outputs) == len(data)
            for datum, output in zip(data, result.loss_fn_outputs, strict=True):
                assert len(output["logprobs"].data) == datum.model_input.length
                assert all(math.isfinite(value) for value in output["logprobs"].data)
        trainer.optim_step(tinker.AdamParams(learning_rate=1e-3)).result()
        after = trainer.forward(data, loss_fn="cross_entropy").result()
        print(f"loss before={before.metrics['loss:sum']} after={after.metrics['loss:sum']}", flush=True)
        assert after.metrics["loss:sum"] < before.metrics["loss:sum"]
        assert after.loss_fn_outputs[1]["loss:sum"].data[0] < before.loss_fn_outputs[1]["loss:sum"].data[0]


if __name__ == "__main__":
    if "MILES_MULTIMODAL_CHECKPOINT" not in os.environ:
        os.environ["MILES_MULTIMODAL_CHECKPOINT"] = snapshot_download(
            BASE_MODEL, revision="9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
        )
    test_qwen3_vl_tinker_images()
