"""Real GEO3K GRPO through the Tinker SDK, including published adapter rollouts.

Uses 64 fixed training problems, 32 held-out validation problems, groups of four,
and up to 48 batches to obtain 16 updates with nonconstant rewards. Metrics and
rollouts are saved to MILES_GEO3K_OUTPUT_DIR. Accuracy improvement is measured,
not asserted: this small run validates the RL path, not statistical convergence.
"""

import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.tinker_gateway import running_gateway
from tests.e2e.lora.tinker_multimodal_gateway import BASE_MODEL, MODEL_REVISION, image_prompt, qwen3_vl_serve_args
from transformers import AutoProcessor

import tinker
from miles.rollout.rm_hub.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy

register_cuda_ci(
    est_time=4200,
    suite="stage-c-4-gpu-h200",
    labels=["multi-lora"],
    hardware=["hopper"],
    nightly=True,
)

DATASET = "hiyouga/geometry3k"
DATASET_REVISION = "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7"
SEED = 2026
GROUP_SIZE = 4
BATCH_SIZE = 4
UPDATES = 16
MAX_TOKENS = 1536


def _record(output_dir, event):
    with (output_dir / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    if event["kind"] != "rollout":
        print("GEO3K " + json.dumps(event), flush=True)


def _load_examples(split, count):
    dataset = load_dataset(DATASET, revision=DATASET_REVISION, split=split)
    indices = random.Random(SEED).sample(range(len(dataset)), count)
    return [{**dataset[index], "id": f"{split}:{index}"} for index in indices]


def _prompt(processor, example, *, blank=False):
    images = [image.convert("RGB") for image in example["images"]]
    if blank:
        images = [Image.new("RGB", image.size, "white") for image in images]
    question = example["problem"].replace("<image>", "").strip()
    return image_prompt(processor, images, question + "\nReason briefly and put your final answer in \\boxed{}.")


def _reward(answer, target):
    extracted = extract_answer(answer)
    return int(
        extracted is not None and (grade_answer_mathd(extracted, target) or grade_answer_sympy(extracted, target))
    )


def _sample(sampler, processor, examples, prompts, *, count, temperature, seed, phase, output_dir):
    futures = [
        sampler.sample(
            prompt,
            num_samples=count,
            sampling_params=tinker.SamplingParams(
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                top_p=1.0,
                top_k=-1,
                seed=seed + index * count,
            ),
        )
        for index, prompt in enumerate(prompts)
    ]
    groups = []
    for example, prompt, future in zip(examples, prompts, futures, strict=True):
        group = []
        for sequence in future.result(timeout=600).sequences:
            assert sequence.tokens and sequence.logprobs is not None
            assert len(sequence.tokens) == len(sequence.logprobs)
            assert all(math.isfinite(x) for x in sequence.logprobs)
            answer = processor.tokenizer.decode(sequence.tokens, skip_special_tokens=True)
            reward = _reward(answer, example["answer"])
            _record(
                output_dir,
                {
                    "kind": "rollout",
                    "phase": phase,
                    "id": example["id"],
                    "problem": example["problem"],
                    "target": example["answer"],
                    "answer": answer,
                    "reward": reward,
                    "stop_reason": sequence.stop_reason,
                    "tokens": sequence.tokens,
                    "logprobs": sequence.logprobs,
                },
            )
            group.append({"prompt": prompt, "sequence": sequence, "reward": reward})
        groups.append(group)
    rewards = [item["reward"] for group in groups for item in group]
    sequences = [item["sequence"] for group in groups for item in group]
    _record(
        output_dir,
        {
            "kind": "sampling",
            "phase": phase,
            "reward": float(np.mean(rewards)),
            "samples": len(rewards),
            "truncated": sum(sequence.stop_reason == "length" for sequence in sequences),
            "mean_completion_tokens": float(np.mean([len(sequence.tokens) for sequence in sequences])),
        },
    )
    return groups


def _rl_datums(groups):
    selected = []
    for group in groups:
        mean = sum(item["reward"] for item in group) / len(group)
        selected.extend((item, item["reward"] - mean) for item in group if item["reward"] != mean)
    # Tinker accumulates sums. Normalize by the number of active completion tokens.
    normalizer = sum(len(item["sequence"].tokens) for item, _ in selected)
    datums = []
    for item, advantage in selected:
        prompt, sequence = item["prompt"], item["sequence"]
        prefix = prompt.length - 1
        datums.append(
            tinker.Datum(
                model_input=prompt.append(tinker.types.EncodedTextChunk(tokens=sequence.tokens[:-1])),
                loss_fn_inputs={
                    "target_tokens": [0] * prefix + sequence.tokens,
                    "logprobs": [0.0] * prefix + sequence.logprobs,
                    "advantages": [0.0] * prefix + [advantage / normalizer] * len(sequence.tokens),
                },
            )
        )
    return datums


def _logprob_check(result, datums):
    differences = []
    assert len(result.loss_fn_outputs) == len(datums)
    for output, datum in zip(result.loss_fn_outputs, datums, strict=True):
        actual = np.asarray(output["logprobs"].data)
        expected = np.asarray(datum.loss_fn_inputs["logprobs"].data)
        active = np.asarray(datum.loss_fn_inputs["advantages"].data) != 0
        assert len(actual) == datum.model_input.length
        assert np.isfinite(actual).all()
        differences.extend((actual[active] - expected[active]).tolist())
    differences = np.asarray(differences)
    return {
        "logprob_mae": float(np.abs(differences).mean()),
        "logprob_p99_abs": float(np.quantile(np.abs(differences), 0.99)),
        "ratio_mean": float(np.exp(differences).mean()),
    }


def _publish(service, trainer, step, output_dir):
    path = trainer.save_weights_for_sampler(name=f"geo3k-{step:03d}").result(timeout=300).path
    _record(output_dir, {"kind": "checkpoint", "step": step, "path": path})
    return service.create_sampling_client(model_path=path)


def _evaluate(sampler, processor, examples, prompts, phase, output_dir):
    groups = _sample(
        sampler,
        processor,
        examples,
        prompts,
        count=1,
        temperature=0.0,
        seed=SEED,
        phase=phase,
        output_dir=output_dir,
    )
    return sum(group[0]["reward"] for group in groups) / len(groups)


def _train(service, trainer, sampler, processor, examples, prompts, output_dir):
    updates, history = 0, []
    for batch_index in range(48):
        start = (batch_index * BATCH_SIZE) % len(examples)
        group = _sample(
            sampler,
            processor,
            examples[start : start + BATCH_SIZE],
            prompts[start : start + BATCH_SIZE],
            count=GROUP_SIZE,
            temperature=1.0,
            seed=SEED + batch_index * 100,
            phase=f"train-{batch_index:03d}",
            output_dir=output_dir,
        )
        datums = _rl_datums(group)
        if not datums:
            _record(output_dir, {"kind": "skipped_constant_rewards", "batch": batch_index})
            continue
        before = trainer.forward(datums, loss_fn="importance_sampling").result(timeout=600)
        metrics = _logprob_check(before, datums)
        _record(output_dir, {"kind": "parity", "step": updates, **metrics})
        # BF16 kernels differ between Megatron and SGLang; large systematic drift
        # would invalidate on-policy RL even if forward/backward did not crash.
        assert metrics["logprob_mae"] < 0.15, metrics
        backward = trainer.forward_backward(datums, loss_fn="importance_sampling").result(timeout=600)
        assert math.isfinite(backward.metrics["loss:sum"])
        _logprob_check(backward, datums)
        optim = trainer.optim_step(tinker.AdamParams(learning_rate=1e-4)).result(timeout=600)
        assert math.isfinite(optim.metrics["grad_norm"]) and optim.metrics["grad_norm"] > 0, optim.metrics
        updates += 1
        record = {
            "kind": "update",
            "step": updates,
            "batch": batch_index,
            "datums": len(datums),
            "reward": float(np.mean([item["reward"] for samples in group for item in samples])),
            "loss": backward.metrics["loss:sum"],
            "grad_norm": optim.metrics["grad_norm"],
            **metrics,
        }
        history.append(record)
        _record(output_dir, record)
        sampler = _publish(service, trainer, updates, output_dir)
        if updates == UPDATES:
            break
    assert updates == UPDATES, f"only {updates} nonconstant-reward updates in 48 batches"
    return sampler, history


def test_qwen3_vl_tinker_geo3k():
    output_dir = Path(os.environ.get("MILES_GEO3K_OUTPUT_DIR") or tempfile.mkdtemp(prefix="miles-geo3k-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = os.environ["MILES_MULTIMODAL_CHECKPOINT"]
    processor = AutoProcessor.from_pretrained(checkpoint)
    train, validation = _load_examples("train", 64), _load_examples("validation", 32)
    train_prompts = [_prompt(processor, row) for row in train]
    eval_prompts = [_prompt(processor, row) for row in validation]
    blank_prompts = [_prompt(processor, row, blank=True) for row in validation]
    metadata = {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "model": BASE_MODEL,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "train_ids": [x["id"] for x in train],
        "validation_ids": [x["id"] for x in validation],
        "group_size": GROUP_SIZE,
        "batch_size": BATCH_SIZE,
        "max_tokens": MAX_TOKENS,
        "learning_rate": 1e-4,
    }
    _record(output_dir, {"kind": "config", **metadata})
    with running_gateway(serve_args=qwen3_vl_serve_args(checkpoint)) as base_url:
        service = tinker.ServiceClient(base_url=base_url, api_key="tml-geo3k-validation")
        trainer = service.create_lora_training_client(
            base_model=BASE_MODEL, rank=8, train_mlp=False, train_unembed=False
        )
        sampler = _publish(service, trainer, 0, output_dir)
        before = _evaluate(sampler, processor, validation, eval_prompts, "eval-before", output_dir)
        blank_before = _evaluate(sampler, processor, validation, blank_prompts, "blank-before", output_dir)
        sampler, updates = _train(service, trainer, sampler, processor, train, train_prompts, output_dir)
        after = _evaluate(sampler, processor, validation, eval_prompts, "eval-after", output_dir)
        blank_after = _evaluate(sampler, processor, validation, blank_prompts, "blank-after", output_dir)
        summary = {
            **metadata,
            "updates": updates,
            "accuracy_before": before,
            "accuracy_after": after,
            "blank_accuracy_before": blank_before,
            "blank_accuracy_after": blank_after,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        _record(
            output_dir,
            {
                "kind": "complete",
                "updates": len(updates),
                "accuracy_before": before,
                "accuracy_after": after,
                "blank_accuracy_before": blank_before,
                "blank_accuracy_after": blank_after,
            },
        )


if __name__ == "__main__":
    if "MILES_MULTIMODAL_CHECKPOINT" not in os.environ:
        os.environ["MILES_MULTIMODAL_CHECKPOINT"] = snapshot_download(BASE_MODEL, revision=MODEL_REVISION)
    test_qwen3_vl_tinker_geo3k()
