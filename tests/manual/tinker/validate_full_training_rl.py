"""Run three GSM8K GRPO updates against a full-training Tinker gateway.

Start the gateway with at least a 4096-token training and sampling context, then run:
python -m tests.manual.tinker.validate_full_training_rl --base-url http://localhost:10613

Uses 8 prompts x 4 completions per step, a 1024-token generation limit, and no KL penalty.
Each step uses different questions; its reward is a batch statistic, not a fixed-set evaluation.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tinker import types
from transformers import AutoTokenizer

from miles.rollout.rm_hub.math_utils import grade_answer_verl
from tests.manual.tinker.validate_full_training import create_full_training_client


def validate(base_url: str, base_model: str, output: str) -> dict:
    client = create_full_training_client(base_url, base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dataset = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42).select(range(24))
    adam = types.AdamParams(learning_rate=1e-5, grad_clip_norm=1.0)
    records, rollouts = [], []
    for step in range(3):
        started = time.monotonic()
        sampler = client.save_weights_and_get_sampling_client(f"rl-step-{step}")
        published = time.monotonic()
        groups = []
        for index in range(step * 8, (step + 1) * 8):
            problem = dataset[index]
            messages = [
                {
                    "role": "user",
                    "content": problem["question"] + "\nExplain briefly and put your final answer in \\boxed{}.",
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
            )
            future = sampler.sample(
                types.ModelInput.from_ints(prompt),
                num_samples=4,
                sampling_params=types.SamplingParams(
                    max_tokens=1024, temperature=1.0, top_p=1.0, top_k=-1, seed=42 + index
                ),
            )
            groups.append((problem, prompt, future))
        batches, rewards, lengths, old_logprobs, offsets = [], [], [], [], []
        informative_groups = 0
        for problem, prompt, future in groups:
            sequences = future.result(timeout=600).sequences
            texts = [tokenizer.decode(seq.tokens, skip_special_tokens=True) for seq in sequences]
            answer = problem["answer"].split("####")[-1].strip()
            group_rewards = np.array([float(grade_answer_verl(text, answer)) for text in texts])
            std = float(group_rewards.std())
            informative_groups += int(std > 0)
            advantages = (group_rewards - group_rewards.mean()) / (std + 1e-8)
            for seq, text, reward, advantage in zip(sequences, texts, group_rewards, advantages, strict=True):
                assert len(seq.tokens) > 0 and len(seq.tokens) == len(seq.logprobs)
                assert np.isfinite(seq.logprobs).all()
                tokens = prompt + seq.tokens
                prefix = len(prompt) - 1
                assert len(tokens) <= 4096, "rollout exceeds validation context"
                batches.append(
                    types.Datum(
                        model_input=types.ModelInput.from_ints(tokens[:-1]),
                        loss_fn_inputs={
                            "target_tokens": types.TensorData(data=tokens[1:], dtype="int64", shape=[len(tokens) - 1]),
                            "logprobs": types.TensorData(
                                data=[0.0] * prefix + seq.logprobs, dtype="float32", shape=[len(tokens) - 1]
                            ),
                            # Average each sequence's token objective, then average the 32 sequences.
                            "advantages": types.TensorData(
                                data=[0.0] * prefix + [float(advantage) / (32 * len(seq.tokens))] * len(seq.tokens),
                                dtype="float32",
                                shape=[len(tokens) - 1],
                            ),
                        },
                    )
                )
                rewards.append(float(reward))
                lengths.append(len(seq.tokens))
                old_logprobs.extend(seq.logprobs)
                offsets.append(prefix)
                rollouts.append(
                    {
                        "step": step + 1,
                        "question": problem["question"],
                        "answer": answer,
                        "response": text,
                        "reward": float(reward),
                        "stop_reason": seq.stop_reason,
                    }
                )
        sampled = time.monotonic()
        assert informative_groups > 0, "no within-group reward variation; cannot validate an RL update"
        print(
            json.dumps(
                {
                    "phase": "sampled",
                    "step": step + 1,
                    "reward": float(np.mean(rewards)),
                    "informative_groups": informative_groups,
                    "mean_generation_tokens": float(np.mean(lengths)),
                }
            ),
            flush=True,
        )
        before = client.forward_backward(
            batches, "ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
        ).result(timeout=600)
        backward_done = time.monotonic()
        trained_logprobs = np.concatenate(
            [row["logprobs"].data[offset:] for row, offset in zip(before.loss_fn_outputs, offsets, strict=True)]
        )
        diff = trained_logprobs - np.array(old_logprobs)
        assert np.isfinite(diff).all() and abs(diff).mean() < 0.5, "sampler/trainer logprobs disagree"
        update = client.optim_step(adam).result(timeout=600)
        optimizer_done = time.monotonic()
        assert np.isfinite(update.metrics["grad_norm"]) and update.metrics["grad_norm"] > 0
        after = client.forward(batches, "ppo").result(timeout=600)
        after_logprobs = np.concatenate(
            [row["logprobs"].data[offset:] for row, offset in zip(after.loss_fn_outputs, offsets, strict=True)]
        )
        shift = after_logprobs - trained_logprobs
        assert np.isfinite(shift).all() and abs(shift).max() > 1e-6, "optimizer did not change the policy"
        record = {
            "step": step + 1,
            "reward": float(np.mean(rewards)),
            "informative_groups": informative_groups,
            "mean_generation_tokens": float(np.mean(lengths)),
            "max_generation_tokens": max(lengths),
            "truncated_fraction": float(np.mean(np.array(lengths) == 1024)),
            "publish_seconds": published - started,
            "sampling_and_reward_seconds": sampled - published,
            "forward_backward_seconds": backward_done - sampled,
            "optimizer_seconds": optimizer_done - backward_done,
            "step_seconds": time.monotonic() - started,
            "grad_norm": update.metrics["grad_norm"],
            "loss": before.metrics["loss:sum"],
            "sampler_trainer_mean_abs_logprob_diff": float(abs(diff).mean()),
            "sampler_trainer_max_abs_logprob_diff": float(abs(diff).max()),
            "initial_ratio_clip_fraction": float(np.mean((np.exp(diff) < 0.8) | (np.exp(diff) > 1.2))),
            "update_mean_abs_logprob_change": float(abs(shift).mean()),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    checkpoint = client.save_state("rl-after-three").result(timeout=600).path
    client.load_state_with_optimizer(checkpoint).result(timeout=600)
    restored = client.forward(batches, "ppo").result(timeout=600)
    restored_logprobs = np.concatenate(
        [row["logprobs"].data[offset:] for row, offset in zip(restored.loss_fn_outputs, offsets, strict=True)]
    )
    np.testing.assert_allclose(restored_logprobs, after_logprobs, rtol=0, atol=1e-4)
    final_sampler = client.save_weights_and_get_sampling_client("rl-step-3")
    final_sample = final_sampler.sample(
        types.ModelInput.from_ints(groups[-1][1]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=64, temperature=0, seed=99),
    ).result(timeout=600)
    assert len(final_sample.sequences[0].tokens) > 0
    result = {
        "model": base_model,
        "dataset": "openai/gsm8k train",
        "steps": records,
        "checkpoint_max_logprob_diff": float(abs(restored_logprobs - after_logprobs).max()),
        "final_sampler_works": True,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({**result, "rollouts": rollouts}, indent=2))
    print("RL_VALIDATION_RESULT=" + json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="rl-validation.json")
    args = parser.parse_args()
    validate(args.base_url, args.base_model, args.output)
