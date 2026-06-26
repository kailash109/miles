"""Minimal multi job grpo example on the training engine using a Tinker style API

Spin up three GRPO LoRA jobs concurrently, each on a different math dataset, and runs the online-RL loop until each has trained TARGET_STEPS steps

sample rollouts -> grade client-side -> submit scored rollouts -> repeat 
"""

from __future__ import annotations

import itertools
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import requests

from miles.rollout.rm_hub.math_utils import grade_answer_verl
from miles.training_engine.client import ServiceClient

BASE_URL = os.environ.get("ENGINE_BASE_URL", "http://localhost:8000")
BASE_MODEL = os.environ.get("ENGINE_BASE_MODEL", "/root/Qwen3-4B/")
TARGET_STEPS = int(os.environ.get("ENGINE_TARGET_STEPS", "50"))
N_SAMPLES = int(os.environ.get("ENGINE_N_SAMPLES", "8"))
PROMPTS_PER_ITER = int(os.environ.get("ENGINE_PROMPTS_PER_ITER", "8"))
MAX_NEW_TOKENS = int(os.environ.get("ENGINE_MAX_NEW_TOKENS", "1024"))
MAX_ROWS = int(os.environ.get("ENGINE_MAX_ROWS", "2000"))
LR = float(os.environ.get("ENGINE_LR", "1e-5"))

INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

# name -> (HF dataset, row -> (question, gold_answer))
DATASETS = {
    "gsm8k": ("zhuzilin/gsm8k", lambda r: (r["messages"][-1]["content"], r["label"])),
    "dapo-math": ("zhuzilin/dapo-math-17k", lambda r: (r["prompt"][-1]["content"], r["label"])),
    "deepscaler": ("agentica-org/DeepScaleR-Preview-Dataset", lambda r: (r["problem"], r["answer"])),
}


def reward(text: str, gold: str) -> float:
    return 1.0 if grade_answer_verl(text, str(gold)) else 0.0


def wait_until_ready(max_wait: float = 1800) -> None:
    """Poll /v1/stats until the engine (coordinator + workers + sglang) is up."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            if requests.get(f"{BASE_URL}/v1/stats", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)


def run_job(name: str, tokenizer) -> None:
    from datasets import load_dataset

    repo, extract = DATASETS[name]
    rows = itertools.cycle(load_dataset(repo, split=f"train[:{MAX_ROWS}]").shuffle(seed=0))

    # Register one LoRA RL job. (Optimizer must match across jobs in the MVP.)
    client = ServiceClient(BASE_URL, timeout=600).create_lora_training_client(
        base_model=BASE_MODEL,
        rank=32,
        alpha=32,
        name=name,
        loss={"type": "grpo"},
        optimizer={"lr": LR},
        budget={"max_steps": TARGET_STEPS},
        scheduling={"min_tokens_per_train_quantum": 256, "min_hot_steps": 0},
    )

    recent = deque(maxlen=256)
    while client.status()["trained_steps"] < TARGET_STEPS:
        for _ in range(PROMPTS_PER_ITER):
            question, gold = extract(next(rows))
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question + INSTRUCTION}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            sampled = client.sample(
                [prompt_ids],
                sampling_params={"temperature": 1.0, "max_new_tokens": MAX_NEW_TOKENS},
                n_samples_per_prompt=N_SAMPLES,
            )
            rewards = [reward(r["text"], gold) for r in sampled["rollouts"]]
            recent.extend(rewards)
            client.submit_scored_rollouts(
                sampled["rollouts"], rewards, adapter_version=sampled["adapter_version"]
            )

        s = client.status()
        print(
            f"[{name}] step {s['trained_steps']}/{TARGET_STEPS}  "
            f"last_loss={s['last_loss']}  mean_reward={sum(recent) / max(1, len(recent)):.3f}",
            flush=True,
        )

    print(f"[{name}] done -> {client.status()['latest_adapter_uri']}", flush=True)


def main() -> None:
    wait_until_ready()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    with ThreadPoolExecutor(max_workers=len(DATASETS)) as pool:
        for future in [pool.submit(run_job, name, tokenizer) for name in DATASETS]:
            future.result()


if __name__ == "__main__":
    main()
