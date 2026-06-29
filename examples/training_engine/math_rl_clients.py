"""Many-adapter GRPO example on the training engine using a Tinker-style API.

Registers NUM_ADAPTERS LoRA jobs that alternate across three math datasets
(dapo-math / deepscaler / openthoughts-math) and drives the online-RL loop for each:

    sample rollouts -> grade client-side -> submit scored rollouts -> repeat

The engine only has a fixed number of hot GPU slots (ENGINE_N_ADAPTERS), so with
many more jobs than slots the coordinator pages adapters in/out. A bounded thread
pool (ENGINE_CONCURRENCY) caps how many jobs sample at once so the single sglang
engine isn't swamped; the remaining jobs run as workers free up.
"""

from __future__ import annotations

import os
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import requests

from miles.rollout.rm_hub.math_utils import grade_answer_verl
from miles.training_engine.client import ServiceClient

BASE_URL = os.environ.get("ENGINE_BASE_URL", "http://localhost:8000")
BASE_MODEL = os.environ.get("ENGINE_BASE_MODEL", "/root/Qwen3-4B/")
NUM_ADAPTERS = int(os.environ.get("ENGINE_NUM_ADAPTERS", "500"))
CONCURRENCY = int(os.environ.get("ENGINE_CONCURRENCY", "16"))
TARGET_STEPS = int(os.environ.get("ENGINE_TARGET_STEPS", "50"))
N_SAMPLES = int(os.environ.get("ENGINE_N_SAMPLES", "8"))
PROMPTS_PER_ITER = int(os.environ.get("ENGINE_PROMPTS_PER_ITER", "8"))
MAX_NEW_TOKENS = int(os.environ.get("ENGINE_MAX_NEW_TOKENS", "1024"))
# Per-job per-step token budget; with the step budget fixed, lowering this lets
# more jobs co-train per step (and vice versa).
TOKENS_PER_UPDATE = int(os.environ.get("ENGINE_TOKENS_PER_UPDATE", "8192"))
MAX_ROWS = int(os.environ.get("ENGINE_MAX_ROWS", "2000"))
LR = float(os.environ.get("ENGINE_LR", "1e-5"))
# Persist checkpoints/adapters to a real (mounted) dir instead of the default
# relative ``engine://<job_id>`` placeholder, which lands in ephemeral container
# storage. On Modal this is the ``/adapter_store`` volume.
OUTPUT_DIR = os.environ.get("ENGINE_OUTPUT_DIR", "/adapter_store")

INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

# name -> (HF dataset, row -> (question, gold_answer))
# gsm8k is intentionally excluded: it's too easy for this policy (reward ~1 from
# the start), so it gives no RL signal. These are harder, verifiable-answer math
# sets. openthoughts-math is open-r1's filtered subset of OpenThoughts-114k that
# carries a gold ``solution`` (the base OpenThoughts has no answer to grade).
DATASETS = {
    "dapo-math": ("zhuzilin/dapo-math-17k", lambda r: (r["prompt"][-1]["content"], r["label"])),
    "deepscaler": ("agentica-org/DeepScaleR-Preview-Dataset", lambda r: (r["problem"], r["answer"])),
    "openthoughts-math": ("open-r1/OpenThoughts-114k-math", lambda r: (r["problem"], r["solution"])),
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


def load_examples(dataset: str) -> list[tuple[str, str]]:
    """Load a dataset once into (question, gold) pairs, shared across its jobs."""
    from datasets import load_dataset

    repo, extract = DATASETS[dataset]
    rows = load_dataset(repo, split=f"train[:{MAX_ROWS}]")
    return [extract(r) for r in rows]


def run_job(name: str, tokenizer, examples: list[tuple[str, str]]) -> None:
    # Register one LoRA RL job. (Optimizer must match across jobs in the MVP.)
    client = ServiceClient(BASE_URL, timeout=600).create_lora_training_client(
        base_model=BASE_MODEL,
        rank=32,
        alpha=32,
        name=name,
        output_uri=f"{OUTPUT_DIR}/{name}",
        loss={"type": "grpo"},
        optimizer={"lr": LR},
        budget={"max_steps": TARGET_STEPS, "tokens_per_update": TOKENS_PER_UPDATE},
        scheduling={"min_tokens_per_train_quantum": 256, "min_hot_steps": 0},
    )

    recent = deque(maxlen=256)
    while client.status()["trained_steps"] < TARGET_STEPS:
        for _ in range(PROMPTS_PER_ITER):
            question, gold = random.choice(examples)
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

    # Load each dataset once; all of its jobs share the (read-only) examples.
    dataset_names = list(DATASETS)
    examples = {name: load_examples(name) for name in dataset_names}
    print(f"[client] loaded datasets: {{ {', '.join(f'{n}: {len(examples[n])}' for n in dataset_names)} }}", flush=True)

    # NUM_ADAPTERS jobs, round-robin across the datasets.
    jobs = [
        (f"{dataset_names[i % len(dataset_names)]}-{i:04d}", dataset_names[i % len(dataset_names)])
        for i in range(NUM_ADAPTERS)
    ]
    print(f"[client] launching {len(jobs)} adapters ({CONCURRENCY} concurrent)", flush=True)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(run_job, name, tokenizer, examples[dataset]) for name, dataset in jobs]
        for future in futures:
            future.result()

    print("[client] all adapters done.", flush=True)


if __name__ == "__main__":
    main()
