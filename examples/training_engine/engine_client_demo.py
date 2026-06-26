"""Demo clients for the training-engine HTTP API.

Waits for the engine to warm up, then opens a few independent clients that each
register a LoRA job and submit a tiny SFT batch over HTTP (no Ray, no torch on
the coordinator path). Each client prints "submitting job" then "received result"
once the engine has trained its job at least one step.

Env knobs:
    ENGINE_BASE_URL     engine HTTP base url       (default http://localhost:8000)
    ENGINE_CLIENT_WAIT  seconds to wait first      (default 300)
    ENGINE_NUM_CLIENTS  how many clients to open   (default 3)
    ENGINE_BASE_MODEL   base model path            (default /root/Qwen3-4B/)
    ENGINE_DEMO_MODE    "sft" or "rl"              (default sft)

In "rl" mode the client exercises the online-RL loop: sample rollouts from the
engine, score them client-side (a trivial demo reward), then submit the scored
rollouts back as a trajectory batch. Requires the engine started with
ENGINE_ENABLE_GENERATION=1.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from miles.training_engine.client import ServiceClient
from miles.training_engine.dataset_worker import build_sft_example
from miles.training_engine.schemas import DatasetSpec

BASE_URL = os.environ.get("ENGINE_BASE_URL", "http://localhost:8000")
WAIT_SECONDS = int(os.environ.get("ENGINE_CLIENT_WAIT", "300"))
NUM_CLIENTS = int(os.environ.get("ENGINE_NUM_CLIENTS", "3"))
BASE_MODEL = os.environ.get("ENGINE_BASE_MODEL", "/root/Qwen3-4B/")
DEMO_MODE = os.environ.get("ENGINE_DEMO_MODE", "sft")
RESULT_TIMEOUT = 300.0


def _wait_until_ready(deadline: float) -> None:
    """Poll /v1/stats until the engine answers (or the deadline passes)."""
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/v1/stats", timeout=5.0)
            if r.status_code == 200:
                print(f"[client] engine reachable: {r.json()}", flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(5.0)
    print("[client] warning: engine not confirmed ready, proceeding anyway", flush=True)


def _make_example(encode, eos_id, idx: int) -> dict:
    ex = build_sft_example(
        job_id="_demo",
        record={"prompt": f"Q: what is {idx} + {idx}?", "completion": f" The answer is {idx + idx}."},
        dataset=DatasetSpec(format="prompt_completion_jsonl"),
        encode=encode,
        eos_id=eos_id,
    )
    return {
        "input_ids": ex.input_ids,
        "attention_mask": ex.attention_mask,
        "loss_mask": ex.loss_mask,
        "labels": ex.labels,
    }


def _wait_for_result(tc, idx: int) -> None:
    deadline = time.time() + RESULT_TIMEOUT
    while time.time() < deadline:
        status = tc.status()
        if status.get("trained_steps", 0) > 0:
            print(f"[client {idx}] received result: {status}", flush=True)
            return
        time.sleep(2.0)
    print(f"[client {idx}] timed out waiting for result", flush=True)


def _run_sft_client(idx: int, encode, eos_id) -> None:
    sc = ServiceClient(BASE_URL)
    print(f"[client {idx}] submitting job (sft)", flush=True)
    tc = sc.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=32,
        alpha=32,
        target_modules="all-linear",
        # Tiny demo examples (~10 loss tokens) must still be schedulable, so drop
        # the per-quantum minimum well below the default 2048.
        scheduling={"min_tokens_per_train_quantum": 1, "min_hot_steps": 0},
    )
    tc.submit_sft_examples([_make_example(encode, eos_id, idx)])
    _wait_for_result(tc, idx)


def _demo_reward(text: str, idx: int) -> float:
    """Trivial client-side reward: prefer responses that mention the answer."""
    return 1.0 if str(idx + idx) in text else 0.0


def _run_rl_client(idx: int, encode, eos_id) -> None:
    sc = ServiceClient(BASE_URL)
    print(f"[client {idx}] submitting job (rl/online)", flush=True)
    tc = sc.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=32,
        alpha=32,
        target_modules="all-linear",
        loss={"type": "grpo"},
        scheduling={"min_tokens_per_train_quantum": 1, "min_hot_steps": 0},
    )

    prompt_ids = encode(f"Q: what is {idx} + {idx}? A:")
    print(f"[client {idx}] sampling rollouts", flush=True)
    sampled = tc.sample(
        [prompt_ids],
        sampling_params={"temperature": 1.0, "max_new_tokens": 32},
        n_samples_per_prompt=4,
    )
    rollouts = sampled["rollouts"]
    rewards = [_demo_reward(r.get("text", ""), idx) for r in rollouts]
    print(f"[client {idx}] scored {len(rollouts)} rollouts, mean_reward={sum(rewards) / max(1, len(rewards)):.2f}", flush=True)

    tc.submit_scored_rollouts(rollouts, rewards, adapter_version=sampled["adapter_version"])
    _wait_for_result(tc, idx)


def _run_client(idx: int, encode, eos_id) -> None:
    if DEMO_MODE == "rl":
        _run_rl_client(idx, encode, eos_id)
    else:
        _run_sft_client(idx, encode, eos_id)


def main() -> None:
    print(f"[client] waiting {WAIT_SECONDS}s for the engine to come up ...", flush=True)
    time.sleep(WAIT_SECONDS)
    _wait_until_ready(time.time() + 120.0)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    def encode(s: str) -> list[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    with ThreadPoolExecutor(max_workers=NUM_CLIENTS) as pool:
        futures = [pool.submit(_run_client, i, encode, tokenizer.eos_token_id) for i in range(NUM_CLIENTS)]
        for f in futures:
            f.result()

    print("[client] all clients done.", flush=True)


if __name__ == "__main__":
    main()
