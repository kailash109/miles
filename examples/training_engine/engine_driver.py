"""Ray-job driver for the central-coordinator continuous MultiLoRA engine.

Allocates a Miles train-only actor group (which builds the real Megatron
MultiLoRA model), prepares each worker's plan executor, then runs the central
loop: the ``TrainingCoordinator`` builds an immutable ``TrainStepPlan``, every
worker executes the same plan, and the coordinator commits/aborts and publishes
adapter artifacts. SFT, external-rollout mode (no SGLang here).
"""

from __future__ import annotations

import asyncio
import json
import os

import ray

from miles.ray.placement_group import allocate_train_group, create_placement_groups
from miles.training_engine.client import build_job_spec
from miles.training_engine.coordinator import make_training_coordinator
from miles.training_engine.dataset_worker import build_sft_examples
from miles.training_engine.results import WorkerStepResult
from miles.training_engine.schemas import BatchingPolicy, DatasetSpec
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger


def _synthetic_records(name: str, n: int = 32) -> list[dict]:
    return [
        {"prompt": f"{name} question {i}: what is {i} + {i}?", "completion": f" The answer is {2 * i}."}
        for i in range(n)
    ]


def _submit_jobs(coordinator, args, output_root: str) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id
    dataset_spec = DatasetSpec(format="prompt_completion_jsonl")

    target_modules = (
        list(args.target_modules)
        if isinstance(args.target_modules, (list, tuple))
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    for name in ("alpha", "beta"):
        spec = build_job_spec(
            base_model=args.hf_checkpoint,
            job_id=name,
            adapter={"name": name, "rank": 8, "alpha": 16, "target_modules": target_modules},
            output_uri=f"{output_root}/{name}",
            dataset={"format": "prompt_completion_jsonl"},
            loss={"type": "sft"},
            budget={"max_steps": 4, "tokens_per_update": 256, "publish_every_steps": 1},
            scheduling={"min_tokens_per_train_quantum": 1, "min_hot_steps": 0},
        )
        ray.get(coordinator.submit_job.remote(spec))
        examples = build_sft_examples(
            job_id=name,
            records=_synthetic_records(name),
            dataset=dataset_spec,
            encode=lambda s: tokenizer.encode(s, add_special_tokens=False),
            eos_id=eos_id,
        )
        ray.get(coordinator.submit_sft_examples.remote(name, examples))
        print(f"[engine] submitted job {name} with {len(examples)} SFT examples", flush=True)


async def main(args) -> None:
    configure_logger()
    pgs = create_placement_groups(args)

    # Legacy hot-slot controller must exist for the model build's load_pending_adapters.
    from miles.ray.multi_lora_controller import create_multi_lora_controller

    legacy_controller = create_multi_lora_controller(  # noqa: F841 (kept alive)
        args.multi_lora_n_adapters, args.lora_rank
    )

    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
        role="actor",
        with_ref=False,
    )
    await actor_model.init()
    await actor_model.prepare_training_engine_workers({})

    batching = BatchingPolicy(
        max_train_tokens_per_step=4096,
        max_adapters_per_step=args.multi_lora_n_adapters,
        base_quantum_tokens=1024,
    )
    coordinator = make_training_coordinator(
        base_model=args.hf_checkpoint,
        max_hot_slots=args.multi_lora_n_adapters,
        batching=batching,
    )

    output_root = os.environ.get("ENGINE_OUTPUT_ROOT", "/root/engine_artifacts")
    _submit_jobs(coordinator, args, output_root)

    print("[engine] entering coordinator loop...", flush=True)
    for _ in range(1000):
        plan = ray.get(coordinator.build_next_plan.remote())
        if plan is None:
            if ray.get(coordinator.all_terminal_or_idle.remote()):
                break
            await asyncio.sleep(batching.idle_sleep_s)
            continue

        result_refs = actor_model.execute_train_step_plan(ray.put(plan))
        ready, not_ready = ray.wait(
            result_refs, num_returns=len(result_refs), timeout=batching.step_timeout_s
        )
        if not_ready:
            results = [WorkerStepResult(plan_id=plan.plan_id, rank=-1, ok=False, error="step_timeout")]
        else:
            results = ray.get(ready)

        commit = ray.get(coordinator.commit_or_abort_plan.remote(plan.plan_id, results))
        for event in commit.published:
            print(f"[engine] {json.dumps(event)}", flush=True)
        if not commit.ok:
            print(f"[engine] step aborted: {commit.error}", flush=True)

    summary = {
        jid: {
            "lifecycle": j.lifecycle.value,
            "trained_steps": j.trained_steps,
            "trained_tokens": j.trained_tokens,
            "latest_published_version": j.latest_published_version,
            "latest_adapter_uri": j.latest_adapter_uri,
        }
        for jid, j in ray.get(coordinator.snapshot_jobs.remote()).items()
    }
    print("[engine] done. summary:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
