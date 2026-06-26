"""Oversubscription stress driver for the continuous MultiLoRA engine.

Submits *many* more LoRA jobs than can fit in GPU slots (default 50,000 vs a
handful of ``--multi-lora-n-adapters`` slots) to exercise the coordinator's slot
paging (onload / preempt) and token-fair scheduling. Only ``n_adapters`` adapters
are ever resident in VRAM; the rest live cold in the coordinator registry, so the
engine never OOMs on job count.

This is a control-plane stress test, not a convergence run: the GPU does real
Megatron steps, but the run is bounded by ``ENGINE_MAX_STEPS`` and we report how
many distinct jobs got trained, how many slot onloads/preemptions happened, and
how many adapters stayed resident.

Env knobs:
    ENGINE_NUM_JOBS         number of jobs to submit          (default 50000)
    ENGINE_MAX_STEPS        max GPU optimizer steps to run     (default 300)
    ENGINE_EXAMPLES_PER_JOB tiny SFT examples per job          (default 1)
    ENGINE_LOG_EVERY        print stats every N steps          (default 20)
    ENGINE_OUTPUT_ROOT      where v0 manifests are written      (default /root/stress_artifacts)
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import ray

from miles.ray.placement_group import allocate_train_group, create_placement_groups
from miles.training_engine.client import build_job_spec
from miles.training_engine.coordinator import make_training_coordinator
from miles.training_engine.dataset_worker import build_sft_example
from miles.training_engine.results import WorkerStepResult
from miles.training_engine.schemas import BatchingPolicy, DatasetSpec, QueueLimits, TrainExample
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger

NUM_JOBS = int(os.environ.get("ENGINE_NUM_JOBS", "50000"))
MAX_STEPS = int(os.environ.get("ENGINE_MAX_STEPS", "300"))
EXAMPLES_PER_JOB = max(1, int(os.environ.get("ENGINE_EXAMPLES_PER_JOB", "1")))
LOG_EVERY = max(1, int(os.environ.get("ENGINE_LOG_EVERY", "20")))


def _target_modules(args) -> list[str]:
    if isinstance(args.target_modules, (list, tuple)):
        return list(args.target_modules)
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _submit_jobs(coordinator, args, output_root: str) -> int:
    """Fire N (job, data) submissions. Returns per-example loss-token count."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    dataset = DatasetSpec(format="prompt_completion_jsonl")

    # Tokenize ONE synthetic example and clone it per job, so submitting 50k jobs
    # costs ~no tokenizer work (the scheduler/paging is what we're stressing).
    template = build_sft_example(
        job_id="_template",
        record={"prompt": "Q: what is 2 + 2?", "completion": " The answer is 4."},
        dataset=dataset,
        encode=lambda s: tokenizer.encode(s, add_special_tokens=False),
        eos_id=tokenizer.eos_token_id,
    )
    per_example_tokens = max(1, int(sum(template.loss_mask)))
    target_modules = _target_modules(args)

    def clone(job_id: str) -> TrainExample:
        return TrainExample(
            job_id=job_id,
            slot=None,
            loss_type="sft",
            adapter_version=None,
            input_ids=list(template.input_ids),
            attention_mask=list(template.attention_mask),
            loss_mask=list(template.loss_mask),
            labels=list(template.labels),
        )

    pending = []
    t0 = time.time()
    for i in range(NUM_JOBS):
        job_id = f"job{i:06d}"
        spec = build_job_spec(
            base_model=args.hf_checkpoint,
            job_id=job_id,
            adapter={"name": job_id, "rank": 8, "alpha": 16, "target_modules": target_modules},
            output_uri=f"{output_root}/{job_id}",
            loss={"type": "sft"},
            budget={
                "max_steps": EXAMPLES_PER_JOB,
                # ~one example per step so multi-example jobs span multiple steps
                # and force the scheduler to rotate/preempt under contention.
                "tokens_per_update": per_example_tokens,
                "publish_every_steps": 10**9,  # skip artifact export in the stress run
            },
            scheduling={"min_tokens_per_train_quantum": 1, "min_hot_steps": 0},
        )
        # Fire-and-forget: the actor processes calls FIFO, so submit_job lands
        # before its submit_sft_examples. Drain periodically to bound the mailbox.
        coordinator.submit_job.remote(spec)
        pending.append(
            coordinator.submit_sft_examples.remote(job_id, [clone(job_id) for _ in range(EXAMPLES_PER_JOB)])
        )
        if len(pending) >= 2000:
            ray.get(pending)
            pending = []
        if (i + 1) % 10000 == 0:
            print(f"[stress] submitted {i + 1}/{NUM_JOBS} jobs ({time.time() - t0:.0f}s)", flush=True)
    ray.get(pending)
    print(f"[stress] submitted all {NUM_JOBS} jobs in {time.time() - t0:.0f}s", flush=True)
    return per_example_tokens


async def main(args) -> None:
    configure_logger()
    pgs = create_placement_groups(args)

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

    n_slots = args.multi_lora_n_adapters
    output_root = os.environ.get("ENGINE_OUTPUT_ROOT", "/root/stress_artifacts")
    os.makedirs(output_root, exist_ok=True)

    # Build the coordinator with a budget sized so one step can fill every slot,
    # and queue limits sized to the stress workload (the default max_jobs=1024
    # backpressure cap would otherwise reject the 1025th job).
    coordinator = make_training_coordinator(
        base_model=args.hf_checkpoint,
        max_hot_slots=n_slots,
        batching=BatchingPolicy(
            max_train_tokens_per_step=10**12,
            max_adapters_per_step=n_slots,
            base_quantum_tokens=64,
            max_batch_wait_s=0.0,
        ),
        limits=QueueLimits(
            max_jobs=NUM_JOBS + n_slots,
            max_ready_tokens_per_job=10**9,
            max_ready_tokens_global=10**13,
        ),
    )

    per_example_tokens = _submit_jobs(coordinator, args, output_root)
    print(
        f"[stress] {NUM_JOBS} jobs submitted, {n_slots} GPU slots "
        f"({NUM_JOBS // max(1, n_slots)}x oversubscription), "
        f"{per_example_tokens} loss tokens/example. Running up to {MAX_STEPS} steps.",
        flush=True,
    )

    t0 = time.time()
    for step in range(MAX_STEPS):
        plan = ray.get(coordinator.build_next_plan.remote())
        if plan is None:
            if ray.get(coordinator.all_terminal_or_idle.remote()):
                print("[stress] all jobs drained.", flush=True)
                break
            await asyncio.sleep(0.01)
            continue

        result_refs = actor_model.execute_train_step_plan(ray.put(plan))
        ready, not_ready = ray.wait(result_refs, num_returns=len(result_refs), timeout=1800.0)
        if not_ready:
            results = [WorkerStepResult(plan_id=plan.plan_id, rank=-1, ok=False, error="step_timeout")]
        else:
            results = ray.get(ready)

        commit = ray.get(coordinator.commit_or_abort_plan.remote(plan.plan_id, results))
        if not commit.ok:
            first_line = (commit.error or "").splitlines()[0] if commit.error else ""
            print(f"[stress] step {step} aborted: {first_line}", flush=True)

        if step % LOG_EVERY == 0:
            s = ray.get(coordinator.stats.remote())
            rate = (step + 1) / max(1e-6, time.time() - t0)
            print(
                f"[stress] step={s['engine_step']} hot={s['hot']}/{s['max_hot_slots']} "
                f"trained={s['distinct_jobs_trained']} completed={s['completed']} "
                f"onloads={s['total_onloads']} preemptions={s['total_preemptions']} "
                f"({rate:.1f} steps/s)",
                flush=True,
            )

    final = ray.get(coordinator.stats.remote())
    final["wall_seconds"] = round(time.time() - t0, 1)
    final["num_jobs_submitted"] = NUM_JOBS
    print("[stress] done. final stats:", flush=True)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
