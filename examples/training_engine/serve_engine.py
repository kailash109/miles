"""Run the continuous MultiLoRA training engine as a service.

Starts the coordinator + Megatron training workers, runs the train-step loop in
a background thread, and serves the thin HTTP API (``miles.training_engine.api_server``).
Clients then connect over HTTP (see ``ServiceClient``) to register LoRA jobs and
submit batches — they never touch Ray.

Env knobs:
    ENGINE_API_HOST          bind host for the HTTP API           (default 0.0.0.0)
    ENGINE_API_PORT          bind port for the HTTP API           (default 8000)
    ENGINE_N_ADAPTERS        resident GPU slots                    (default 16)
    ENGINE_ENABLE_GENERATION bring up sglang + enable /sample      (default 0)
    ENGINE_SYNC_EVERY        resync adapters -> sglang every N steps (default 1)

Online RL (generation) requires the run to be configured with a colocated
sglang rollout (NOT ``--debug-train-only``) and a fixed ``--sglang-router-port``
so the generator can reach the router. See ``run_engine_serve.sh``.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

import ray

from miles.ray.placement_group import allocate_train_group, create_placement_groups
from miles.training_engine.coordinator import make_training_coordinator
from miles.training_engine.results import WorkerStepResult
from miles.training_engine.schemas import BatchingPolicy, QueueLimits
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger

API_HOST = os.environ.get("ENGINE_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("ENGINE_API_PORT", "8000"))
ENABLE_GENERATION = os.environ.get("ENGINE_ENABLE_GENERATION", "0") == "1"
SYNC_EVERY = max(1, int(os.environ.get("ENGINE_SYNC_EVERY", "1")))


async def _setup(args):
    """Bring up coordinator + workers (+ optional sglang). Returns (actor_model, coordinator, generator)."""
    configure_logger()

    if ENABLE_GENERATION and args.colocate:
        # If someone runs generation colocated (single GPU), force both offloads
        # off so the trainer and sglang stay resident (no offload dance). The
        # recommended path is DISAGG (separate GPUs), where offload already
        # defaults off and this branch is a no-op.
        if args.offload_train or args.offload_rollout:
            print("[engine] colocated generation: forcing offload off (both resident)", flush=True)
        args.offload_train = False
        args.offload_rollout = False

    pgs = create_placement_groups(args)

    from miles.ray.multi_lora_controller import create_multi_lora_controller

    _legacy_controller = create_multi_lora_controller(args.multi_lora_n_adapters, args.lora_rank)  # noqa: F841

    # Online-RL generation: bring up the sglang rollout engines first so the
    # train workers connect to them during update_weights (mirrors train_multi_lora).
    rollout_manager = None
    if ENABLE_GENERATION:
        from miles.ray.placement_group import create_rollout_manager

        rollout_manager, _ = create_rollout_manager(args, pgs["rollout"])

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

    generator = None
    if ENABLE_GENERATION:
        from miles.training_engine.generation import SglangGenerator

        from miles.utils.http_utils import get_host_info

        await actor_model.set_rollout_manager(rollout_manager)
        # Push base + adapter weights into sglang so /sample uses current policy.
        await actor_model.update_weights()
        router_port = getattr(args, "sglang_router_port", None)
        if router_port is None:
            raise ValueError("generation requires a fixed --sglang-router-port so the generator can reach the router")
        # Router binds to the node IP (matches _start_router's get_host_info()[1]),
        # not 127.0.0.1, so use the same here.
        router_ip = get_host_info()[1]
        generator = SglangGenerator(f"http://{router_ip}:{router_port}")
        print(f"[engine] generation enabled; router=http://{router_ip}:{router_port}", flush=True)

    n_slots = args.multi_lora_n_adapters
    coordinator = make_training_coordinator(
        base_model=args.hf_checkpoint,
        max_hot_slots=n_slots,
        batching=BatchingPolicy(max_adapters_per_step=n_slots),
        limits=QueueLimits(),
    )
    return actor_model, coordinator, generator


# Serializes every update_weights() call (background loop + on-demand /sync_weights)
# so two NCCL weight syncs never overlap on the train actors.
_WEIGHT_SYNC_LOCK = threading.Lock()


def _sync_adapters_to_sglang(actor_model, coordinator, loaded: set) -> None:
    """Reconcile the multi_lora_controller with the coordinator's HOT adapters,
    then push to sglang via update_weights.

    Bridges the coordinator (which owns slot assignment) to the
    multi_lora_controller (which update_weights reads): register every HOT job's
    adapter ACTIVE at its model slot, drain adapters that left HOT so they get
    unloaded from sglang, then sync. ``loaded`` is the live set the /sample
    handler reads to decide whether to route to an adapter; updated incrementally
    (add/discard are atomic) so the API never sees an empty window.
    """
    from miles.ray.multi_lora_controller import get_multi_lora_controller

    controller = get_multi_lora_controller()
    hot = ray.get(coordinator.hot_adapters.remote())
    hot_by_name = {h["name"]: h for h in hot}

    with _WEIGHT_SYNC_LOCK:
        # Register/refresh resident adapters as ACTIVE at their coordinator slot.
        for h in hot:
            ray.get(controller.set_engine_adapter.remote(h["name"], h["rank"], h["alpha"], h["slot"]))

        # Adapters we synced before that are no longer hot -> drain (unload from sglang).
        departed = [name for name in loaded if name not in hot_by_name]
        for name in departed:
            ray.get(controller.drain_engine_adapter.remote(name))
            loaded.discard(name)  # stop routing /sample to it before the unload lands

        # Push ACTIVE adapters into sglang and unload DRAINED ones.
        asyncio.run(actor_model.update_weights())

        # Forget drained adapters now that sglang has unloaded them; mark hot ones loaded.
        for name in departed:
            ray.get(controller.remove_engine_adapter.remote(name))
        for name in hot_by_name:
            loaded.add(name)


def _sync_one_job(actor_model, coordinator, loaded: set, job_id: str) -> int | None:
    """Synchronously force one job's current adapter into sglang. Returns the
    published ``adapter_version`` once loaded, or None if the job isn't resident
    (not yet trained, or paged out). This is the ``save_weights`` primitive: after
    it returns, ``/sample`` for this job routes to its trained adapter.
    """
    from miles.ray.multi_lora_controller import get_multi_lora_controller

    info = ray.get(coordinator.hot_adapter.remote(job_id))
    if info is None:
        return None
    controller = get_multi_lora_controller()
    with _WEIGHT_SYNC_LOCK:
        ray.get(controller.set_engine_adapter.remote(info["name"], info["rank"], info["alpha"], info["slot"]))
        asyncio.run(actor_model.update_weights())
        loaded.add(info["name"])
    print(f"[engine] synced adapter {info['name']} (v{info['version']}) into sglang on demand", flush=True)
    return int(info["version"])


def _run_training_loop(
    actor_model, coordinator, stop: threading.Event, *, sync_weights: bool, loaded_adapters: set
) -> None:
    """Continuously drain the coordinator: build plan -> execute -> commit.

    When ``sync_weights`` (generation enabled), pushes freshly-trained adapters
    into sglang after committed steps so /sample uses the current policy.
    """
    steps_since_sync = 0
    while not stop.is_set():
        plan = ray.get(coordinator.build_next_plan.remote())
        if plan is None:
            time.sleep(0.01)
            continue

        print(f"[coordinator] running jobs: {list(plan.selected_jobs)}", flush=True)
        result_refs = actor_model.execute_train_step_plan(ray.put(plan))
        ready, not_ready = ray.wait(result_refs, num_returns=len(result_refs), timeout=1800.0)
        if not_ready:
            results = [WorkerStepResult(plan_id=plan.plan_id, rank=-1, ok=False, error="step_timeout")]
        else:
            results = ray.get(ready)

        commit = ray.get(coordinator.commit_or_abort_plan.remote(plan.plan_id, results))
        if not commit.ok:
            first_line = (commit.error or "").splitlines()[0] if commit.error else ""
            print(f"[engine] step aborted: {first_line}", flush=True)
            continue

        if sync_weights:
            steps_since_sync += 1
            if steps_since_sync >= SYNC_EVERY:
                _sync_adapters_to_sglang(actor_model, coordinator, loaded_adapters)
                steps_since_sync = 0


def main() -> None:
    args = parse_args()
    actor_model, coordinator, generator = asyncio.run(_setup(args))

    # Adapter names currently loaded in sglang; shared between the train loop
    # (writer) and the /sample handler (reader) so sampling routes to an adapter
    # only once it's actually loaded.
    loaded_adapters: set = set()

    stop = threading.Event()
    loop_thread = threading.Thread(
        target=_run_training_loop,
        args=(actor_model, coordinator, stop),
        kwargs={"sync_weights": generator is not None, "loaded_adapters": loaded_adapters},
        daemon=True,
    )
    loop_thread.start()

    from miles.training_engine.api_server import serve

    # On-demand "save weights": force a specific job's adapter into sglang.
    sync_weights_fn = None
    if generator is not None:
        sync_weights_fn = lambda job_id: _sync_one_job(actor_model, coordinator, loaded_adapters, job_id)  # noqa: E731

    print(f"[engine] serving training API on http://{API_HOST}:{API_PORT} (base_model={args.hf_checkpoint})", flush=True)
    print(
        f"[engine] coordinator ready for jobs (base_model={args.hf_checkpoint}, "
        f"slots={args.multi_lora_n_adapters}, generation={'on' if generator is not None else 'off'})",
        flush=True,
    )
    try:
        serve(coordinator, generator, loaded_adapters, sync_weights_fn=sync_weights_fn, host=API_HOST, port=API_PORT)
    finally:
        stop.set()


if __name__ == "__main__":
    main()
