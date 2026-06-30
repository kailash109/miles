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
from miles.utils import tracking_utils
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger

API_HOST = os.environ.get("ENGINE_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("ENGINE_API_PORT", "8000"))
ENABLE_GENERATION = os.environ.get("ENGINE_ENABLE_GENERATION", "0") == "1"
SYNC_EVERY = max(1, int(os.environ.get("ENGINE_SYNC_EVERY", "1")))
# How long the coordinator waits for more jobs/tokens to accumulate before
# dispatching a step, so multiple adapters co-pack into one step instead of
# training one job at a time. Higher = wider batches, more per-step latency.
MAX_BATCH_WAIT_S = float(os.environ.get("ENGINE_MAX_BATCH_WAIT_S", "2.0"))
# How often the training loop logs its plan-build cadence (attempts/s).
_SCHED_TICK_REPORT_S = float(os.environ.get("ENGINE_SCHED_TICK_REPORT_S", "2.0"))
# Persistent HF-PEFT adapter store (disk-load into sglang for paged-out jobs).
ADAPTER_STORE = os.environ.get("ENGINE_ADAPTER_STORE") or None


async def _setup(args):
    """Bring up coordinator + workers (+ optional sglang).

    Returns (actor_model, coordinator, generator, rollout_manager).
    """
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
        # Bound concurrent /generate requests the same way the rollout path sizes
        # its semaphore (server concurrency across all rollout engines). Env
        # ENGINE_GEN_CONCURRENCY overrides; falls back to the generator default.
        gen_concurrency = None
        _env_gen_conc = os.environ.get("ENGINE_GEN_CONCURRENCY")
        if _env_gen_conc:
            gen_concurrency = int(_env_gen_conc)
        elif getattr(args, "sglang_server_concurrency", 0):
            gen_concurrency = (
                args.sglang_server_concurrency
                * max(1, args.rollout_num_gpus)
                // max(1, args.rollout_num_gpus_per_engine)
            )
        generator = SglangGenerator(f"http://{router_ip}:{router_port}", max_concurrency=gen_concurrency)
        print(
            f"[engine] generation enabled; router=http://{router_ip}:{router_port} "
            f"gen_concurrency={generator.max_concurrency}",
            flush=True,
        )

    n_slots = args.multi_lora_n_adapters
    if ADAPTER_STORE:
        os.makedirs(ADAPTER_STORE, exist_ok=True)
    # Per-step training token budget. The greedy scheduler packs jobs until this
    # is exhausted, so it (not slot count) caps how many runnable adapters batch
    # into one step. The budget is sharded across the data-parallel trainers, so
    # the default scales with BOTH slots and DP size -- keeping the per-GPU load
    # constant as trainers are added, so more trainers => more adapters/step
    # rather than the same count spread thinner. Override with
    # ENGINE_MAX_TRAIN_TOKENS_PER_STEP.
    tp = max(1, int(getattr(args, "tensor_model_parallel_size", 1) or 1))
    pp = max(1, int(getattr(args, "pipeline_model_parallel_size", 1) or 1))
    cp = max(1, int(getattr(args, "context_parallel_size", 1) or 1))
    world = max(1, int(args.actor_num_nodes) * int(args.actor_num_gpus_per_node))
    dp_size = max(1, world // (tp * pp * cp))
    _max_tokens_env = os.environ.get("ENGINE_MAX_TRAIN_TOKENS_PER_STEP", "")
    max_train_tokens_per_step = int(_max_tokens_env) if _max_tokens_env else n_slots * 131072 * dp_size
    # "deficit" (default) packs few jobs with a large per-job target; "waterfill"
    # admits as many jobs as the budget seats with a small base grant each, then
    # fairly water-fills the rest -- better slot saturation with many small jobs.
    scheduler = os.environ.get("ENGINE_SCHEDULER", "deficit")
    coordinator = make_training_coordinator(
        base_model=args.hf_checkpoint,
        max_hot_slots=n_slots,
        batching=BatchingPolicy(
            max_adapters_per_step=n_slots,
            max_batch_wait_s=MAX_BATCH_WAIT_S,
            max_train_tokens_per_step=max_train_tokens_per_step,
            scheduler=scheduler,
        ),
        limits=QueueLimits(),
        adapter_store=ADAPTER_STORE,
    )
    print(
        f"[engine] batching: slots={n_slots} dp_size={dp_size} "
        f"max_train_tokens_per_step={max_train_tokens_per_step} "
        f"max_batch_wait_s={MAX_BATCH_WAIT_S} scheduler={scheduler}",
        flush=True,
    )
    return actor_model, coordinator, generator, rollout_manager, _legacy_controller


# Serializes weight syncs against each other AND against in-flight /sample
# generations: the sync takes the write side (pausing + flushing sglang safely),
# /sample takes the read side. Without this, a sync that pauses sglang while a
# generation is in flight deadlocks on flush_cache.
from miles.training_engine.api_server import ReadWriteLock

_WEIGHT_SYNC_LOCK = ReadWriteLock()


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

    with _WEIGHT_SYNC_LOCK.write_lock():
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


def _sync_one_job(actor_model, coordinator, rollout_manager, loaded: set, job_id: str) -> int | None:
    """Synchronously force one job's current adapter into sglang (``save_weights``).

    HOT job  -> GPU tensor push (freshest weights from the model slot).
    COLD job -> flush the inference write + load the persisted HF-PEFT from disk.
    Returns the adapter_version once loaded, or None if it can't be served (e.g.
    never trained / never persisted).
    """
    from miles.ray.multi_lora_controller import get_multi_lora_controller

    info = ray.get(coordinator.hot_adapter.remote(job_id))
    if info is not None:
        controller = get_multi_lora_controller()
        with _WEIGHT_SYNC_LOCK.write_lock():
            ray.get(controller.set_engine_adapter.remote(info["name"], info["rank"], info["alpha"], info["slot"]))
            asyncio.run(actor_model.update_weights())
            loaded.add(info["name"])
        print(f"[engine] synced HOT adapter {info['name']} (v{info['version']}) into sglang", flush=True)
        return int(info["version"])

    # COLD: serve from the persistent adapter store (disk-load).
    if not ADAPTER_STORE or rollout_manager is None:
        return None
    try:
        job = ray.get(coordinator.get_job.remote(job_id))
    except KeyError:
        return None
    name = job.spec.adapter.name
    path = f"{ADAPTER_STORE.rstrip('/')}/{job_id}"
    # Ensure the eviction write for this job has flushed before we read it.
    asyncio.run(actor_model.wait_adapter_persisted(job_id))
    if not os.path.exists(os.path.join(path, "adapter_model.safetensors")):
        return None  # never persisted (e.g. never evicted while dirty)
    with _WEIGHT_SYNC_LOCK.write_lock():
        ray.get(rollout_manager.load_lora_adapter_on_engines.remote(name, path))
        loaded.add(name)
    print(f"[engine] disk-loaded COLD adapter {name} (v{job.latest_published_version}) from {path}", flush=True)
    return int(job.latest_published_version)


def _init_engine_wandb(args) -> None:
    """Init the driver-side W&B run and bind per-adapter metrics to ``engine/step``.

    The engine has a single metric writer (this driver loop), so each adapter's
    curves live under their own ``{job_id}/...`` section in W&B, all on a shared
    engine-step x-axis. Per-job binding happens lazily as jobs first appear.
    """
    tracking_utils.init_tracking(args, primary=True)
    if args.use_wandb:
        import wandb

        wandb.define_metric("engine/step")
        wandb.define_metric("train/*", step_metric="engine/step")


_SAMPLE_COLUMNS = ["engine_step", "adapter", "reward", "response_len", "prompt", "response"]
_MAX_SAMPLE_ROWS = int(os.environ.get("ENGINE_LOG_SAMPLES_MAX_ROWS", "200"))


def _log_step_metrics(
    args, plan, results, defined_jobs: set, sample_rows: list, time_between_steps: float | None = None
) -> None:
    """Forward this step's aggregate + per-adapter metrics to W&B/tracking.

    Per-adapter loss/reward/tokens/response-length arrive in
    ``WorkerStepResult.metrics`` keyed ``{job_id}/train/*`` (DP-reduced on the
    last pipeline stage); the aggregate ``loss``/``grad_norm`` are remapped under
    the ``train/`` section. Decoded sample rollouts (rank 0, env-gated) ride in
    ``_rollout_samples`` and are rendered as a rolling ``rollouts/samples`` table.

    ``time_between_steps`` is the wall-clock gap (s) since the previous committed
    step on the coordinator side; ``None`` for the first step.
    """
    merged: dict = {}
    for r in results:
        if r.ok and r.metrics:
            merged.update(r.metrics)
    if not merged and time_between_steps is None:
        return

    samples = merged.pop("_rollout_samples", None)

    log_dict: dict = {}
    for key, val in merged.items():
        if key in ("update_successful", "skipped"):
            continue
        if key == "loss":
            log_dict["train/loss"] = val
        elif key == "grad_norm":
            log_dict["train/grad_norm"] = val
        else:
            log_dict[key] = val  # already namespaced per adapter, e.g. "{job}/train/loss"

    if args.use_wandb:
        import wandb

        for job_id in plan.selected_jobs:
            if job_id not in defined_jobs:
                wandb.define_metric(f"{job_id}/*", step_metric="engine/step")
                defined_jobs.add(job_id)

    log_dict["train/num_jobs"] = len(plan.selected_jobs)
    log_dict["train/num_runnable_jobs"] = plan.num_runnable_jobs
    if time_between_steps is not None:
        log_dict["train/time_between_steps_s"] = time_between_steps
    log_dict["engine/step"] = plan.engine_step
    tracking_utils.log(args, log_dict, step_key="engine/step")

    if samples and args.use_wandb:
        import wandb

        sample_rows.extend(samples)
        del sample_rows[:-_MAX_SAMPLE_ROWS]  # keep only the most recent rows
        table = wandb.Table(columns=_SAMPLE_COLUMNS)
        for row in sample_rows:
            table.add_data(*(row.get(col) for col in _SAMPLE_COLUMNS))
        wandb.log({"rollouts/samples": table})


def _run_training_loop(
    actor_model, coordinator, stop: threading.Event, *, args, sync_weights: bool, loaded_adapters: set
) -> None:
    """Continuously drain the coordinator: build plan -> execute -> commit.

    When ``sync_weights`` (generation enabled), pushes freshly-trained adapters
    into sglang after committed steps so /sample uses the current policy.
    """
    steps_since_sync = 0
    defined_jobs: set = set()
    sample_rows: list = []
    # Scheduler-cadence instrumentation: the loop should TRY to build a plan as
    # fast as possible. We count attempts/dispatches and report the rate every
    # _SCHED_TICK_REPORT_S so the per-attempt rate doesn't flood the logs (and
    # printing itself doesn't slow the hot loop).
    attempts = dispatched = 0
    last_report = time.time()
    # Wall-clock timestamp of the previous committed step, for "time between
    # steps" (the actual coordinator-side cadence of executed training steps).
    last_step_time: float | None = None
    while not stop.is_set():
        plan = ray.get(coordinator.build_next_plan.remote())
        attempts += 1
        if plan is not None:
            dispatched += 1
        now = time.time()
        if now - last_report >= _SCHED_TICK_REPORT_S:
            elapsed = now - last_report
            print(
                f"[engine] scheduler cadence: {attempts / elapsed:.0f} build attempts/s "
                f"over {elapsed:.1f}s ({attempts} attempts, {dispatched} dispatched, "
                f"{attempts - dispatched} idle)",
                flush=True,
            )
            attempts = dispatched = 0
            last_report = now
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

        step_time = time.time()
        time_between_steps = step_time - last_step_time if last_step_time is not None else None
        last_step_time = step_time

        _log_step_metrics(args, plan, results, defined_jobs, sample_rows, time_between_steps)

        if sync_weights:
            steps_since_sync += 1
            if steps_since_sync >= SYNC_EVERY:
                _sync_adapters_to_sglang(actor_model, coordinator, loaded_adapters)
                steps_since_sync = 0


def main() -> None:
    args = parse_args()
    actor_model, coordinator, generator, rollout_manager, controller = asyncio.run(_setup(args))

    # Per-adapter + aggregate training metrics land in W&B from this driver loop.
    _init_engine_wandb(args)

    # Adapter names currently loaded in sglang; shared between the train loop
    # (writer) and the /sample handler (reader) so sampling routes to an adapter
    # only once it's actually loaded.
    loaded_adapters: set = set()

    stop = threading.Event()
    loop_thread = threading.Thread(
        target=_run_training_loop,
        args=(actor_model, coordinator, stop),
        kwargs={"args": args, "sync_weights": generator is not None, "loaded_adapters": loaded_adapters},
        daemon=True,
    )
    loop_thread.start()

    from miles.training_engine.api_server import serve

    # On-demand "save weights": force a specific job's adapter into sglang.
    sync_weights_fn = None
    if generator is not None:
        sync_weights_fn = lambda job_id: _sync_one_job(  # noqa: E731
            actor_model, coordinator, rollout_manager, loaded_adapters, job_id
        )

    print(f"[engine] serving training API on http://{API_HOST}:{API_PORT} (base_model={args.hf_checkpoint})", flush=True)
    print(f"[engine] controller: {controller}", flush=True)
    print(
        f"[engine] coordinator ready for jobs (base_model={args.hf_checkpoint}, "
        f"slots={args.multi_lora_n_adapters}, generation={'on' if generator is not None else 'off'})",
        flush=True,
    )
    try:
        serve(
            coordinator,
            generator,
            loaded_adapters,
            sync_weights_fn=sync_weights_fn,
            weight_sync_lock=_WEIGHT_SYNC_LOCK,
            host=API_HOST,
            port=API_PORT,
        )
    finally:
        stop.set()


if __name__ == "__main__":
    main()
