"""Central training coordinator: the single writer of all engine state.

Owns the job registry, batch store (leases), fairness scheduler, slot table,
adapter-version state, and plan lifecycle. It emits immutable ``TrainStepPlan``s
and commits/aborts them globally. Workers never mutate global state.

Plain class (unit-testable) wrapped as a Ray actor by ``make_training_coordinator``.
torch-free; Ray is only used lazily via the batch store's ``put_fn``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .artifact_store import ArtifactStore
from .batch_store import BatchStore
from .plan import ExportRequest, SlotOnload, SlotPreemption, TrainStepPlan
from .results import (
    CommitResult,
    CreateJobResponse,
    SubmitBatchResponse,
    WorkerStepResult,
)
from .scheduler import ContinuousTrainingScheduler
from .schemas import (
    BatchingPolicy,
    Execution,
    ExternalTrajectoryBatch,
    Lifecycle,
    QueueLimits,
    Readiness,
    Residency,
    TERMINAL_LIFECYCLES,
    TrainExample,
    TrainingJobRuntime,
    TrainingJobSpec,
    count_action_tokens,
    is_runnable,
    new_plan_id,
    validate_trajectory_batch,
)


@dataclass
class PlanRecord:
    plan: TrainStepPlan
    state: str = "leased"  # leased | committed | aborted


class TrainingCoordinator:
    def __init__(
        self,
        *,
        base_model: str,
        max_hot_slots: int,
        batching: BatchingPolicy | None = None,
        limits: QueueLimits | None = None,
        batch_store: BatchStore | None = None,
        artifact_store: ArtifactStore | None = None,
        adapter_store: str | None = None,
    ):
        self.base_model = base_model
        self.max_hot_slots = max_hot_slots
        # Root dir for persisted HF-PEFT adapters (inference disk-load); None disables.
        self.adapter_store = adapter_store
        self.batching = batching or BatchingPolicy()
        self.limits = limits or QueueLimits()
        self.batch_store = batch_store or BatchStore()
        self.artifact_store = artifact_store or ArtifactStore()
        self.scheduler = ContinuousTrainingScheduler(
            base_quantum_tokens=self.batching.base_quantum_tokens
        )

        self.jobs: dict[str, TrainingJobRuntime] = {}
        self.free_slots: set[int] = set(range(max_hot_slots))
        self.slot_owner: dict[int, str] = {}
        self.engine_step: int = 0
        self.plan_store: dict[str, PlanRecord] = {}
        self._optimizer_spec = None  # homogeneous optimizer across jobs (MVP)
        self.total_onloads: int = 0  # slot loads (cold->hot) across all steps
        self.total_preemptions: int = 0  # slot evictions (hot->cold) across all steps

    # -- ingestion -----------------------------------------------------------

    def submit_job(self, spec: TrainingJobSpec) -> CreateJobResponse:
        if spec.base_model != self.base_model:
            raise ValueError(f"job base_model {spec.base_model} != engine {self.base_model}")
        if spec.job_id in self.jobs:
            raise ValueError(f"duplicate job_id: {spec.job_id}")
        if len(self.jobs) >= self.limits.max_jobs:
            raise ValueError("max_jobs reached")
        if spec.adapter.rank <= 0 or spec.adapter.alpha <= 0:
            raise ValueError("adapter rank/alpha must be positive")
        # MVP: a single homogeneous optimizer config across jobs.
        if self._optimizer_spec is None:
            self._optimizer_spec = spec.optimizer
        elif spec.optimizer != self._optimizer_spec:
            raise ValueError("per-job optimizer config is not supported in MVP")

        job = TrainingJobRuntime(spec=spec)
        initial_uri = self.artifact_store.create_v0(spec)
        job.latest_published_version = 0
        job.latest_adapter_uri = initial_uri
        self.jobs[spec.job_id] = job
        return CreateJobResponse(job_id=spec.job_id, adapter_version=0, adapter_uri=initial_uri)

    def submit_trajectory_batch(self, batch: ExternalTrajectoryBatch) -> SubmitBatchResponse:
        job = self.jobs[batch.job_id]
        validate_trajectory_batch(job, batch)
        token_count = count_action_tokens(batch.action_mask)
        if self.batch_store.client_batch_exists(batch.job_id, batch.client_batch_id):
            existing = self.batch_store.client_id_index[(batch.job_id, batch.client_batch_id)]
            return SubmitBatchResponse(accepted=True, batch_id=existing, token_count=token_count)
        rejected = self._check_backpressure(job, token_count)
        if rejected is not None:
            return rejected
        batch_id = self.batch_store.put(
            job_id=batch.job_id,
            payload=batch,
            token_count=token_count,
            adapter_version=batch.adapter_version,
            client_batch_id=batch.client_batch_id,
        )
        self._on_data_ready(job, token_count)
        return SubmitBatchResponse(accepted=True, batch_id=batch_id, token_count=token_count)

    def submit_sft_examples(
        self, job_id: str, examples: list[TrainExample], client_batch_id: str | None = None
    ) -> SubmitBatchResponse:
        job = self.jobs[job_id]
        token_count = int(sum(int(sum(ex.loss_mask)) for ex in examples))
        if self.batch_store.client_batch_exists(job_id, client_batch_id):
            existing = self.batch_store.client_id_index[(job_id, client_batch_id)]
            return SubmitBatchResponse(accepted=True, batch_id=existing, token_count=token_count)
        rejected = self._check_backpressure(job, token_count)
        if rejected is not None:
            return rejected
        batch_id = self.batch_store.put(
            job_id=job_id,
            payload=examples,
            token_count=token_count,
            adapter_version=job.latest_published_version,
            client_batch_id=client_batch_id,
        )
        self._on_data_ready(job, token_count)
        return SubmitBatchResponse(accepted=True, batch_id=batch_id, token_count=token_count)

    def _check_backpressure(self, job, token_count) -> SubmitBatchResponse | None:
        if job.ready_train_tokens + token_count > self.limits.max_ready_tokens_per_job:
            return SubmitBatchResponse(accepted=False, reason="job_backpressure")
        if self.batch_store.global_ready_tokens + token_count > self.limits.max_ready_tokens_global:
            return SubmitBatchResponse(accepted=False, reason="engine_backpressure")
        return None

    def _on_data_ready(self, job: TrainingJobRuntime, token_count: int) -> None:
        job.ready_train_tokens += token_count
        if job.readiness == Readiness.EMPTY:
            job.readiness = Readiness.READY
        if job.last_ready_at is None:
            job.last_ready_at = time.time()

    # -- planning ------------------------------------------------------------

    def build_next_plan(self) -> TrainStepPlan | None:
        now = time.time()
        self.scheduler.accrue_deficits(self.jobs)

        if not self._should_dispatch(now):
            return None

        selected = self.scheduler.select_training_jobs(
            self.jobs,
            token_budget=self.batching.max_train_tokens_per_step,
            max_adapters=self.batching.max_adapters_per_step,
        )
        print(f"[coordinator] selected jobs: {selected}", flush=True)

        if not selected:
            return None

        loss_type = self.jobs[selected[0].job_id].spec.loss.type
        selected = [s for s in selected if self.jobs[s.job_id].spec.loss.type == loss_type]

        plan_id = new_plan_id()
        preemptions, onloads, job_to_slot = self._assign_slots(selected)

        print(f"[coordinator] job_to_slot: {job_to_slot}", flush=True)

        selected = [s for s in selected if s.job_id in job_to_slot]
        if not selected:
            return None

        leases = {}
        for s in selected:
            leases[s.job_id] = tuple(
                self.batch_store.lease_for_plan(s.job_id, s.target_tokens, plan_id)
            )
        if not any(leases.values()):
            return None

        plan = TrainStepPlan(
            plan_id=plan_id,
            engine_step=self.engine_step,
            loss_type=loss_type,
            selected_jobs=tuple(s.job_id for s in selected),
            target_tokens={s.job_id: s.target_tokens for s in selected},
            loss_weights={s.job_id: 1.0 for s in selected},
            job_to_loss={s.job_id: self.jobs[s.job_id].spec.loss for s in selected},
            job_to_slot=job_to_slot,
            job_to_optimizer_step={j: self.jobs[j].optimizer_step for j in job_to_slot},
            job_to_latest_published_version={
                j: self.jobs[j].latest_published_version for j in job_to_slot
            },
            preemptions=tuple(preemptions),
            onloads=tuple(onloads),
            leases=leases,
            publish_after_step=tuple(j for j in job_to_slot if self._should_publish(j)),
            exports=tuple(
                ExportRequest(
                    job_id=j,
                    slot=job_to_slot[j],
                    output_uri=self.jobs[j].spec.output_uri,
                    rank=self.jobs[j].spec.adapter.rank,
                    alpha=self.jobs[j].spec.adapter.alpha,
                    version=self.jobs[j].optimizer_step + 1,
                )
                for j in job_to_slot
                if self._should_publish(j)
            ),
        )
        self.plan_store[plan_id] = PlanRecord(plan=plan, state="leased")
        self._mark_jobs_leased(plan)
        return plan

    def _should_dispatch(self, now: float) -> bool:
        runnable = [j for j in self.jobs.values() if is_runnable(j)]
        if not runnable:
            return False
        ready_tokens = sum(j.ready_train_tokens for j in runnable)
        if ready_tokens >= self.batching.max_train_tokens_per_step:
            return True
        oldest = min((j.last_ready_at for j in runnable if j.last_ready_at is not None), default=now)
        if now - oldest >= self.batching.max_batch_wait_s:
            return True
        # Workers are otherwise idle (one active plan max, driver only builds when
        # none is active), so a minimum viable plan should dispatch.
        return True

    def _assign_slots(self, selected):
        preemptions: list[SlotPreemption] = []
        onloads: list[SlotOnload] = []
        job_to_slot: dict[str, int] = {}
        free = set(self.free_slots)
        chosen_victim_ids: set[str] = set()

        for s in selected:
            job = self.jobs[s.job_id]
            if job.residency == Residency.HOT and job.slot is not None:
                job_to_slot[job.job_id] = job.slot
                continue

            if free:
                slot = min(free)
                free.remove(slot)
            else:
                hot = [
                    j
                    for j in self.jobs.values()
                    if j.residency == Residency.HOT
                    and j.slot is not None
                    and j.job_id not in chosen_victim_ids
                    and j.job_id not in job_to_slot
                ]
                victim = self.scheduler.choose_preemption_victim(hot, job, self.engine_step)
                if victim is None:
                    continue
                chosen_victim_ids.add(victim.job_id)
                slot = victim.slot
                # Persist only if the adapter changed since its last disk write.
                persist = victim.optimizer_step > victim.persisted_step
                inference_uri = (
                    f"{self.adapter_store.rstrip('/')}/{victim.job_id}"
                    if (self.adapter_store and persist)
                    else None
                )
                preemptions.append(
                    SlotPreemption(
                        job_id=victim.job_id,
                        slot=slot,
                        checkpoint_uri=self._internal_checkpoint_uri(victim),
                        persist=persist,
                        inference_uri=inference_uri,
                        rank=victim.spec.adapter.rank,
                        alpha=victim.spec.adapter.alpha,
                        target_modules=tuple(victim.spec.adapter.target_modules),
                    )
                )

            onloads.append(
                SlotOnload(
                    job_id=job.job_id,
                    slot=slot,
                    source_uri=job.cold_checkpoint_uri or job.spec.adapter.adapter_uri,
                    rank=job.spec.adapter.rank,
                    alpha=job.spec.adapter.alpha,
                    target_modules=tuple(job.spec.adapter.target_modules),
                )
            )
            job_to_slot[job.job_id] = slot

        return preemptions, onloads, job_to_slot

    def _should_publish(self, job_id: str) -> bool:
        job = self.jobs[job_id]
        next_step = job.optimizer_step + 1
        return next_step % max(1, job.spec.budget.publish_every_steps) == 0

    def _internal_checkpoint_uri(self, job: TrainingJobRuntime) -> str:
        return f"{job.spec.output_uri.rstrip('/')}/internal/checkpoints/v{job.optimizer_step:06d}"

    def _mark_jobs_leased(self, plan: TrainStepPlan) -> None:
        for job_id in plan.selected_jobs:
            job = self.jobs[job_id]
            job.readiness = Readiness.LEASED
            job.execution = Execution.ACTIVE_STEP
            job.leased_train_tokens = sum(l.token_count for l in plan.leases.get(job_id, ()))

    # -- commit / abort ------------------------------------------------------

    def commit_or_abort_plan(
        self, plan_id: str, results: list[WorkerStepResult]
    ) -> CommitResult:
        record = self.plan_store[plan_id]
        plan = record.plan

        if not results or not all(r.ok for r in results):
            self.batch_store.abort_plan(plan_id)
            self._release_leased_jobs(plan)
            record.state = "aborted"
            errors = "; ".join(r.error for r in results if r and r.error) or "no_results"
            return CommitResult(ok=False, error=errors)

        self.batch_store.commit_plan(plan_id)
        self.engine_step += 1

        # Step-aggregate loss (batch-mean over all jobs this step); the workers
        # all-reduce it across DP, so any ok result carries the same value.
        step_metrics = next((r.metrics for r in results if r.metrics), {})
        step_loss = step_metrics.get("loss")

        for job_id in plan.selected_jobs:
            job = self.jobs[job_id]
            trained = sum(l.token_count for l in plan.leases.get(job_id, ()))
            job.optimizer_step += 1
            job.trained_steps += 1
            job.trained_tokens += trained
            if step_loss is not None:
                job.last_loss = float(step_loss)
            job.ready_train_tokens = max(0, job.ready_train_tokens - trained)
            job.deficit_tokens -= trained
            job.consecutive_steps += 1
            job.leased_train_tokens = 0
            job.execution = Execution.IDLE
            job.readiness = Readiness.READY if job.ready_train_tokens > 0 else Readiness.EMPTY
            if job.readiness == Readiness.EMPTY:
                job.last_ready_at = None
            job.pending_publish_steps.add(job.optimizer_step)

        self._apply_slot_state_after_success(plan)

        published = self._publish_completed(plan, results)
        self._maybe_complete_budget_jobs(plan)

        record.state = "committed"
        return CommitResult(ok=True, published=published)

    def _release_leased_jobs(self, plan: TrainStepPlan) -> None:
        for job_id in plan.selected_jobs:
            job = self.jobs[job_id]
            job.execution = Execution.IDLE
            job.leased_train_tokens = 0
            job.readiness = Readiness.READY if job.ready_train_tokens > 0 else Readiness.EMPTY

    def _apply_slot_state_after_success(self, plan: TrainStepPlan) -> None:
        self.total_preemptions += len(plan.preemptions)
        self.total_onloads += len(plan.onloads)
        for op in plan.preemptions:
            victim = self.jobs[op.job_id]
            victim.residency = Residency.COLD
            victim.slot = None
            victim.cold_checkpoint_uri = op.checkpoint_uri
            if op.persist:
                # The worker wrote the training ckpt + HF adapter for this step.
                victim.persisted_step = victim.optimizer_step
            victim.hot_since_engine_step = None
            victim.consecutive_steps = 0
            self.slot_owner.pop(op.slot, None)
            self.free_slots.add(op.slot)
        for op in plan.onloads:
            job = self.jobs[op.job_id]
            job.residency = Residency.HOT
            job.slot = op.slot
            job.hot_since_engine_step = self.engine_step
            self.slot_owner[op.slot] = op.job_id
            self.free_slots.discard(op.slot)

    def _publish_completed(self, plan: TrainStepPlan, results: list[WorkerStepResult]) -> list[dict]:
        published: list[dict] = []
        for job_id in plan.publish_after_step:
            job = self.jobs[job_id]
            step = job.optimizer_step
            files = sorted(
                {f for r in results for f in r.written_files.get(job_id, [])}
            )
            try:
                uri = self.artifact_store.finalize_manifest(
                    job.spec,
                    step,
                    files,
                    trained_steps=job.trained_steps,
                    trained_tokens=job.trained_tokens,
                )
            except Exception as exc:  # publish failure: leave version unchanged
                job.last_error = f"publish failed: {exc}"
                continue
            job.latest_materialized_step = max(job.latest_materialized_step, step)
            job.latest_published_version = max(job.latest_published_version, step)
            job.latest_adapter_uri = uri
            job.pending_publish_steps.discard(step)
            published.append(
                {
                    "event": "adapter_ready",
                    "job_id": job_id,
                    "adapter_version": step,
                    "adapter_uri": uri,
                    "trained_tokens": job.trained_tokens,
                }
            )
        return published

    def _maybe_complete_budget_jobs(self, plan: TrainStepPlan) -> None:
        for job_id in plan.selected_jobs:
            job = self.jobs[job_id]
            budget = job.spec.budget
            done = (budget.max_steps is not None and job.trained_steps >= budget.max_steps) or (
                budget.max_train_tokens is not None and job.trained_tokens >= budget.max_train_tokens
            )
            if done:
                job.lifecycle = Lifecycle.COMPLETED
                if job.slot is not None:
                    self.slot_owner.pop(job.slot, None)
                    self.free_slots.add(job.slot)
                    job.slot = None
                    job.residency = Residency.COLD

    # -- status --------------------------------------------------------------

    def snapshot_jobs(self) -> dict[str, TrainingJobRuntime]:
        return dict(self.jobs)

    def hot_adapter(self, job_id: str) -> dict | None:
        """Resident-adapter info for one job, or None if it isn't HOT.

        Used by the synchronous save/sync path: only a HOT job has its adapter
        weights live in a model slot to push into sglang.
        """
        job = self.jobs.get(job_id)
        if job is None or job.residency != Residency.HOT or job.slot is None:
            return None
        return {
            "name": job.spec.adapter.name,
            "slot": job.slot,
            "rank": job.spec.adapter.rank,
            "alpha": job.spec.adapter.alpha,
            "version": job.latest_published_version,
        }

    def hot_adapters(self) -> list[dict]:
        """Resident (HOT) adapters and their physical slots, for syncing into the
        inference engine (the bridge to ``multi_lora_controller`` / sglang).

        Each entry's ``slot`` is the model's MultiLoRALinear slot where this job's
        trained adapter weights live, so the weight updater can expose + push it.
        """
        out = []
        for job in self.jobs.values():
            if job.residency == Residency.HOT and job.slot is not None:
                out.append(
                    {
                        "name": job.spec.adapter.name,
                        "slot": job.slot,
                        "rank": job.spec.adapter.rank,
                        "alpha": job.spec.adapter.alpha,
                    }
                )
        return out

    def get_job(self, job_id: str) -> TrainingJobRuntime:
        return self.jobs[job_id]

    def stats(self) -> dict:
        """Aggregate engine/scheduling stats (cheap; safe to poll during a run)."""
        hot = sum(1 for j in self.jobs.values() if j.residency == Residency.HOT)
        completed = sum(1 for j in self.jobs.values() if j.lifecycle == Lifecycle.COMPLETED)
        trained = sum(1 for j in self.jobs.values() if j.trained_steps > 0)
        running = sum(1 for j in self.jobs.values() if j.lifecycle == Lifecycle.RUNNING)
        return {
            "engine_step": self.engine_step,
            "num_jobs": len(self.jobs),
            "running": running,
            "completed": completed,
            "hot": hot,
            "max_hot_slots": self.max_hot_slots,
            "distinct_jobs_trained": trained,
            "total_onloads": self.total_onloads,
            "total_preemptions": self.total_preemptions,
            "global_ready_tokens": self.batch_store.global_ready_tokens,
        }

    def all_terminal_or_idle(self) -> bool:
        for job in self.jobs.values():
            if job.lifecycle in TERMINAL_LIFECYCLES:
                continue
            if is_runnable(job) or job.execution == Execution.ACTIVE_STEP:
                return False
            if job.ready_train_tokens > 0:
                return False
        return True


def make_training_coordinator(*, base_model: str, max_hot_slots: int, **kwargs):
    """Wrap ``TrainingCoordinator`` as a Ray actor (Ray imported lazily)."""
    import ray

    return ray.remote(num_cpus=0)(TrainingCoordinator).remote(
        base_model=base_model, max_hot_slots=max_hot_slots, **kwargs
    )
