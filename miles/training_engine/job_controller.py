"""Durable logical-job controller for the continuous training engine.

This owns *logical* jobs and maps them onto a bounded pool of physical hot
slots. It deliberately does NOT touch Megatron tensors — all GPU-affecting
transitions are driven by the runner/pager and reflected back here.

Implemented as a plain class so the state machine is unit-testable without a
Ray cluster. ``make_training_job_controller`` wraps it as a Ray actor for the
live engine (Ray is imported lazily so this module stays import-light).
"""

from __future__ import annotations

from .schemas import (
    TrainingJobRuntime,
    TrainingJobSpec,
    TrainingJobState,
    lora_config_hash,
)


class TrainingJobController:
    def __init__(self, base_model: str, max_hot_slots: int):
        if max_hot_slots <= 0:
            raise ValueError("max_hot_slots must be positive")
        self.base_model = base_model
        self.max_hot_slots = max_hot_slots
        self.jobs: dict[str, TrainingJobRuntime] = {}
        self.hot_job_by_slot: dict[int, str] = {}
        self.free_slots: set[int] = set(range(max_hot_slots))
        self.engine_step: int = 0

    # -- submission / queries ------------------------------------------------

    def submit_job(self, spec: TrainingJobSpec) -> str:
        self._validate_new_job(spec)
        if spec.job_id in self.jobs:
            raise ValueError(f"duplicate job_id: {spec.job_id}")
        rt = TrainingJobRuntime(spec=spec)
        # SFT jobs with a dataset URI and external-RL jobs both start by waiting
        # for data; the dataset worker / ingestion endpoint flips them ready.
        rt.state = TrainingJobState.WAITING_FOR_DATA
        self.jobs[spec.job_id] = rt
        return spec.job_id

    def get_job(self, job_id: str) -> TrainingJobRuntime:
        return self.jobs[job_id]

    def snapshot_jobs(self) -> dict[str, TrainingJobRuntime]:
        # For MVP returning the live dataclasses is fine; production should
        # return compact views or persist snapshots to a DB/object store.
        return dict(self.jobs)

    def lora_config_hash(self, job_id: str) -> str:
        return lora_config_hash(self.jobs[job_id].spec.adapter)

    # -- data readiness ------------------------------------------------------

    def mark_train_batch_ready(self, job_id: str, batch_id: str, token_count: int) -> None:
        rt = self.jobs[job_id]
        rt.ready_batch_ids.append(batch_id)
        rt.ready_train_tokens += int(token_count)
        if rt.state in {
            TrainingJobState.WAITING_FOR_DATA,
            TrainingJobState.COLD_READY,
            TrainingJobState.HOT_IDLE,
        }:
            rt.state = TrainingJobState.TRAIN_READY

    # -- slot lifecycle ------------------------------------------------------

    def reserve_free_slot(self, job_id: str) -> int | None:
        if not self.free_slots:
            return None
        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.hot_job_by_slot[slot] = job_id
        rt = self.jobs[job_id]
        rt.slot = slot
        rt.state = TrainingJobState.LOADING
        return slot

    def mark_hot(self, job_id: str, slot: int) -> None:
        rt = self.jobs[job_id]
        rt.slot = slot
        rt.hot_since_engine_step = self.engine_step
        rt.state = TrainingJobState.HOT_IDLE

    def accrue_deficits(self, base_quantum_tokens: int) -> None:
        """Grant scheduling credit to runnable jobs (server-side mutation).

        Must run on the controller (not on a snapshot): when the controller is a
        Ray actor, ``snapshot_jobs`` returns copies, so accruing there would be
        lost. ``select_training_jobs`` then reads a fresh snapshot read-only.
        """
        from .scheduler import accrue_deficits

        accrue_deficits(self.jobs, base_quantum_tokens)

    def mark_active_step(self, job_ids: list[str]) -> None:
        for job_id in job_ids:
            self.jobs[job_id].state = TrainingJobState.ACTIVE_STEP

    def commit_step(self, updates: dict[str, dict]) -> None:
        self.engine_step += 1
        for job_id, u in updates.items():
            rt = self.jobs[job_id]
            trained = int(u.get("trained_tokens", 0))
            rt.trained_steps += 1
            rt.trained_tokens += trained
            rt.current_adapter_version += 1
            rt.ready_train_tokens = max(0, rt.ready_train_tokens - trained)
            rt.dirty_since_publish = True
            rt.consecutive_steps += 1
            rt.deficit_tokens -= trained
            rt.state = (
                TrainingJobState.HOT_IDLE
                if rt.ready_train_tokens > 0
                else TrainingJobState.WAITING_FOR_DATA
            )

    def mark_cold(self, job_id: str, checkpoint_uri: str) -> None:
        rt = self.jobs[job_id]
        if rt.slot is not None:
            self.hot_job_by_slot.pop(rt.slot, None)
            self.free_slots.add(rt.slot)
        rt.slot = None
        rt.cold_checkpoint_uri = checkpoint_uri
        rt.hot_since_engine_step = None
        rt.consecutive_steps = 0
        rt.state = (
            TrainingJobState.TRAIN_READY
            if rt.ready_train_tokens > 0
            else TrainingJobState.COLD_READY
        )

    # -- publishing / completion --------------------------------------------

    def mark_published(self, job_id: str, adapter_uri: str) -> None:
        rt = self.jobs[job_id]
        rt.latest_adapter_uri = adapter_uri
        rt.dirty_since_publish = False

    def complete_job(self, job_id: str, adapter_uri: str | None = None) -> None:
        rt = self.jobs[job_id]
        if adapter_uri is not None:
            rt.latest_adapter_uri = adapter_uri
        if rt.slot is not None:
            self.hot_job_by_slot.pop(rt.slot, None)
            self.free_slots.add(rt.slot)
            rt.slot = None
        rt.state = TrainingJobState.COMPLETED

    def cancel_job(self, job_id: str) -> None:
        rt = self.jobs[job_id]
        if rt.slot is not None:
            self.hot_job_by_slot.pop(rt.slot, None)
            self.free_slots.add(rt.slot)
            rt.slot = None
        rt.state = TrainingJobState.CANCELLED

    def fail_job(self, job_id: str, error: str) -> None:
        rt = self.jobs[job_id]
        rt.state = TrainingJobState.FAILED
        rt.last_error = error

    # -- budgets -------------------------------------------------------------

    def budget_exhausted(self, job_id: str) -> bool:
        rt = self.jobs[job_id]
        budget = rt.spec.budget
        if budget.max_steps is not None and rt.trained_steps >= budget.max_steps:
            return True
        if budget.max_train_tokens is not None and rt.trained_tokens >= budget.max_train_tokens:
            return True
        return False

    # -- internal ------------------------------------------------------------

    def _validate_new_job(self, spec: TrainingJobSpec) -> None:
        if spec.base_model != self.base_model:
            raise ValueError(
                f"job base_model {spec.base_model} != engine base_model {self.base_model}"
            )
        if spec.adapter.rank <= 0:
            raise ValueError("adapter rank must be positive")
        if spec.adapter.alpha <= 0:
            raise ValueError("adapter alpha must be positive")


def make_training_job_controller(base_model: str, max_hot_slots: int):
    """Return a Ray actor handle wrapping ``TrainingJobController``.

    Ray is imported here so importing this module never requires Ray.
    """
    import ray

    return ray.remote(num_cpus=0)(TrainingJobController).remote(base_model, max_hot_slots)
