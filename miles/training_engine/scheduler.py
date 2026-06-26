"""Deficit-weighted fair scheduler (pure: no torch, no ray, no model state).

The coordinator owns a scheduler instance and calls ``accrue_deficits`` /
``select_training_jobs`` / ``choose_preemption_victim``. Fairness is measured in
*trained tokens*: credit accrues to runnable jobs and is spent when they train.
"""

from __future__ import annotations

from .schemas import (
    Lifecycle,
    Readiness,
    Residency,
    SelectedJob,
    TrainingJobRuntime,
    can_preempt,
    is_runnable,
)


def _accrues_credit(job: TrainingJobRuntime) -> bool:
    # Runnable, or waiting with ready data (so backlog jobs keep earning share).
    return (
        job.lifecycle == Lifecycle.RUNNING
        and job.readiness == Readiness.READY
        and job.ready_train_tokens > 0
    )


def accrue_deficits(jobs: dict[str, TrainingJobRuntime], base_quantum_tokens: int) -> None:
    for job in jobs.values():
        if _accrues_credit(job):
            job.deficit_tokens += base_quantum_tokens * job.spec.scheduling.priority


class ContinuousTrainingScheduler:
    def __init__(self, *, base_quantum_tokens: int):
        if base_quantum_tokens <= 0:
            raise ValueError("base_quantum_tokens must be positive")
        self.base_quantum_tokens = base_quantum_tokens

    def accrue_deficits(self, jobs: dict[str, TrainingJobRuntime]) -> None:
        accrue_deficits(jobs, self.base_quantum_tokens)

    def _cold_load_penalty(self, job: TrainingJobRuntime) -> int:
        # Discourage loading a cold job for a tiny one-step quantum.
        return 0 if job.residency == Residency.HOT else self.base_quantum_tokens

    def select_training_jobs(
        self,
        jobs: dict[str, TrainingJobRuntime],
        *,
        token_budget: int,
        max_adapters: int,
    ) -> list[SelectedJob]:
        candidates = [j for j in jobs.values() if is_runnable(j)]
        candidates = [
            j
            for j in candidates
            if j.deficit_tokens >= j.spec.scheduling.min_tokens_per_train_quantum
            and j.consecutive_steps < j.spec.scheduling.max_consecutive_steps
        ]

        # Primary key: effective deficit after a cold-load penalty (so cold jobs
        # need more accrued credit to justify a load). Then prefer hot jobs, then
        # least-recently-run, then higher priority.
        candidates.sort(
            key=lambda j: (
                -(j.deficit_tokens - self._cold_load_penalty(j)),
                0 if j.residency == Residency.HOT else 1,
                j.consecutive_steps,
                -j.spec.scheduling.priority,
            )
        )

        selected: list[SelectedJob] = []
        remaining = token_budget
        for job in candidates:
            if len(selected) >= max_adapters or remaining <= 0:
                break
            target = min(
                remaining,
                job.spec.budget.tokens_per_update,
                job.ready_train_tokens,
                job.deficit_tokens,
            )
            if target < job.spec.scheduling.min_tokens_per_train_quantum:
                continue
            selected.append(SelectedJob(job_id=job.spec.job_id, target_tokens=int(target)))
            remaining -= int(target)
        return selected

    def choose_preemption_victim(
        self,
        hot_jobs: list[TrainingJobRuntime],
        incoming_job: TrainingJobRuntime,
        engine_step: int,
    ) -> TrainingJobRuntime | None:
        candidates = [
            j
            for j in hot_jobs
            if j.spec.job_id != incoming_job.spec.job_id and can_preempt(j, engine_step)
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda j: (
                j.ready_train_tokens > 0,  # idle (no data) first
                j.deficit_tokens,  # least under-served first
                -j.consecutive_steps,  # longest hogger first
                j.spec.scheduling.priority,  # lowest priority first
            )
        )
        return candidates[0]
