"""Deficit-weighted fair scheduler for continuous MultiLoRA training.

Pure-Python control-plane logic: no torch, no ray. Credit (deficit) accrues to
runnable jobs and is spent in trained tokens, so a single job with a huge
backlog cannot monopolize the engine.
"""

from __future__ import annotations

from .schemas import TrainingJobRuntime, TrainingJobState


def accrue_deficits(jobs: dict[str, TrainingJobRuntime], base_quantum_tokens: int) -> None:
    """Grant scheduling credit to runnable jobs, weighted by priority.

    Module-level so it can run either on a live jobs dict (in-process) or, more
    importantly, *inside the controller actor* (where mutating a returned
    snapshot wouldn't persist because Ray hands back deserialized copies).
    """
    for job in jobs.values():
        if (
            job.state
            in {
                TrainingJobState.TRAIN_READY,
                TrainingJobState.HOT_IDLE,
                TrainingJobState.COLD_READY,
            }
            and job.ready_train_tokens > 0
        ):
            job.deficit_tokens += base_quantum_tokens * job.spec.scheduling.priority


class ContinuousTrainingScheduler:
    def __init__(
        self,
        *,
        train_tokens_per_step: int,
        max_adapters_per_step: int,
        base_quantum_tokens: int,
    ):
        if train_tokens_per_step <= 0:
            raise ValueError("train_tokens_per_step must be positive")
        if max_adapters_per_step <= 0:
            raise ValueError("max_adapters_per_step must be positive")
        if base_quantum_tokens <= 0:
            raise ValueError("base_quantum_tokens must be positive")
        self.train_tokens_per_step = train_tokens_per_step
        self.max_adapters_per_step = max_adapters_per_step
        self.base_quantum_tokens = base_quantum_tokens

    def accrue_deficits(self, jobs: dict[str, TrainingJobRuntime]) -> None:
        """Grant scheduling credit to runnable jobs, weighted by priority."""
        accrue_deficits(jobs, self.base_quantum_tokens)

    def select_training_jobs(
        self,
        jobs: dict[str, TrainingJobRuntime],
    ) -> list[tuple[str, int]]:
        """Return [(job_id, target_tokens)] to train in the next quantum."""
        candidates: list[TrainingJobRuntime] = []
        for job in jobs.values():
            if job.state in {
                TrainingJobState.FAILED,
                TrainingJobState.CANCELLED,
                TrainingJobState.COMPLETED,
            }:
                continue
            if job.state == TrainingJobState.ACTIVE_STEP:
                continue
            min_quantum = job.spec.scheduling.min_tokens_per_train_quantum
            if job.ready_train_tokens < min_quantum:
                continue
            if job.deficit_tokens < min_quantum:
                continue
            candidates.append(job)

        # Highest deficit first; prefer already-hot jobs when fairness is close;
        # then jobs that have run fewer consecutive steps; then higher priority.
        candidates.sort(
            key=lambda j: (
                -j.deficit_tokens,
                0 if j.slot is not None else 1,
                j.consecutive_steps,
                -j.spec.scheduling.priority,
            )
        )

        selected: list[tuple[str, int]] = []
        remaining = self.train_tokens_per_step
        for job in candidates:
            if len(selected) >= self.max_adapters_per_step:
                break
            if remaining <= 0:
                break
            if job.consecutive_steps >= job.spec.scheduling.max_consecutive_steps:
                continue
            want = min(
                remaining,
                job.spec.budget.tokens_per_update,
                job.ready_train_tokens,
                job.deficit_tokens,
            )
            if want < job.spec.scheduling.min_tokens_per_train_quantum:
                continue
            selected.append((job.spec.job_id, int(want)))
            remaining -= int(want)
        return selected

    def choose_preemption_victim(
        self,
        hot_jobs: list[TrainingJobRuntime],
        incoming_job: TrainingJobRuntime,
        engine_step: int,
    ) -> TrainingJobRuntime | None:
        """Pick a hot job to evict so ``incoming_job`` can load, or None."""
        candidates: list[TrainingJobRuntime] = []
        for job in hot_jobs:
            if job.spec.job_id == incoming_job.spec.job_id:
                continue
            if not job.spec.scheduling.preemptible:
                continue
            if job.state == TrainingJobState.ACTIVE_STEP:
                continue
            if job.hot_since_engine_step is not None:
                hot_steps = engine_step - job.hot_since_engine_step
                if hot_steps < job.spec.scheduling.min_hot_steps:
                    continue
            candidates.append(job)

        if not candidates:
            return None

        # Prefer evicting idle (no ready data) jobs, then under-credited jobs,
        # then jobs that have hogged the slot the longest, then lower priority.
        candidates.sort(
            key=lambda j: (
                j.ready_train_tokens > 0,
                j.deficit_tokens,
                -j.consecutive_steps,
                j.spec.scheduling.priority,
            )
        )
        return candidates[0]
