"""Slot-Saturating Fair Waterfill scheduler (pure: no torch, no ray, no model state).

An alternative to the deficit-weighted scheduler in ``scheduler.py``. Where that
one packs *few jobs with a large target each* (it spends a whole job's accrued
deficit before moving on), this one is built to **saturate slots**: admit as many
jobs as the budget can seat with a small viable grant each, then fairly water-fill
the leftover token budget across the admitted jobs.

Two phases per round:

* **Phase A -- admission.** Rank runnable jobs by an additive score (long-run
  fairness deficit + tail-latency/anti-starvation waiting penalties - cold-load
  and hogging penalties) and admit them, each with a small ``min_grant``, until we
  run out of slots or the budget can't seat another minimum grant.
* **Phase B -- token allocation.** Give every admitted job its ``min_grant``, then
  water-fill the remaining tokens proportionally to priority, capped per job so one
  stale/high-deficit job can't eat the whole round.

Select via ``BatchingPolicy.scheduler = "waterfill"``; the coordinator drives it
through the same surface as the deficit scheduler (``accrue`` / mostly
``select_training_jobs`` / ``choose_preemption_victim``).

Indivisible-record caveat: ``BatchStore.lease_for_plan`` leases *whole* records.
When the coordinator passes ``queue_views`` we size the per-job admission cost to
the next whole record (``first_batch_tokens``) so the step budget stays a hard
ceiling rather than being silently overshot. With large indivisible submissions
this naturally admits fewer jobs -- the real fix for saturation is splitting data
into smaller records upstream (see module docstring in ``batch_store.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from .schemas import (
    Execution,
    Lifecycle,
    Readiness,
    Residency,
    SelectedJob,
    TrainingJobRuntime,
    can_preempt,
)

if TYPE_CHECKING:
    from .batch_store import JobQueueView


@dataclass(frozen=True)
class FairWaterfillConfig:
    base_quantum_tokens: int = 2048
    min_tokens_per_job: int = 512
    cold_load_penalty_tokens: int = 2048
    wait_penalty_tokens_per_s: float = 2048.0
    age_penalty_tokens_per_s: float = 256.0
    # Caps accrued deficit at this many per-job updates so an idle-then-busy job
    # can't hoard unbounded credit and starve everyone when it returns.
    deficit_cap_updates: int = 4

    def __post_init__(self) -> None:
        if self.base_quantum_tokens <= 0:
            raise ValueError("base_quantum_tokens must be positive")
        if self.min_tokens_per_job <= 0:
            raise ValueError("min_tokens_per_job must be positive")


class SlotSaturatingFairScheduler:
    def __init__(self, cfg: FairWaterfillConfig | None = None):
        self.cfg = cfg or FairWaterfillConfig()
        self._last_accrual_round: int | None = None

    # -- credit accrual ------------------------------------------------------

    def accrue(self, jobs: dict[str, TrainingJobRuntime], *, round_id: int) -> None:
        """Accrue credit once per actual scheduling round, not once per poll.

        The coordinator may call ``build_next_plan`` many times during a single
        batching wait window; deduping on ``round_id`` keeps credit accrual tied
        to rounds so a job's share doesn't inflate just because we polled often.
        """
        if round_id == self._last_accrual_round:
            return
        for job in jobs.values():
            if not self._accrues_credit(job):
                continue
            weight = max(1, job.spec.scheduling.priority)
            cap = self.cfg.deficit_cap_updates * job.spec.budget.tokens_per_update * weight
            job.deficit_tokens = min(
                cap, job.deficit_tokens + self.cfg.base_quantum_tokens * weight
            )
        self._last_accrual_round = round_id

    @staticmethod
    def _accrues_credit(job: TrainingJobRuntime) -> bool:
        return (
            job.lifecycle == Lifecycle.RUNNING
            and job.readiness == Readiness.READY
            and job.ready_train_tokens > 0
        )

    # -- grant sizing --------------------------------------------------------

    def _minimum_grant(self, job: TrainingJobRuntime, qv: "JobQueueView | None") -> int:
        """Smallest token grant worth admitting this job with.

        Without a queue view we use the configured engine-level floor. With one we
        charge the next *whole* record (records are indivisible), so the admission
        cost matches what ``lease_for_plan`` will actually hand out and the step
        budget stays a hard ceiling.
        """
        desired = min(
            self.cfg.min_tokens_per_job,
            job.ready_train_tokens,
            max(1, job.spec.budget.tokens_per_update),
        )
        if qv is not None and qv.first_batch_tokens:
            return int(min(qv.first_batch_tokens, job.ready_train_tokens))
        return int(desired)

    def _max_grant(self, job: TrainingJobRuntime, qv: "JobQueueView | None") -> int:
        """Per-round ceiling for one job: never below its min grant, never above
        its available data or per-update size, and softly bounded by accrued
        deficit so a single job can't swallow the whole water-fill."""
        weight = max(1, job.spec.scheduling.priority)
        deficit_ceiling = max(
            self._minimum_grant(job, qv),
            job.deficit_tokens + self.cfg.base_quantum_tokens * weight,
        )
        return int(
            min(
                job.ready_train_tokens,
                max(self._minimum_grant(job, qv), job.spec.budget.tokens_per_update),
                deficit_ceiling,
            )
        )

    # -- admission scoring ---------------------------------------------------

    def _admission_score(
        self,
        job: TrainingJobRuntime,
        *,
        now: float,
        avg_wait_s: float,
        free_slots_available: bool,
    ) -> float:
        ready_at = job.last_ready_at if job.last_ready_at is not None else now
        wait_s = max(0.0, now - ready_at)
        wait_excess = max(0.0, wait_s - avg_wait_s)

        # Only discourage cold jobs when seating one would cost a load/preemption;
        # with empty slots there's no reason to avoid filling them.
        cold_penalty = 0.0
        if job.residency != Residency.HOT and not free_slots_available:
            cold_penalty = float(self.cfg.cold_load_penalty_tokens)

        consecutive_penalty = 0.0
        if job.consecutive_steps >= job.spec.scheduling.max_consecutive_steps:
            consecutive_penalty = 10.0 * self.cfg.base_quantum_tokens

        return (
            float(job.deficit_tokens)
            + self.cfg.wait_penalty_tokens_per_s * wait_excess
            + self.cfg.age_penalty_tokens_per_s * wait_s
            - cold_penalty
            - consecutive_penalty
        )

    def _is_schedulable(self, job: TrainingJobRuntime, qv: "JobQueueView | None") -> bool:
        return (
            job.lifecycle == Lifecycle.RUNNING
            and job.readiness == Readiness.READY
            and job.execution == Execution.IDLE
            and job.ready_train_tokens >= self._minimum_grant(job, qv)
        )

    # -- selection -----------------------------------------------------------

    def select_training_jobs(
        self,
        jobs: dict[str, TrainingJobRuntime],
        *,
        token_budget: int,
        max_adapters: int,
        max_hot_slots: int | None = None,
        free_slots: int = 0,
        now: float | None = None,
        queue_views: "Mapping[str, JobQueueView] | None" = None,
    ) -> list[SelectedJob]:
        import time

        if now is None:
            now = time.time()
        if max_hot_slots is None:
            max_hot_slots = max_adapters

        def qv_of(job_id: str) -> "JobQueueView | None":
            return queue_views.get(job_id) if queue_views is not None else None

        runnable = [j for j in jobs.values() if self._is_schedulable(j, qv_of(j.job_id))]
        if not runnable or token_budget <= 0 or max_adapters <= 0:
            return []

        waits = [max(0.0, now - (j.last_ready_at if j.last_ready_at is not None else now)) for j in runnable]
        avg_wait_s = sum(waits) / max(1, len(waits))
        free_slots_available = free_slots > 0

        target_count = min(max_adapters, max_hot_slots, len(runnable))

        candidates = sorted(
            runnable,
            key=lambda j: self._admission_score(
                j, now=now, avg_wait_s=avg_wait_s, free_slots_available=free_slots_available
            ),
            reverse=True,
        )

        # Phase A: admit as many jobs as feasible with a small base grant.
        selected: list[TrainingJobRuntime] = []
        grants: dict[str, int] = {}
        caps: dict[str, int] = {}
        remaining = int(token_budget)
        for job in candidates:
            if len(selected) >= target_count:
                break
            qv = qv_of(job.job_id)
            min_grant = self._minimum_grant(job, qv)
            if min_grant <= 0 or remaining < min_grant:
                # Can't seat this job's minimum; keep scanning -- a later (smaller)
                # job may still fit the leftover budget.
                continue
            selected.append(job)
            grants[job.job_id] = min_grant
            caps[job.job_id] = max(min_grant, self._max_grant(job, qv))
            remaining -= min_grant

        if not selected:
            return []

        # Phase B: fair water-fill the remaining tokens.
        self._waterfill(selected, grants, caps, remaining)

        return [
            SelectedJob(job_id=j.job_id, target_tokens=int(grants[j.job_id]))
            for j in selected
            if grants[j.job_id] > 0
        ]

    def _waterfill(
        self,
        selected: list[TrainingJobRuntime],
        grants: dict[str, int],
        caps: dict[str, int],
        remaining: int,
    ) -> None:
        if remaining <= 0:
            return
        job_by_id = {j.job_id: j for j in selected}
        active = {jid for jid in job_by_id if grants[jid] < caps[jid]}

        while remaining > 0 and active:
            total_weight = sum(max(1, job_by_id[jid].spec.scheduling.priority) for jid in active)
            if total_weight <= 0:
                break
            # Snapshot the pass budget so each job's proportional share is computed
            # from the same base -- otherwise jobs visited earlier would eat into
            # later jobs' share and bias allocation away from the priority ratio.
            pass_budget = remaining
            progressed = False
            for jid in list(active):
                weight = max(1, job_by_id[jid].spec.scheduling.priority)
                headroom = caps[jid] - grants[jid]
                if headroom <= 0:
                    active.discard(jid)
                    continue
                share = max(1, int(pass_budget * weight / total_weight))
                add = min(share, headroom, remaining)
                if add > 0:
                    grants[jid] += add
                    remaining -= add
                    progressed = True
                if grants[jid] >= caps[jid]:
                    active.discard(jid)
                if remaining <= 0:
                    break
            if not progressed:
                break

    # -- preemption ----------------------------------------------------------

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
