"""CPU tests for the Slot-Saturating Fair Waterfill scheduler (no torch/ray)."""

from __future__ import annotations

from miles.training_engine.batch_store import JobQueueView
from miles.training_engine.scheduler import make_scheduler
from miles.training_engine.scheduler_waterfill import (
    FairWaterfillConfig,
    SlotSaturatingFairScheduler,
)
from miles.training_engine.schemas import BatchingPolicy, Readiness, Residency
from tests.training_engine.helpers import make_runtime


def _sched(**cfg) -> SlotSaturatingFairScheduler:
    return SlotSaturatingFairScheduler(FairWaterfillConfig(**cfg))


def test_factory_selects_waterfill():
    sched = make_scheduler(BatchingPolicy(scheduler="waterfill"))
    assert isinstance(sched, SlotSaturatingFairScheduler)


def test_saturates_many_jobs_with_small_grants():
    # 48 jobs, budget only fits a base grant each: all should be admitted (vs the
    # deficit scheduler, which would pack far fewer with large per-job targets).
    sched = _sched(min_tokens_per_job=512)
    jobs = {
        f"j{i}": make_runtime(f"j{i}", ready_train_tokens=8192, tokens_per_update=8192)
        for i in range(48)
    }
    for j in jobs.values():
        j.deficit_tokens = 2048
    selected = sched.select_training_jobs(
        jobs, token_budget=50_000, max_adapters=50, max_hot_slots=50, free_slots=50, now=1000.0
    )
    assert len(selected) == 48
    assert sum(s.target_tokens for s in selected) <= 50_000


def test_budget_caps_admission_count():
    sched = _sched(min_tokens_per_job=1000)
    jobs = {
        f"j{i}": make_runtime(f"j{i}", ready_train_tokens=8192, tokens_per_update=8192)
        for i in range(50)
    }
    # Budget only seats ~5 minimum grants of 1000.
    selected = sched.select_training_jobs(
        jobs, token_budget=5000, max_adapters=50, max_hot_slots=50, free_slots=50, now=1000.0
    )
    assert 1 <= len(selected) <= 5
    assert sum(s.target_tokens for s in selected) <= 5000


def test_waterfill_distributes_leftover_by_priority():
    # Two jobs, plenty of budget; extra tokens split ~1:3 by priority.
    sched = _sched(base_quantum_tokens=2048, min_tokens_per_job=512)
    a = make_runtime("a", ready_train_tokens=10**9, tokens_per_update=10**9, priority=1)
    b = make_runtime("b", ready_train_tokens=10**9, tokens_per_update=10**9, priority=3)
    a.deficit_tokens = 10**9
    b.deficit_tokens = 10**9
    selected = sched.select_training_jobs(
        {"a": a, "b": b},
        token_budget=100_000,
        max_adapters=8,
        max_hot_slots=8,
        free_slots=8,
        now=1000.0,
    )
    grants = {s.job_id: s.target_tokens for s in selected}
    assert sum(grants.values()) <= 100_000
    ratio = grants["b"] / max(1, grants["a"])
    assert 2.0 < ratio < 4.0


def test_max_grant_caps_a_single_job():
    # One job can't eat the whole budget: capped by deficit_tokens + base_quantum.
    sched = _sched(base_quantum_tokens=2048, min_tokens_per_job=512)
    a = make_runtime("a", ready_train_tokens=10**9, tokens_per_update=10**9)
    a.deficit_tokens = 4000
    selected = sched.select_training_jobs(
        {"a": a}, token_budget=100_000, max_adapters=8, max_hot_slots=8, free_slots=8, now=1000.0
    )
    assert len(selected) == 1
    assert selected[0].target_tokens == 4000 + 2048  # deficit + base_quantum


def test_skips_jobs_without_min_grant_data():
    sched = _sched(min_tokens_per_job=512)
    empty = make_runtime("empty", ready_train_tokens=0, readiness=Readiness.EMPTY)
    empty.deficit_tokens = 10**6
    selected = sched.select_training_jobs(
        {"empty": empty},
        token_budget=10_000,
        max_adapters=8,
        max_hot_slots=8,
        free_slots=8,
        now=1000.0,
    )
    assert selected == []


def test_long_waiting_job_admitted_first():
    # A job past its wait window outscores a fresh, equal-deficit job for the one
    # available slot via the waiting-time penalty.
    sched = _sched(wait_penalty_tokens_per_s=2048.0, age_penalty_tokens_per_s=256.0)
    fresh = make_runtime("fresh", ready_train_tokens=8192, tokens_per_update=8192)
    stale = make_runtime("stale", ready_train_tokens=8192, tokens_per_update=8192)
    fresh.deficit_tokens = 2048
    stale.deficit_tokens = 2048
    fresh.last_ready_at = 1000.0
    stale.last_ready_at = 900.0  # waited 100s longer
    selected = sched.select_training_jobs(
        {"fresh": fresh, "stale": stale},
        token_budget=8192,  # room for exactly one base grant
        max_adapters=1,
        max_hot_slots=1,
        free_slots=1,
        now=1000.0,
    )
    assert [s.job_id for s in selected] == ["stale"]


def test_queue_view_keeps_budget_a_hard_ceiling():
    # Indivisible 8192-token records: with min_tokens_per_job=512 the naive grant
    # would be 512, but lease_for_plan would hand out a whole 8192 record. The
    # queue view makes admission charge the real record size, so fewer jobs are
    # admitted and the budget is never overshot.
    sched = _sched(min_tokens_per_job=512)
    jobs = {
        f"j{i}": make_runtime(f"j{i}", ready_train_tokens=8192, tokens_per_update=8192)
        for i in range(10)
    }
    for j in jobs.values():
        j.deficit_tokens = 0
    views = {
        jid: JobQueueView(jid, available_tokens=8192, available_batches=1, first_batch_tokens=8192, oldest_ready_at=1.0)
        for jid in jobs
    }
    selected = sched.select_training_jobs(
        jobs,
        token_budget=30_000,
        max_adapters=50,
        max_hot_slots=50,
        free_slots=50,
        now=1000.0,
        queue_views=views,
    )
    # Each admitted job costs a full 8192-token record -> at most 3 fit in 30k.
    assert len(selected) <= 3
    assert all(s.target_tokens >= 8192 for s in selected)
    assert sum(s.target_tokens for s in selected) <= 30_000


def test_accrue_is_once_per_round():
    sched = _sched(base_quantum_tokens=1000)
    a = make_runtime("a", ready_train_tokens=500)
    jobs = {"a": a}
    sched.accrue(jobs, round_id=0)
    sched.accrue(jobs, round_id=0)  # same round: no double credit
    sched.accrue(jobs, round_id=0)
    assert a.deficit_tokens == 1000
    sched.accrue(jobs, round_id=1)  # new round: accrues again
    assert a.deficit_tokens == 2000


def test_accrue_caps_deficit():
    sched = _sched(base_quantum_tokens=1000, deficit_cap_updates=4)
    a = make_runtime("a", ready_train_tokens=500, tokens_per_update=2000, priority=1)
    jobs = {"a": a}
    for r in range(100):
        sched.accrue(jobs, round_id=r)
    assert a.deficit_tokens == 4 * 2000 * 1  # capped at deficit_cap_updates * U * w


def test_preemption_victim_prefers_idle_low_deficit():
    sched = _sched()
    incoming = make_runtime("incoming", ready_train_tokens=2000)
    busy = make_runtime("busy", ready_train_tokens=2000, residency=Residency.HOT, slot=0)
    idle = make_runtime("idle", ready_train_tokens=0, residency=Residency.HOT, slot=1)
    busy.deficit_tokens = 5000
    idle.deficit_tokens = 5000
    victim = sched.choose_preemption_victim([busy, idle], incoming, engine_step=10)
    assert victim.job_id == "idle"
