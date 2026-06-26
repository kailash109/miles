"""CPU tests for the deficit-weighted scheduler (no torch/ray)."""

from __future__ import annotations

from miles.training_engine.scheduler import ContinuousTrainingScheduler
from miles.training_engine.schemas import Readiness, Residency
from tests.training_engine.helpers import make_runtime


def _sched(base: int = 1000) -> ContinuousTrainingScheduler:
    return ContinuousTrainingScheduler(base_quantum_tokens=base)


def test_accrue_only_credits_ready_jobs():
    sched = _sched(1000)
    a = make_runtime("a", ready_train_tokens=500)
    b = make_runtime("b", ready_train_tokens=0, readiness=Readiness.EMPTY)
    sched.accrue_deficits({"a": a, "b": b})
    assert a.deficit_tokens == 1000
    assert b.deficit_tokens == 0


def test_accrue_scales_with_priority():
    sched = _sched(1000)
    a = make_runtime("a", ready_train_tokens=500, priority=1)
    b = make_runtime("b", ready_train_tokens=500, priority=3)
    sched.accrue_deficits({"a": a, "b": b})
    assert (a.deficit_tokens, b.deficit_tokens) == (1000, 3000)


def test_select_respects_budget_and_max_adapters():
    sched = _sched(1000)
    jobs = {
        f"j{i}": make_runtime(f"j{i}", ready_train_tokens=4000, tokens_per_update=4000)
        for i in range(4)
    }
    for j in jobs.values():
        j.deficit_tokens = 4000
    selected = sched.select_training_jobs(jobs, token_budget=5000, max_adapters=8)
    # Budget 5000: first job takes 4000, second capped to 1000 (>= min_quantum).
    assert sum(s.target_tokens for s in selected) <= 5000
    assert len(selected) <= 2


def test_select_skips_non_runnable():
    sched = _sched(1000)
    a = make_runtime("a", ready_train_tokens=0, readiness=Readiness.EMPTY)
    a.deficit_tokens = 5000
    selected = sched.select_training_jobs({"a": a}, token_budget=5000, max_adapters=8)
    assert selected == []


def test_hot_jobs_preferred_when_deficit_close():
    sched = _sched(1000)
    cold = make_runtime("cold", ready_train_tokens=2000, residency=Residency.COLD)
    hot = make_runtime("hot", ready_train_tokens=2000, residency=Residency.HOT, slot=0)
    cold.deficit_tokens = 2000
    hot.deficit_tokens = 2000
    selected = sched.select_training_jobs(
        {"cold": cold, "hot": hot}, token_budget=2000, max_adapters=1
    )
    # Cold-load penalty + hot tiebreak -> hot job wins the single slot.
    assert [s.job_id for s in selected] == ["hot"]


def test_preemption_victim_prefers_idle_low_deficit():
    sched = _sched(1000)
    incoming = make_runtime("incoming", ready_train_tokens=2000)
    busy = make_runtime("busy", ready_train_tokens=2000, residency=Residency.HOT, slot=0)
    idle = make_runtime("idle", ready_train_tokens=0, residency=Residency.HOT, slot=1)
    busy.deficit_tokens = 5000
    idle.deficit_tokens = 5000
    victim = sched.choose_preemption_victim([busy, idle], incoming, engine_step=10)
    assert victim.job_id == "idle"


def test_weighted_fair_share_over_time():
    # Spend the full accrued deficit each selection (target uncapped), so the
    # system is not overloaded and trained tokens track the accrual weights.
    sched = _sched(300)
    a = make_runtime("a", ready_train_tokens=10**9, priority=1)
    b = make_runtime("b", ready_train_tokens=10**9, priority=2)
    jobs = {"a": a, "b": b}
    trained = {"a": 0, "b": 0}
    for _ in range(400):
        sched.accrue_deficits(jobs)
        for s in sched.select_training_jobs(jobs, token_budget=10**9, max_adapters=1):
            job = jobs[s.job_id]
            job.deficit_tokens -= s.target_tokens
            job.ready_train_tokens -= s.target_tokens
            trained[s.job_id] += s.target_tokens
    ratio = trained["b"] / max(1, trained["a"])
    assert 1.6 < ratio < 2.4  # ~2:1 by priority
