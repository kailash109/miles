"""CPU tests for the deficit-weighted fair scheduler (no torch/ray)."""

from __future__ import annotations

from miles.training_engine.scheduler import ContinuousTrainingScheduler
from miles.training_engine.schemas import TrainingJobState
from tests.training_engine.helpers import make_runtime


def _simulate(jobs, scheduler, rounds):
    """Run accrue+select for many rounds, spending the selected job's credit."""
    for _ in range(rounds):
        scheduler.accrue_deficits(jobs)
        for job_id, tokens in scheduler.select_training_jobs(jobs):
            rt = jobs[job_id]
            rt.deficit_tokens -= tokens
            rt.ready_train_tokens -= tokens
            rt.trained_tokens += tokens


def test_equal_priority_is_fair():
    jobs = {
        "a": make_runtime("a", ready_train_tokens=10**9, priority=1),
        "b": make_runtime("b", ready_train_tokens=10**9, priority=1),
    }
    sched = ContinuousTrainingScheduler(
        train_tokens_per_step=10**9,
        max_adapters_per_step=1,
        base_quantum_tokens=1000,
    )
    _simulate(jobs, sched, rounds=400)
    ta, tb = jobs["a"].trained_tokens, jobs["b"].trained_tokens
    total = ta + tb
    assert total > 0
    assert abs(ta - tb) / total < 0.1, (ta, tb)


def test_priority_weighting_approximates_ratio():
    jobs = {
        "hi": make_runtime("hi", ready_train_tokens=10**9, priority=2),
        "lo": make_runtime("lo", ready_train_tokens=10**9, priority=1),
    }
    sched = ContinuousTrainingScheduler(
        train_tokens_per_step=10**9,
        max_adapters_per_step=1,
        base_quantum_tokens=1000,
    )
    _simulate(jobs, sched, rounds=600)
    ratio = jobs["hi"].trained_tokens / max(1, jobs["lo"].trained_tokens)
    assert 1.7 <= ratio <= 2.3, ratio


def test_selection_requires_ready_data_and_credit():
    sched = ContinuousTrainingScheduler(
        train_tokens_per_step=10**9, max_adapters_per_step=8, base_quantum_tokens=1000
    )
    # No ready data -> not selected even after accrual.
    jobs = {"a": make_runtime("a", ready_train_tokens=0)}
    sched.accrue_deficits(jobs)
    assert sched.select_training_jobs(jobs) == []

    # Ready data but below min quantum -> not selected.
    jobs = {
        "a": make_runtime("a", ready_train_tokens=10, min_tokens_per_train_quantum=2048)
    }
    sched.accrue_deficits(jobs)
    assert sched.select_training_jobs(jobs) == []


def test_max_consecutive_steps_blocks_hogging():
    sched = ContinuousTrainingScheduler(
        train_tokens_per_step=10**9, max_adapters_per_step=8, base_quantum_tokens=1000
    )
    job = make_runtime("a", ready_train_tokens=10**9, max_consecutive_steps=3)
    job.consecutive_steps = 3
    jobs = {"a": job}
    sched.accrue_deficits(jobs)
    assert sched.select_training_jobs(jobs) == []


def test_active_step_not_selected():
    sched = ContinuousTrainingScheduler(
        train_tokens_per_step=10**9, max_adapters_per_step=8, base_quantum_tokens=1000
    )
    job = make_runtime("a", ready_train_tokens=10**9, state=TrainingJobState.ACTIVE_STEP)
    job.deficit_tokens = 10**6
    jobs = {"a": job}
    assert sched.select_training_jobs(jobs) == []


def test_preemption_victim_rules():
    sched = ContinuousTrainingScheduler(
        train_tokens_per_step=10**9, max_adapters_per_step=8, base_quantum_tokens=1000
    )
    incoming = make_runtime("incoming", ready_train_tokens=10**9)

    idle = make_runtime("idle", ready_train_tokens=0, slot=0, state=TrainingJobState.HOT_IDLE)
    idle.hot_since_engine_step = 0
    busy = make_runtime("busy", ready_train_tokens=10**9, slot=1, state=TrainingJobState.HOT_IDLE)
    busy.hot_since_engine_step = 0
    locked = make_runtime("locked", ready_train_tokens=0, slot=2, state=TrainingJobState.ACTIVE_STEP)
    locked.hot_since_engine_step = 0
    nonpre = make_runtime("nonpre", ready_train_tokens=0, slot=3, preemptible=False)
    nonpre.hot_since_engine_step = 0

    victim = sched.choose_preemption_victim(
        [idle, busy, locked, nonpre], incoming_job=incoming, engine_step=100
    )
    # Prefer the idle (no ready data) preemptible job; never active/non-preemptible.
    assert victim is idle


def test_min_hot_steps_protects_recent_loads():
    sched = ContinuousTrainingScheduler(
        train_tokens_per_step=10**9, max_adapters_per_step=8, base_quantum_tokens=1000
    )
    incoming = make_runtime("incoming", ready_train_tokens=10**9)
    fresh = make_runtime("fresh", ready_train_tokens=0, slot=0, min_hot_steps=5)
    fresh.hot_since_engine_step = 98  # only 2 steps hot at engine_step=100
    assert sched.choose_preemption_victim([fresh], incoming, engine_step=100) is None
