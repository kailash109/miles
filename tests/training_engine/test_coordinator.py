"""CPU tests for the central TrainingCoordinator (no torch; ray bypassed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from miles.training_engine.batch_store import BatchStore
from miles.training_engine.coordinator import TrainingCoordinator
from miles.training_engine.results import WorkerStepResult
from miles.training_engine.schemas import (
    BatchingPolicy,
    ExternalTrajectoryBatch,
    Lifecycle,
    QueueLimits,
    Readiness,
    Residency,
)
from tests.training_engine.helpers import make_example, make_spec


def _coord(tmp_path: Path, *, max_hot_slots=2, limits=None) -> TrainingCoordinator:
    return TrainingCoordinator(
        base_model="test-model",
        max_hot_slots=max_hot_slots,
        batching=BatchingPolicy(
            max_train_tokens_per_step=10**9,
            max_adapters_per_step=8,
            base_quantum_tokens=2048,
            max_batch_wait_s=0.0,
        ),
        limits=limits,
        batch_store=BatchStore(put_fn=lambda x: x),
    )


def _spec(tmp_path: Path, job_id: str, **kw):
    return make_spec(job_id, base_model="test-model", output_uri=str(tmp_path / job_id), **kw)


def test_submit_job_publishes_v0(tmp_path):
    c = _coord(tmp_path)
    resp = c.submit_job(_spec(tmp_path, "a"))
    assert resp.adapter_version == 0
    manifest = json.loads(Path(resp.adapter_uri).read_text())
    assert manifest["status"] == "READY" and manifest["is_base"] is True
    assert c.jobs["a"].latest_published_version == 0


def test_duplicate_job_rejected(tmp_path):
    c = _coord(tmp_path)
    c.submit_job(_spec(tmp_path, "a"))
    with pytest.raises(ValueError, match="duplicate"):
        c.submit_job(_spec(tmp_path, "a"))


def test_submit_sft_examples_marks_ready(tmp_path):
    c = _coord(tmp_path)
    c.submit_job(_spec(tmp_path, "a"))
    resp = c.submit_sft_examples("a", [make_example("a", 0, n=4) for _ in range(3)])
    assert resp.accepted and resp.token_count == 12
    job = c.jobs["a"]
    assert job.readiness == Readiness.READY and job.ready_train_tokens == 12


def test_backpressure_rejects(tmp_path):
    c = _coord(tmp_path, limits=QueueLimits(max_ready_tokens_per_job=5))
    c.submit_job(_spec(tmp_path, "a"))
    resp = c.submit_sft_examples("a", [make_example("a", 0, n=4) for _ in range(3)])
    assert not resp.accepted and resp.reason == "job_backpressure"


def test_trajectory_idempotency(tmp_path):
    c = _coord(tmp_path)
    c.submit_job(_spec(tmp_path, "a", loss_type="grpo"))

    def mk():
        return ExternalTrajectoryBatch(
            job_id="a",
            adapter_version=0,
            input_ids=[[1, 2, 3, 4]],
            attention_mask=[[1, 1, 1, 1]],
            action_mask=[[0, 1, 1, 1]],
            old_logprobs=[[0.0, 0.0, 0.0, 0.0]],
            client_batch_id="retry-1",
        )

    r1 = c.submit_trajectory_batch(mk())
    r2 = c.submit_trajectory_batch(mk())
    assert r1.batch_id == r2.batch_id
    assert c.jobs["a"].ready_train_tokens == r1.token_count  # not double counted


def test_build_plan_assigns_slot_and_commit_advances(tmp_path):
    c = _coord(tmp_path)
    c.submit_job(_spec(tmp_path, "a"))
    c.submit_sft_examples("a", [make_example("a", 0, n=4) for _ in range(3)])

    plan = c.build_next_plan()
    assert plan is not None
    assert plan.selected_jobs == ("a",)
    assert "a" in plan.job_to_slot
    assert len(plan.onloads) == 1 and plan.onloads[0].slot == plan.job_to_slot["a"]
    # Leased -> not re-leasable.
    assert c.jobs["a"].readiness == Readiness.LEASED

    c.commit_or_abort_plan(plan.plan_id, [WorkerStepResult(plan_id=plan.plan_id, rank=0, ok=True)])
    job = c.jobs["a"]
    assert job.optimizer_step == 1 and job.trained_steps == 1 and job.trained_tokens == 12
    assert job.residency == Residency.HOT and job.slot is not None
    assert job.readiness == Readiness.EMPTY  # data exhausted


def test_abort_releases_and_does_not_advance(tmp_path):
    c = _coord(tmp_path)
    c.submit_job(_spec(tmp_path, "a"))
    c.submit_sft_examples("a", [make_example("a", 0, n=4) for _ in range(3)])
    plan = c.build_next_plan()

    res = c.commit_or_abort_plan(
        plan.plan_id, [WorkerStepResult(plan_id=plan.plan_id, rank=0, ok=False, error="boom")]
    )
    assert not res.ok
    job = c.jobs["a"]
    assert job.optimizer_step == 0 and job.trained_steps == 0
    assert job.readiness == Readiness.READY  # data released
    # Re-buildable after abort.
    assert c.build_next_plan() is not None


def test_publish_advances_version_when_files_exist(tmp_path):
    c = _coord(tmp_path)
    c.submit_job(_spec(tmp_path, "a"))
    c.submit_sft_examples("a", [make_example("a", 0, n=4)])
    plan = c.build_next_plan()
    assert plan.publish_after_step == ("a",)

    # Simulate a worker having written adapter files on disk.
    f = tmp_path / "a" / "checkpoints" / "step_1" / "adapter_model.safetensors"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("weights")

    res = c.commit_or_abort_plan(
        plan.plan_id,
        [WorkerStepResult(plan_id=plan.plan_id, rank=0, ok=True, written_files={"a": [str(f)]})],
    )
    assert res.ok
    job = c.jobs["a"]
    assert job.latest_published_version == 1
    manifest = json.loads(Path(job.latest_adapter_uri).read_text())
    assert manifest["status"] == "READY" and manifest["version"] == 1


def test_publish_failure_leaves_version_unchanged(tmp_path):
    c = _coord(tmp_path)
    c.submit_job(_spec(tmp_path, "a"))
    c.submit_sft_examples("a", [make_example("a", 0, n=4)])
    plan = c.build_next_plan()

    # Worker reports a file that does not exist -> finalize fails.
    res = c.commit_or_abort_plan(
        plan.plan_id,
        [WorkerStepResult(plan_id=plan.plan_id, rank=0, ok=True, written_files={"a": ["/nope.bin"]})],
    )
    assert res.ok  # step committed
    assert c.jobs["a"].optimizer_step == 1
    assert c.jobs["a"].latest_published_version == 0  # publish failed, unchanged


def test_central_preemption_when_no_free_slot(tmp_path):
    c = _coord(tmp_path, max_hot_slots=1)
    # job a already hot+idle in slot 0 (data exhausted).
    c.submit_job(_spec(tmp_path, "a"))
    a = c.jobs["a"]
    a.residency = Residency.HOT
    a.slot = 0
    a.hot_since_engine_step = 0
    c.free_slots = set()
    c.slot_owner = {0: "a"}

    c.submit_job(_spec(tmp_path, "b"))
    c.submit_sft_examples("b", [make_example("b", 0, n=4) for _ in range(3)])

    plan = c.build_next_plan()
    assert plan.selected_jobs == ("b",)
    assert len(plan.preemptions) == 1 and plan.preemptions[0].job_id == "a"
    assert plan.job_to_slot["b"] == 0

    c.commit_or_abort_plan(plan.plan_id, [WorkerStepResult(plan_id=plan.plan_id, rank=0, ok=True)])
    assert c.jobs["a"].residency == Residency.COLD and c.jobs["a"].slot is None
    assert c.jobs["b"].residency == Residency.HOT and c.jobs["b"].slot == 0


def test_oversubscription_cycles_jobs_through_slots(tmp_path):
    # 50 jobs, 2 slots: hot residency must never exceed slot count, and every
    # job must eventually train + complete (slots are freed and reused).
    c = _coord(tmp_path, max_hot_slots=2)
    n_jobs = 50
    for i in range(n_jobs):
        jid = f"j{i:03d}"
        c.submit_job(_spec(tmp_path, jid, max_steps=1))
        c.submit_sft_examples(jid, [make_example(jid, 0, n=4)])

    max_hot_seen = 0
    for _ in range(10_000):
        plan = c.build_next_plan()
        if plan is None:
            assert c.all_terminal_or_idle()
            break
        assert len(plan.job_to_slot) <= 2  # never plan more than slots
        results = [WorkerStepResult(plan_id=plan.plan_id, rank=0, ok=True)]
        c.commit_or_abort_plan(plan.plan_id, results)
        max_hot_seen = max(max_hot_seen, sum(1 for j in c.jobs.values() if j.residency == Residency.HOT))

    stats = c.stats()
    assert max_hot_seen <= 2  # VRAM-bound: never more than slot count resident
    assert stats["completed"] == n_jobs  # all processed despite 25x oversubscription
    assert stats["distinct_jobs_trained"] == n_jobs
    assert stats["total_onloads"] == n_jobs  # each job loaded once


def test_max_steps_completes_job(tmp_path):
    c = _coord(tmp_path)
    c.submit_job(_spec(tmp_path, "a", max_steps=1))
    c.submit_sft_examples("a", [make_example("a", 0, n=4) for _ in range(5)])
    plan = c.build_next_plan()
    c.commit_or_abort_plan(plan.plan_id, [WorkerStepResult(plan_id=plan.plan_id, rank=0, ok=True)])
    assert c.jobs["a"].lifecycle == Lifecycle.COMPLETED
    assert 0 in c.free_slots  # slot freed on completion
