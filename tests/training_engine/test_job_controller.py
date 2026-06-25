"""CPU tests for the logical job controller state machine (no torch/ray)."""

from __future__ import annotations

import pytest

from miles.training_engine.job_controller import TrainingJobController
from miles.training_engine.schemas import TrainingJobState
from tests.training_engine.helpers import make_spec


def _controller(slots: int = 2) -> TrainingJobController:
    return TrainingJobController(base_model="test-model", max_hot_slots=slots)


def test_submit_and_duplicate_and_base_model_check():
    c = _controller()
    job_id = c.submit_job(make_spec("a"))
    assert job_id == "a"
    assert c.get_job("a").state == TrainingJobState.WAITING_FOR_DATA

    with pytest.raises(ValueError, match="duplicate"):
        c.submit_job(make_spec("a"))

    with pytest.raises(ValueError, match="base_model"):
        c.submit_job(make_spec("b", base_model="other-model"))


def test_data_ready_transitions_state():
    c = _controller()
    c.submit_job(make_spec("a"))
    c.mark_train_batch_ready("a", "batch1", 100)
    rt = c.get_job("a")
    assert rt.state == TrainingJobState.TRAIN_READY
    assert rt.ready_train_tokens == 100


def test_slot_reserve_hot_commit_cold_cycle():
    c = _controller(slots=1)
    c.submit_job(make_spec("a"))
    c.mark_train_batch_ready("a", "b", 500)

    slot = c.reserve_free_slot("a")
    assert slot == 0
    assert c.reserve_free_slot("a") is None  # no free slots left
    c.mark_hot("a", slot)
    assert c.get_job("a").state == TrainingJobState.HOT_IDLE

    c.mark_active_step(["a"])
    assert c.get_job("a").state == TrainingJobState.ACTIVE_STEP

    c.commit_step({"a": {"trained_tokens": 200}})
    rt = c.get_job("a")
    assert rt.trained_steps == 1
    assert rt.trained_tokens == 200
    assert rt.current_adapter_version == 1
    assert rt.ready_train_tokens == 300
    assert rt.dirty_since_publish is True
    # Still has data -> HOT_IDLE.
    assert rt.state == TrainingJobState.HOT_IDLE

    c.mark_cold("a", "/tmp/ckpt")
    rt = c.get_job("a")
    assert rt.slot is None
    assert 0 in c.free_slots
    # Had ready tokens -> TRAIN_READY after offload.
    assert rt.state == TrainingJobState.TRAIN_READY
    assert rt.cold_checkpoint_uri == "/tmp/ckpt"


def test_commit_without_remaining_data_waits():
    c = _controller(slots=1)
    c.submit_job(make_spec("a"))
    c.mark_train_batch_ready("a", "b", 100)
    c.reserve_free_slot("a")
    c.mark_hot("a", 0)
    c.commit_step({"a": {"trained_tokens": 100}})
    assert c.get_job("a").state == TrainingJobState.WAITING_FOR_DATA


def test_publish_and_budget_and_complete():
    c = _controller(slots=1)
    c.submit_job(make_spec("a", max_steps=1))
    c.mark_train_batch_ready("a", "b", 100)
    c.reserve_free_slot("a")
    c.mark_hot("a", 0)
    c.commit_step({"a": {"trained_tokens": 100}})

    assert c.budget_exhausted("a") is True
    c.mark_published("a", "/tmp/v1")
    assert c.get_job("a").dirty_since_publish is False

    c.complete_job("a", "/tmp/v1")
    rt = c.get_job("a")
    assert rt.state == TrainingJobState.COMPLETED
    assert rt.slot is None
    assert 0 in c.free_slots
