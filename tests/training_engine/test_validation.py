"""CPU tests for trajectory validation and policy-lag enforcement (no torch)."""

from __future__ import annotations

import pytest

from miles.training_engine.schemas import (
    ExternalTrajectoryBatch,
    TrainingJobRuntime,
    TrajectoryValidationError,
    count_action_tokens,
    lora_config_hash,
    validate_trajectory_batch,
)
from tests.training_engine.helpers import make_spec


def _job(current_version: int, max_policy_lag: int = 4) -> TrainingJobRuntime:
    rt = TrainingJobRuntime(spec=make_spec("job", loss_type="grpo", max_policy_lag=max_policy_lag))
    rt.current_adapter_version = current_version
    return rt


def _batch(version: int, *, n_seq: int = 2, seq: int = 3, **kwargs) -> ExternalTrajectoryBatch:
    ids = [[1] * seq for _ in range(n_seq)]
    mask = [[1] * seq for _ in range(n_seq)]
    logp = [[0.0] * seq for _ in range(n_seq)]
    fields = dict(
        job_id="job",
        adapter_version=version,
        input_ids=ids,
        attention_mask=mask,
        action_mask=mask,
        old_logprobs=logp,
    )
    fields.update(kwargs)
    return ExternalTrajectoryBatch(**fields)


def test_accepts_current_and_within_lag():
    job = _job(current_version=10, max_policy_lag=4)
    validate_trajectory_batch(job, _batch(10))
    validate_trajectory_batch(job, _batch(8))
    validate_trajectory_batch(job, _batch(6))  # exactly at the lag boundary


def test_rejects_stale_and_future_versions():
    job = _job(current_version=10, max_policy_lag=4)
    with pytest.raises(TrajectoryValidationError, match="policy lag"):
        validate_trajectory_batch(job, _batch(5))
    with pytest.raises(TrajectoryValidationError, match="future"):
        validate_trajectory_batch(job, _batch(11))


def test_rejects_shape_mismatch():
    job = _job(current_version=1)
    bad = _batch(1)
    bad.action_mask = [[1, 1]]  # wrong shape vs input_ids
    with pytest.raises(TrajectoryValidationError, match="shapes differ"):
        validate_trajectory_batch(job, bad)


def test_rejects_old_logprobs_shape_mismatch():
    job = _job(current_version=1)
    bad = _batch(1)
    bad.old_logprobs = [[0.0, 0.0]]
    with pytest.raises(TrajectoryValidationError, match="old_logprobs"):
        validate_trajectory_batch(job, bad)


def test_rejects_empty_action_mask():
    job = _job(current_version=1)
    bad = _batch(1)
    bad.action_mask = [[0, 0, 0], [0, 0, 0]]
    with pytest.raises(TrajectoryValidationError, match="at least one token"):
        validate_trajectory_batch(job, bad)


def test_rejects_lora_config_hash_mismatch():
    job = _job(current_version=1)
    with pytest.raises(TrajectoryValidationError, match="lora_config_hash"):
        validate_trajectory_batch(job, _batch(1, lora_config_hash="deadbeef"))


def test_accepts_matching_lora_config_hash():
    job = _job(current_version=1)
    good_hash = lora_config_hash(job.spec.adapter)
    validate_trajectory_batch(job, _batch(1, lora_config_hash=good_hash))


def test_count_action_tokens_handles_nested_lists():
    assert count_action_tokens([[1, 0, 1], [1, 1, 0]]) == 4
    assert count_action_tokens([1, 0, 1, 1]) == 3
    assert count_action_tokens(None) == 0
