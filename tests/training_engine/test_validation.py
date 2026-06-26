"""CPU tests for trajectory-batch validation against the published version."""

from __future__ import annotations

import pytest

from miles.training_engine.schemas import (
    ExternalTrajectoryBatch,
    TrajectoryValidationError,
    count_action_tokens,
    lora_config_hash,
    validate_trajectory_batch,
)
from tests.training_engine.helpers import make_runtime


def _batch(job_id="j", version=0, n=4, **kw):
    return ExternalTrajectoryBatch(
        job_id=job_id,
        adapter_version=version,
        input_ids=[list(range(n))],
        attention_mask=[[1] * n],
        action_mask=[[0, 1, 1, 1][:n]],
        old_logprobs=[[0.0] * n],
        **kw,
    )


def test_valid_batch_passes():
    job = make_runtime("j", max_policy_lag=4)
    job.latest_published_version = 2
    validate_trajectory_batch(job, _batch("j", version=2))


def test_future_version_rejected():
    job = make_runtime("j")
    job.latest_published_version = 1
    with pytest.raises(TrajectoryValidationError, match="future or unpublished"):
        validate_trajectory_batch(job, _batch("j", version=5))


def test_policy_lag_exceeded_rejected():
    job = make_runtime("j", max_policy_lag=2)
    job.latest_published_version = 10
    with pytest.raises(TrajectoryValidationError, match="policy lag"):
        validate_trajectory_batch(job, _batch("j", version=3))


def test_lora_hash_mismatch_rejected():
    job = make_runtime("j")
    job.latest_published_version = 0
    with pytest.raises(TrajectoryValidationError, match="lora_config_hash"):
        validate_trajectory_batch(job, _batch("j", version=0, lora_config_hash="deadbeef"))


def test_matching_lora_hash_passes():
    job = make_runtime("j")
    job.latest_published_version = 0
    good = lora_config_hash(job.spec.adapter)
    validate_trajectory_batch(job, _batch("j", version=0, lora_config_hash=good))


def test_shape_mismatch_rejected():
    job = make_runtime("j")
    job.latest_published_version = 0
    bad = _batch("j", version=0)
    bad.attention_mask = [[1, 1]]  # wrong length
    with pytest.raises(TrajectoryValidationError, match="shapes differ"):
        validate_trajectory_batch(job, bad)


def test_count_action_tokens():
    assert count_action_tokens([[0, 1, 1], [1, 0, 0]]) == 3
