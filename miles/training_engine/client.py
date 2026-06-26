"""Thin client for the central training coordinator.

``build_job_spec`` is a pure helper (import-light). ``TrainingClient`` wraps a
Ray ``TrainingCoordinator`` handle; all calls are remote.
"""

from __future__ import annotations

from typing import Any

from .results import CreateJobResponse, SubmitBatchResponse
from .schemas import (
    AdapterSpec,
    BudgetSpec,
    DatasetSpec,
    ExternalTrajectoryBatch,
    LossSpec,
    OptimizerSpec,
    SchedulingSpec,
    TrainingJobSpec,
    new_job_id,
)


def build_job_spec(
    *,
    base_model: str,
    adapter: dict[str, Any],
    output_uri: str,
    user_id: str = "default",
    job_id: str | None = None,
    base_model_revision: str | None = None,
    dataset: dict[str, Any] | None = None,
    loss: dict[str, Any] | None = None,
    optimizer: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    scheduling: dict[str, Any] | None = None,
) -> TrainingJobSpec:
    return TrainingJobSpec(
        job_id=job_id or new_job_id(),
        user_id=user_id,
        base_model=base_model,
        base_model_revision=base_model_revision,
        adapter=AdapterSpec(**adapter),
        dataset=DatasetSpec(**(dataset or {})),
        loss=LossSpec(**(loss or {"type": "sft"})),
        optimizer=OptimizerSpec(**(optimizer or {})),
        budget=BudgetSpec(**(budget or {})),
        scheduling=SchedulingSpec(**(scheduling or {})),
        output_uri=output_uri,
    )


class TrainingClient:
    """Wraps a Ray ``TrainingCoordinator`` handle."""

    def __init__(self, coordinator):
        self._coordinator = coordinator

    def create_lora_training_job(
        self, *, base_model: str, adapter: dict[str, Any], output_uri: str, **kwargs: Any
    ) -> CreateJobResponse:
        import ray

        spec = build_job_spec(
            base_model=base_model, adapter=adapter, output_uri=output_uri, **kwargs
        )
        return ray.get(self._coordinator.submit_job.remote(spec))

    def submit_trajectory_batch(self, batch: ExternalTrajectoryBatch) -> SubmitBatchResponse:
        import ray

        return ray.get(self._coordinator.submit_trajectory_batch.remote(batch))

    def get_job(self, job_id: str):
        import ray

        return ray.get(self._coordinator.get_job.remote(job_id))
