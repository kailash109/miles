"""Thin in-process client for the continuous training engine.

In the MVP the client talks directly to the Ray controller handle and the
in-process stores; an HTTP transport (see ``api_server.py``) can be layered on
later without changing call sites. ``build_job_spec`` is a pure helper for
constructing a validated ``TrainingJobSpec`` and is import-light.
"""

from __future__ import annotations

from typing import Any

from .schemas import (
    AdapterSpec,
    BudgetSpec,
    DatasetSpec,
    ExternalTrajectoryBatch,
    LossSpec,
    OptimizerSpec,
    SchedulingSpec,
    TrainingJobRuntime,
    TrainingJobSpec,
    new_job_id,
    validate_trajectory_batch,
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
    """Construct a TrainingJobSpec from plain dicts (API-style payloads)."""
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
    """In-process client wrapping a Ray controller handle + trajectory store."""

    def __init__(self, controller, trajectory_store):
        self._controller = controller
        self._trajectory_store = trajectory_store

    def create_lora_training_job(
        self,
        *,
        base_model: str,
        adapter: dict[str, Any],
        output_uri: str,
        **kwargs: Any,
    ) -> str:
        import ray

        spec = build_job_spec(
            base_model=base_model, adapter=adapter, output_uri=output_uri, **kwargs
        )
        return ray.get(self._controller.submit_job.remote(spec))

    def get_job(self, job_id: str) -> TrainingJobRuntime:
        import ray

        return ray.get(self._controller.get_job.remote(job_id))

    def submit_trajectory_batch(self, batch: ExternalTrajectoryBatch) -> dict[str, Any]:
        import ray

        job = ray.get(self._controller.get_job.remote(batch.job_id))
        validate_trajectory_batch(job, batch)
        batch_id, token_count = self._trajectory_store.put(batch)
        ray.get(
            self._controller.mark_train_batch_ready.remote(batch.job_id, batch_id, token_count)
        )
        return {"accepted": True, "batch_id": batch_id, "token_count": token_count}

    def cancel_job(self, job_id: str) -> None:
        import ray

        ray.get(self._controller.cancel_job.remote(job_id))
