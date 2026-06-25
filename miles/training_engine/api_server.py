"""Optional HTTP surface for the training engine (deferred per plan §25).

This is intentionally not wired into the MVP control flow; it provides the
endpoint shapes from the plan for when an external HTTP boundary is needed.
FastAPI is imported lazily inside ``build_app`` so importing this module never
requires the web stack.
"""

from __future__ import annotations

from typing import Any

from .client import build_job_spec
from .schemas import ExternalTrajectoryBatch, validate_trajectory_batch


def build_app(controller, trajectory_store):
    """Build a FastAPI app exposing the training-engine endpoints."""
    import ray
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="miles continuous training engine")

    @app.post("/v1/training/jobs")
    def create_job(payload: dict[str, Any]):
        spec = build_job_spec(**payload)
        try:
            job_id = ray.get(controller.submit_job.remote(spec))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"job_id": job_id}

    @app.get("/v1/training/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            job = ray.get(controller.get_job.remote(job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        return {
            "job_id": job.spec.job_id,
            "state": job.state.value,
            "trained_steps": job.trained_steps,
            "trained_tokens": job.trained_tokens,
            "current_adapter_version": job.current_adapter_version,
            "latest_adapter_uri": job.latest_adapter_uri,
        }

    @app.post("/v1/training/jobs/{job_id}/trajectory_batches")
    def submit_trajectory_batch(job_id: str, payload: dict[str, Any]):
        batch = ExternalTrajectoryBatch(job_id=job_id, **payload)
        job = ray.get(controller.get_job.remote(job_id))
        try:
            validate_trajectory_batch(job, batch)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        batch_id, token_count = trajectory_store.put(batch)
        ray.get(controller.mark_train_batch_ready.remote(job_id, batch_id, token_count))
        return {"accepted": True, "batch_id": batch_id, "token_count": token_count}

    @app.post("/v1/training/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        ray.get(controller.cancel_job.remote(job_id))
        return {"cancelled": True}

    return app
