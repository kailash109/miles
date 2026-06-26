"""Optional HTTP surface over the central coordinator (deferred per design).

Endpoint shapes only; FastAPI is imported lazily so importing this module never
requires the web stack.
"""

from __future__ import annotations

from typing import Any

from .client import build_job_spec
from .schemas import ExternalTrajectoryBatch


def build_app(coordinator):
    """Build a FastAPI app exposing the training-engine endpoints over a coordinator."""
    import ray
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="miles continuous training engine")

    @app.post("/v1/training/jobs")
    def create_job(payload: dict[str, Any]):
        spec = build_job_spec(**payload)
        try:
            resp = ray.get(coordinator.submit_job.remote(spec))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"job_id": resp.job_id, "adapter_version": resp.adapter_version, "adapter_uri": resp.adapter_uri}

    @app.post("/v1/training/jobs/{job_id}/trajectory_batches")
    def submit_trajectory_batch(job_id: str, payload: dict[str, Any]):
        batch = ExternalTrajectoryBatch(job_id=job_id, **payload)
        try:
            resp = ray.get(coordinator.submit_trajectory_batch.remote(batch))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"accepted": resp.accepted, "batch_id": resp.batch_id, "token_count": resp.token_count, "reason": resp.reason}

    @app.get("/v1/training/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            job = ray.get(coordinator.get_job.remote(job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        return {
            "job_id": job.spec.job_id,
            "lifecycle": job.lifecycle.value,
            "trained_steps": job.trained_steps,
            "latest_published_version": job.latest_published_version,
            "latest_adapter_uri": job.latest_adapter_uri,
        }

    return app
