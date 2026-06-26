"""HTTP interface over central coordinator (analogous to vllm generation interface). 
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any

from .client import build_job_spec
from .schemas import ExternalTrajectoryBatch, TrainExample


class ReadWriteLock:
    """Writer-preferring read/write lock.

    Online RL has many concurrent ``/sample`` generations (readers) and an
    occasional weight sync (writer). The writer must pause + flush sglang, which
    deadlocks if any generation is in flight, so the sync has to wait for all
    samples to drain and block new ones until it finishes. Writer-preferring so
    continuous sampling can't starve the sync.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer_active = False
        self._writers_waiting = 0

    @contextlib.contextmanager
    def read_lock(self):
        with self._cond:
            while self._writer_active or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextlib.contextmanager
    def write_lock(self):
        with self._cond:
            self._writers_waiting += 1
            while self._writer_active or self._readers > 0:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer_active = True
        try:
            yield
        finally:
            with self._cond:
                self._writer_active = False
                self._cond.notify_all()


def _train_example_from_dict(job_id: str, ex: dict[str, Any]) -> TrainExample:
    """Build a TrainExample from a client dict, filling sensible defaults so the
    client only has to send the token tensors."""
    input_ids = ex["input_ids"]
    return TrainExample(
        job_id=job_id,
        slot=ex.get("slot"),
        loss_type=ex.get("loss_type", "sft"),
        adapter_version=ex.get("adapter_version"),
        input_ids=input_ids,
        attention_mask=ex.get("attention_mask") or [1] * len(input_ids),
        loss_mask=ex["loss_mask"],
        labels=ex.get("labels"),
    )


def build_app(coordinator, generator=None, loaded_adapters=None, sync_weights_fn=None, weight_sync_lock=None):
    """Build a FastAPI app exposing the training-engine endpoints over a coordinator.

    ``generator`` (a ``SglangGenerator``) enables the online-RL ``/sample``
    endpoint; if ``None``, sampling returns 503 (engine started train-only).
    ``loaded_adapters`` is a live set of adapter names currently loaded in sglang
    (maintained by the serve loop); ``/sample`` only sets ``lora_path`` when the
    job's adapter is in it, otherwise it samples the base model (correct for a
    fresh, untrained adapter whose first rollout is from the initial policy).
    ``sync_weights_fn(job_id) -> version | None`` synchronously forces a job's
    adapter into sglang (the ``save_weights`` primitive) for ``/sync_weights``.
    ``weight_sync_lock`` (a ``ReadWriteLock``) serializes generation against weight
    syncs: ``/sample`` holds the read side so a sync (write side) can't pause +
    flush sglang while a generation is in flight (which would deadlock).
    """
    import ray
    from fastapi import FastAPI, HTTPException

    if loaded_adapters is None:
        loaded_adapters = set()
    sample_lock = weight_sync_lock.read_lock if weight_sync_lock is not None else contextlib.nullcontext

    app = FastAPI(title="miles continuous training engine")

    @app.post("/v1/training/jobs")
    def create_job(payload: dict[str, Any]):
        spec = build_job_spec(**payload)
        try:
            resp = ray.get(coordinator.submit_job.remote(spec))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        print(f"[coordinator] received job {resp.job_id} (base_model={spec.base_model})", flush=True)
        return {"job_id": resp.job_id, "adapter_version": resp.adapter_version, "adapter_uri": resp.adapter_uri}

    @app.post("/v1/training/jobs/{job_id}/sft_examples")
    def submit_sft_examples(job_id: str, payload: dict[str, Any]):
        examples = [_train_example_from_dict(job_id, ex) for ex in payload["examples"]]
        try:
            resp = ray.get(
                coordinator.submit_sft_examples.remote(job_id, examples, payload.get("client_batch_id"))
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        return {"accepted": resp.accepted, "batch_id": resp.batch_id, "token_count": resp.token_count, "reason": resp.reason}

    @app.post("/v1/training/jobs/{job_id}/trajectory_batches")
    def submit_trajectory_batch(job_id: str, payload: dict[str, Any]):
        batch = ExternalTrajectoryBatch(job_id=job_id, **payload)
        try:
            resp = ray.get(coordinator.submit_trajectory_batch.remote(batch))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        return {"accepted": resp.accepted, "batch_id": resp.batch_id, "token_count": resp.token_count, "reason": resp.reason}

    @app.post("/v1/training/jobs/{job_id}/sample")
    def sample(job_id: str, payload: dict[str, Any]):
        """Online-RL generation: prompts -> rollouts with the job's current adapter.

        Reward-agnostic: the client scores the returned rollouts and resubmits
        them via ``/trajectory_batches``.
        """
        if generator is None:
            raise HTTPException(status_code=503, detail="generation not enabled on this engine")
        try:
            job = ray.get(coordinator.get_job.remote(job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        # Only route to the adapter once it's actually loaded in sglang; otherwise
        # sample the base model (the fresh/untrained adapter == base policy).
        adapter_name = job.spec.adapter.name
        lora_path = adapter_name if adapter_name in loaded_adapters else None
        # Hold the read side so an in-flight weight sync waits for this generation
        # to finish (and blocks new ones) before it pauses + flushes sglang.
        with sample_lock():
            rollouts = generator.generate(
                input_ids_list=payload["prompts"],
                sampling_params=payload.get("sampling_params"),
                lora_path=lora_path,
                n_samples_per_prompt=int(payload.get("n_samples_per_prompt", 1)),
            )
        return {
            "job_id": job_id,
            "adapter_version": job.latest_published_version,
            "rollouts": rollouts,
        }

    @app.post("/v1/training/jobs/{job_id}/sync_weights")
    def sync_weights(job_id: str):
        """Synchronously force this job's current adapter into sglang so a
        subsequent /sample routes to it (the ``save_weights`` primitive)."""
        if sync_weights_fn is None:
            raise HTTPException(status_code=503, detail="generation not enabled on this engine")
        version = sync_weights_fn(job_id)
        if version is None:
            raise HTTPException(
                status_code=409,
                detail="job not resident; it must have trained at least one step and not be paged out",
            )
        return {"job_id": job_id, "adapter_version": version, "loaded": True}

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
            "trained_tokens": job.trained_tokens,
            "last_loss": job.last_loss,
            "latest_published_version": job.latest_published_version,
            "latest_adapter_uri": job.latest_adapter_uri,
        }

    @app.get("/v1/stats")
    def stats():
        return ray.get(coordinator.stats.remote())

    return app


def serve(
    coordinator,
    generator=None,
    loaded_adapters=None,
    *,
    sync_weights_fn=None,
    weight_sync_lock=None,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Run the HTTP API for ``coordinator`` (blocking)."""
    import uvicorn

    uvicorn.run(
        build_app(coordinator, generator, loaded_adapters, sync_weights_fn, weight_sync_lock),
        host=host,
        port=port,
    )
