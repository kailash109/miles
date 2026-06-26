"""Clients for the central training coordinator.

``build_job_spec`` is a pure helper (import-light). ``TrainingClient`` wraps a
Ray ``TrainingCoordinator`` handle directly (in-cluster use). ``ServiceClient`` /
``LoraTrainingClient`` are HTTP clients that talk to a running ``api_server`` so
a user process can submit jobs/data without touching Ray (Tinker-style).
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


class ServiceClient:
    """HTTP entrypoint to a running training engine (one engine == one base model).

    Usage::

        sc = ServiceClient("http://engine-host:8000")
        tc = sc.create_lora_training_client(base_model="/root/Qwen3-4B", rank=32, alpha=32)
        tc.submit_sft_examples([{"input_ids": [...], "loss_mask": [...], "labels": [...]}])
    """

    def __init__(self, base_url: str, *, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import requests

        resp = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> dict[str, Any]:
        import requests

        resp = requests.get(f"{self.base_url}{path}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def create_lora_training_client(
        self,
        *,
        base_model: str,
        rank: int,
        alpha: int,
        target_modules: list[str] | str = "all-linear",
        output_uri: str | None = None,
        name: str | None = None,
        loss: dict[str, Any] | None = None,
        optimizer: dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
        scheduling: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LoraTrainingClient:
        job_id = kwargs.pop("job_id", None) or new_job_id()
        adapter = {
            "name": name or job_id,
            "rank": rank,
            "alpha": alpha,
            "target_modules": [target_modules] if isinstance(target_modules, str) else list(target_modules),
        }
        payload: dict[str, Any] = {
            "base_model": base_model,
            "job_id": job_id,
            "adapter": adapter,
            "output_uri": output_uri or f"engine://{job_id}",
        }
        for key, value in (("loss", loss), ("optimizer", optimizer), ("budget", budget), ("scheduling", scheduling)):
            if value is not None:
                payload[key] = value
        payload.update(kwargs)
        resp = self._post("/v1/training/jobs", payload)
        return LoraTrainingClient(self, job_id=resp["job_id"], base_model=base_model)

    def stats(self) -> dict[str, Any]:
        return self._get("/v1/stats")


class LoraTrainingClient:
    """Handle to a single LoRA job on a specific engine; submits data + reads status."""

    def __init__(self, service: ServiceClient, *, job_id: str, base_model: str):
        self._service = service
        self.job_id = job_id
        self.base_model = base_model

    def submit_sft_examples(
        self, examples: list[dict[str, Any]], *, client_batch_id: str | None = None
    ) -> dict[str, Any]:
        return self._service._post(
            f"/v1/training/jobs/{self.job_id}/sft_examples",
            {"examples": examples, "client_batch_id": client_batch_id},
        )

    def submit_trajectory_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._service._post(f"/v1/training/jobs/{self.job_id}/trajectory_batches", batch)

    def sample(
        self,
        prompts: list[list[int]],
        *,
        sampling_params: dict[str, Any] | None = None,
        n_samples_per_prompt: int = 1,
    ) -> dict[str, Any]:
        """Online-RL generation: returns rollouts to score client-side.

        Response: ``{job_id, adapter_version, rollouts: [{prompt_ids,
        response_ids, response_logprobs, text}, ...]}``. Score each rollout and
        call :meth:`submit_scored_rollouts` to train on it.
        """
        return self._service._post(
            f"/v1/training/jobs/{self.job_id}/sample",
            {
                "prompts": prompts,
                "sampling_params": sampling_params or {},
                "n_samples_per_prompt": n_samples_per_prompt,
            },
        )

    def submit_scored_rollouts(
        self,
        rollouts: list[dict[str, Any]],
        rewards: list[float],
        *,
        adapter_version: int,
        client_batch_id: str | None = None,
    ) -> dict[str, Any]:
        """Turn sampled rollouts + client-computed rewards into an RL trajectory batch.

        Builds the prompt+response sequences, an action mask over the response
        tokens, and per-token rewards (broadcast from the per-rollout scalar),
        then submits as an ``ExternalTrajectoryBatch``.
        """
        if len(rollouts) != len(rewards):
            raise ValueError(f"rollouts ({len(rollouts)}) and rewards ({len(rewards)}) length mismatch")

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        action_mask: list[list[int]] = []
        old_logprobs: list[list[float]] = []
        per_token_rewards: list[list[float]] = []
        for rollout, reward in zip(rollouts, rewards, strict=True):
            prompt_ids = rollout["prompt_ids"]
            response_ids = rollout["response_ids"]
            seq = list(prompt_ids) + list(response_ids)
            n_resp = len(response_ids)
            input_ids.append(seq)
            attention_mask.append([1] * len(seq))
            action_mask.append([0] * len(prompt_ids) + [1] * n_resp)
            # old_logprobs aligned to the full sequence (0 on prompt positions).
            old_logprobs.append([0.0] * len(prompt_ids) + list(rollout["response_logprobs"]))
            per_token_rewards.append([0.0] * len(prompt_ids) + [float(reward)] * n_resp)

        batch = {
            "adapter_version": adapter_version,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "action_mask": action_mask,
            "old_logprobs": old_logprobs,
            "rewards": per_token_rewards,
            "client_batch_id": client_batch_id,
        }
        return self.submit_trajectory_batch(batch)

    def save_weights_and_get_sampler(self) -> "LoraTrainingClient":
        """Force this job's current adapter into the inference engine, then return
        a sampler (this same client) guaranteed to route /sample to that adapter.

        Tinker-style: after this returns, ``sample(...)`` uses the trained weights
        instead of falling back to the base model. Requires the job to be resident
        (trained >=1 step and not paged out); raises if not.
        """
        resp = self._service._post(f"/v1/training/jobs/{self.job_id}/sync_weights", {})
        self.synced_version = resp.get("adapter_version")
        return self

    def status(self) -> dict[str, Any]:
        return self._service._get(f"/v1/training/jobs/{self.job_id}")
