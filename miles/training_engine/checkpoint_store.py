"""Internal training-checkpoint store for preemption/resume (Phase 3).

Stores per-rank adapter + optimizer state so a logical job can be evicted from
a hot slot and later resumed (possibly in a different slot). Runs inside the
Megatron training process; URIs are treated as local directories in the MVP
(swap for an object store in Phase 4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot
from megatron.core import mpu

from .schemas import TrainingJobRuntime


def _rank_tag() -> str:
    return f"tp{mpu.get_tensor_model_parallel_rank()}_pp{mpu.get_pipeline_model_parallel_rank()}"


class CheckpointStore:
    def write_training_checkpoint(
        self,
        job: TrainingJobRuntime,
        adapter_state: dict[str, Any],
        optimizer_state: dict[str, Any],
    ) -> str:
        version = job.current_adapter_version
        uri = f"{job.spec.output_uri.rstrip('/')}/internal/checkpoints/v{version:06d}"
        out_dir = Path(uri)
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "job_id": job.spec.job_id,
            "adapter_version": version,
            "rank": job.spec.adapter.rank,
            "alpha": job.spec.adapter.alpha,
            "target_modules": list(job.spec.adapter.target_modules),
            "adapter_megatron": adapter_state,
            "optimizer": optimizer_state,
            "trained_steps": job.trained_steps,
            "trained_tokens": job.trained_tokens,
        }
        torch.save(payload, out_dir / f"checkpoint_{_rank_tag()}.pt")
        return uri

    def load_training_checkpoint(self, uri: str) -> dict[str, Any]:
        path = Path(uri) / f"checkpoint_{_rank_tag()}.pt"
        if not path.exists():
            raise FileNotFoundError(f"training checkpoint not found: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_adapter_init(self, adapter_uri: str) -> dict[str, Any]:
        """Load an external adapter (Megatron shard) for warm-start init."""
        shard = Path(adapter_uri) / f"adapter_megatron_{_rank_tag()}.pt"
        if not shard.exists():
            raise FileNotFoundError(
                f"no Megatron adapter shard for this rank under {adapter_uri} ({shard.name})"
            )
        return {"adapter_megatron": torch.load(shard, map_location="cpu", weights_only=False)}

    def extract_megatron_adapter_state(self, *, model, slot: int) -> dict[str, Any]:
        """Capture this rank's adapter tensors for ``slot`` (CPU clones)."""
        state: dict[str, Any] = {}
        with expose_adapter_slot(model, slot):
            chunks = model if isinstance(model, list) else [model]
            for chunk in chunks:
                for name, param in chunk.named_parameters():
                    if ".adapter." in name:
                        state[name] = param.data.detach().cpu().clone()
        return state
