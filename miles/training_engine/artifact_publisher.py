"""Publish immutable, HF/PEFT-compatible adapter artifacts per version.

In external-rollout mode this replaces any direct SGLang weight push: the
trainer only emits artifacts, and external inference systems decide when to load
them. Reuses the existing ``save_multi_lora_checkpoints`` exporter so we do not
reimplement Megatron-Bridge adapter export. Runs inside the GPU training process.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from megatron.core import mpu

from miles.backends.megatron_utils.update_weight.multi_lora_sync import (
    save_multi_lora_checkpoints,
)
from miles.utils.adapter_config import AdapterConfig

from .schemas import TrainingJobRuntime, lora_config_hash


class AdapterArtifactPublisher:
    def __init__(self, args):
        self.args = args

    def publish(self, job: TrainingJobRuntime, model) -> str:
        """Export the job's current adapter slot to a versioned artifact dir.

        Layout (reusing the existing writer)::

            {output_uri}/checkpoints/step_{version}/
              adapter_megatron_tp{tp}_pp{pp}.pt
              adapter_model.safetensors
              adapter_config.json
              metadata.json   (added here)
        """
        assert job.slot is not None, "cannot publish a job with no hot slot"
        version = job.current_adapter_version
        output_root = Path(job.spec.output_uri)

        config = AdapterConfig(
            name=job.spec.adapter.name,
            rank=job.spec.adapter.rank,
            alpha=job.spec.adapter.alpha,
            data="",
            dir=output_root,
            slot=job.slot,
        )
        save_multi_lora_checkpoints(self.args, model, version, {job.spec.adapter.name: config})

        version_dir = output_root / "checkpoints" / f"step_{version}"
        self._write_metadata(job, version_dir)
        return str(version_dir)

    def _write_metadata(self, job: TrainingJobRuntime, version_dir: Path) -> None:
        is_writer = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
            and mpu.get_data_parallel_rank() == 0
        )
        if not is_writer:
            return
        version_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "job_id": job.spec.job_id,
            "adapter_version": job.current_adapter_version,
            "base_model": job.spec.base_model,
            "base_model_revision": job.spec.base_model_revision,
            "lora_config_hash": lora_config_hash(job.spec.adapter),
            "rank": job.spec.adapter.rank,
            "alpha": job.spec.adapter.alpha,
            "target_modules": list(job.spec.adapter.target_modules),
            "trained_steps": job.trained_steps,
            "trained_tokens": job.trained_tokens,
        }
        (version_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
