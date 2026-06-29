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


def _rank_tag() -> str:
    # Include the data-parallel rank: with the distributed optimizer (always on),
    # optimizer state is sharded across DP, so each DP rank owns a distinct shard
    # and must write to a distinct file. Without the dp tag, DP>1 ranks would
    # clobber each other's preemption checkpoints at the same path.
    dp_rank = mpu.get_data_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    return f"tp{tp_rank}_pp{pp_rank}_dp{dp_rank}"


class CheckpointStore:
    def write_training_checkpoint_to(
        self,
        uri: str,
        adapter_state: dict[str, Any],
        optimizer_state: dict[str, Any],
    ) -> str:
        """Write this rank's adapter + optimizer shard to an explicit URI.

        The coordinator supplies the URI in ``SlotPreemption.checkpoint_uri``, so
        the worker doesn't need the job object.
        """
        out_dir = Path(uri)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"adapter_megatron": adapter_state, "optimizer": optimizer_state}
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
