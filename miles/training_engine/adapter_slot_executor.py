"""Worker-local slot executor: applies a plan's slot ops and exports adapters.

Unlike the old policyful pager, this has no scheduling or victim-selection logic.
It only *executes* the immutable plan: preempt (checkpoint + clear) victim slots,
load incoming slots, and export dirty adapters for publishing. Runs inside the
Megatron training process.
"""

from __future__ import annotations

from pathlib import Path

from megatron.bridge.peft.multi_lora_layers import (
    clear_adapter_slot,
    init_adapter_slot,
    load_adapter,
)

from miles.backends.megatron_utils.multi_lora import (
    capture_optimizer_state_for_adapter,
    restore_optimizer_state_for_adapter,
    zero_optimizer_state_for_adapter,
)
from miles.backends.megatron_utils.update_weight.multi_lora_sync import save_multi_lora_checkpoints
from miles.utils.adapter_config import AdapterConfig

from .checkpoint_store import CheckpointStore
from .plan import SlotOnload, SlotPreemption, TrainStepPlan


class AdapterSlotExecutor:
    def __init__(self, args, model, optimizer, *, writer=None, hf_iterator=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_store = CheckpointStore()
        # Inference persistence (generation mode only): the coalescing writer and
        # the HF weight iterator used to export a slot to HF-PEFT.
        self.writer = writer
        self.hf_iterator = hf_iterator

    def prepare_slots(self, plan: TrainStepPlan) -> None:
        for op in plan.preemptions:
            self.checkpoint_and_clear(op)
        for op in plan.onloads:
            self.load(op)

    def load(self, op: SlotOnload) -> None:
        init_adapter_slot(self.model, op.slot, rank=op.rank, alpha=op.alpha)
        if op.source_uri:
            # Ensure any in-flight inference write for this job has flushed so the
            # training checkpoint we read back is complete/consistent.
            if self.writer is not None:
                self.writer.flush(op.job_id)
            state = self.checkpoint_store.load_training_checkpoint(op.source_uri)
            if "adapter_megatron" in state:
                load_adapter(self.model, op.slot, state["adapter_megatron"])
            if "optimizer" in state:
                restore_optimizer_state_for_adapter(
                    self.optimizer, self.model, op.slot, state["optimizer"]
                )
        self.optimizer.reload_model_params()

    def checkpoint_and_clear(self, op: SlotPreemption) -> None:
        # Single eviction export, gated by the coordinator's dirty check: persist
        # only if the adapter changed since its last disk write.
        if op.persist:
            adapter_state = self.checkpoint_store.extract_megatron_adapter_state(
                model=self.model, slot=op.slot
            )
            optimizer_state = capture_optimizer_state_for_adapter(self.optimizer, self.model, op.slot)
            # Training checkpoint (Megatron + optimizer) is written synchronously —
            # training resume depends on it being durable.
            self.checkpoint_store.write_training_checkpoint_to(
                op.checkpoint_uri, adapter_state=adapter_state, optimizer_state=optimizer_state
            )
            # Inference HF-PEFT: snapshot now (slot still live), write async.
            if op.inference_uri and self.writer is not None and self.hf_iterator is not None:
                from .adapter_export import capture_slot_hf_cpu

                hf_tensors = capture_slot_hf_cpu(self.hf_iterator, self.model, op.slot, op.rank)
                self.writer.submit(
                    op.job_id,
                    op.inference_uri,
                    hf_tensors,
                    rank=op.rank,
                    alpha=op.alpha,
                    target_modules=op.target_modules,
                )
        clear_adapter_slot(self.model, op.slot)
        zero_optimizer_state_for_adapter(self.optimizer, self.model, op.slot)
        self.optimizer.reload_model_params()

    def write_dirty_adapter_files(self, plan: TrainStepPlan) -> dict[str, list[str]]:
        """Export each ``plan.exports`` slot to its versioned dir; return files per job."""
        written: dict[str, list[str]] = {}
        for ex in plan.exports:
            config = AdapterConfig(
                name=ex.job_id, rank=ex.rank, alpha=ex.alpha, data="", dir=Path(ex.output_uri), slot=ex.slot
            )
            save_multi_lora_checkpoints(self.args, self.model, ex.version, {ex.job_id: config})
            step_dir = Path(ex.output_uri) / "checkpoints" / f"step_{ex.version}"
            written[ex.job_id] = sorted(str(p) for p in step_dir.rglob("*") if p.is_file())
        return written
