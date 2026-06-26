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
    def __init__(self, args, model, optimizer):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_store = CheckpointStore()

    def prepare_slots(self, plan: TrainStepPlan) -> None:
        for op in plan.preemptions:
            self.checkpoint_and_clear(op)
        for op in plan.onloads:
            self.load(op)

    def load(self, op: SlotOnload) -> None:
        init_adapter_slot(self.model, op.slot, rank=op.rank, alpha=op.alpha)
        if op.source_uri:
            state = self.checkpoint_store.load_training_checkpoint(op.source_uri)
            if "adapter_megatron" in state:
                load_adapter(self.model, op.slot, state["adapter_megatron"])
            if "optimizer" in state:
                restore_optimizer_state_for_adapter(
                    self.optimizer, self.model, op.slot, state["optimizer"]
                )
        self.optimizer.reload_model_params()

    def checkpoint_and_clear(self, op: SlotPreemption) -> None:
        adapter_state = self.checkpoint_store.extract_megatron_adapter_state(
            model=self.model, slot=op.slot
        )
        optimizer_state = capture_optimizer_state_for_adapter(self.optimizer, self.model, op.slot)
        self.checkpoint_store.write_training_checkpoint_to(
            op.checkpoint_uri, adapter_state=adapter_state, optimizer_state=optimizer_state
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
