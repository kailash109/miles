"""Map many logical jobs onto fewer physical MultiLoRA slots (Phase 3).

The pager performs the GPU-affecting slot transitions: load a job's adapter +
optimizer state into a slot, preempt (save + clear) a job, and complete a job.
Runs inside the Megatron training process.
"""

from __future__ import annotations

import ray
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

from .schemas import TrainingJobRuntime


class AdapterPager:
    def __init__(self, args, model, optimizer, controller, scheduler, checkpoint_store):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        # Always a Ray actor handle (see make_training_job_controller).
        self.controller = controller
        self.scheduler = scheduler
        self.checkpoint_store = checkpoint_store

    def ensure_hot(self, job: TrainingJobRuntime, all_jobs: dict[str, TrainingJobRuntime]) -> int:
        if job.slot is not None:
            return job.slot

        slot = ray.get(self.controller.reserve_free_slot.remote(job.spec.job_id))
        if slot is None:
            victim = self._choose_victim(job, all_jobs)
            if victim is None:
                raise RuntimeError("no free slot and no preemptible victim")
            self.preempt(victim)
            slot = ray.get(self.controller.reserve_free_slot.remote(job.spec.job_id))
            assert slot is not None, "slot still unavailable after preemption"

        self._load_job_into_slot(job, slot)
        ray.get(self.controller.mark_hot.remote(job.spec.job_id, slot))
        return slot

    def preempt(self, job: TrainingJobRuntime) -> str:
        assert job.slot is not None
        checkpoint_uri = self._save_hot_job_checkpoint(job)
        self._clear_slot(job.slot)
        ray.get(self.controller.mark_cold.remote(job.spec.job_id, checkpoint_uri))
        return checkpoint_uri

    # -- GPU operations ------------------------------------------------------

    def _load_job_into_slot(self, job: TrainingJobRuntime, slot: int) -> None:
        init_adapter_slot(
            self.model,
            slot,
            rank=job.spec.adapter.rank,
            alpha=job.spec.adapter.alpha,
        )

        state = None
        if job.cold_checkpoint_uri is not None:
            state = self.checkpoint_store.load_training_checkpoint(job.cold_checkpoint_uri)
        elif job.spec.adapter.init == "adapter_uri" and job.spec.adapter.adapter_uri:
            state = self.checkpoint_store.load_adapter_init(job.spec.adapter.adapter_uri)

        if state is not None:
            if "adapter_megatron" in state:
                load_adapter(self.model, slot, state["adapter_megatron"])
            if "optimizer" in state:
                restore_optimizer_state_for_adapter(
                    self.optimizer, self.model, slot, state["optimizer"]
                )

        self.optimizer.reload_model_params()

    def _save_hot_job_checkpoint(self, job: TrainingJobRuntime) -> str:
        adapter_state = self.checkpoint_store.extract_megatron_adapter_state(
            model=self.model,
            slot=job.slot,
        )
        optimizer_state = capture_optimizer_state_for_adapter(self.optimizer, self.model, job.slot)
        return self.checkpoint_store.write_training_checkpoint(
            job=job,
            adapter_state=adapter_state,
            optimizer_state=optimizer_state,
        )

    def _clear_slot(self, slot: int) -> None:
        clear_adapter_slot(self.model, slot)
        zero_optimizer_state_for_adapter(self.optimizer, self.model, slot)
        self.optimizer.reload_model_params()

    def _choose_victim(self, incoming_job, all_jobs):
        hot = [j for j in all_jobs.values() if j.slot is not None]
        engine_step = max((j.trained_steps for j in all_jobs.values()), default=0)
        return self.scheduler.choose_preemption_victim(
            hot_jobs=hot,
            incoming_job=incoming_job,
            engine_step=engine_step,
        )
