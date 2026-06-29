"""Immutable per-step plan emitted by the coordinator and executed by workers.

A ``TrainStepPlan`` fully describes one all-or-nothing Megatron training quantum:
which jobs train, in which slots, with what data, plus the slot onloads/preemptions
to perform first. Workers receive the *same* plan and must not mutate global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .schemas import LossSpec


@dataclass(frozen=True)
class SlotPreemption:
    job_id: str
    slot: int
    checkpoint_uri: str
    # Unified eviction export: persist only if the adapter changed since its last
    # disk write (dirty). When True the worker writes the training checkpoint AND
    # (if inference_uri set) the HF-PEFT adapter, from one slot snapshot.
    persist: bool = True
    inference_uri: str | None = None
    rank: int = 0
    alpha: int = 0
    target_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlotOnload:
    job_id: str
    slot: int
    source_uri: str | None
    rank: int
    alpha: int
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class BatchLease:
    batch_id: str
    job_id: str
    token_count: int
    payload_ref: Any  # Ray ObjectRef (or the payload itself in CPU tests)
    adapter_version: int


@dataclass(frozen=True)
class ExportRequest:
    """Tells a worker to export a slot's adapter to a versioned artifact dir."""

    job_id: str
    slot: int
    output_uri: str
    rank: int
    alpha: int
    version: int


@dataclass(frozen=True)
class TrainStepPlan:
    plan_id: str
    engine_step: int
    loss_type: Literal["sft", "grpo", "ppo", "dpo"]

    selected_jobs: tuple[str, ...]
    target_tokens: dict[str, int]
    loss_weights: dict[str, float]
    job_to_loss: dict[str, LossSpec]

    job_to_slot: dict[str, int]
    job_to_optimizer_step: dict[str, int]
    job_to_latest_published_version: dict[str, int]

    preemptions: tuple[SlotPreemption, ...] = ()
    onloads: tuple[SlotOnload, ...] = ()
    leases: dict[str, tuple[BatchLease, ...]] = field(default_factory=dict)

    publish_after_step: tuple[str, ...] = ()
    exports: tuple[ExportRequest, ...] = ()

    # Diagnostic: how many jobs were runnable at build time (>= selected_jobs).
    # Logged per step to compare scheduler supply vs. what actually batched.
    num_runnable_jobs: int = 0
