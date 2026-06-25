"""Typed contracts for the continuous-batched MultiLoRA training engine.

These are intentionally torch/ray-free so the control plane (controller,
scheduler, packer layout, validation) can be imported and unit-tested on a
CPU-only machine without the GPU stack installed.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, Sequence


class TrainingJobState(str, Enum):
    QUEUED = "queued"
    WAITING_FOR_DATA = "waiting_for_data"
    TRAIN_READY = "train_ready"
    LOADING = "loading"
    HOT_IDLE = "hot_idle"
    ACTIVE_STEP = "active_step"
    PREEMPTING = "preempting"
    COLD_READY = "cold_ready"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = {
    TrainingJobState.COMPLETED,
    TrainingJobState.FAILED,
    TrainingJobState.CANCELLED,
}

# States in which a job is eligible to accrue scheduling credit / be selected.
RUNNABLE_STATES = {
    TrainingJobState.TRAIN_READY,
    TrainingJobState.HOT_IDLE,
    TrainingJobState.COLD_READY,
}


# ---------------------------------------------------------------------------
# Job specification (immutable, client-supplied)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    rank: int
    alpha: int
    target_modules: list[str]
    init: Literal["random", "adapter_uri"] = "random"
    adapter_uri: Optional[str] = None


@dataclass(frozen=True)
class DatasetSpec:
    uri: str | None = None
    format: Literal[
        "prompt_completion_jsonl",
        "prompt_jsonl",
        "external_trajectory_batches",
    ] = "external_trajectory_batches"
    prompt_key: str = "prompt"
    completion_key: str | None = "completion"


@dataclass(frozen=True)
class LossSpec:
    type: Literal["sft", "dpo", "grpo", "ppo"]
    clip_eps: float = 0.2
    kl_coef: float = 0.0
    kl_ref: Literal["base", "provided", "none"] = "base"
    advantage_source: Literal[
        "provided",
        "trainer_from_rewards",
        "trainer_from_group_rewards",
    ] = "provided"


@dataclass(frozen=True)
class OptimizerSpec:
    type: Literal["adamw"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class BudgetSpec:
    max_steps: int | None = None
    max_train_tokens: int | None = None
    tokens_per_update: int = 8192
    checkpoint_every_steps: int = 25
    publish_every_steps: int = 1
    max_policy_lag: int = 4


@dataclass(frozen=True)
class SchedulingSpec:
    priority: int = 1
    preemptible: bool = True
    min_hot_steps: int = 2
    max_consecutive_steps: int = 8
    min_tokens_per_train_quantum: int = 2048


@dataclass(frozen=True)
class TrainingJobSpec:
    job_id: str
    user_id: str
    base_model: str
    base_model_revision: str | None
    adapter: AdapterSpec
    dataset: DatasetSpec
    loss: LossSpec
    optimizer: OptimizerSpec
    budget: BudgetSpec
    scheduling: SchedulingSpec
    output_uri: str


# ---------------------------------------------------------------------------
# Mutable per-job runtime state (engine-owned)
# ---------------------------------------------------------------------------


@dataclass
class TrainingJobRuntime:
    spec: TrainingJobSpec
    state: TrainingJobState = TrainingJobState.QUEUED

    # Physical hot slot; None when cold/not loaded.
    slot: int | None = None
    hot_since_engine_step: int | None = None

    # Progress.
    trained_steps: int = 0
    trained_tokens: int = 0
    current_adapter_version: int = 0
    latest_adapter_uri: str | None = None

    # Queue / accounting.
    ready_train_tokens: int = 0
    ready_batch_ids: list[str] = field(default_factory=list)
    deficit_tokens: int = 0
    consecutive_steps: int = 0

    # Checkpoint / offload.
    cold_checkpoint_uri: str | None = None
    dirty_since_publish: bool = False

    # Error / reporting.
    last_error: str | None = None

    @property
    def job_id(self) -> str:
        return self.spec.job_id

    def is_runnable(self) -> bool:
        return self.state in RUNNABLE_STATES and self.ready_train_tokens > 0


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class ExternalTrajectoryBatch:
    """Tokenized RL data submitted by an external rollout/inference system."""

    job_id: str
    adapter_version: int

    # Compatibility fields; validated against hashes stored at job creation.
    base_model_hash: str | None = None
    tokenizer_hash: str | None = None
    lora_config_hash: str | None = None

    # Shape [num_sequences, seq_len] (tensors) or ragged list[list[...]].
    input_ids: Any = None
    attention_mask: Any = None
    action_mask: Any = None

    # Behavior-policy logprobs from external inference.
    old_logprobs: Any = None

    # RL supervision (optional depending on advantage_source).
    rewards: Any | None = None
    advantages: Any | None = None
    returns: Any | None = None
    group_ids: list[str] | None = None
    ref_logprobs: Any | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainExample:
    """Internal per-sequence training example.

    Both the SFT dataset worker and external RL ingestion normalize to this so
    the packer does not care about the data source.
    """

    job_id: str
    slot: int | None
    loss_type: Literal["sft", "grpo", "ppo", "dpo"]
    adapter_version: int | None

    input_ids: list[int]
    attention_mask: list[int]
    loss_mask: list[int]

    # SFT.
    labels: list[int] | None = None

    # RL.
    old_logprobs: list[float] | None = None
    rewards: list[float] | None = None
    advantages: list[float] | None = None
    returns: list[float] | None = None
    ref_logprobs: list[float] | None = None
    group_id: str | None = None


# ---------------------------------------------------------------------------
# Helpers + validation
# ---------------------------------------------------------------------------


def new_job_id(prefix: str = "job") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def make_batch_id(job_id: str) -> str:
    return f"{job_id}:{uuid.uuid4().hex[:12]}:{int(time.time() * 1000)}"


def lora_config_hash(adapter: AdapterSpec) -> str:
    """Stable hash of the parts of a LoRA config that affect compatibility."""
    import hashlib

    payload = "|".join(
        [
            str(adapter.rank),
            str(adapter.alpha),
            ",".join(sorted(adapter.target_modules)),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _shape(value: Any) -> tuple[int, ...]:
    """Best-effort shape for tensors, numpy arrays, or nested python lists."""
    if value is None:
        return ()
    shape_attr = getattr(value, "shape", None)
    if shape_attr is not None:
        return tuple(int(d) for d in shape_attr)
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        first = value[0]
        if isinstance(first, (list, tuple)):
            return (len(value), len(first))
        return (len(value),)
    raise TypeError(f"cannot determine shape of {type(value)!r}")


def count_action_tokens(action_mask: Any) -> int:
    """Sum of an action/loss mask given as tensor, array, or nested lists."""
    if action_mask is None:
        return 0
    sum_attr = getattr(action_mask, "sum", None)
    if sum_attr is not None and not isinstance(action_mask, (list, tuple)):
        return int(sum_attr())
    total = 0
    for row in action_mask:
        if isinstance(row, (list, tuple)):
            total += int(sum(row))
        else:
            total += int(row)
    return total


class TrajectoryValidationError(ValueError):
    """Raised when an external trajectory batch fails validation."""


def validate_trajectory_batch(job: TrainingJobRuntime, batch: ExternalTrajectoryBatch) -> None:
    """Enforce the disaggregated-RL ingestion contract (plan §5, §17)."""
    spec = job.spec

    if batch.job_id != spec.job_id:
        raise TrajectoryValidationError(
            f"batch.job_id {batch.job_id!r} != job {spec.job_id!r}"
        )

    # Policy-lag window. A batch cannot come from a future adapter version, and
    # cannot be staler than max_policy_lag versions behind the current one.
    if batch.adapter_version > job.current_adapter_version:
        raise TrajectoryValidationError(
            f"future adapter_version {batch.adapter_version} > "
            f"current {job.current_adapter_version}"
        )
    lag = job.current_adapter_version - batch.adapter_version
    if lag > spec.budget.max_policy_lag:
        raise TrajectoryValidationError(
            f"policy lag {lag} exceeds max_policy_lag {spec.budget.max_policy_lag}"
        )

    # Compatibility hashes (only checked when both sides provide them).
    expected_lora_hash = lora_config_hash(spec.adapter)
    if batch.lora_config_hash is not None and batch.lora_config_hash != expected_lora_hash:
        raise TrajectoryValidationError(
            f"lora_config_hash mismatch: {batch.lora_config_hash} != {expected_lora_hash}"
        )

    # Shapes.
    ids_shape = _shape(batch.input_ids)
    attn_shape = _shape(batch.attention_mask)
    action_shape = _shape(batch.action_mask)
    if not (ids_shape == attn_shape == action_shape):
        raise TrajectoryValidationError(
            f"input_ids/attention_mask/action_mask shapes differ: "
            f"{ids_shape} / {attn_shape} / {action_shape}"
        )
    old_shape = _shape(batch.old_logprobs)
    if old_shape != action_shape:
        raise TrajectoryValidationError(
            f"old_logprobs shape {old_shape} != action_mask shape {action_shape}"
        )

    if count_action_tokens(batch.action_mask) <= 0:
        raise TrajectoryValidationError("action_mask must select at least one token")


def validate_sequence_lengths(example: TrainExample) -> None:
    n = len(example.input_ids)
    for name in ("attention_mask", "loss_mask"):
        value = getattr(example, name)
        if len(value) != n:
            raise ValueError(
                f"TrainExample.{name} length {len(value)} != input_ids length {n}"
            )
    if example.loss_type == "sft":
        if example.labels is None or len(example.labels) != n:
            raise ValueError("SFT TrainExample requires labels matching input_ids length")
    else:
        if example.old_logprobs is None or len(example.old_logprobs) != n:
            raise ValueError("RL TrainExample requires old_logprobs matching input_ids length")


def sum_loss_mask(examples: Sequence[TrainExample]) -> int:
    return int(sum(int(sum(ex.loss_mask)) for ex in examples))
