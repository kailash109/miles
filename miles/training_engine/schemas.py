"""Typed contracts for the continuous-batched MultiLoRA training engine.


"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, Sequence


# ---------------------------------------------------------------------------
# Orthogonal job state
# ---------------------------------------------------------------------------


class Lifecycle(str, Enum):
    RUNNING = "running"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Residency(str, Enum):
    COLD = "cold"
    LOADING = "loading"
    HOT = "hot"
    PREEMPTING = "preempting"


class Readiness(str, Enum):
    EMPTY = "empty"
    READY = "ready"
    LEASED = "leased"


class Execution(str, Enum):
    IDLE = "idle"
    ACTIVE_STEP = "active_step"


TERMINAL_LIFECYCLES = {Lifecycle.COMPLETED, Lifecycle.FAILED, Lifecycle.CANCELLED}


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
# Engine policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchingPolicy:
    max_train_tokens_per_step: int = 8192
    max_adapters_per_step: int = 8
    base_quantum_tokens: int = 2048
    min_tokens_per_job: int = 512
    max_batch_wait_s: float = 0.25
    cost_metric: Literal["loss_tokens", "sequence_tokens"] = "loss_tokens"
    idle_sleep_s: float = 0.01
    step_timeout_s: float = 1800.0


@dataclass(frozen=True)
class QueueLimits:
    max_jobs: int = 1024
    max_ready_tokens_per_job: int = 2_000_000
    max_ready_tokens_global: int = 50_000_000
    max_batches_per_submit: int = 1024


# ---------------------------------------------------------------------------
# Mutable per-job runtime state (coordinator-owned)
# ---------------------------------------------------------------------------


@dataclass
class TrainingJobRuntime:
    spec: TrainingJobSpec

    lifecycle: Lifecycle = Lifecycle.RUNNING
    residency: Residency = Residency.COLD
    readiness: Readiness = Readiness.EMPTY
    execution: Execution = Execution.IDLE

    slot: int | None = None
    ready_train_tokens: int = 0
    leased_train_tokens: int = 0

    deficit_tokens: int = 0
    consecutive_steps: int = 0
    hot_since_engine_step: int | None = None

    trained_steps: int = 0
    trained_tokens: int = 0

    # Internal counter (advances on a committed step) vs the *public* version,
    # which only advances once a durable artifact manifest is READY.
    optimizer_step: int = 0
    latest_materialized_step: int = 0
    latest_published_version: int = 0
    latest_adapter_uri: str | None = None
    pending_publish_steps: set[int] = field(default_factory=set)

    cold_checkpoint_uri: str | None = None
    last_ready_at: float | None = None
    last_error: str | None = None
    last_loss: float | None = None
    # optimizer_step last persisted to disk (training ckpt + HF-PEFT) on eviction
    # (-1 = never). Skips rewriting an unchanged adapter.
    persisted_step: int = -1

    @property
    def job_id(self) -> str:
        return self.spec.job_id


def is_runnable(job: TrainingJobRuntime) -> bool:
    return (
        job.lifecycle == Lifecycle.RUNNING
        and job.readiness == Readiness.READY
        and job.execution == Execution.IDLE
        and job.ready_train_tokens >= job.spec.scheduling.min_tokens_per_train_quantum
    )


def can_preempt(job: TrainingJobRuntime, engine_step: int) -> bool:
    if job.lifecycle != Lifecycle.RUNNING:
        return False
    if job.residency != Residency.HOT or job.slot is None:
        return False
    if job.execution != Execution.IDLE:
        return False
    if not job.spec.scheduling.preemptible:
        return False
    if job.hot_since_engine_step is not None:
        if engine_step - job.hot_since_engine_step < job.spec.scheduling.min_hot_steps:
            return False
    return True


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class ExternalTrajectoryBatch:
    """Tokenized RL data submitted by an external rollout/inference system."""

    job_id: str
    adapter_version: int

    base_model_hash: str | None = None
    tokenizer_hash: str | None = None
    lora_config_hash: str | None = None

    input_ids: Any = None
    attention_mask: Any = None
    action_mask: Any = None
    old_logprobs: Any = None

    rewards: Any | None = None
    advantages: Any | None = None
    returns: Any | None = None
    group_ids: list[str] | None = None
    ref_logprobs: Any | None = None

    # Makes submissions idempotent across client retries.
    client_batch_id: str | None = None


@dataclass
class PromptBatch:
    """Prompt-only submission for online RL.

    The client supplies only tokenized prompts; the engine generates rollouts
    with the job's current adapter (via sglang), returns them to the client to
    score, and the client resubmits the scored rollouts as an
    ``ExternalTrajectoryBatch``. Reward computation stays client-side.
    """

    job_id: str
    prompts: list[list[int]]  # tokenized prompt token ids, one list per prompt
    sampling_params: dict[str, Any] = field(default_factory=dict)
    n_samples_per_prompt: int = 1


@dataclass
class TrainExample:
    """Internal per-sequence training example (SFT and RL normalize to this)."""

    job_id: str
    slot: int | None
    loss_type: Literal["sft", "grpo", "ppo", "dpo"]
    adapter_version: int | None

    input_ids: list[int]
    attention_mask: list[int]
    loss_mask: list[int]

    # Next-token targets (-100 at ignored positions / sequence end). Used by SFT
    # cross-entropy and by RL current-policy logprob gathering.
    labels: list[int] | None = None

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


def new_plan_id() -> str:
    return f"plan_{uuid.uuid4().hex[:12]}"


def lora_config_hash(adapter: AdapterSpec) -> str:
    import hashlib

    payload = "|".join(
        [str(adapter.rank), str(adapter.alpha), ",".join(sorted(adapter.target_modules))]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _shape(value: Any) -> tuple[int, ...]:
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
    if action_mask is None:
        return 0
    sum_attr = getattr(action_mask, "sum", None)
    if sum_attr is not None and not isinstance(action_mask, (list, tuple)):
        return int(sum_attr())
    total = 0
    for row in action_mask:
        total += int(sum(row)) if isinstance(row, (list, tuple)) else int(row)
    return total


class TrajectoryValidationError(ValueError):
    """Raised when an external trajectory batch fails validation."""


def validate_trajectory_batch(job: TrainingJobRuntime, batch: ExternalTrajectoryBatch) -> None:
    # support disagg inference pipeline
    spec = job.spec

    if batch.job_id != spec.job_id:
        raise TrajectoryValidationError(f"batch.job_id {batch.job_id!r} != job {spec.job_id!r}")

    # Validate against the public/published version, not the internal step.
    if batch.adapter_version > job.latest_published_version:
        raise TrajectoryValidationError(
            f"future or unpublished adapter_version {batch.adapter_version} > "
            f"published {job.latest_published_version}"
        )
    lag = job.latest_published_version - batch.adapter_version
    if lag > spec.budget.max_policy_lag:
        raise TrajectoryValidationError(
            f"policy lag {lag} exceeds max_policy_lag {spec.budget.max_policy_lag}"
        )

    expected_lora_hash = lora_config_hash(spec.adapter)
    if batch.lora_config_hash is not None and batch.lora_config_hash != expected_lora_hash:
        raise TrajectoryValidationError(
            f"lora_config_hash mismatch: {batch.lora_config_hash} != {expected_lora_hash}"
        )

    ids_shape = _shape(batch.input_ids)
    attn_shape = _shape(batch.attention_mask)
    action_shape = _shape(batch.action_mask)
    if not (ids_shape == attn_shape == action_shape):
        raise TrajectoryValidationError(
            f"input_ids/attention_mask/action_mask shapes differ: "
            f"{ids_shape} / {attn_shape} / {action_shape}"
        )
    if _shape(batch.old_logprobs) != action_shape:
        raise TrajectoryValidationError(
            f"old_logprobs shape {_shape(batch.old_logprobs)} != action_mask {action_shape}"
        )
    if count_action_tokens(batch.action_mask) <= 0:
        raise TrajectoryValidationError("action_mask must select at least one token")


def validate_sequence_lengths(example: TrainExample) -> None:
    n = len(example.input_ids)
    for name in ("attention_mask", "loss_mask"):
        if len(getattr(example, name)) != n:
            raise ValueError(f"TrainExample.{name} length != input_ids length {n}")
    if example.labels is not None and len(example.labels) != n:
        raise ValueError("TrainExample.labels length != input_ids length")


@dataclass(frozen=True)
class SelectedJob:
    job_id: str
    target_tokens: int
