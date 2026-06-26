"""Shared CPU-only factories for training-engine tests (no torch/ray)."""

from __future__ import annotations

from miles.training_engine.schemas import (
    AdapterSpec,
    BudgetSpec,
    DatasetSpec,
    Execution,
    Lifecycle,
    LossSpec,
    OptimizerSpec,
    Readiness,
    Residency,
    SchedulingSpec,
    TrainExample,
    TrainingJobRuntime,
    TrainingJobSpec,
)


def make_spec(
    job_id: str,
    *,
    base_model: str = "test-model",
    loss_type: str = "sft",
    priority: int = 1,
    preemptible: bool = True,
    rank: int = 8,
    alpha: int = 16,
    tokens_per_update: int = 10**9,
    min_tokens_per_train_quantum: int = 1,
    max_consecutive_steps: int = 10**6,
    min_hot_steps: int = 0,
    max_policy_lag: int = 4,
    max_steps: int | None = None,
    publish_every_steps: int = 1,
    output_uri: str | None = None,
) -> TrainingJobSpec:
    return TrainingJobSpec(
        job_id=job_id,
        user_id="u",
        base_model=base_model,
        base_model_revision=None,
        adapter=AdapterSpec(name=job_id, rank=rank, alpha=alpha, target_modules=["q_proj"]),
        dataset=DatasetSpec(format="external_trajectory_batches"),
        loss=LossSpec(type=loss_type),
        optimizer=OptimizerSpec(),
        budget=BudgetSpec(
            tokens_per_update=tokens_per_update,
            max_policy_lag=max_policy_lag,
            max_steps=max_steps,
            publish_every_steps=publish_every_steps,
        ),
        scheduling=SchedulingSpec(
            priority=priority,
            preemptible=preemptible,
            min_hot_steps=min_hot_steps,
            max_consecutive_steps=max_consecutive_steps,
            min_tokens_per_train_quantum=min_tokens_per_train_quantum,
        ),
        output_uri=output_uri or f"/tmp/{job_id}",
    )


def make_runtime(
    job_id: str,
    *,
    ready_train_tokens: int = 0,
    residency: Residency = Residency.COLD,
    readiness: Readiness = Readiness.READY,
    execution: Execution = Execution.IDLE,
    lifecycle: Lifecycle = Lifecycle.RUNNING,
    slot: int | None = None,
    **spec_kwargs,
) -> TrainingJobRuntime:
    rt = TrainingJobRuntime(spec=make_spec(job_id, **spec_kwargs))
    rt.ready_train_tokens = ready_train_tokens
    rt.residency = residency
    rt.readiness = readiness
    rt.execution = execution
    rt.lifecycle = lifecycle
    rt.slot = slot
    return rt


def make_example(
    job_id: str,
    slot: int,
    *,
    n: int,
    loss_type: str = "sft",
    n_loss_tokens: int | None = None,
) -> TrainExample:
    loss_tokens = n if n_loss_tokens is None else n_loss_tokens
    loss_mask = [0] * (n - loss_tokens) + [1] * loss_tokens
    kwargs = dict(
        job_id=job_id,
        slot=slot,
        loss_type=loss_type,
        adapter_version=0,
        input_ids=list(range(n)),
        attention_mask=[1] * n,
        loss_mask=loss_mask,
    )
    if loss_type == "sft":
        kwargs["labels"] = [(1 if loss_mask[i] else -100) for i in range(n)]
    else:
        kwargs["old_logprobs"] = [0.0] * n
    return TrainExample(**kwargs)
