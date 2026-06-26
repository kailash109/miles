"""Result/response value types exchanged between client, coordinator, and workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkerStepResult:
    """Returned by each Megatron worker after executing a TrainStepPlan."""

    plan_id: str
    rank: int
    ok: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    # Per-job adapter files written this step (rank-local), for the publisher.
    written_files: dict[str, list[str]] = field(default_factory=dict)
    error: str | None = None


@dataclass
class CommitResult:
    ok: bool
    error: str | None = None
    published: list[dict] = field(default_factory=list)  # AdapterReady events


@dataclass
class CreateJobResponse:
    job_id: str
    adapter_version: int
    adapter_uri: str | None


@dataclass
class SubmitBatchResponse:
    accepted: bool
    batch_id: str | None = None
    token_count: int = 0
    reason: str | None = None


@dataclass
class WorkerHealth:
    rank: int
    current_plan_id: str | None
    state: str
