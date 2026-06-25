"""Lightweight in-memory event log for the training engine.

Control-plane only (no torch/ray). Production can replace ``EventLog`` with a
streaming/persistent sink without changing emit sites.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Event names (plan §21).
JOB_CREATED = "job.created"
JOB_QUEUED = "job.queued"
JOB_LOADED_TO_SLOT = "job.loaded_to_slot"
JOB_TRAIN_STEP_STARTED = "job.train_step_started"
JOB_TRAIN_STEP_FINISHED = "job.train_step_finished"
JOB_ADAPTER_PUBLISHED = "job.adapter_published"
JOB_PREEMPTED = "job.preempted"
JOB_RESUMED = "job.resumed"
JOB_COMPLETED = "job.completed"
JOB_FAILED = "job.failed"


@dataclass(frozen=True)
class Event:
    name: str
    job_id: str | None
    ts: float
    data: dict[str, Any] = field(default_factory=dict)


class EventLog:
    def __init__(self, max_events: int = 100_000):
        self._events: list[Event] = []
        self._max_events = max_events

    def emit(self, name: str, job_id: str | None = None, **data: Any) -> Event:
        event = Event(name=name, job_id=job_id, ts=time.time(), data=data)
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]
        return event

    def for_job(self, job_id: str, *, after_ts: float = 0.0) -> list[Event]:
        return [e for e in self._events if e.job_id == job_id and e.ts > after_ts]

    def all(self) -> list[Event]:
        return list(self._events)
